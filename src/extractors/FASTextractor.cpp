#include "extractors/FASTextractor.h"

namespace TRACKING_BENCH
{

    FASTextractor::FASTextractor(const int img_width, const int img_height, const int cell_size,const int n_pyr_levels, const int threshold):
    Extractor((int)(img_width/cell_size + 1) * (int)(img_height/cell_size + 1),2,n_pyr_levels),m_cell_size(cell_size),m_img_height(img_height),m_img_width(img_width),
    grid_n_cols_(ceil(static_cast<double>(img_width)/cell_size)),
    grid_n_rows_(ceil(static_cast<double>(img_height)/cell_size)),
    grid_occupancy_(grid_n_cols_*grid_n_rows_, false),
    detection_threshold(threshold)
    {
    }

    void FASTextractor::operator()(cv::InputArray _image, cv::InputArray mask,
                                   std::vector<cv::KeyPoint> &_keypoints,cv::OutputArray descriptors)
    {
        if(_image.empty())
            return;

        cv::Mat image = _image.getMat();
        assert(image.type() == CV_8UC1 );

        // Pre-compute the scale pyramid
        ComputePyramid(image);
        _keypoints.clear();
        std::vector<cv::KeyPoint> gridPoints;
        gridPoints.resize(nFeatures);

        for(int L=0; L<nLevels; ++L)
        {
            const int scale = (1<<L);
            std::vector<fast::fast_xy> fast_corners;
#if __SSE2__
            fast::fast_corner_detect_10_sse2(
                    (fast::fast_byte*) mvImagePyramid.at(L).data, mvImagePyramid.at(L).cols,
                    mvImagePyramid.at(L).rows, mvImagePyramid.at(L).cols, 20, fast_corners);
#elif HAVE_FAST_NEON
            fast::fast_corner_detect_9_neon(
          (fast::fast_byte*) mvImagePyramid.at(L).data, mvImagePyramid.at(L).cols,
          mvImagePyramid.at(L).rows, mvImagePyramid.at(L).cols, 20, fast_corners);
#else
      fast::fast_corner_detect_10(
          (fast::fast_byte*) mvImagePyramid.at(L).data, mvImagePyramid.at(L).cols,
          mvImagePyramid.at(L).rows, mvImagePyramid.at(L).cols, 20, fast_corners);
#endif
            std::vector<int> scores, nm_corners;
            fast::fast_corner_score_10((fast::fast_byte*) mvImagePyramid.at(L).data, mvImagePyramid.at(L).cols, fast_corners, 20, scores);
            fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

            for(auto it=nm_corners.begin(), ite=nm_corners.end(); it!=ite; ++it)
            {
                fast::fast_xy& xy = fast_corners.at(*it);
                const int k = static_cast<int>((xy.y*scale)/m_cell_size)*grid_n_cols_
                              + static_cast<int>((xy.x*scale)/m_cell_size);
                if(grid_occupancy_[k])
                    continue;
                const float score = shiTomasiScore(mvImagePyramid.at(L), xy.x, xy.y);
                if(score > gridPoints.at(k).response)
                    gridPoints.at(k) = cv::KeyPoint(xy.x * mvScaleFactor.at(L), xy.y * mvScaleFactor.at(L),1,0,score,nLevels);
            }
        }

        // Create feature for every corner that has high enough corner score
        std::for_each(gridPoints.begin(), gridPoints.end(), [&](cv::KeyPoint& c) {
            if(c.response > detection_threshold)
                _keypoints.emplace_back(c);
        });

        resetGrid();
    }

    void FASTextractor::resetGrid()
    {
        std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
    }

    void FASTextractor::setGridOccpuancy(const cv::Point2d &px)
    {
        grid_occupancy_.at(
                static_cast<int>(px.y/m_cell_size)*grid_n_cols_
                + static_cast<int>(px.x/m_cell_size)) = true;
    }

    void FASTextractor::setExistingFeatures(const std::vector<cv::KeyPoint> &fts)
    {
        for(auto i:fts)
        {
            grid_occupancy_.at(
                    static_cast<int>(i.pt.y/m_cell_size)*grid_n_cols_
                    + static_cast<int>(i.pt.x/m_cell_size)) = true;
        }
    }

    float FASTextractor::shiTomasiScore(const cv::Mat &img, int u, int v)
    {
        assert(img.type() == CV_8UC1);

        float dXX = 0.0;
        float dYY = 0.0;
        float dXY = 0.0;
        const int halfbox_size = 4;
        const int box_size = 2*halfbox_size;
        const int box_area = box_size*box_size;
        const int x_min = u-halfbox_size;
        const int x_max = u+halfbox_size;
        const int y_min = v-halfbox_size;
        const int y_max = v+halfbox_size;

        if(x_min < 1 || x_max >= img.cols-1 || y_min < 1 || y_max >= img.rows-1)
            return 0.0; // patch is too close to the boundary

        const int stride = img.step.p[0];
        for( int y=y_min; y<y_max; ++y )
        {
            const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
            const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
            const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
            const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
            for(int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom)
            {
                float dx = *ptr_right - *ptr_left;
                float dy = *ptr_bottom - *ptr_top;
                dXX += dx*dx;
                dYY += dy*dy;
                dXY += dx*dy;
            }
        }

        // Find and return smaller eigenvalue:
        dXX = dXX / (2.0 * box_area);
        dYY = dYY / (2.0 * box_area);
        dXY = dXY / (2.0 * box_area);
        return 0.5 * (dXX + dYY - sqrt( (dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY) ));
    }

    void FASTextractor::AddPoints(const cv::_InputArray &image, const std::vector<cv::KeyPoint> &exitPoints,
                                  std::vector<cv::KeyPoint> &newPoints, const cv::_OutputArray &descriptors)
    {
        setExistingFeatures(exitPoints);
        (*this)(image, cv::Mat(), newPoints, descriptors);
    }
}

