#include "extractors/FASTextractor.h"

namespace TRACKING_BENCH
{

    FASTExtractor::FASTExtractor() = default;

    void FASTExtractor::operator()(std::vector<cv::Mat>& images,
            std::vector<float>& invScaleFactor,
            int nFeatures,
            float threshold,
            std::vector<cv::KeyPoint> &keyPoints,
            cv::OutputArray descriptors,
            bool reset)
    {
        if(images.empty() || images.at(0).empty())
            return;
        const int cell_size = (int)sqrtf((float)images.at(0).cols *  (float)images.at(0).rows / (float)nFeatures);
        const int grid_n_cols = (int)((float)images.at(0).cols / (float)cell_size);
        const int grid_n_rows = (int)((float)images.at(0).rows / (float)cell_size);

        if(reset)
        {
            grid_occupancy_.resize(grid_n_cols*grid_n_rows, false);
        }

        assert(images.at(0).type() == CV_8UC1 );

        keyPoints.clear();
        std::vector<cv::KeyPoint> gridPoints;
        gridPoints.resize(nFeatures);

        for(int L=0; L<images.size(); ++L)
        {
            std::vector<fast::fast_xy> fast_corners;
#if __SSE2__
            fast::fast_corner_detect_10_sse2(
                    (fast::fast_byte*) images.at(L).data, images.at(L).cols,
                    images.at(L).rows, images.at(L).cols, 20, fast_corners);
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
            fast::fast_corner_score_10((fast::fast_byte*) images.at(L).data, images.at(L).cols, fast_corners, 20, scores);
            fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

            for(int & nm_corner : nm_corners)
            {
                fast::fast_xy& xy = fast_corners.at(nm_corner);
                const int k = static_cast<int>(((float)xy.y*invScaleFactor.at(L))/(float)cell_size)*grid_n_cols
                              + static_cast<int>(((float)xy.x*invScaleFactor.at(L))/(float)cell_size);
                // cell is occupancy
                if(grid_occupancy_[k])
                    continue;
                const float score = shiTomasiScore(images.at(L), xy.x, xy.y);
                if(score > gridPoints.at(k).response)
                {
                    gridPoints.at(k) = cv::KeyPoint((float)xy.x * invScaleFactor.at(L),
                                                      (float)xy.y * invScaleFactor.at(L),
                                                      1,0,score,L);
                }
            }
        }

        // Create feature for every corner that has high enough corner score
        std::for_each(gridPoints.begin(), gridPoints.end(), [&](cv::KeyPoint& c) {
            if(c.response > (float)threshold)
            {
                keyPoints.emplace_back(c);
            }
        });

        resetGrid();
    }

    void FASTExtractor::resetGrid()
    {
        std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
    }

    float FASTExtractor::shiTomasiScore(const cv::Mat &img, int u, int v)
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

        const int stride = (int)img.step.p[0];
        for( int y=y_min; y<y_max; ++y )
        {
            const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
            const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
            const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
            const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
            for(int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom)
            {
                float dx = (float)*ptr_right - (float)*ptr_left;
                float dy = (float)*ptr_bottom - (float)*ptr_top;
                dXX += dx*dx;
                dYY += dy*dy;
                dXY += dx*dy;
            }
        }

        // Find and return smaller eigenvalue:
        dXX = dXX / (2.f * box_area);
        dYY = dYY / (2.f * box_area);
        dXY = dXY / (2.f * box_area);
        return 0.5f * (dXX + dYY - sqrtf( (dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY) ));
    }

    void FASTExtractor::AddPoints(std::vector<cv::Mat>& images,
                                  std::vector<float>& mvScaleFactor,
                                  int nFeatures,
                                  float threshold,
                                  const std::vector<cv::KeyPoint> &exitPoints,
                                  std::vector<cv::KeyPoint> &newPoints,
                                  cv::OutputArray &descriptors)
    {
        const int cell_size = (int)sqrtf((float)images.at(0).cols *  (float)images.at(0).rows / (float)nFeatures);
        const int grid_n_cols = (int)((float)images.at(0).cols / (float)cell_size);
        const int grid_n_rows = (int)((float)images.at(0).rows / (float)cell_size);
        grid_occupancy_.resize(grid_n_cols*grid_n_rows, false);

        for(const auto& i:exitPoints)
        {
            grid_occupancy_.at(
                    static_cast<int>(i.pt.y/(float)cell_size)*grid_n_cols
                    + static_cast<int>(i.pt.x/(float)cell_size)) = true;
        }

        operator()(images, mvScaleFactor, nFeatures, threshold, newPoints, descriptors, false);
    }
}

