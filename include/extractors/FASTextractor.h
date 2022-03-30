#ifndef TRACKING_BENCH_FASTEXTRACTOR_H
#define TRACKING_BENCH_FASTEXTRACTOR_H
#include <vector>
#include "extractor.h"
#include "third_part/fast_lib/include/fast/fast.h"

namespace TRACKING_BENCH
{
    class FASTextractor : public Extractor
    {
    public:
        FASTextractor(const int img_width,
                const int img_height,
                const int cell_size,
                const int n_pyr_levels,
                const int threshold);
        ~FASTextractor() = default;
        void operator()(cv::InputArray image, cv::InputArray mask,
                        std::vector<cv::KeyPoint> &keypoints,
                        cv::OutputArray descriptors) override;
        void AddPoints(cv::InputArray image, const std::vector<cv::KeyPoint> &exitPoints,
                       std::vector<cv::KeyPoint> &newPoints, cv::OutputArray &descriptors) override;
        void resetGrid();
        inline int getCellIndex(int x, int y, int level)
        {
            const int scale = (1<<level);
            return (scale*y)/m_cell_size*grid_n_cols_ + (scale*x)/m_cell_size;
        }
        /// Flag the grid cell as occupied
        void setGridOccpuancy(const cv::Point2d& px);
        /// Set grid cells of existing features as occupied
        void setExistingFeatures(const std::vector<cv::KeyPoint>& fts);

    private:
        int m_img_width;
        int m_img_height;
        int m_cell_size;

        const int grid_n_cols_;
        const int grid_n_rows_;
        std::vector<bool> grid_occupancy_;
        float detection_threshold;
        float shiTomasiScore(const cv::Mat& img, int u, int v);
    };

}

#endif //TRACKING_BENCH_FASTEXTRACTOR_H
