#ifndef TRACKING_BENCH_FASTEXTRACTOR_H
#define TRACKING_BENCH_FASTEXTRACTOR_H
#include <vector>
#include <list>
#include <opencv/cv.h>
#include "third_part/fast_lib/include/fast/fast.h"

namespace TRACKING_BENCH
{

    class FASTExtractor
    {
    public:
        FASTExtractor();

        ~FASTExtractor() = default;
        // detection
        void operator()(std::vector<cv::Mat>& images,
                        std::vector<float>& mvScaleFactor,
                        int targetNum,
                        float threshold,
                        std::vector<cv::KeyPoint> &keypoints,
                        cv::OutputArray descriptors,
                        bool reset = true);

        void AddPoints(std::vector<cv::Mat>& images,
                       std::vector<float>& mvScaleFactor,
                       int targetNum,
                       float threshold,
                       const std::vector<cv::KeyPoint> &exitPoints,
                       std::vector<cv::KeyPoint> &newPoints,
                       cv::OutputArray &descriptors);

        void resetGrid();


    private:

        std::vector<bool> grid_occupancy_;
        static float shiTomasiScore(const cv::Mat& img, int u, int v);
    };

}

#endif //TRACKING_BENCH_FASTEXTRACTOR_H
