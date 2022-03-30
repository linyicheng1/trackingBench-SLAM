#ifndef TRACKING_BENCH_EXTRACTOR_H
#define TRACKING_BENCH_EXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>

namespace TRACKING_BENCH
{

    const int EDGE_THRESHOLD = 19;
    class Extractor
    {
    public:
        explicit Extractor(int nfeatures, float scaleFactor, int nlevels);
        ~Extractor() = default;
        virtual void AddPoints(cv::InputArray image, const std::vector<cv::KeyPoint> &exitPoints,
                               std::vector<cv::KeyPoint> &newPoints, cv::OutputArray &descriptors){}
        virtual void operator()(cv::InputArray image, cv::InputArray mask,
                        std::vector<cv::KeyPoint> &keypoints,
                        cv::OutputArray descriptors){}

        // base common functions
        int inline GetLevels() const
        {
            return nLevels;
        }
        float inline GetScaleFactor()
        {
            return scaleFactor;
        }

        std::vector<float> inline GetScaleFactors()
        {
            return mvScaleFactor;
        }

        std::vector<float> inline GetInverseScaleFactors()
        {
            return mvInvScaleFactor;
        }
        void ComputePyramid(cv::Mat image);
        int nFeatures;
        int nLevels;
        float scaleFactor;
        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<cv::Mat> mvImagePyramid;
    };
}

#endif //TRACKING_BENCH_EXTRACTOR_H
