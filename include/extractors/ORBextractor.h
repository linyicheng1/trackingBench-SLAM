#ifndef TRACKING_BENCH_ORBEXTRACTOR_H
#define TRACKING_BENCH_ORBEXTRACTOR_H
#include <vector>
#include <list>
#include <opencv/cv.h>

namespace TRACKING_BENCH
{
    class ExtractorNode
    {
    public:
        ExtractorNode():bNoMore(false){}

        void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

        std::vector<cv::KeyPoint> vKeys;
        std::vector<cv::KeyPoint> vExitKeys;
        cv::Point2i UL, UR, BL, BR;
        std::list<ExtractorNode>::iterator lit;
        bool bNoMore;
    };

    class ORBExtractor
    {
    public:
        enum
        {
            HARRIS_SCORE = 0, FAST_SCORE = 1
        };

        ORBExtractor();

        ~ORBExtractor() = default;

        // Compute the ORB features and descriptors on an image.
        // ORB are dispersed on the image using an octree.
        // Mask is ignored in the current implementation.
        void operator()(std::vector<cv::Mat> &images,
                        std::vector<float> mvScaleFactor,
                        int targetNum,
                        float initTh,
                        float minTH,
                        std::vector<cv::KeyPoint> &keypoints,
                        cv::Mat& descriptors);
        void AddPoints(std::vector<cv::Mat>& images,
                       std::vector<float>& mvScaleFactor,
                       int targetNum,
                       float initTh,
                       float minTH,
                       const std::vector<cv::KeyPoint> &exitPoints,
                       std::vector<cv::KeyPoint> &newPoints,
                       cv::OutputArray &descriptors);



        std::vector<float> inline GetScaleSigmaSquares()
        {
            return mvLevelSigma2;
        }

        std::vector<float> inline GetInverseScaleSigmaSquares()
        {
            return mvInvLevelSigma2;
        }

    protected:


        void ComputeKeyPointsOctTree(std::vector<cv::Mat>& images,
                                     std::vector<float>& mvScaleFactor,
                                     int targetNum,
                                     float initTh,
                                     float minTH,
                                     std::vector<std::vector<cv::KeyPoint> > &allKeypoints,
                                    const std::vector<cv::KeyPoint>& exitKeyPoints);

        std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys,const std::vector<cv::KeyPoint>& exitKeys,
                                                    const int &minX,const int &maxX, const int &minY, const int &maxY,
                                                    const int &nFeatures, const int &level);


        std::vector<cv::Point> pattern;


        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;
        std::vector<int> mnFeaturesPerLevel;

        std::vector<int> umax;


    };
}
#endif //TRACKING_BENCH_ORBEXTRACTOR_H
