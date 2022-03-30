#ifndef TRACKING_BENCH_ORBEXTRACTOR_H
#define TRACKING_BENCH_ORBEXTRACTOR_H
#include "extractor.h"

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

    class ORBextractor:public Extractor
    {
    public:
        enum
        {
            HARRIS_SCORE = 0, FAST_SCORE = 1
        };

        ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                     int iniThFAST, int minThFAST);

        ~ORBextractor() = default;

        // Compute the ORB features and descriptors on an image.
        // ORB are dispersed on the image using an octree.
        // Mask is ignored in the current implementation.
        void operator()(cv::InputArray image, cv::InputArray mask,
                        std::vector<cv::KeyPoint> &keypoints,
                        cv::OutputArray descriptors) override;
        void AddPoints(cv::InputArray image, const std::vector<cv::KeyPoint> &exitPoints,
                               std::vector<cv::KeyPoint> &newPoints, cv::OutputArray &descriptors) override;



        std::vector<float> inline GetScaleSigmaSquares()
        {
            return mvLevelSigma2;
        }

        std::vector<float> inline GetInverseScaleSigmaSquares()
        {
            return mvInvLevelSigma2;
        }

    protected:


        void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> > &allKeypoints, const std::vector<cv::KeyPoint>& exitKeyPoints);

        std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys,const std::vector<cv::KeyPoint>& exitKeys,
                                                    const int &minX,const int &maxX, const int &minY, const int &maxY,
                                                    const int &nFeatures, const int &level);

        void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> > &allKeypoints);

        std::vector<cv::Point> pattern;



        int iniThFAST;
        int minThFAST;

        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;
        std::vector<int> mnFeaturesPerLevel;

        std::vector<int> umax;


    };
}
#endif //TRACKING_BENCH_ORBEXTRACTOR_H
