#ifndef TRACKING_BENCH_MATCHER_H
#define TRACKING_BENCH_MATCHER_H

#include <vector>
#include "opencv2/features2d.hpp"
#include <memory>

namespace TRACKING_BENCH
{
    class Frame;
    class MapPoint;
    class Map;
    class KeyFrame;

    class Matcher
    {
    public:
        Matcher();
        ~Matcher() = default;
        //  match parameter
        int TH_LOW = 50;
        int TH_HIGH = 100;
        int HISTO_LENGTH = 30;
        bool checkOrientation = true;
        float nRatio{};

        // F1 current frame
        // OpenCV NN
        void setNNParam(int high, float ratio){TH_HIGH = high; nRatio = ratio;}
        std::vector<cv::DMatch> searchByNN(Frame* F1, Frame* F2, int MinLevel, int MaxLevel, bool MapPointOnly = false);
        std::vector<cv::DMatch> searchByNN(Map* map, Frame* F1, bool Projection = true);
        // OpenCV BF
        std::vector<cv::DMatch> searchByBF(Frame* F1, Frame* F2, bool MapPointOnly = false);
        std::vector<cv::DMatch> searchByBF(Map* map, Frame* F1);
        // Violence
        void setViolenceParam(int low, int high, int length, bool check, float ratio){TH_LOW = low; TH_HIGH = high; HISTO_LENGTH = length;checkOrientation=check;nRatio=ratio;}
        std::vector<cv::DMatch> searchByViolence(Frame* F1, Frame* F2, bool MapPointOnly = false);
        std::vector<cv::DMatch> searchByViolence(Map* map, Frame* F1);

        // Projection + Feature Point
        std::vector<cv::DMatch> searchByProjection(Frame* F1, Frame* F2, bool MapPointOnly = false);
        std::vector<cv::DMatch> searchByProjection(Map* map, Frame* F1);

        // Projection + Feature Alignment
        std::vector<cv::DMatch> searchByFeatureAlignment(Frame* F1, Frame* F2, bool MapPointOnly = false);
        std::vector<cv::DMatch> searchByFeatureAlignment(Map* map, Frame* F1);
        // Bow accelerate ORB only
        std::vector<cv::DMatch> searchByBow(Frame* F1, Frame* F2, bool MapPointOnly = false);
        std::vector<cv::DMatch> searchByBow(Map* map, Frame* F2);
        // Optical flow
        std::vector<cv::DMatch> searchByOPFlow(Frame* F1, Frame* F2, bool MapPointOnly = false);
        std::vector<cv::DMatch> searchByOPFlow(Map* map, Frame* F2);
        // Direct
        std::vector<cv::DMatch> searchByDirect(Frame* F1, Frame* F2, bool MapPointOnly = false, bool ph = false);
        std::vector<cv::DMatch> searchByDirect(Map* M, Frame* F1);

    private:
        cv::Ptr<cv::FlannBasedMatcher> m_flann_matcher;
        std::shared_ptr<cv::BFMatcher> m_bf_matcher;
        static int descriptorDistance(const cv::Mat& a, const cv::Mat& b);
        static void computeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
    };
}

#endif //TRACKING_BENCH_MATCHER_H
