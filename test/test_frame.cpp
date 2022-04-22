#include "types/Frame.h"
#include "opencv2/opencv.hpp"
#include "extractors/extractor.h"
#include "extractors/ORBextractor.h"
#include "extractors/FASTextractor.h"

using namespace TRACKING_BENCH;

void test_extractor()
{
    cv::Mat gray;
    // gray
    gray = cv::imread("../data/1.jpg", CV_8UC1);
    Extractor* extractor =new ORBextractor(1000, 1.2, 8, 20, 7);
    Distortion distortion;
    Frame frame(gray, 0, extractor, &distortion);

    cv::Mat show = gray.clone();
    auto pts = frame.getKeysUn();
    cv::drawKeypoints(gray, pts, show);
    cv::imshow("img", show);

    cv::Mat show2 = gray.clone();
    auto pts2 = frame.getKeysUn();
    std::vector<cv::KeyPoint> pts_part;
    pts_part.resize(pts2.size()/5 - 1);
    for(int i = 0;i < pts2.size()/5 - 1; i++)
    {
        pts_part.at(i) = pts2.at(5*i);
    }
    cv::drawKeypoints(gray, pts_part, show2);
    cv::imshow("img-2", show2);

    std::vector<cv::KeyPoint> pts_new;
    cv::Mat des;
    extractor->AddPoints(gray, pts_part, pts_new, des);
    for(auto & i : pts_new)
    {
        pts_part.emplace_back(i);
    }
    std::cout<<"finally size: "<<pts_part.size()<<" target size: "<<pts.size()<<" add size: "<<pts_new.size()<<std::endl;
    cv::Mat show3 = gray.clone();
    cv::drawKeypoints(gray, pts_new, show3);
    cv::imshow("img-3", show3);
    cv::waitKey(0);
}

//void test_fast()
//{
//    cv::Mat gray;
//    // gray
//    gray = cv::imread("../data/1.jpg", CV_8UC1);
//    Extractor* extractor =new FASTextractor(gray.cols, gray.rows, 20, 5, 30);
//    Distortion distortion;
//    Frame frame(gray, 0, extractor, &distortion);
//
//    cv::Mat show = gray.clone();
//    auto pts = frame.getKeysUn();
//    cv::drawKeypoints(gray, pts, show);
//    cv::imshow("img", show);
//
//    cv::Mat show2 = gray.clone();
//    auto pts2 = frame.getKeysUn();
//    std::vector<cv::KeyPoint> pts_part;
//    pts_part.resize(pts2.size()/5 - 1);
//    for(int i = 0;i < pts2.size()/5 - 1; i++)
//    {
//        pts_part.at(i) = pts2.at(5*i);
//    }
//    cv::drawKeypoints(gray, pts_part, show2);
//    cv::imshow("img-2", show2);
//
//    std::vector<cv::KeyPoint> pts_new;
//    cv::Mat des;
//    extractor->AddPoints(gray, pts_part, pts_new, des);
//    for(auto & i : pts_new)
//    {
//        pts_part.emplace_back(i);
//    }
//    std::cout<<"finally size: "<<pts_part.size()<<" target size: "<<pts.size()<<" add size: "<<pts_new.size()<<std::endl;
//    cv::Mat show3 = gray.clone();
//    cv::drawKeypoints(gray, pts_new, show3);
//    cv::imshow("img-3", show3);
//    cv::waitKey(0);
//}

int main()
{
    test_extractor();
//    test_fast();
}

