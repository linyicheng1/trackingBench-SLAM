#include "matchers/matcher.h"
#include "opencv2/opencv.hpp"
#include "camera/CameraModel.h"
#include "types/Frame.h"
//#include "extractors/FASTextractor.h"
#include "extractors/ORBextractor.h"
#include "types/Map.h"
#include "types/MapPoint.h"
#include "mapping/LocalBA.h"
#include <memory>
#include <Eigen/Geometry>

using namespace TRACKING_BENCH;


void test_match_des()
{
    // 472
    cv::Mat img1 = cv::imread("/home/lyc/code/slam_bench/MH_05_difficult/mav0/cam0/data/1403638541627829504.png", CV_8UC1);
    cv::Mat img2 = cv::imread("/home/lyc/code/slam_bench/MH_05_difficult/mav0/cam1/data/1403638541627829504.png", CV_8UC1);
    // 513
    cv::Mat img3 = cv::imread("/home/lyc/code/slam_bench/MH_05_difficult/mav0/cam0/data/1403638543677829376.png", CV_8UC1);

    auto camera_ptr = std::make_shared<PinholeCamera>(img1.cols, img1.rows,458.654, 457.296, 367.215, 248.375);
    auto frame1_ptr = std::make_shared<Frame>(img1, 0, 5, 0.8, camera_ptr);
    auto frame2_ptr = std::make_shared<Frame>(img2, 0, 5, 0.8, camera_ptr);
    auto frame3_ptr = std::make_shared<Frame>(img3, 0, 5, 0.8, camera_ptr);

    auto extractor_ptr = std::make_shared<ORBExtractor>();
    std::vector<cv::KeyPoint> keypoints1,keypoints2,keypoints3;
    cv::Mat descriptors1, descriptors2, descriptors3;

    extractor_ptr->operator()(frame1_ptr->GetImagePyramid(),
                              frame1_ptr->GetScaleFactors(),
                              1000,
                              80,
                              30,
                              keypoints1,
                              descriptors1);

    frame1_ptr->SetKeys(keypoints1, frame1_ptr, descriptors1);
    cv::drawKeypoints(img1, keypoints1, img1);

    extractor_ptr->operator()(frame2_ptr->GetImagePyramid(),
                              frame2_ptr->GetScaleFactors(),
                              1000,
                              80,
                              30,
                              keypoints2,
                              descriptors2);

    frame2_ptr->SetKeys(keypoints2, frame2_ptr, descriptors2);
    cv::drawKeypoints(img2, keypoints2, img2);

    extractor_ptr->operator()(frame3_ptr->GetImagePyramid(),
                              frame3_ptr->GetScaleFactors(),
                              1000,
                              80,
                              30,
                              keypoints3,
                              descriptors3);

    frame3_ptr->SetKeys(keypoints3, frame3_ptr, descriptors3);
    cv::drawKeypoints(img3, keypoints3, img3);

    auto matcher_ptr = std::make_shared<Matcher>();
    // OpenCV
    auto matches = matcher_ptr->searchByBF(frame1_ptr, frame2_ptr, 0, 5, 10, 30);
    // Violence
//    frame2_ptr->AssignFeaturesToGrid();
//    matcher_ptr->setViolenceParam(30, 100, 30, true, 5);
//    auto matches = matcher_ptr->searchByViolence(frame1_ptr, frame2_ptr, 0, 5, 50);

    // Dow
//    DBoW2::BowVector bowVector1, bowVector2;
//    DBoW2::FeatureVector featureVector1, featureVector2;
//    auto vocabulary = std::make_shared<ORBVocabulary>();
//    bool load = vocabulary->loadFromTextFile("/home/lyc/code/slam_bench/trackingBench-SLAM/Vocabulary/ORBvoc.txt");
//    if(!load)
//    {
//        cerr <<"Failed to open vocabulary "<<std::endl;
//    }
//
//    frame1_ptr->SetBow(vocabulary);
//    frame2_ptr->SetBow(vocabulary);
//    matcher_ptr->setBowParam(30, 100, 30, true, 5);
//    auto matches = matcher_ptr->searchByBow(frame1_ptr, frame2_ptr);
    // optical flow
//    std::vector<cv::Point2f> pts;
//    auto matches = matcher_ptr->searchByOPFlow(frame1_ptr, frame2_ptr, pts, true, true);
//    keypoints1.resize(pts.size());
//    for (int i = 0; i < pts.size();i ++)
//    {
//        keypoints1.at(i).pt = pts.at(i);
//    }

    // projection
    auto map_ptr = std::make_shared<Map>();
    Eigen::Matrix4f t_bs0, t_bs1, t2;
    Eigen::Quaternionf q0{0.238726, 4.461980,-1.680894,0.578817};
    Eigen::Vector3f x0{-0.758387,-0.347718,-0.496942};

    t_bs0 << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
          0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
          -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
          0.0, 0.0, 0.0, 1.0;
    t_bs1 << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
          0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
          -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
          0.0, 0.0, 0.0, 1.0;


    frame1_ptr->SetPose(Eigen::Matrix4f::Identity());
    frame2_ptr->SetPose(t_bs0.inverse() * t_bs1);
    frame3_ptr->SetPose(t2);
    LocalBA localBa;
    for (auto &match:matches)
    {
        Eigen::Vector3f pos = localBa.LinearTriangle(
                frame1_ptr->GetKey(match.queryIdx)->px_un,
                frame2_ptr->GetKey(match.trainIdx)->px_un,
                frame1_ptr->GetPose(),
                frame2_ptr->GetPose()
                );
        // add map point
        auto pt_ptr = std::make_shared<MapPoint>(pos, map_ptr, frame1_ptr, frame1_ptr->GetKey(match.queryIdx));
        map_ptr->AddMapPoint(pt_ptr);
        frame1_ptr->AddMapPoint(pt_ptr, match.queryIdx);
    }
//    matcher_ptr->setProjectionParam(30, 100, 30, true, 5);
    matches = matcher_ptr->searchByProjection(frame1_ptr, frame2_ptr);

    cv::Mat show;
    cv::drawMatches(img1,keypoints1, img3, keypoints3, matches, show);

    cv::imshow("img1", img1);
    cv::imshow("img2", img2);
    cv::imshow("img3", img3);
    cv::imshow("show", show);
    cv::waitKey(0);
}

int main()
{
    test_match_des();
    return 0;
}
