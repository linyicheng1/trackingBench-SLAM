#include "matchers/matcher.h"
#include "opencv2/opencv.hpp"

void test_match_des()
{
    cv::Mat img1 = cv::imread("/home/lyc/code/slam_bench/MH_05_difficult/mav0/cam0/data/1403638541627829504.png");
    cv::Mat img2 = cv::imread("/home/lyc/code/slam_bench/MH_05_difficult/mav0/cam1/data/1403638541627829504.png");

    cv::imshow("img1", img1);
    cv::imshow("img2", img2);
    cv::waitKey(0);
}

int main()
{
    test_match_des();
    return 0;
}
