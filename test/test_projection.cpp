#include "matchers/matcher.h"
#include "opencv2/opencv.hpp"
#include "camera/CameraModel.h"
#include "types/Frame.h"
//#include "extractors/FASTextractor.h"
#include "extractors/ORBextractor.h"
#include "types/Map.h"
#include "types/MapPoint.h"
#include "mapping/LocalBA.h"
#include "Viewer.h"
#include <memory>
#include <Eigen/Geometry>
#include <thread>
#include <sophus/se3.hpp>
#include <iostream>
#include <stdint.h>
#include <unordered_set>
#include "g2o/stuff/sampler.h"
#include<suitesparse/cholmod.h>

using namespace TRACKING_BENCH;


// simple direct
using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;
// paths
string left_file = "/home/lyc/share/slambook2-master/slambook2-master/ch8/left.png";
string disparity_file = "/home/lyc/code/slam_bench/trackingBench-SLAM/data/disparity.png";


// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

/// class for accumulator jacobians in parallel
class JacobianAccumulator {
public:
    JacobianAccumulator(
            const cv::Mat &img1_,
            const cv::Mat &img2_,
            const VecVector2d &px_ref_,
            const vector<double> depth_ref_,
            Sophus::SE3d &T21_) :
            img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {
        projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
    }

    /// accumulate jacobians in a range
    void accumulate_jacobian(const cv::Range &range);

    /// get hessian matrix
    Matrix6d hessian() const { return H; }

    /// get bias
    Vector6d bias() const { return b; }

    /// get total cost
    double cost_func() const { return cost; }

    /// get projected points
    VecVector2d projected_points() const { return projection; }

    /// reset h, b, cost to zero
    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points

    std::mutex hessian_mutex;
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21
);

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21
);

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
//    std::cout<<"f1: "<<(1 - xx) * (1 - yy)<<" x1: "<<(int)data[0]<<std::endl;
//    std::cout<<"f2: "<<xx * (1 - yy)<<" x2: "<<(int)data[1]<<std::endl;
//    std::cout<<"f3: "<<(1 - xx) * yy<<" x3: "<<(int)data[img.step]<<std::endl;
//    std::cout<<"f4: "<<xx * yy<<" x4: "<<(int)data[img.step + 1]<<std::endl;
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}


void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,//new
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21) {

    const int iterations = 10;
    double cost = 0, lastCost = 0;
    auto t1 = chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++) {
        jaco_accu.reset();
        cv::parallel_for_(cv::Range(0, px_ref.size()),
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);
        T21 = Sophus::SE3d::exp(-update) * T21;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        if (update.norm() < 1e-3) {
            // converge
            break;
        }

        lastCost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }

    cout << "T21 = \n" << T21.matrix() << endl;
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "direct method for single layer: " << time_used.count() << endl;

    // plot the projected pixels here
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i) {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0) {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("current", img2_show);
    cv::waitKey(0);
}

void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) {

    // parameters
    const int half_patch_size = 2;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i < range.end; i++) {

        // compute the projection in the second image
        Eigen::Vector3d point_ref =
                depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
        Eigen::Vector3d point_cur = T21 * point_ref;
        if (point_cur[2] < 0)   // depth invalid
            continue;

        float u = fx * point_cur[0] / point_cur[2] + cx, v = fy * point_cur[1] / point_cur[2] + cy;
        if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size ||
            v > img2.rows - half_patch_size)
            continue;

        projection[i] = Eigen::Vector2d(u, v);
        double X = point_ref[0], Y = point_ref[1], Z = point_ref[2],
                Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
        cnt_good++;

        // and compute error and jacobian
        for (int y = -half_patch_size; y < half_patch_size; y++)
            for (int x = -half_patch_size; x < half_patch_size; x++) {

                double error = GetPixelValue(img2, u + x, v + y) -
                               GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y);

                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                        0.5 * (GetPixelValue(img1, px_ref[i][0] + 1 + x, px_ref[i][1] + y) - GetPixelValue(img1, px_ref[i][0] - 1 + x, px_ref[i][1] + y)),
                        0.5 * (GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + 1 + y) - GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] - 1 + y))
                );

                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
                std::cout<<" H "<<J * J.transpose()<<std::endl;
                std::cout<<" b "<<-error * J<<std::endl;
            }
        std::cout<<"H: "<<hessian<<std::endl;
        std::cout<<"bias: "<<bias<<std::endl;
    }

    if (cnt_good) {
        // set hessian, bias and cost
        unique_lock<mutex> lck(hessian_mutex);
        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good;
    }
    //std::cout<<" mean cost: "<<std::endl;
}

void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21)
{
    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }

}


void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}

void LoadKittiGroundTruth(const string& file, vector<Eigen::Matrix3d>& R, vector<Eigen::Vector3d>& t)
{
    ifstream fTimes;
    fTimes.open(file.c_str());
    R.reserve(1000);
    t.reserve(1000);

    while(!fTimes.eof())
    {
        Eigen::Vector3d t0;
        Eigen::Matrix3d R0;

        string s;
        for (int i = 0; i < 3; i ++)
        {
            for (int j = 0;j < 3; j ++)
            {
                getline(fTimes,s,' ');
                R0(i, j) =  std::atof(s.c_str());
            }
            if (i == 2)
                getline(fTimes,s);
            else
                getline(fTimes,s,' ');
            t0(i) = std::atof(s.c_str());
        }
        R.emplace_back(R0.transpose());
        t.emplace_back(- R0.transpose() * t0);
        //getline(fTimes,s);
    }
}



void test_projection()
{

    // Load Images Paths
    std::vector<string> imageLeft;
    std::vector<string> imageRight;
    std::vector<double> timeStamp;
    LoadImages( "/media/lyc/ 其他/dataset/sequences/00",
                imageLeft, imageRight, timeStamp);
    string left_file = "/home/lyc/code/slam_bench/trackingBench-SLAM/data/left.png";
    string right_file = "/home/lyc/code/slam_bench/trackingBench-SLAM/data/right.png";
    string disparity_file = "/home/lyc/code/slam_bench/trackingBench-SLAM/data/disparity.png";
    // Load ground truth
    cv::Mat disparity = cv::imread(disparity_file, 0);
    std::vector<Eigen::Matrix3d> gtR;
    std::vector<Eigen::Vector3d> gtT;
    LoadKittiGroundTruth("/home/lyc/share/data_odometry_poses/dataset/poses/00.txt",
                         gtR, gtT);

    //
    // 内参
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // 间距
    double d = 0.573;

    auto extractor_ptr = std::make_shared<ORBExtractor>();
    auto local_ba = std::make_shared<LocalBA>();

    std::shared_ptr<Frame> key_frame, last_frame;
    auto matcher_ptr = std::make_shared<Matcher>();
    auto map_ptr = std::make_shared<TRACKING_BENCH::Map>();

    // viewer
    auto viewer = std::make_shared<Viewer>();
    viewer->SetCameraPos(Eigen::Matrix4f::Identity());
    viewer->Run();

    // view gt pos
    std::vector<cv::KeyPoint> keyPoints_ref;
    std::vector<Eigen::Matrix4f> kfs_pos;

    // for images
    for (int i = 0;i < 2;i +=1)
    {
        cv::Mat imLeft, imRight, show;
        std::vector<cv::KeyPoint> keyPoints_r, keyPoints_l;
        cv::Mat descriptors_r, descriptors_l;
        // 0. read image to frame
        imLeft = cv::imread(imageLeft[i], CV_LOAD_IMAGE_UNCHANGED);
        static auto camera_ptr = std::make_shared<PinholeCamera>(imLeft.cols, imLeft.rows,fx, fy, cx, cy);
        cv::Mat out;

        auto cur_frame_ptr = std::make_shared<Frame>(imLeft, 0, 5, 0.6, camera_ptr);
        if(last_frame != nullptr)
        {
            cur_frame_ptr->SetPose(last_frame->GetPose());
        }
        else
        {
            Eigen::Matrix4f pos = Eigen::Matrix4f::Identity();
            pos.block<3, 3>(0, 0) = gtR.at(i).cast<float>();
            pos.block<3, 1>(0 ,3) = gtT.at(i).cast<float>();
            cur_frame_ptr->SetPose(pos);
            viewer->SetCameraPos(cur_frame_ptr->GetPoseInverse());
        }

        // 1. match points
        if (last_frame != nullptr)
        {
            // match with last frame
//            auto matches = matcher_ptr->searchByBow(cur_frame_ptr, cur_frame_ptr, true);
//            auto matches = matcher_ptr->searchByBF(cur_frame_ptr, key_frame, 0, 5, 10, 30);
//            auto matches = matcher_ptr->searchByViolence(cur_frame_ptr, key_frame, 0, 5, 50);
            // matches by optical flow
//            std::vector<cv::Point2f> pts;
//            std::vector<cv::KeyPoint> kps;
//            auto matches = matcher_ptr->searchByOPFlow(cur_frame_ptr, last_frame, pts, true, true);
//            kps.reserve(pts.size());
//            for (auto& pt:pts)
//            {
//                cv::KeyPoint kp;
//                kp.pt = pt;
//                kps.emplace_back(kp);
//            }
//            cur_frame_ptr->SetKeys(kps, cur_frame_ptr);
            // matches by projection frame

//            extractor_ptr->operator()(cur_frame_ptr->GetImagePyramid(),
//                                      cur_frame_ptr->GetScaleFactors(),
//                                      2000,
//                                      80,
//                                      30,
//                                      keyPoints_l,
//                                      descriptors_l);
//
//            cur_frame_ptr->SetKeys(keyPoints_l, cur_frame_ptr, descriptors_l);
//            cur_frame_ptr->AssignFeaturesToGrid();

            Eigen::Matrix4f pos = Eigen::Matrix4f::Identity();
            pos.block<3, 3>(0, 0) = gtR.at(i).cast<float>();
            pos.block<3, 1>(0 ,3) = gtT.at(i).cast<float>();
            pos(2, 3) = -0.85;
            cur_frame_ptr->SetPose(last_frame->GetPose());//pos);//last_frame->GetPose());

//            matcher_ptr->setProjectionParam(30, 50, 30, true, 30);
//            auto matches = matcher_ptr->searchByProjection(cur_frame_ptr, key_frame);

            // matches by projection map
//            matcher_ptr->setProjectionParam(30, 50, 30, true, 20);
//            auto matches = matcher_ptr->searchByProjection(map_ptr, cur_frame_ptr, 0.6);
//
//            // assign map point
//            for (auto &item:matches)
//            {
//                // last frame
////                shared_ptr<MapPoint> mp = last_frame->GetMapPoint(item.trainIdx);
//                // key frame
////                shared_ptr<MapPoint> mp = key_frame->GetMapPoint(item.trainIdx);
//                // map
//                shared_ptr<MapPoint> mp = map_ptr->GetAllMapPoints().at(item.trainIdx);
//                if (mp != nullptr)
//                    cur_frame_ptr->AddMapPoint(mp, item.queryIdx);
//            }
//
//            for (auto& m:matches)
//            {
//                m.trainIdx =  map_ptr->GetAllMapPoints().at(m.trainIdx)->GetReferenceFeature()->idxF;
//            }

            // direct
            matcher_ptr->setDirectParam(4, 0, 20, 20, 0.01);
            auto matches = matcher_ptr->searchByDirect(map_ptr, cur_frame_ptr, last_frame);
            // pnp
            cur_frame_ptr->SetPose(Eigen::Matrix4f::Identity());
            local_ba->PoseOptimization(cur_frame_ptr);
            std::cout<<"current pose Twc: "<<cur_frame_ptr->GetPoseInverse()<<std::endl;
            // feature align
            cv::Mat show2 = cur_frame_ptr->GetImagePyramid()[0].clone();
            cv::cvtColor(show2,show2,CV_GRAY2BGR);
            for (const auto& f:cur_frame_ptr->GetKeys())
            {
                cv::line(show2,
                         cv::Point((int)f->px.x(), (int)f->px.y()),
                         cv::Point((int)f->point->GetReferenceFeature()->px.x(), (int)f->point->GetReferenceFeature()->px.y()),
                         cv::Scalar(255,0,0), 2);
            }
            cv::imshow("image align", show2);
            cv::waitKey(0);

//            cv::drawMatches(imLeft,
//                            keyPoints_l,
//                            key_frame->GetImagePyramid().at(0),
//                            keyPoints_ref, matches, show);
//            cv::imshow("match", show);


            // 2. pose optimization
//            local_ba->PoseOptimization(cur_frame_ptr);

            std::cout<<"current pose Twc: "<<cur_frame_ptr->GetPoseInverse()<<std::endl;
            std::cout<<"ground truth pose Twc: "<<pos.inverse()<<std::endl;
        }


        kfs_pos.emplace_back(cur_frame_ptr->GetPoseInverse());
        viewer->SetKeyFrames(kfs_pos);
        if(i % 10 == 0)
        {// 3. key frame

            extractor_ptr->operator()(cur_frame_ptr->GetImagePyramid(),
                                      cur_frame_ptr->GetScaleFactors(),
                                      2000,
                                      80,
                                      30,
                                      keyPoints_l,
                                      descriptors_l);

            cur_frame_ptr->SetKeys(keyPoints_l, cur_frame_ptr, descriptors_l);
            cur_frame_ptr->AssignFeaturesToGrid();
            imRight = cv::imread(imageRight[i], CV_LOAD_IMAGE_UNCHANGED);
//            cv::imshow("right", imRight);
            static auto right_camera_ptr = std::make_shared<PinholeCamera>(imLeft.cols, imLeft.rows, fx, fy, cx, cy);
//            right_camera_ptr->UndistortImage(imRight, imRight);
            // 4. add map points by stereo
            auto right_frame_ptr = std::make_shared<Frame>(imRight, 0, 5, 0.6, right_camera_ptr);
//            extractor_ptr->operator()(cur_frame_ptr->GetImagePyramid(),
//                                      cur_frame_ptr->GetScaleFactors(),
//                                      2000,
//                                      80,
//                                      30,
//                                      keyPoints_r,
//                                      descriptors_r);
//            right_frame_ptr->SetKeys(keyPoints_r, right_frame_ptr, descriptors_r);

            auto depth = local_ba->AddMapPointsByStereo(cur_frame_ptr, right_frame_ptr, d*fx, fx);
            float sum_err = 0, cnt = 0;
            for (size_t j = 0; j < depth.size(); j ++)
            {
                if(depth[j] > 0)
                {

                    int u = (int)cur_frame_ptr->GetKey(j)->kp.pt.x;
                    int v = (int)cur_frame_ptr->GetKey(j)->kp.pt.y;

                    double disp = disparity.at<uchar>(v, u);
                    disp = (disp)/fx;
                    float err = abs(depth[j]/5 - d/disp);
                    sum_err += err;
                    cnt ++;

                    depth[j] = d/disp;

                    Eigen::Vector3f norm;
                    norm[0] = (u-cx)/fx;
                    norm[1] = (v-cy)/fy;
                    norm[2] = 1;

//                    const auto& pt = cur_frame_ptr->GetKey(j);
//                    norm = camera_ptr->Cam2World(pt->px);

                    const Eigen::Matrix3f& R = cur_frame_ptr->GetRotation();
                    Eigen::Vector3f t = cur_frame_ptr->GetTranslation();
//                    t.x() += 0.5;
                    auto mp = std::make_shared<MapPoint>(R * norm * depth[j] + t, map_ptr, cur_frame_ptr, cur_frame_ptr->GetKey(j), cur_frame_ptr->GetDescriptor(j));

                    cur_frame_ptr->AddMapPoint(mp, j);
                    map_ptr->AddMapPoint(mp);
                    map_ptr->AddKeyFrame(cur_frame_ptr);
                }
            }
            std::cout<<" error: "<<sum_err<<" mean: "<<sum_err/cnt<<" cnt: "<<cnt<<std::endl;
            viewer->SetMapPoints(map_ptr->GetAllMapPoints(), cur_frame_ptr->GetMapPointMatches());
            keyPoints_ref = keyPoints_l;
            key_frame = cur_frame_ptr;
        }
        last_frame = cur_frame_ptr;
        // visualization
        cv::waitKey(2);
    }
    for(int i = 0;i < gtR.size();i += 1)
    {
        Eigen::Matrix4f pos = Eigen::Matrix4f::Identity();
        pos.block<3, 3>(0, 0) = gtR.at(i).cast<float>().transpose();
        pos.block<3, 1>(0 ,3) = -  pos.block<3, 3>(0, 0) * gtT.at(i).cast<float>();
        kfs_pos.emplace_back(pos);
    }
    cv::waitKey(0);
    viewer->RequestFinish();
}


int test_direct()
{
    std::vector<string> imageLeft;
    std::vector<string> imageRight;
    std::vector<double> timeStamp;

    LoadImages( "/media/lyc/ 其他/dataset/sequences/00",
                imageLeft, imageRight, timeStamp);

    cv::Mat left_img = cv::imread(imageLeft[0], 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 1;
    int boarder = 2;
    VecVector2d pixels_ref;
    vector<double> depth_ref;
    for (int i = 0; i < nPoints; i++)
    {
        int x = 130;//rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = 73;//rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }
    // estimates 01~05.png's pose using this information
    Sophus::SE3d T_cur_ref;

    //for (int i = 1; i < 6; i++) {  // 1~10
    cv::Mat img = cv::imread(imageLeft[1], 0);
    // try single layer by uncomment this line
    // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    //}
    return 0;
}

void test_direct2()
{
    std::vector<string> imageLeft;
    std::vector<string> imageRight;
    std::vector<double> timeStamp;

    LoadImages( "/media/lyc/ 其他/dataset/sequences/00",
                imageLeft, imageRight, timeStamp);

    auto matcher_ptr = std::make_shared<Matcher>();

    cv::Mat left_img = cv::imread(imageLeft[0], 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;
    auto map_ptr = std::make_shared<TRACKING_BENCH::Map>();

    auto camera_ptr = std::make_shared<PinholeCamera>(left_img.cols, left_img.rows,fx, fy, cx, cy);
    cv::Mat img = cv::imread(imageLeft[1], 0);

    auto ref_frame_ptr = std::make_shared<Frame>(left_img, 0, 4, 0.5, camera_ptr);
    auto cur_frame_ptr = std::make_shared<Frame>(img, 0, 4, 0.5, camera_ptr);

    std::vector<cv::KeyPoint> kps;
    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++)
    {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
        kps.emplace_back(cv::Point2f((float)x,(float)y),0);
    }
    ref_frame_ptr->SetKeys(kps, ref_frame_ptr);

    for (int i = 0;i < nPoints; i++)
    {
        Eigen::Vector3f norm;
        norm[0] = (pixels_ref.at(i).x() - cx)/fx;
        norm[1] = (pixels_ref.at(i).y() - cy)/fy;
        norm[2] = 1;
        auto mp = std::make_shared<MapPoint>(norm * depth_ref[i], map_ptr, ref_frame_ptr, ref_frame_ptr->GetKey(i), cv::Mat());

        ref_frame_ptr->AddMapPoint(mp, i);
        map_ptr->AddMapPoint(mp);
    }
    ref_frame_ptr->SetPose(Eigen::Matrix4f::Identity());
    Eigen::Matrix4f init_gauss = Eigen::Matrix4f::Identity();
    cur_frame_ptr->SetPose(init_gauss);

    matcher_ptr->setDirectParam(3, 0, 10, 5, 0.0001);
    auto T_cur_ref = matcher_ptr->SparseImageAlign(cur_frame_ptr, ref_frame_ptr);
    std::cout<<"T: "<<T_cur_ref<<std::endl;

}


void test_image_align()
{
    std::vector<string> imageLeft;
    std::vector<string> imageRight;
    std::vector<double> timeStamp;

    LoadImages( "/media/lyc/ 其他/dataset/sequences/00",
                imageLeft, imageRight, timeStamp);

    auto matcher_ptr = std::make_shared<Matcher>();

    cv::Mat left_img = cv::imread(imageLeft[0], 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 1;
    int boarder = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;
    auto map_ptr = std::make_shared<TRACKING_BENCH::Map>();

    auto camera_ptr = std::make_shared<PinholeCamera>(left_img.cols, left_img.rows,fx, fy, cx, cy);
    cv::Mat img = cv::imread(imageLeft[1], 0);

    auto ref_frame_ptr = std::make_shared<Frame>(left_img, 0, 4, 0.5, camera_ptr);
    auto cur_frame_ptr = std::make_shared<Frame>(img, 0, 4, 0.5, camera_ptr);

    std::vector<cv::KeyPoint> kps;
    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++)
    {
        int x = 130;//rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = 73;//rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
        kps.emplace_back(cv::Point2f((float)x,(float)y),0, 0, 0, 1);
    }
    ref_frame_ptr->SetKeys(kps, ref_frame_ptr);

    ref_frame_ptr->SetPose(Eigen::Matrix4f::Identity());
    Eigen::Matrix4f init_gauss = Eigen::Matrix4f::Identity();
    cur_frame_ptr->SetPose(init_gauss);
    for (int i = 0;i < nPoints; i++)
    {
        Eigen::Vector3f norm;
        norm[0] = (pixels_ref.at(i).x() - cx)/fx;
        norm[1] = (pixels_ref.at(i).y() - cy)/fy;
        norm[2] = 1;
        auto mp = std::make_shared<MapPoint>(norm * depth_ref[i], map_ptr, ref_frame_ptr, ref_frame_ptr->GetKey(i), cv::Mat());

        ref_frame_ptr->AddMapPoint(mp, i);
        map_ptr->AddMapPoint(mp);

        Eigen::Vector2f px_cur(130, 73);

        cv::Mat show1 = cur_frame_ptr->GetImagePyramid()[0].clone();
        cv::Mat show2 = ref_frame_ptr->GetImagePyramid()[0].clone();
        cv::cvtColor(show1, show1, CV_GRAY2BGR);
        cv::cvtColor(show2, show2, CV_GRAY2BGR);
        cv::circle(show2, cv::Point((int)px_cur[0], (int)px_cur[1]), 5, cv::Scalar(255, 0, 0), 5);

        std::cout<<"before "<<std::endl<<px_cur<<std::endl;
        matcher_ptr->FindMatchDirect(mp, cur_frame_ptr, px_cur);
        std::cout<<"after "<<std::endl<<px_cur<<std::endl;

        cv::circle(show1, cv::Point((int)px_cur[0], (int)px_cur[1]), 5, cv::Scalar(255, 0, 0), 5);

        cv::imshow("before", show2);
        cv::imshow("after", show1);
        cv::waitKey(0);
    }

    ref_frame_ptr->SetPose(Eigen::Matrix4f::Identity());
    cur_frame_ptr->SetPose(Eigen::Matrix4f::Identity());

//    const cv::Mat& cur_img, uint8_t* ref_patch_with_border, uint8_t* ref_patch, const int n_iter, Eigen::Vector2f& cur_ps_estimate, bool no_simd = false
    int half_patch_size = 4;
    uint8_t patch[half_patch_size*half_patch_size*4] __attribute__((aligned(16)));
    uint8_t patch_with_border[(half_patch_size+1)*(half_patch_size+1)*4] __attribute__((aligned(16)));

    Eigen::Vector2f px_scaled(65, 36.5);
    const int cur_step = ref_frame_ptr->GetImagePyramid()[1].step.p[0];

    for(int y=0; y<(half_patch_size+1)*2; ++y)
    {
        uint8_t *it = (uint8_t *) ref_frame_ptr->GetImagePyramid()[1].data + ((int)px_scaled[1] + y - half_patch_size-1) * cur_step + (int)px_scaled[0] - half_patch_size-1;
        for (int x = 0; x < (half_patch_size+1)*2; ++x, ++it)
        {
            int id = x + y*(half_patch_size+1)*2;
            patch_with_border[id] = *it;
        }
    }

    uint8_t* patch_ptr = patch;
    for(int y = 1; y < half_patch_size*2 + 1; ++y, patch_ptr += half_patch_size*2)
    {
        uint8_t* y_ptr = patch_with_border + y * (half_patch_size*2 + 2) + 1;
        for(int x = 0; x < half_patch_size*2; ++x)
            patch_ptr[x] = y_ptr[x];
    }

    cv::Mat show1 = cur_frame_ptr->GetImagePyramid()[1].clone();
    cv::Mat show2 = ref_frame_ptr->GetImagePyramid()[1].clone();
    cv::cvtColor(show1, show1, CV_GRAY2BGR);
    cv::cvtColor(show2, show2, CV_GRAY2BGR);

    cv::circle(show2, cv::Point((int)px_scaled[0], (int)px_scaled[1]), 5, cv::Scalar(255, 0, 0), 5);

    std::cout<<" before: "<<px_scaled<<std::endl;
    matcher_ptr->Align2D(cur_frame_ptr->GetImagePyramid()[1], patch_with_border, patch, 40, px_scaled);
    std::cout<<" align: "<<px_scaled<<std::endl;
    cv::circle(show1, cv::Point((int)px_scaled[0], (int)px_scaled[1]), 5, cv::Scalar(255, 0, 0), 5);

    cv::imshow("before", show2);
    cv::imshow("after", show1);
    cv::waitKey(0);
}

int main()
{
    //test_direct();
    //test_direct2();
    test_projection();
//    test_image_align();
    return 0;
}
