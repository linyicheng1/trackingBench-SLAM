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

#include <iostream>
#include <stdint.h>
#include <unordered_set>
#include "g2o/stuff/sampler.h"
#include<suitesparse/cholmod.h>

using namespace TRACKING_BENCH;

void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps);


void LoadGroundTruth(const string& file, vector<double>& timeStamp, vector<Eigen::Quaterniond>& q, vector<Eigen::Vector3d>& t)
{
    ifstream fTimes;
    fTimes.open(file.c_str());
    timeStamp.reserve(50000);
    q.reserve(50000);
    t.reserve(50000);

    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        getline(fTimes,s,',');
        if(!s.empty())
        {
            timeStamp.emplace_back((std::atof(s.c_str()))/1e9);
            Eigen::Vector3d t_rs;
            Eigen::Quaterniond q_rs;

            getline(fTimes,s,',');
            t_rs.x() = std::atof(s.c_str());
            getline(fTimes,s,',');
            t_rs.y() = std::atof(s.c_str());
            getline(fTimes,s,',');
            t_rs.z() = std::atof(s.c_str());
            t.emplace_back(t_rs);

            getline(fTimes,s,',');
            q_rs.w() = std::atof(s.c_str());
            getline(fTimes,s,',');
            q_rs.x() = std::atof(s.c_str());
            getline(fTimes,s,',');
            q_rs.y() = std::atof(s.c_str());
            getline(fTimes,s,',');
            q_rs.z() = std::atof(s.c_str());
            q.emplace_back(q_rs);
        }
    }
}

Eigen::Matrix4d GetTRB(double time, const vector<double>& timeStamp, const vector<Eigen::Quaterniond>& q, const vector<Eigen::Vector3d>& t, Eigen::Matrix4d trans =  Eigen::Matrix4d::Identity())
{
    auto item = std::lower_bound(timeStamp.begin(), timeStamp.end(), time);
    if(*item >= time && *(item - 1) <= time)
    {
        auto id = (int)(item - timeStamp.begin());

        double k;
        if(*item == time)
            k = 1;
        else if(*(item - 1) == time)
            k = 0;
        else
            k = (double)((time - *(item - 1)) / (*item - *(item - 1)));
        Eigen::Quaterniond q1 = q[id];
        Eigen::Quaterniond q0 = q[id-1];
        Eigen::Quaterniond dq = q0.inverse() * q1;
        dq.normalize();
        const double dPhi = 2 * acos(dq.w());
        Eigen::Vector3d u = Eigen::Vector3d(dq.x(), dq.y(), dq.z()) / (sin(dPhi * 0.5f));
        Eigen::Quaterniond q_rs = q0 * Eigen::Quaterniond(
                Eigen::AngleAxisd(dPhi * k, u)
                );

        Eigen::Vector3d t_rs = k * t[id] + (1 - k) * t[id-1];
        // rs --> rb
        {
            Eigen::Matrix4d T_bs;
            T_bs << 1.0, 0.0, 0.0,  7.48903e-02,
                    0.0, 1.0, 0.0, -1.84772e-02,
                    0.0, 0.0, 1.0, -1.20209e-01,
                    0.0, 0.0, 0.0,  1.0;
            Eigen::Matrix4d T_rs = Eigen::Matrix4d::Identity();
            T_rs.block<3, 3>(0, 0) = q_rs.toRotationMatrix();
            T_rs.block<3, 1>(0, 3) = t_rs;

            Eigen::Matrix4d T_rb = T_rs * T_bs.inverse();
            return T_rb * trans;
        }
    }
}

void test_vo_1()
{
    std::vector<string> imageLeft;
    std::vector<string> imageRight;
    std::vector<double> timeStamp;
    LoadImages( "/home/lyc/code/slam_bench/MH_05_difficult/mav0/cam0/data",
                "/home/lyc/code/slam_bench/MH_05_difficult/mav0/cam1/data",
                "/home/lyc/code/slam_bench/MH_05_difficult/mav0/cam0/data.csv",
                imageLeft, imageRight, timeStamp);

    std::vector<double> gtTimeStamp;
    std::vector<Eigen::Quaterniond> gtQ;
    std::vector<Eigen::Vector3d> gtT;
    LoadGroundTruth("/home/lyc/code/slam_bench/MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv",
                    gtTimeStamp, gtQ, gtT);



    auto extractor_ptr = std::make_shared<ORBExtractor>();
    auto local_ba = std::make_shared<LocalBA>();

    std::shared_ptr<Frame> key_frame, last_frame;
    auto matcher_ptr = std::make_shared<Matcher>();
    auto map_ptr = std::make_shared<Map>();
    auto viewer = std::make_shared<Viewer>();
    viewer->SetCameraPos(Eigen::Matrix4f::Identity());
    viewer->Run();

//    std::vector<Eigen::Matrix4f> gt;
//    for(int i = 0;i < gtTimeStamp.size();i += 20)
//    {
//        gt.emplace_back(GetTRB(gtTimeStamp.at(i), gtTimeStamp, gtQ, gtT).cast<float>());
//    }
//    viewer->SetKeyFrames(gt);
//    sleep(-1);

//    auto vocabulary = std::make_shared<ORBVocabulary>();
//    bool load = vocabulary->loadFromTextFile("/home/lyc/code/slam_bench/trackingBench-SLAM/Vocabulary/ORBvoc.txt");
//    if(!load)
//    {
//        cerr <<"Failed to open vocabulary "<<std::endl;
//    }

    Eigen::Matrix4d T_lr, T_br, T_bl;
    T_br << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
            0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
            -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
            0.0, 0.0, 0.0, 1.0;
    T_bl << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
            0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
            -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
            0.0, 0.0, 0.0, 1.0;
    T_lr = T_bl.inverse() * T_br;

    std::vector<cv::KeyPoint> keyPoints_ref;
    std::vector<Eigen::Matrix4f> kfs_pos;
    for(int i = 0;i < gtTimeStamp.size();i += 200)
    {
        kfs_pos.emplace_back(GetTRB(gtTimeStamp.at(i), gtTimeStamp, gtQ, gtT).cast<float>());
    }
    for (int i = 540;i < 542;i +=1)
    {
        cv::Mat imLeft, imRight, show;
        std::vector<cv::KeyPoint> keyPoints_r, keyPoints_l;
        cv::Mat descriptors_r, descriptors_l;
        // 0. read image to frame
        imLeft = cv::imread(imageLeft[i], CV_LOAD_IMAGE_UNCHANGED);
        static auto camera_ptr = std::make_shared<PinholeCamera>(imLeft.cols, imLeft.rows,458.654, 457.296, 367.215, 248.375,-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);
        cv::Mat out;
        camera_ptr->UndistortImage(imLeft, imLeft);

        auto cur_frame_ptr = std::make_shared<Frame>(imLeft, 0, 5, 0.8, camera_ptr);
        if(last_frame != nullptr)
        {
            cur_frame_ptr->SetPose(last_frame->GetPose());

        }
        else
        {
            cur_frame_ptr->SetPose(GetTRB(timeStamp.at(i), gtTimeStamp, gtQ, gtT, T_bl).inverse().cast<float>());
            viewer->SetCameraPos(cur_frame_ptr->GetPoseInverse());
            std::cout<<"ground truth pose Twc: "<<GetTRB(timeStamp.at(i), gtTimeStamp, gtQ, gtT, T_bl)<<std::endl;

        }

        extractor_ptr->operator()(cur_frame_ptr->GetImagePyramid(),
                                  cur_frame_ptr->GetScaleFactors(),
                                  2000,
                                  80,
                                  30,
                                  keyPoints_l,
                                  descriptors_l);

        cur_frame_ptr->SetKeys(keyPoints_l, cur_frame_ptr, descriptors_l);

//        DBoW2::BowVector bowVector;
//        DBoW2::FeatureVector featureVector;
//        cur_frame_ptr->SetBow(vocabulary);
//        matcher_ptr->setBowParam(30, 100, 30, true, 5);
        // 1. match points
        if (last_frame != nullptr)
        {
            // match with last frame
//            auto matches = matcher_ptr->searchByBow(cur_frame_ptr, key_frame);
            auto matches = matcher_ptr->searchByNN(cur_frame_ptr, key_frame, 0, 5, 10, 30);

            cv::drawMatches(imLeft,keyPoints_l,
                            key_frame->GetImagePyramid().at(0),
                            keyPoints_ref, matches, show);

            // assign map point
            for (auto &item:matches)
            {
                shared_ptr<MapPoint> mp = key_frame->GetMapPoint(item.trainIdx);
                if (mp  != nullptr)
                    cur_frame_ptr->AddMapPoint(mp, item.queryIdx);
            }

//            cur_frame_ptr->SetPose(GetTRB(timeStamp.at(i), gtTimeStamp, gtQ, gtT, T_bl).inverse().cast<float>());
            // 2. pose optimization
            local_ba->PoseOptimization(cur_frame_ptr);
            std::cout<<"current pose Twc: "<<cur_frame_ptr->GetPoseInverse()<<std::endl;
            std::cout<<"ground truth pose Twc: "<<GetTRB(timeStamp.at(i), gtTimeStamp, gtQ, gtT, T_bl)<<std::endl;
//            cur_frame_ptr->SetPose(GetTRB(timeStamp.at(i), gtTimeStamp, gtQ, gtT, T_bl).inverse().cast<float>());
        }


        kfs_pos.emplace_back(cur_frame_ptr->GetPoseInverse());
        viewer->SetKeyFrames(kfs_pos);
        //if(i % 5 == 0)
        {// 3. key frame

            imRight = cv::imread(imageRight[i], CV_LOAD_IMAGE_UNCHANGED);
            static auto right_camera_ptr = std::make_shared<PinholeCamera>(imLeft.cols, imLeft.rows,457.587, 456.134, 379.999, 255.238,-0.28368365,  0.07451284, -0.00010473, -3.55590700e-05);
            right_camera_ptr->UndistortImage(imRight, imRight);
            // 4. add map points by stereo
            auto right_frame_ptr = std::make_shared<Frame>(imRight, 0, 5, 0.8, right_camera_ptr);
            extractor_ptr->operator()(cur_frame_ptr->GetImagePyramid(),
                                      cur_frame_ptr->GetScaleFactors(),
                                      2000,
                                      80,
                                      30,
                                      keyPoints_r,
                                      descriptors_r);
            right_frame_ptr->SetKeys(keyPoints_r, right_frame_ptr, descriptors_r);

            auto depth = local_ba->AddMapPointsByStereo(cur_frame_ptr, right_frame_ptr, 47.91, 435.2);

            for (size_t j = 0; j < depth.size(); j ++)
            {
                if(depth[j] > 0)
                {
                    Eigen::Vector3f norm;
                    const auto& pt = cur_frame_ptr->GetKey(j);
                    norm << camera_ptr->Cam2World(pt->px);

                    const Eigen::Matrix3f& R = cur_frame_ptr->GetRotation();
                    const Eigen::Vector3f& t = cur_frame_ptr->GetTranslation();
                    auto mp = std::make_shared<MapPoint>(R * norm * depth[j] + t, map_ptr, cur_frame_ptr, cur_frame_ptr->GetKey(j));

                    cur_frame_ptr->AddMapPoint(mp, j);
                    map_ptr->AddMapPoint(mp);
                }
            }
//            local_ba->PoseOptimization(cur_frame_ptr);
//            std::cout<<"opt pose: "<<cur_frame_ptr->GetPoseInverse()<<std::endl;
//            std::cout<<"ground truth pose Twc: "<<GetTRB(timeStamp.at(i), gtTimeStamp, gtQ, gtT, T_bl)<<std::endl;
            //map_ptr->AddKeyFrame(cur_frame_ptr);
            // only keep 20 frames
            //map_ptr->RemoveOldFrames(1);
            // test visualization
            viewer->SetMapPoints(map_ptr->GetAllMapPoints(), cur_frame_ptr->GetMapPointMatches());


            keyPoints_ref = keyPoints_l;
            key_frame = cur_frame_ptr;
        }
        last_frame = cur_frame_ptr;
        // visualization
        if(!show.empty())
        {
            cv::imshow("show", show);
        }
        cv::waitKey(200);
    }
    cv::waitKey(0);
    viewer->RequestFinish();
}

using namespace Eigen;
using namespace std;
class Sample {
public:
    static int uniform(int from, int to) { return static_cast<int>(g2o::Sampler::uniformRand(from, to)); }
};

void test_PoseOptimization()
{
    LocalBA local_ba;
    std::vector<string> imageLeft;
    std::vector<string> imageRight;
    std::vector<double> timeStamp;
    LoadImages( "/home/lyc/code/slam_bench/MH_05_difficult/mav0/cam0/data",
                "/home/lyc/code/slam_bench/MH_05_difficult/mav0/cam1/data",
                "/home/lyc/code/slam_bench/MH_05_difficult/mav0/cam0/data.csv",
                imageLeft, imageRight, timeStamp);

    cv::Mat m = cv::imread(imageLeft[100], CV_LOAD_IMAGE_UNCHANGED);
    auto camera_ptr = std::make_shared<PinholeCamera>(m.cols, m.rows,458.654, 457.296, 367.215, 248.375);
    auto frame = std::make_shared<Frame>(m, 0, 5, 0.8, camera_ptr);
    auto map_ptr = std::make_shared<TRACKING_BENCH::Map>();
    // add map point
    Eigen::AngleAxisf rot(M_PI/2, Eigen::Vector3f(0,0,1));
    Eigen::Matrix3f R_wc = rot.toRotationMatrix();
    Eigen::Vector3f t_wc{0, 0.1, 0.1};
    frame->SetPose(Eigen::Matrix4f::Identity());

    std::vector<cv::KeyPoint> kps;
    std::vector<Eigen::Vector3f> pts;
    for (size_t i = 0;i < 200;i ++)
    {
        Eigen::Vector3f pt_w(
                (float)(g2o::Sampler::uniformRand(0., 1.)-0.5)*3,
                (float)(g2o::Sampler::uniformRand(0., 1.)-0.5),
                (float)(g2o::Sampler::uniformRand(0., 1.)-0.5)+3);
        Eigen::Vector3f pt_c = R_wc*pt_w + t_wc;
        auto uv = camera_ptr->World2Cam(pt_c);
        cv::KeyPoint keyPoint(uv.x(), uv.y(), 1);
        kps.emplace_back(keyPoint);
        pts.emplace_back(pt_w);
    }
    frame->SetKeys(kps, frame);
    int i = 0;
    for (auto key:frame->GetKeys())
    {
        auto mp = std::make_shared<MapPoint>(pts[i], map_ptr, frame, key);
        frame->AddMapPoint(mp, i);
        i ++;
    }

    local_ba.PoseOptimization(frame);

    std::cout<<"set R: "<<R_wc<<std::endl;
    std::cout<<"opt R: "<<frame->GetPose().block<3, 3>(0, 0)<<std::endl;
    std::cout<<"set t: "<<t_wc<<std::endl;
    std::cout<<"opt t: "<<frame->GetPose().block<3, 1>(0 ,3)<<std::endl;
}

void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);


        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            if(p.w() > 255)
            {
                glPointSize(3);
                glColor3f(1, 0, 0);
                glVertex3d(p[0], p[1], p[2]);
            }
            else
            {
                glPointSize(2);
                glColor3f(p[3], p[3], p[3]);
                glVertex3d(p[0], p[1], p[2]);
            }
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}

void test_triangle()
{
    // 文件路径
    string left_file = "/home/lyc/code/slam_bench/trackingBench-SLAM/data/left.png";
    string right_file = "/home/lyc/code/slam_bench/trackingBench-SLAM/data/right.png";
    string disparity_file = "/home/lyc/code/slam_bench/trackingBench-SLAM/data/disparity.png";
    // 内参
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // 间距
    double d = 0.573;
    std::vector<cv::KeyPoint> keyPoints_r, keyPoints_l;
    cv::Mat descriptors_r, descriptors_l;
    auto local_ba = std::make_shared<LocalBA>();
    // 读取图像
    cv::Mat left = cv::imread(left_file, 0);
    cv::Mat right = cv::imread(right_file, 0);
    cv::Mat disparity = cv::imread(disparity_file, 0); // disparty 为CV_8U,单位为像素
    auto extractor_ptr = std::make_shared<ORBExtractor>();

    static auto camera_ptr = std::make_shared<PinholeCamera>(left.cols, left.rows,fx, fy, cx, cy);
    auto left_frame_ptr = std::make_shared<Frame>(left, 0, 5, 0.8, camera_ptr);
    extractor_ptr->operator()(left_frame_ptr->GetImagePyramid(),
                              left_frame_ptr->GetScaleFactors(),
                              2000,
                              80,
                              30,
                              keyPoints_r,
                              descriptors_r);
    left_frame_ptr->SetKeys(keyPoints_r, left_frame_ptr, descriptors_r);


    auto right_frame_ptr = std::make_shared<Frame>(right, 0, 5, 0.8, camera_ptr);
    extractor_ptr->operator()(right_frame_ptr->GetImagePyramid(),
                              right_frame_ptr->GetScaleFactors(),
                              2000,
                              80,
                              30,
                              keyPoints_r,
                              descriptors_r);
    right_frame_ptr->SetKeys(keyPoints_r, right_frame_ptr, descriptors_r);

    auto tri_depth = local_ba->AddMapPointsByStereo(left_frame_ptr, right_frame_ptr, d*fx, fx);

    // 生成点云
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;
    int cnt = 0;
    float sum_err = 0;
    for(int i = 0;i < tri_depth.size();i ++)
    {
        if(tri_depth.at(i) > 0 && !std::isnan(-tri_depth[i]))
        {
            int u = (int)left_frame_ptr->GetKey(i)->kp.pt.x;
            int v = (int)left_frame_ptr->GetKey(i)->kp.pt.y;
            double disp = disparity.at<uchar>(v, u);
            disp = (disp)/fx;
            double depth = d/disp;

            Vector4d point(0, 0, 0, 1000); // 前三维为xyz,第四维为颜色
            point[0] = (u-cx)/fx*tri_depth[i];
            point[1] = (v-cy)/fy*tri_depth[i];
            point[2] = tri_depth[i];

            Vector4d point2(0, 0, 0, left.at<uchar>(v, u) / 255.0);
            point2[0] = (u-cx)/fx*depth;
            point2[1] = (v-cy)/fy*depth;
            point2[2] = depth;

            pointcloud.push_back(point);
            pointcloud.push_back(point2);

            float my_depth = tri_depth[i];
            float err = abs(depth - tri_depth[i]);
            sum_err += err;
            cnt ++;
        }
    }
    std::cout<<"mean error: "<< sum_err / cnt <<std::endl;


    // TODO 根据双目模型计算点云
    // 如果你的机器慢，请把后面的v++和u++改成v+=2, u+=2
    for (int v = 0; v < left.rows; v++)
        for (int u = 0; u < left.cols; u++) {
            if(u == 469 && v == 211)
                std::cout<<"here"<<std::endl;
            Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0); // 前三维为xyz,第四维为颜色

            // start your code here (~6 lines)
            // 根据双目模型计算 point 的位置
            double disp = disparity.at<uchar>(v, u);
            disp = (disp)/fx;
            double depth = d/disp;
            point[0] = (u-cx)/fx*depth;
            point[1] = (v-cy)/fy*depth;
            point[2] = depth;
            //pointcloud.push_back(point);
            // end your code here
        }

    // 画出点云
    showPointCloud(pointcloud);
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

void drawProjectionError(const std::shared_ptr<Frame>& F, const Eigen::Matrix4f& T_cw, const Eigen::Matrix4f& T_cw2, bool delay = false)
{
    cv::Mat show = F->GetImagePyramid().at(0).clone();
    cv::cvtColor(show, show, cv::COLOR_GRAY2BGR);

    int cnt = 0;
    for (const auto& mp:F->GetMapPointMatches())
    {
        if (mp != nullptr)
        {
            const Eigen::Vector3f pw = mp->GetWorldPos();
            const Eigen::Vector3f pc = T_cw.block<3, 3>(0, 0) * pw + T_cw.block<3, 1>(0, 3);
            const Eigen::Vector2f uv = F->GetCameraModel()->World2Cam(pc);

            const Eigen::Vector3f pc2 = T_cw2.block<3, 3>(0, 0) * pw + T_cw2.block<3, 1>(0, 3);
            const Eigen::Vector2f uv2 = F->GetCameraModel()->World2Cam(pc2);

            cv::Point p0(F->GetKey(cnt)->px.x(), F->GetKey(cnt)->px.y());
            cv::Point p1(uv.x(), uv.y());
//            cv::Point p2(uv2.x(), uv2.y());

            cv::circle(show, p0, 8,cv::Scalar(0, 0, 255), -1);
            cv::circle(show, p1, 2,cv::Scalar(0, 255, 0), -1);
//            cv::circle(show, p2, 2,cv::Scalar(255, 0, 0), -1);

        }
        cnt ++;
    }

    cv::imshow("projection", show);
    if (delay)
        cv::waitKey(0);
}

void test_kitti()
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

    // test ground truth
//    std::vector<Eigen::Matrix4f> gt;
//    for(int i = 0;i < gtR.size();i += 1)
//    {
//        Eigen::Matrix4f pos = Eigen::Matrix4f::Identity();
//        pos.block<3, 3>(0, 0) = gtR.at(i).cast<float>();
//        pos.block<3, 1>(0 ,3) = gtT.at(i).cast<float>();
//        gt.emplace_back(pos);
//    }
//    viewer->SetKeyFrames(gt);
//    sleep(-1);

    auto vocabulary = std::make_shared<ORBVocabulary>();
    bool load = vocabulary->loadFromTextFile("/home/lyc/code/slam_bench/trackingBench-SLAM/Vocabulary/ORBvoc.txt");
    if(!load)
    {
        cerr <<"Failed to open vocabulary "<<std::endl;
    }

    // view gt pos
    std::vector<cv::KeyPoint> keyPoints_ref;
    std::vector<Eigen::Matrix4f> kfs_pos;


    // for images
    for (int i = 0;i < timeStamp.size();i +=1)
    {
        cv::Mat imLeft, imRight, show;
        std::vector<cv::KeyPoint> keyPoints_r, keyPoints_l;
        cv::Mat descriptors_r, descriptors_l;
        // 0. read image to frame
        imLeft = cv::imread(imageLeft[i], CV_LOAD_IMAGE_UNCHANGED);
        static auto camera_ptr = std::make_shared<PinholeCamera>(imLeft.cols, imLeft.rows,fx, fy, cx, cy);
        cv::Mat out;
//        camera_ptr->UndistortImage(imLeft, imLeft);

        auto cur_frame_ptr = std::make_shared<Frame>(imLeft, 0, 5, 0.8, camera_ptr);
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
//            std::cout<<"ground truth pose Twc: "<<GetTRB(timeStamp.at(i), gtTimeStamp, gtQ, gtT)<<std::endl;

        }


        DBoW2::BowVector bowVector;
        DBoW2::FeatureVector featureVector;
        cur_frame_ptr->SetBow(vocabulary);
        matcher_ptr->setBowParam(50, 100, 30, true, 6);
        // 1. match points
        if (last_frame != nullptr)
        {
            // match with last frame
//            auto matches = matcher_ptr->searchByBow(cur_frame_ptr, cur_frame_ptr, true);
//            auto matches = matcher_ptr->searchByBF(cur_frame_ptr, key_frame, 0, 5, 10, 30);
//            auto matches = matcher_ptr->searchByViolence(cur_frame_ptr, key_frame, 0, 5, 50);
            std::vector<cv::Point2f> pts;
            std::vector<cv::KeyPoint> kps;
            auto matches = matcher_ptr->searchByOPFlow(cur_frame_ptr, last_frame, pts, true, true);
            kps.reserve(pts.size());
            for (auto& pt:pts)
            {
                cv::KeyPoint kp;
                kp.pt = pt;
                kps.emplace_back(kp);
            }
            cur_frame_ptr->SetKeys(kps, cur_frame_ptr);

//            cv::drawMatches(imLeft,kps,
//                            key_frame->GetImagePyramid().at(0),
//                            keyPoints_ref, matches, show);
//            cv::imshow("match", show);

            // assign map point
            for (auto &item:matches)
            {
                shared_ptr<MapPoint> mp = last_frame->GetMapPoint(item.trainIdx);

                if (mp != nullptr)
                    cur_frame_ptr->AddMapPoint(mp, item.queryIdx);
            }

            Eigen::Matrix4f pos = Eigen::Matrix4f::Identity();
            pos.block<3, 3>(0, 0) = gtR.at(i).cast<float>();
            pos.block<3, 1>(0 ,3) = gtT.at(i).cast<float>();


//            cur_frame_ptr->SetPose(pos);

//            {// TODO TEST
//                cur_frame_ptr->SetPose(Eigen::Matrix4f::Identity());
//                for (int id = 0;id < cur_frame_ptr->GetKeys().size();id ++)
//                {
//                    shared_ptr<MapPoint> mp = cur_frame_ptr->GetMapPoint(id);
//                    if(mp != nullptr)
//                    {
//                        mp->SetWorldPos(gtR.at(i).cast<float>()*mp->GetWorldPos() + gtT.at(i).cast<float>());
////                        cur_frame_ptr->GetKey(id)->px.x() = (cur_frame_ptr->GetKey(id)->kp.pt.x - cx ) / fx;
////                        cur_frame_ptr->GetKey(id)->px.y() = (cur_frame_ptr->GetKey(id)->kp.pt.y - cy ) / fy;
//                    }
//                }
//            }
            // 2. pose optimization
            local_ba->PoseOptimization(cur_frame_ptr);

            std::cout<<"current pose Twc: "<<cur_frame_ptr->GetPoseInverse()<<std::endl;
            std::cout<<"ground truth pose Twc: "<<pos.inverse()<<std::endl;
//            drawProjectionError(cur_frame_ptr, pos, cur_frame_ptr->GetPose());
//            cur_frame_ptr->SetPose(GetTRB(timeStamp.at(i), gtTimeStamp, gtQ, gtT, T_bl).inverse().cast<float>());
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

            imRight = cv::imread(imageRight[i], CV_LOAD_IMAGE_UNCHANGED);
//            cv::imshow("right", imRight);
            static auto right_camera_ptr = std::make_shared<PinholeCamera>(imLeft.cols, imLeft.rows, fx, fy, cx, cy);
//            right_camera_ptr->UndistortImage(imRight, imRight);
            // 4. add map points by stereo
            auto right_frame_ptr = std::make_shared<Frame>(imRight, 0, 5, 0.8, right_camera_ptr);
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

                    //depth[j] = d/disp;

                    Eigen::Vector3f norm;
                    norm[0] = (u-cx)/fx;
                    norm[1] = (v-cy)/fy;
                    norm[2] = 1;

//                    const auto& pt = cur_frame_ptr->GetKey(j);
//                    norm = camera_ptr->Cam2World(pt->px);

                    const Eigen::Matrix3f& R = cur_frame_ptr->GetRotation();
                    Eigen::Vector3f t = cur_frame_ptr->GetTranslation();
//                    t.x() += 0.5;
                    auto mp = std::make_shared<MapPoint>(R * norm * depth[j] + t, map_ptr, cur_frame_ptr, cur_frame_ptr->GetKey(j));

                    cur_frame_ptr->AddMapPoint(mp, j);
                    map_ptr->AddMapPoint(mp);
                }
            }
            std::cout<<" error: "<<sum_err<<" mean: "<<sum_err/cnt<<" cnt: "<<cnt<<std::endl;
//            local_ba->PoseOptimization(cur_frame_ptr);
//            std::cout<<"opt pose: "<<cur_frame_ptr->GetPoseInverse()<<std::endl;
//            std::cout<<"ground truth pose Twc: "<<GetTRB(timeStamp.at(i), gtTimeStamp, gtQ, gtT, T_bl)<<std::endl;
            //map_ptr->AddKeyFrame(cur_frame_ptr);
            // only keep 20 frames
            //map_ptr->RemoveOldFrames(1);
            // test visualization
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

int main()
{

//    test_PoseOptimization();
//    test_triangle();
//    test_vo_1();
    test_kitti();
    return 0;
}

void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);

    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        getline(fTimes,s,',');
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
            vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);
        }
    }
}