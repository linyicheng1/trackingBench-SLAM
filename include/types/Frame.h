#ifndef TRACKING_BENCH_FRAME_H
#define TRACKING_BENCH_FRAME_H

#include "opencv2/core/core.hpp"
#include <utility>
#include <vector>
#include <set>
#include <mutex>
#include "third_part/DBoW2/DBoW2/BowVector.h"
#include "third_part/DBoW2/DBoW2/FeatureVector.h"
#include "third_part/DBoW2/DBoW2/FORB.h"
#include "third_part/DBoW2/DBoW2/TemplatedVocabulary.h"
#include <Eigen/Core>
#include <memory>

namespace TRACKING_BENCH
{

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64
    const int EDGE_THRESHOLD = 19;
    typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;

    class Frame;
    class Map;
    class MapPoint;
    class CameraModel;
    class Feature
    {
    public:
        std::shared_ptr<Frame> frame;
        std::shared_ptr<MapPoint> point;
        Eigen::Vector3f f;

        cv::KeyPoint kp;
        Eigen::Vector2f px;
        Eigen::Vector2f px_un;
        int idxF;
        Feature(std::shared_ptr<Frame> _frame, cv::KeyPoint& _kp, Eigen::Vector3f _f, int id):
                frame(std::move(_frame)),kp(_kp),px(_kp.pt.x, _kp.pt.y),f(std::move(_f)),idxF(id)
        {
        }

        Feature(std::shared_ptr<Frame> _frame, std::shared_ptr<MapPoint> _point, cv::KeyPoint& _kp, Eigen::Vector3f _f, int id):
            frame(std::move(_frame)),kp(_kp),point(std::move(_point)),px(_kp.pt.x, _kp.pt.y),f(std::move(_f)),idxF(id)
        {
        }

        Feature(const Feature&obj):
                frame(obj.frame),kp(obj.kp),point(obj.point),px(obj.px),f(obj.f),idxF(obj.idxF){}
    };

    class Frame
    {
    public:
        Frame(const Frame& frame);
        Frame(const cv::Mat &imGray, const double &timeStamp, int level,
              float scale,std::shared_ptr<CameraModel> camera);
        long unsigned int GetId()const{return mnId;}
        // pose
        void SetPose(const Eigen::Matrix4f& Tcw);
        Eigen::Matrix4f GetPose();
        Eigen::Matrix4f GetPoseInverse();
        Eigen::Vector3f GetCameraCenter();
        Eigen::Matrix3f GetRotation();
        Eigen::Vector3f GetTranslation();

        // features
        void SetKeys(std::vector<cv::KeyPoint>& pts,const std::shared_ptr<Frame>& frame, cv::Mat mDescriptors = cv::Mat(), bool unDistort = false);
        void AddKeys(std::vector<Feature>& pts, cv::Mat mDescriptors = cv::Mat(), bool unDistort = false);
        void UnDistortPoints();
        void UnDistortImage();
        std::vector<std::shared_ptr<Feature>>& GetKeys(){return mvKeys;}
        std::shared_ptr<Feature>& GetKey(size_t id){return mvKeys.at(id);}
        std::vector<cv::Point2f>& GetCVKeys(){return mvCVKeys;}
        cv::Mat GetDescriptors() const{return mDescriptors;}
        std::vector<cv::Mat> GetVectorDescriptors() const;
        cv::Mat GetDescriptor(int id) const{return mDescriptors.row(id);}
        bool GetOutlier(size_t id){return mvbOutlier.at(id);}
        void SetOutlier(size_t id, bool state){mvbOutlier.at(id)=state;}
        std::vector<float> inline GetScaleSigmaSquares()
        {
            return mvLevelSigma2;
        }
        std::vector<float> inline GetInverseScaleSigmaSquares()
        {
            return mvInvLevelSigma2;
        }
        void AssignFeaturesToGrid();
        std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, int minLevel=-1, int maxLevel=-1) const;
        bool PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY) const;

        void SetBow(const std::shared_ptr<ORBVocabulary>& voc);
        DBoW2::BowVector& GetBowVector(){return mBowVec;}
        DBoW2::FeatureVector& GetFeatureVector(){return mFeatVec;}

        // MapPoints
        std::set<std::shared_ptr<MapPoint>> GetMapPoints();
        std::vector<std::shared_ptr<MapPoint>> GetMapPointMatches();
        int TrackedMapPoint(const int &minObs);
        std::shared_ptr<MapPoint> GetMapPoint(const size_t &idx);

        void AddMapPoint(std::shared_ptr<MapPoint>& pMP, const size_t& idx);
        void AddMapPoints(const std::shared_ptr<Frame>& ref, const std::vector<cv::DMatch>& matches);
        void AddMapPoints(const std::shared_ptr<Map>& map, const std::vector<cv::DMatch>& matches);
        void EraseMapPointMatch(const size_t &idx);
        void EraseMapPointMatch(const std::shared_ptr<MapPoint>& pMP);
        void ReplaceMapPointMatch(const size_t &idx, std::shared_ptr<MapPoint> pMP);

        // key frame
        int GetMaxLevel() const{return nLevels;}
        bool IsKeyFrame() const{return mbKeyFrame;}
        void SetKeyFrame(){mbKeyFrame = true;}
        void ComputePyramid(cv::Mat image);

        // base common functions
        int inline GetLevels() const{return nLevels;}
        float inline GetScaleFactor() const{return scaleFactor;}
        std::vector<float> inline GetScaleFactors(){return mvScaleFactor;}
        std::vector<float> inline GetInverseScaleFactors(){return mvInvScaleFactor;}
        std::vector<cv::Mat>& GetImagePyramid(){return mvImagePyramid;}
        cv::Mat Equalize();
        cv::Mat GetImage(){return mvImagePyramid[0];}
        // camera
        std::shared_ptr<CameraModel> GetCameraModel(){return mpCamera;}
        bool IsInImage(const float& x, const float& y)const;
        bool IsInFrustum(const std::shared_ptr<MapPoint>& pMP, int &level, float &x, float &y, float &view);

        inline static void JacobianXYZ2uv(
                const cv::Mat& xyz_in_f,
                Eigen::Matrix<double, 2, 6>& J
                )
        {
            const double x = xyz_in_f.at<double>(0);
            const double y = xyz_in_f.at<double>(1);
            const double z_inv = 1. / xyz_in_f.at<double>(2);
            const double z_inv_2 = z_inv * z_inv;

            J(0,0) = -z_inv;
            J(0, 1) = 0.0;
            J(0, 2) = x * z_inv_2;
            J(0, 3) = y * J(0, 2);
            J(0, 4) = -(1.0 + x*J(0, 2));
            J(0, 5) = y * z_inv;

            J(1, 0) = 0.0;
            J(1, 1) = -z_inv;
            J(1, 2) = y * z_inv_2;
            J(1, 3) = 1.0 + y * J(1, 2);
            J(1, 4) = -J(0, 3);
            J(1, 5) = - x * z_inv;
        }

    protected:
        static long unsigned int nNextId;
        long unsigned int mnId{};
        double mTimeStamp{};
        bool mbKeyFrame{};

        // pose
        // SE3 Pose and camera center
        Eigen::Matrix4f mTcw;
        Eigen::Matrix4f mTwc;
        Eigen::Vector3f mOw;// == mtwc
        std::mutex mMutexPose;

        // features
        std::vector<std::shared_ptr<Feature>> mvKeys;
        // for OpenCV optical flow
        std::vector<cv::Point2f> mvCVKeys;
        cv::Mat mDescriptors;
        std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
        // DBow for orb match
        DBoW2::BowVector mBowVec;
        DBoW2::FeatureVector mFeatVec;

        // MapPoints
        std::vector<std::shared_ptr<MapPoint>> mvpMapPoints;
        std::mutex mMutexFeatures;

        // camera model
        std::shared_ptr<CameraModel> mpCamera = nullptr;

        // frame
        int nFeatures{};
        int nLevels;
        float scaleFactor;
        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;
        std::vector<cv::Mat> mvImagePyramid;
        std::vector<float> mvbOutlier;
        cv::Mat mImage;
        float mfGridElementWidthInv{};
        float mfGridElementHeightInv{};

    };
}


#endif //TRACKING_BENCH_FRAME_H
