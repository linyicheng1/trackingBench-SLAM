#ifndef TRACKING_BENCH_FRAME_H
#define TRACKING_BENCH_FRAME_H

#include "extractors/extractor.h"
#include "opencv2/core/core.hpp"
#include <vector>
#include <set>
#include <mutex>

namespace TRACKING_BENCH
{

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

    class Frame;
    class MapPoint;
    class CameraModel
    {
    public:
        CameraModel() = default;
        CameraModel(const CameraModel &model);
        CameraModel(cv::Mat k, cv::Mat &distCoef);
        int UnDistortKeyPoints();
        void ComputeImageBounds(const cv::Mat &imLeft);
        cv::Mat mK;
        float fx;
        float fy;
        float cx;
        float cy;
        float inv_fx;
        float inv_fy;
        cv::Mat mDistCoef;
        int mnMinX;
        int mnMinY;
        int mnMaxX;
        int mnMaxY;

        int mnGridCols;
        int mnGridRows;
        float mfGridElementWidthInv;
        float mfGridElementHeightInv;
    };

    class Frame
    {
    public:

        Frame() = default;
        Frame(const Frame& frame);
        Frame(const cv::Mat &imGray, const double &timeStamp, Extractor* extractor, CameraModel* distortion);
        // pose
        void SetPose(const cv::Mat& Tcw);
        cv::Mat GetPose();
        cv::Mat GetPoseInverse();
        cv::Mat GetCameraCenter();
        cv::Mat GetRotation();
        cv::Mat GetTranslation();

        // features
        std::vector<cv::KeyPoint>& getKeysUn(){return mvKeysUn;}
        std::vector<cv::KeyPoint>& getKeys(){return mvKeys;}
        cv::Mat getDescriptors(){return mDescriptors;}
        void ExtractPoint(const cv::Mat &img);
        void AssignFeaturesToGrid();
        std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, int minLevel=-1, int maxLevel=-1) const;
        bool PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY);

        // MapPoints
        std::set<MapPoint*> GetMapPoints();
        std::vector<MapPoint*> GetMapPointMatches();
        int TrackedMapPoint(const int &minObs);
        MapPoint* GetMapPoint(const size_t &idx);

        int getMaxLevel(){return mpExtractor->nLevels;}
        long unsigned int getId() const{return mnId;}

    protected:
        static long unsigned int nNextId;
        long unsigned int mnId{};
        double mTimeStamp{};

        // pose
        // SE3 Pose and camera center
        cv::Mat mTcw;
        cv::Mat mTwc;
        cv::Mat mOw;// == mtwc
        std::mutex mMutexPose;

        // features
        Extractor* mpExtractor{};
        std::vector<cv::KeyPoint> mvKeys;
        std::vector<cv::KeyPoint> mvKeysUn;
        cv::Mat mDescriptors;
        std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

        // MapPoints
        std::vector<MapPoint*> mvpMapPoints;
        std::mutex mMutexFeatures;

        // camera model
        CameraModel* mpCamera{};



    };
}


#endif //TRACKING_BENCH_FRAME_H
