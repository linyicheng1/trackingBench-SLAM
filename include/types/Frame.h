#ifndef TRACKING_BENCH_FRAME_H
#define TRACKING_BENCH_FRAME_H

#include "extractors/extractor.h"
#include "opencv2/core/core.hpp"
#include <vector>

namespace TRACKING_BENCH
{
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
        //
        Frame() = default;
        Frame(const Frame& frame);
        Frame(const cv::Mat &imGray, const double &timeStamp, Extractor* extractor, CameraModel* distortion);

        void SetPose(const cv::Mat& Tcw);
        std::vector<cv::KeyPoint>& getKeysUn(){return mvKeysUn;}
        std::vector<cv::KeyPoint>& getKeys(){return mvKeys;}
        cv::Mat getDescriptors(){return mDescriptors;}

        std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

    private:
        static long unsigned int nNextId;
        long unsigned int mnId;
        double mTimeStamp;
        Extractor* mpExtractor;
        CameraModel* mpCamera;
        int N;// Number of KeyPoints
        std::vector<cv::KeyPoint> mvKeys;
        std::vector<cv::KeyPoint> mvKeysUn;
        std::vector<MapPoint*> mvpMapPoints;
        std::vector<std::size_t> mGrid;
        cv::Mat mDescriptors;

        // Rotation, translation and camera center
        cv::Mat mTcw;
        cv::Mat mRcw;
        cv::Mat mtcw;
        cv::Mat mRwc;
        cv::Mat mOw;// == mtwc
        int FRAME_GRID_ROWS = 48;
        int FRAME_GRID_COLS = 64;

        void ExtractPoint(const cv::Mat &img);
    };
}


#endif //TRACKING_BENCH_FRAME_H
