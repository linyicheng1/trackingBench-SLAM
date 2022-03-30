#ifndef TRACKING_BENCH_VIEWER_H
#define TRACKING_BENCH_VIEWER_H
#include "types/Frame.h"

#include <mutex>
#include <string>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <unistd.h>

namespace TRACK_BENCH
{
//    class Viewer
//    {
//    public:
//        explicit Viewer();
//
//        // Main thread function. Draw points, keyframes, the current camera pose and the last processed
//        // frame. Drawing is refreshed according to the camera fps. We use Pangolin.
//        void Run();
//
//        // draw
//        void SetCameraPos(position Twc);
//        void SetMapPoints(const std::vector<point3d>& map, std::vector<point3d> ref);
//        void SetKeyFrames(std::vector<position> vpKFs);
//
//        void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
//        void DrawMapPoints();
//        void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
//        void RequestFinish();
//        void RequestStop();
//        bool isFinished();
//        bool isStopped();
//        void Release();
//    private:
//
//        bool Stop();
//        // 1/fps in ms
//        double mT;
//        float mImageWidth, mImageHeight;
//
//        float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
//
//        bool CheckFinish();
//        void SetFinish();
//        bool mbFinishRequested;
//        bool mbFinished;
//        std::mutex mMutexFinish;
//
//        bool mbStopped;
//        bool mbStopRequested;
//        std::mutex mMutexStop;
//        cv::Mat im;
//
//        // viewer
//        pangolin::OpenGlMatrix Twc;
//        std::vector<point3d> vpMPs;
//        std::vector<point3d> vpRefMPs;
//        std::vector<position> vpKFs;
//    };
}

#endif //TRACKING_BENCH_VIEWER_H