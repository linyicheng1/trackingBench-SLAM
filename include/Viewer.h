#ifndef TRACKING_BENCH_VIEWER_H
#define TRACKING_BENCH_VIEWER_H
#include "type.h"
#include <mutex>
#include <string>
#include <opencv2/opencv.hpp>

namespace TRACK_BENCH
{
    class Viewer
    {
    public:
        explicit Viewer();

        // Main thread function. Draw points, keyframes, the current camera pose and the last processed
        // frame. Drawing is refreshed according to the camera fps. We use Pangolin.
        void Run();

        void RequestFinish();

        void RequestStop();

        bool isFinished();

        bool isStopped();

        void Release();

    private:

        bool Stop();


        // 1/fps in ms
        double mT;
        float mImageWidth, mImageHeight;

        float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

        bool CheckFinish();
        void SetFinish();
        bool mbFinishRequested;
        bool mbFinished;
        std::mutex mMutexFinish;

        bool mbStopped;
        bool mbStopRequested;
        std::mutex mMutexStop;
        cv::Mat im;
    };
}

#endif //TRACKING_BENCH_VIEWER_H