#include "Viewer.h"
#include <pangolin/pangolin.h>
#include <mutex>
#include <unistd.h>

namespace TRACK_BENCH
{

    Viewer::Viewer():
        mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false)
    {
        mT = 1e3/20;
        mImageWidth = 640;
        mImageHeight = 480;
        mViewpointX = 0;
        mViewpointY = -10;
        mViewpointZ = 0.1;
        mViewpointF = 2000;
    }

    void Viewer::Run()
    {
        mbFinished = false;
        mbStopped = false;

        pangolin::CreateWindowAndBind("Tracking Bench Viewer",1024,768);

        // 3D Mouse handler requires depth testing to be enabled
        glEnable(GL_DEPTH_TEST);

        // Issue specific OpenGl we might need
        glEnable (GL_BLEND);
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
        pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
        pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
        pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
        pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);
        pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);

        // Define Camera Render Object (for view / scene browsing)
        pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
        );

        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View& d_cam = pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
                .SetHandler(new pangolin::Handler3D(s_cam));

        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();

        cv::namedWindow("Current Frame");

        bool bFollow = true;
        bool bLocalizationMode = false;

        while(true)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            // 1. get Twc and follow the trajectories
            s_cam.Follow(Twc);
            // 2. activate the camera
            d_cam.Activate(s_cam);
            // 3. set white background
            glClearColor(1.0f,1.0f,1.0f,1.0f);
            // 4. draw some things

            // 5. finish it
            pangolin::FinishFrame();
            // draw tracking image
            if(!im.empty())
            {
                cv::imshow("Current Frame",im);
                cv::waitKey((int)mT);
            }
            if(Stop())
            {
                while(isStopped())
                {
                    usleep(3000);
                }
            }
            if(CheckFinish())
                break;
        }
        SetFinish();
    }

    void Viewer::RequestFinish()
    {
        std::unique_lock<std::mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    bool Viewer::CheckFinish()
    {
        std::unique_lock<std::mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void Viewer::SetFinish()
    {
        std::unique_lock<std::mutex> lock(mMutexFinish);
        mbFinished = true;
    }

    bool Viewer::isFinished()
    {
        std::unique_lock<std::mutex> lock(mMutexFinish);
        return mbFinished;
    }

    void Viewer::RequestStop()
    {
        std::unique_lock<std::mutex> lock(mMutexStop);
        if(!mbStopped)
            mbStopRequested = true;
    }

    bool Viewer::isStopped()
    {
        std::unique_lock<std::mutex> lock(mMutexStop);
        return mbStopped;
    }

    bool Viewer::Stop()
    {
        std::unique_lock<std::mutex> lock(mMutexStop);
        std::unique_lock<std::mutex> lock2(mMutexFinish);

        if(mbFinishRequested)
            return false;
        else if(mbStopRequested)
        {
            mbStopped = true;
            mbStopRequested = false;
            return true;
        }

        return false;

    }

    void Viewer::Release()
    {
        std::unique_lock<std::mutex> lock(mMutexStop);
        mbStopped = false;
    }

}