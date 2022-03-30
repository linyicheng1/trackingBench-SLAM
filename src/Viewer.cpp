//#include "Viewer.h"
//
//namespace TRACK_BENCH
//{
//
//    Viewer::Viewer():
//        mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false)
//    {
//        mT = 1e3/20;
//        mImageWidth = 640;
//        mImageHeight = 480;
//        mViewpointX = 0;
//        mViewpointY = -10;
//        mViewpointZ = 0.1;
//        mViewpointF = 2000;
//    }
//
//    void Viewer::Run()
//    {
//        mbFinished = false;
//        mbStopped = false;
//
//        pangolin::CreateWindowAndBind("Tracking Bench Viewer",1024,768);
//
//        // 3D Mouse handler requires depth testing to be enabled
//        glEnable(GL_DEPTH_TEST);
//
//        // Issue specific OpenGl we might need
//        glEnable (GL_BLEND);
//        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//
//        pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
//        pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
//        pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
//        pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
//        pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);
//        pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);
//
//        // Define Camera Render Object (for view / scene browsing)
//        pangolin::OpenGlRenderState s_cam(
//                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
//                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
//        );
//
//        // Add named OpenGL viewport to window and provide 3D Handler
//        pangolin::View& d_cam = pangolin::CreateDisplay()
//                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
//                .SetHandler(new pangolin::Handler3D(s_cam));
//
//
//        Twc.SetIdentity();
//
//        cv::namedWindow("Current Frame");
//
//        bool bFollow = true;
//        bool bLocalizationMode = false;
//
//        while(true)
//        {
//            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//            // 1. get Twc and follow the trajectories
//            s_cam.Follow(Twc);
//            // 2. activate the camera
//            d_cam.Activate(s_cam);
//            // 3. set white background
//            glClearColor(1.0f,1.0f,1.0f,1.0f);
//            // 4. draw some things
//            DrawCurrentCamera(Twc);
//            DrawMapPoints();
//            DrawKeyFrames(true, true);
//            // 5. finish it
//            pangolin::FinishFrame();
//            // draw tracking image
//            if(!im.empty())
//            {
//                cv::imshow("Current Frame",im);
//                cv::waitKey((int)mT);
//            }
//            if(Stop())
//            {
//                while(isStopped())
//                {
//                    usleep(3000);
//                }
//            }
//            if(CheckFinish())
//                break;
//        }
//        SetFinish();
//    }
//
//    void Viewer::RequestFinish()
//    {
//        std::unique_lock<std::mutex> lock(mMutexFinish);
//        mbFinishRequested = true;
//    }
//
//    bool Viewer::CheckFinish()
//    {
//        std::unique_lock<std::mutex> lock(mMutexFinish);
//        return mbFinishRequested;
//    }
//
//    void Viewer::SetFinish()
//    {
//        std::unique_lock<std::mutex> lock(mMutexFinish);
//        mbFinished = true;
//    }
//
//    bool Viewer::isFinished()
//    {
//        std::unique_lock<std::mutex> lock(mMutexFinish);
//        return mbFinished;
//    }
//
//    void Viewer::RequestStop()
//    {
//        std::unique_lock<std::mutex> lock(mMutexStop);
//        if(!mbStopped)
//            mbStopRequested = true;
//    }
//
//    bool Viewer::isStopped()
//    {
//        std::unique_lock<std::mutex> lock(mMutexStop);
//        return mbStopped;
//    }
//
//    bool Viewer::Stop()
//    {
//        std::unique_lock<std::mutex> lock(mMutexStop);
//        std::unique_lock<std::mutex> lock2(mMutexFinish);
//
//        if(mbFinishRequested)
//            return false;
//        else if(mbStopRequested)
//        {
//            mbStopped = true;
//            mbStopRequested = false;
//            return true;
//        }
//
//        return false;
//
//    }
//
//    void Viewer::Release()
//    {
//        std::unique_lock<std::mutex> lock(mMutexStop);
//        mbStopped = false;
//    }
//
//    void Viewer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
//    {
//        const float &w = 0.15f;
//        const float h = w*0.75f;
//        const float z = w*0.6f;
//
//        glPushMatrix();
//
//#ifdef HAVE_GLES
//        glMultMatrixf(Twc.m);
//#else
//        glMultMatrixd(Twc.m);
//#endif
//        glLineWidth(2);
//        glColor3f(0.0f,1.0f,0.0f);
//        glBegin(GL_LINES);
//        glVertex3f(0,0,0);
//        glVertex3f(w,h,z);
//        glVertex3f(0,0,0);
//        glVertex3f(w,-h,z);
//        glVertex3f(0,0,0);
//        glVertex3f(-w,-h,z);
//        glVertex3f(0,0,0);
//        glVertex3f(-w,h,z);
//
//        glVertex3f(w,h,z);
//        glVertex3f(w,-h,z);
//
//        glVertex3f(-w,h,z);
//        glVertex3f(-w,-h,z);
//
//        glVertex3f(-w,h,z);
//        glVertex3f(w,h,z);
//
//        glVertex3f(-w,-h,z);
//        glVertex3f(w,-h,z);
//        glEnd();
//
//        glPopMatrix();
//    }
//
//    void Viewer::DrawMapPoints()
//    {
//        if(vpMPs.empty())
//            return;
//
//        glPointSize(2);
//        glBegin(GL_POINTS);
//        // black
//        glColor3f(0.0,0.0,0.0);
//
//        for(const auto & vpMP : vpMPs)
//        {
//            glVertex3f(vpMP(0),vpMP(1),vpMP(2));
//        }
//        glEnd();
//        glPointSize(2);
//        glBegin(GL_POINTS);
//        glColor3f(1.0,0.0,0.0);
//        // red
//        for(const auto & vpMP : vpRefMPs)
//        {
//            glVertex3f(vpMP(0),vpMP(1),vpMP(2));
//        }
//        glEnd();
//    }
//
//    void Viewer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
//    {
//        const float &w = 0.1f;
//        const float h = w*0.75f;
//        const float z = w*0.6f;
//
//        if(bDrawKF)
//        {
//            for(const auto& pKF : vpKFs)
//            {
//                glPushMatrix();
//                glMultMatrixf(pKF.data());
//                glLineWidth(1);
//                glColor3f(0.0f,0.0f,1.0f);
//                glBegin(GL_LINES);
//                glVertex3f(0,0,0);
//                glVertex3f(w,h,z);
//                glVertex3f(0,0,0);
//                glVertex3f(w,-h,z);
//                glVertex3f(0,0,0);
//                glVertex3f(-w,-h,z);
//                glVertex3f(0,0,0);
//                glVertex3f(-w,h,z);
//
//                glVertex3f(w,h,z);
//                glVertex3f(w,-h,z);
//
//                glVertex3f(-w,h,z);
//                glVertex3f(-w,-h,z);
//
//                glVertex3f(-w,h,z);
//                glVertex3f(w,h,z);
//
//                glVertex3f(-w,-h,z);
//                glVertex3f(w,-h,z);
//                glEnd();
//
//                glPopMatrix();
//            }
//        }
//
////        if(bDrawGraph)
////        {
////            glLineWidth(1);
////            glColor4f(0.0f,1.0f,0.0f,0.6f);
////            glBegin(GL_LINES);
////
////            for(size_t i=0; i<vpKFs.size(); i++)
////            {
////                // Covisibility Graph
////                const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
////                cv::Mat Ow = vpKFs[i]->GetCameraCenter();
////                if(!vCovKFs.empty())
////                {
////                    for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
////                    {
////                        if((*vit)->mnId<vpKFs[i]->mnId)
////                            continue;
////                        cv::Mat Ow2 = (*vit)->GetCameraCenter();
////                        glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
////                        glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
////                    }
////                }
////
////                // Spanning tree
////                KeyFrame* pParent = vpKFs[i]->GetParent();
////                if(pParent)
////                {
////                    cv::Mat Owp = pParent->GetCameraCenter();
////                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
////                    glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
////                }
////
////                // Loops
////                set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
////                for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
////                {
////                    if((*sit)->mnId<vpKFs[i]->mnId)
////                        continue;
////                    cv::Mat Owl = (*sit)->GetCameraCenter();
////                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
////                    glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
////                }
////            }
////
////            glEnd();
////        }
//    }
//
//    void Viewer::SetCameraPos(position T)
//    {
//        for(int i = 0;i < 16; i++)
//        {
//            Twc.m[i] = T(i);
//        }
//    }
//
//    void Viewer::SetMapPoints(const std::vector<point3d>& map, std::vector<point3d> ref)
//    {
//        vpMPs.clear();
//        vpRefMPs.clear();
//        for(const auto& pt:map)
//        {
//            vpMPs.emplace_back(pt);
//        }
//        for(const auto& pt:ref)
//        {
//            vpRefMPs.emplace_back(pt);
//        }
//    }
//
//    void Viewer::SetKeyFrames(std::vector<position> KFs)
//    {
//        vpKFs.clear();
//        for(const auto& p:KFs)
//        {
//            vpKFs.emplace_back(p);
//        }
//    }
//
//}