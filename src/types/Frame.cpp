#include "types/Frame.h"
#include "types/MapPoint.h"
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace TRACKING_BENCH
{
    long unsigned int Frame::nNextId = 0;
    CameraModel::CameraModel(const CameraModel &d):
    fx(d.fx),fy(d.fy),cx(d.cx),cy(d.cy),inv_fx(d.inv_fx),inv_fy(d.inv_fy)
    {
    }

    CameraModel::CameraModel(cv::Mat k, cv::Mat &distCoef)
    {
        fx = k.at<float>(0,0);
        fy = k.at<float>(1,1);
        cx = k.at<float>(0,2);
        cy = k.at<float>(1, 2);
        inv_fx = 1.0f / fx;
        inv_fy = 1.0f / fy;
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);
    }

    int CameraModel::UnDistortKeyPoints()
    {

    }

    void CameraModel::ComputeImageBounds(const cv::Mat &imLeft)
    {
        if(mDistCoef.at<float>(0)!=0.0)
        {
            cv::Mat mat(4,2,CV_32F);
            mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
            mat.at<float>(1,0)=(float)imLeft.cols; mat.at<float>(1,1)=0.0;
            mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=(float)imLeft.rows;
            mat.at<float>(3,0)=(float)imLeft.cols; mat.at<float>(3,1)=(float)imLeft.rows;

            // Undistort corners
            mat=mat.reshape(2);
            cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
            mat=mat.reshape(1);

            mnMinX = (int)fminf(mat.at<float>(0,0),mat.at<float>(2,0));
            mnMaxX = (int)fmaxf(mat.at<float>(1,0),mat.at<float>(3,0));
            mnMinY = (int)fminf(mat.at<float>(0,1),mat.at<float>(1,1));
            mnMaxY = (int)fmaxf(mat.at<float>(2,1),mat.at<float>(3,1));

        }
        else
        {
            mnMinX = 0.0f;
            mnMaxX = imLeft.cols;
            mnMinY = 0.0f;
            mnMaxY = imLeft.rows;
        }
    }

    Frame::Frame(const TRACKING_BENCH::Frame &frame)
    {

    }

    Frame::Frame(const cv::Mat &imGray, const double &timeStamp, TRACKING_BENCH::Extractor *extractor, CameraModel* camera):
    mpExtractor(extractor), mpCamera(camera)
    {
        mnId = nNextId++;
        ExtractPoint(imGray);
        mpCamera->UnDistortKeyPoints();
        mvpMapPoints = std::vector<MapPoint*>(mvKeys.size(), static_cast<MapPoint*>(nullptr));
    }

    void Frame::SetPose(const cv::Mat& Tcw)
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        mTcw = Tcw.clone();
        cv::Mat Rcw = mTcw.rowRange(0, 3).colRange(0, 3);
        cv::Mat tcw = mTcw.rowRange(0,3).col(3);
        cv::Mat Rwc = Rcw.t();
        mOw = -Rwc * tcw;

        mTwc = cv::Mat::eye(4, 4, Tcw.type());
        Rwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
        mOw.copyTo(mTwc.rowRange(0, 3).col(3));
    }
    cv::Mat Frame::GetPose()
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return mTcw.clone();
    }

    cv::Mat Frame::GetPoseInverse()
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return mTwc.clone();
    }

    cv::Mat Frame::GetCameraCenter()
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return mOw.clone();
    }

    cv::Mat Frame::GetRotation()
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return mTwc.rowRange(0, 3).colRange(0, 3).clone();
    }

    cv::Mat Frame::GetTranslation()
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return mTwc.rowRange(0, 3).col(3).clone();
    }

    void Frame::ExtractPoint(const cv::Mat &img)
    {
        (*mpExtractor)(img, cv::Mat(), mvKeysUn, mDescriptors);
    }
    void Frame::AssignFeaturesToGrid()
    {
        int nReserve = (int)(0.5f * (float)mvKeysUn.size() / (float)(FRAME_GRID_ROWS * FRAME_GRID_COLS));
        for (auto & i : mGrid)
            for (auto & j : i)
                j.reserve(nReserve);
        for(size_t i=0; i< mvKeysUn.size(); i ++)
        {
            const cv::KeyPoint &kp = mvKeysUn[i];
            int nGridPosX, nGridPosY;
            if(PosInGrid(kp, nGridPosX, nGridPosY))
                mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }
    std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel,
                                                 const int maxLevel) const
    {
        std::vector<size_t> vIndices;
        vIndices.reserve(mvKeysUn.size());

        const int nMinCellX = std::max(0,(int)std::floor((x-(float)mpCamera->mnMinX-r)*mpCamera->mfGridElementWidthInv));
        if(nMinCellX>=FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = std::min((int)FRAME_GRID_COLS-1,(int)std::ceil((x-(float)mpCamera->mnMinX+r)*mpCamera->mfGridElementWidthInv));
        if(nMaxCellX<0)
            return vIndices;

        const int nMinCellY = std::max(0,(int)std::floor((y-(float)mpCamera->mnMinY-r)*mpCamera->mfGridElementHeightInv));
        if(nMinCellY>=FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = std::min((int)FRAME_GRID_ROWS-1,(int)std::ceil((y-(float)mpCamera->mnMinY+r)*mpCamera->mfGridElementHeightInv));
        if(nMaxCellY<0)
            return vIndices;

        const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

        for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
        {
            for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
            {
                const std::vector<size_t> vCell = mGrid[ix][iy];
                if(vCell.empty())
                    continue;

                for(unsigned long j : vCell)
                {
                    const cv::KeyPoint &kpUn = mvKeysUn[j];
                    if(bCheckLevels)
                    {
                        if(kpUn.octave<minLevel)
                            continue;
                        if(maxLevel>=0)
                            if(kpUn.octave>maxLevel)
                                continue;
                    }

                    const float distx = kpUn.pt.x-x;
                    const float disty = kpUn.pt.y-y;

                    if(std::fabs(distx)<r && std::fabs(disty)<r)
                        vIndices.push_back(j);
                }
            }
        }

        return vIndices;
    }

    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
    {
        posX = round((kp.pt.x - (float)mpCamera->mnMinX) * mpCamera->mfGridElementWidthInv);
        posY = round((kp.pt.y - (float)mpCamera->mnMinY) * (float)mpCamera->mfGridElementHeightInv);

        if(posX<0||posX>=FRAME_GRID_COLS || posY < 0 || posY>= FRAME_GRID_ROWS)
            return false;
        return true;
    }

    std::set<MapPoint *> Frame::GetMapPoints()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        std::set<MapPoint*> s;
        for(auto & mvpMapPoint : mvpMapPoints)
        {
            if(!mvpMapPoint)
                continue;
            MapPoint* pMP = mvpMapPoint;
            if(!pMP->isBad())
                s.insert(pMP);
        }
        return s;
    }

    std::vector<MapPoint *> Frame::GetMapPointMatches()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mvpMapPoints;
    }

    int Frame::TrackedMapPoint(const int &minObs)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        int nPoints = 0;
        const bool bCheckObs = minObs > 0;
        for(int i = 0;i < mvKeysUn.size();i ++)
        {
            MapPoint* pMP = mvpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(bCheckObs)
                    {
                        if(mvpMapPoints[i]->Observations() > minObs)
                            nPoints ++;
                    }
                    else
                        nPoints ++;
                }
            }
        }
        return nPoints;
    }

    MapPoint *Frame::GetMapPoint(const size_t &idx)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mvpMapPoints[idx];
    }




}

