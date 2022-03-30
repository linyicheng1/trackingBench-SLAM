#include "types/Frame.h"

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

    Frame::Frame(const cv::Mat &imGray, const double &timeStamp, TRACKING_BENCH::Extractor *extractor, Distortion* distortion):
    mpExtractor(extractor), mpDistortion(distortion)
    {
        mnId = nNextId++;
        ExtractPoint(imGray);
        N = (int)mvKeys.size();
        mpCamera->UnDistortKeyPoints();
        mvpMapPoints = std::vector<MapPoint*>(N, static_cast<MapPoint*>(nullptr));
    }

    void Frame::ExtractPoint(const cv::Mat &img)
    {
        (*mpExtractor)(img, cv::Mat(), mvKeysUn, mDescriptors);
    }

    void Frame::SetPose(const cv::Mat& Tcw)
    {
        mTcw = Tcw.clone();
        mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
        mRwc = mRcw.t();
        mtcw = mTcw.rowRange(0,3).col(3);
        mOw = -mRcw.t() * mtcw;
    }

    std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel,
                                                 const int maxLevel) const
    {
        std::vector<size_t> vIndices;
        vIndices.reserve(N);

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


}

