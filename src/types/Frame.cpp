#include "types/Frame.h"
#include "types/MapPoint.h"
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <utility>
#include "camera/CameraModel.h"
#include "types/Map.h"

namespace TRACKING_BENCH
{
    long unsigned int Frame::nNextId = 0;

    Frame::Frame(const cv::Mat &imGray, const double &timeStamp, const int level,
                 const float scale, std::shared_ptr<CameraModel> camera):
            mTimeStamp(timeStamp),  mpCamera(std::move(camera)),nLevels(level),scaleFactor(scale)
    {
        mnId = nNextId++;
        mvScaleFactor.resize(nLevels, 1);
        mvInvScaleFactor.resize(nLevels, 1);
        mvLevelSigma2.resize(nLevels, 1);
        mvInvLevelSigma2.resize(nLevels, 1);
        mvImagePyramid.resize(nLevels);
        for(int i = 1;i < nLevels; i++)
        {
            mvScaleFactor.at(i) = mvScaleFactor.at(i-1) * scale;
            mvInvScaleFactor.at(i) = mvInvScaleFactor.at(i-1) / scale;
            mvLevelSigma2.at(i) = mvScaleFactor.at(i) * mvScaleFactor.at(i);
            mvInvLevelSigma2.at(i) = 1.f / mvLevelSigma2.at(i);
        }
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_COLS) / (float)imGray.cols;
        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_ROWS) / (float)imGray.rows;;
        ComputePyramid(imGray);
    }

    //Copy Constructor
    Frame::Frame(const Frame &frame):
            mTimeStamp(frame.mTimeStamp), mvKeys(frame.mvKeys),
            mDescriptors(frame.mDescriptors.clone()),
            mvpMapPoints(frame.mvpMapPoints), mnId(frame.mnId),
            nFeatures(frame.nFeatures),nLevels(frame.nLevels),mbKeyFrame(frame.mbKeyFrame),
            scaleFactor(frame.scaleFactor),mvImagePyramid(frame.mvImagePyramid.begin(), frame.mvImagePyramid.end())
    {
        for(int i=0;i<FRAME_GRID_COLS;i++)
            for(int j=0; j<FRAME_GRID_ROWS; j++)
                mGrid[i][j]=frame.mGrid[i][j];

        SetPose(frame.mTcw);
    }

    // pose
    void Frame::SetPose(const Eigen::Matrix4f& Tcw)
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        mTcw = Tcw;
        Eigen::Matrix3f Rcw = mTcw.block<3, 3>(0,0);
        Eigen::Vector3f tcw = mTcw.block<3, 1>(0 ,3);
        mOw = -Rcw.transpose() * tcw;
        mTwc = Eigen::Matrix4f::Identity();
        mTwc.block<3, 3>(0,0) = Rcw.transpose();
        mTwc.block<3, 1>(0 ,3) = mOw;
    }

    Eigen::Matrix4f Frame::GetPose()
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return mTcw;
    }

    Eigen::Matrix4f Frame::GetPoseInverse()
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return mTwc;
    }

    Eigen::Vector3f Frame::GetCameraCenter()
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return mOw;
    }

    Eigen::Matrix3f Frame::GetRotation()
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return mTwc.block<3, 3>(0, 0);
    }

    Eigen::Vector3f Frame::GetTranslation()
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return mTwc.block<3, 1>(0, 3);
    }

    // feature
    void Frame::SetKeys(std::vector<cv::KeyPoint>& pts,const std::shared_ptr<Frame>& frame, cv::Mat descriptors, bool unDistort)
    {
        mDescriptors = std::move(descriptors);
        mvKeys.clear();
        mvKeys.reserve(pts.size());
        mvCVKeys.clear();
        mvCVKeys.reserve(pts.size());
        for(auto &pt:pts)
        {
            auto f = std::make_shared<Feature>(nullptr, pt, Eigen::Vector3f::Zero(), (int)mvKeys.size());
            if(unDistort)
            {
                f->px_un = mpCamera->UndistortPoint(f->px).block<2, 1>(0, 0);
            }
            mvKeys.emplace_back(f);
            mvCVKeys.emplace_back(pt.pt);
        }
        mvpMapPoints.resize(pts.size(), nullptr);
        mvbOutlier.resize(pts.size(), false);
    }

    void Frame::AddKeys(std::vector<Feature>& pts, cv::Mat descriptors, bool unDistort)
    {
        mDescriptors = std::move(descriptors);
        mvKeys.reserve(pts.size() + mvKeys.size());
        mvCVKeys.reserve(pts.size() + mvKeys.size());
        for(auto &pt:pts)
        {
            if(unDistort)
            {
                pt.px_un = mpCamera->UndistortPoint(pt.px).block<2, 1>(0, 0);
            }
            // mvKeys.emplace_back(pt);
        }
    }

    void Frame::UnDistortPoints()
    {
        for(auto &pt:mvKeys)
        {
            pt->px_un = mpCamera->UndistortPoint(pt->px).block<2, 1>(0, 0);
        }
    }

    void Frame::UnDistortImage()
    {
        mpCamera->UndistortImage(mImage, mImage);
    }

    std::vector<cv::Mat> Frame::GetVectorDescriptors() const
    {
        std::vector<cv::Mat> vCurrentDesc;
        vCurrentDesc.reserve(mDescriptors.rows);
        for (int j = 0; j < mDescriptors.rows; j ++)
            vCurrentDesc.push_back(mDescriptors.row(j));
        return vCurrentDesc;
    }

    void Frame::AssignFeaturesToGrid()
    {
        int nReserve = (int)(0.5f * (float)mvKeys.size() / (float)(FRAME_GRID_ROWS * FRAME_GRID_COLS));
        for (auto & i : mGrid)
            for (auto & j : i)
                j.reserve(nReserve);
        for(size_t i=0; i< mvKeys.size(); i ++)
        {
            const auto &kp = mvKeys[i];
            int nGridPosX, nGridPosY;
            if(PosInGrid(kp->kp, nGridPosX, nGridPosY))
                mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel,
                                                 const int maxLevel) const
    {
        std::vector<size_t> vIndices;
        vIndices.reserve(mvKeys.size());

        const int nMinCellX = std::max(0,(int)std::floor((x-r)*mfGridElementWidthInv));
        if(nMinCellX>=FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = std::min((int)FRAME_GRID_COLS-1,(int)std::ceil((x+r)*mfGridElementWidthInv));
        if(nMaxCellX<0)
            return vIndices;

        const int nMinCellY = std::max(0,(int)std::floor((y-r)*mfGridElementHeightInv));
        if(nMinCellY>=FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = std::min((int)FRAME_GRID_ROWS-1,(int)std::ceil((y+r)*mfGridElementHeightInv));
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
                    const auto &kpUn = mvKeys[j];
                    if(bCheckLevels)
                    {
                        if(kpUn->kp.octave<minLevel)
                            continue;
                        if(maxLevel>=0)
                            if(kpUn->kp.octave>maxLevel)
                                continue;
                    }

                    const float distx = kpUn->kp.pt.x-x;
                    const float disty = kpUn->kp.pt.y-y;

                    if(std::fabs(distx)<r && std::fabs(disty)<r)
                        vIndices.push_back(j);
                }
            }
        }
        return vIndices;
    }

    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) const
    {
        posX = (int)round((kp.pt.x) * mfGridElementWidthInv);
        posY = (int)round((kp.pt.y) * mfGridElementHeightInv);

        if(posX<0||posX>=FRAME_GRID_COLS || posY < 0 || posY>= FRAME_GRID_ROWS)
            return false;
        return true;
    }

    void Frame::SetBow(const std::shared_ptr<ORBVocabulary>& voc)
    {
        voc->transform(GetVectorDescriptors(), mBowVec, mFeatVec, 4);
    }

    std::set<std::shared_ptr<MapPoint>> Frame::GetMapPoints()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        std::set<std::shared_ptr<MapPoint>> s;
        for(auto & mvpMapPoint : mvpMapPoints)
        {
            if(!mvpMapPoint)
                continue;
            if(!mvpMapPoint->isBad())
                s.insert(mvpMapPoint);
        }
        return s;
    }

    std::vector<std::shared_ptr<MapPoint>> Frame::GetMapPointMatches()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mvpMapPoints;
    }

    int Frame::TrackedMapPoint(const int &minObs)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        int nPoints = 0;
        const bool bCheckObs = minObs > 0;
        for(const auto& pMP:mvpMapPoints)
        {
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(bCheckObs)
                    {
                        if(pMP->Observations() > minObs)
                            nPoints ++;
                    }
                    else
                        nPoints ++;
                }
            }
        }
        return nPoints;
    }

    std::shared_ptr<MapPoint> Frame::GetMapPoint(const size_t &idx)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mvpMapPoints[idx];
    }

    void Frame::AddMapPoint(std::shared_ptr<MapPoint>& pMP, const size_t &idx)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = pMP;
    }

    void Frame::AddMapPoints(const std::shared_ptr<Frame>& ref, const std::vector<cv::DMatch>& matches)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        for(const auto& match:matches)
        {
            mvpMapPoints[match.trainIdx] = ref->GetMapPoint(match.queryIdx);
        }
    }

    void Frame::AddMapPoints(const std::shared_ptr<Map>& map, const std::vector<cv::DMatch>& matches)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        for(const auto& match:matches)
        {
            mvpMapPoints[match.trainIdx] = map->GetAllMapPoints().at(match.queryIdx);
        }
    }

    void Frame::EraseMapPointMatch(const size_t &idx)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = nullptr;
    }

    void Frame::EraseMapPointMatch(const std::shared_ptr<MapPoint>& pMP)
    {
        int idx = pMP->GetIndexInFrame(static_cast<std::shared_ptr<Frame>>(this));
        if(idx >= 0)
            mvpMapPoints[idx] = static_cast<std::shared_ptr<MapPoint>>(nullptr);
    }

    void Frame::ReplaceMapPointMatch(const size_t &idx, std::shared_ptr<MapPoint> pMP)
    {
        mvpMapPoints[idx] = std::move(pMP);
    }

    bool Frame::IsInImage(const float &x, const float &y) const
    {
        // TODO
        return false;//mpCamera->IsInFrame(Eigen::Vector2f(x, y));
    }

    bool Frame::IsInFrustum(const std::shared_ptr<MapPoint>& pMP, int &level, float &x, float &y, float &viewingCosLimit)
    {
        // 3D in world
        Eigen::Vector3f P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const Eigen::Vector3f Pc = mTcw.block<3, 3>(0, 0) * P + mTcw.block<3, 1>(0 ,3);

        // Check positive depth
        if(Pc[2] < 0.0f)
            return false;

        // image
        auto uv = mpCamera->World2Cam(Pc);
        // TODO
//        if(!mpCamera->IsInFrame(uv))
//            return false;

        // check distance
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const Eigen::Vector3f PO = P - mOw;
        const float dist = PO.norm();

        if(dist < minDistance || dist > maxDistance)
            return false;
        // check viewing angle
        Eigen::Vector3f Pn = pMP->GetNormal();
        const float viewCos = PO.dot(Pn) / dist;

        if(viewCos < viewingCosLimit)
            return false;
        // predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist, static_cast<std::shared_ptr<Frame>>(this));
        // data used by the tracking

        x = (float)uv[0];
        y = (float)uv[1];
        level = nPredictedLevel;
        viewingCosLimit = viewCos;

        return true;
    }

    void Frame::ComputePyramid(cv::Mat image)
    {
        for (int level = 0; level < nLevels; ++level)
        {
            float scale = mvInvScaleFactor[level];
            cv::Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
            cv::Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
            cv::Mat temp(wholeSize, image.type()), masktemp;
            mvImagePyramid[level] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

            // Compute the resized image
            if( level != 0 )
            {
                cv::resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, CV_INTER_LINEAR);

                copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               cv::BORDER_REFLECT_101+cv::BORDER_ISOLATED);
            }
            else
            {
                copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               cv::BORDER_REFLECT_101);
            }
        }
    }

    cv::Mat Frame::Equalize()
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(mvImagePyramid[0], mImage);
        return mImage;
    }




}

