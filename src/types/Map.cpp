#include "types/Map.h"
#include "types/MapPoint.h"
#include "types/Frame.h"

namespace TRACKING_BENCH
{

    Map::Map():mnMaxKFid(0)
    {
    }

    Map::~Map()
    {
    }

    void Map::AddKeyFrame(std::shared_ptr<Frame> pKF)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        mspKeyFrames.insert(pKF);
        if(pKF->GetId())
        {
            mnMaxKFid = pKF->GetId();
        }
    }

    void Map::AddMapPoint(const std::shared_ptr<MapPoint>& pMP)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        mspMapPoints.insert(pMP);
    }

    void Map::EraseMapPoint(const std::shared_ptr<MapPoint>& pMP)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        mspMapPoints.erase(pMP);
    }

    void Map::EraseKeyFrame(std::shared_ptr<Frame> pKF)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        mspKeyFrames.erase(pKF);
    }

    void Map::EraseMapPointSafe(const std::shared_ptr<MapPoint>& pMP)
    {
        auto kfs = pMP->GetObservations();
        for(auto &kf:kfs)
        {
            kf.first->EraseMapPointMatch(pMP);
        }
        mspMapPoints.erase(pMP);
    }

    void Map::EraseKeyFrameSafe(const std::shared_ptr<Frame>& pKF)
    {
        auto pts = pKF->GetMapPointMatches();
        for(auto &pt:pts)
        {
            EraseMapPointSafe(pt);
        }
        mspKeyFrames.erase(pKF);
    }

    std::vector<std::shared_ptr<Frame>> Map::GetAllKeyFrames()
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        std::vector<std::shared_ptr<Frame>> vKF(mspKeyFrames.begin(), mspKeyFrames.end());
        return vKF;
    }

    std::vector<std::shared_ptr<MapPoint>> Map::GetAllMapPoints()
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        std::vector<std::shared_ptr<MapPoint>> vKP(mspMapPoints.begin(), mspMapPoints.end());
        return vKP;
    }

    long unsigned int Map::MapPointsInMap()
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        return mspMapPoints.size();
    }

    long unsigned int Map::KeyFramesInMap()
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        return mspKeyFrames.size();
    }

    void Map::clear()
    {
        mspKeyFrames.clear();
        mspMapPoints.clear();
        mnMaxKFid = 0;
    }

    long unsigned int Map::GetMaxKFid()
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        return mnMaxKFid;
    }

    void Map::RemoveOldFrames(int num)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        if(mspKeyFrames.size() > num)
        {
            const int del_num = (int)mspKeyFrames.size() - num;
            for (int i = 0;i < del_num;i ++)
            {
                mspKeyFrames.erase(mspKeyFrames.end());
            }
        }
    }
}
