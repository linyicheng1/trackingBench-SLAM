#include "types/Map.h"
#include "types/KeyFrame.h"
#include "types/MapPoint.h"

namespace TRACKING_BENCH
{

    Map::Map():mnMaxKFid(0)
    {
    }

    Map::~Map()
    {
    }

    void Map::AddKeyFrame(KeyFrame* pKF)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        mspKeyFrames.insert(pKF);
        if(pKF->getId())
        {
            mnMaxKFid = pKF->getId();
        }
    }

    void Map::AddMapPoint(MapPoint* pMP)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        mspMapPoints.insert(pMP);
    }

    void Map::EraseMapPoint(MapPoint* pMP)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        mspMapPoints.erase(pMP);
    }

    void Map::EraseKeyFrame(KeyFrame* pKF)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        mspKeyFrames.erase(pKF);
    }

    std::vector<KeyFrame *> Map::GetAllKeyFrames()
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        std::vector<KeyFrame*> vKF(mspKeyFrames.begin(), mspKeyFrames.end());
        return vKF;
    }

    std::vector<MapPoint *> Map::GetAllMapPoints()
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        std::vector<MapPoint*> vKP(mspMapPoints.begin(), mspMapPoints.end());
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
        for(auto mp : mspMapPoints)
        {
            delete mp;
        }
        for(auto kf : mspKeyFrames)
        {
            delete kf;
        }
        mspKeyFrames.clear();
        mspMapPoints.clear();
        mnMaxKFid = 0;
    }

    long unsigned int Map::GetMaxKFid()
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        return mnMaxKFid;
    }
}
