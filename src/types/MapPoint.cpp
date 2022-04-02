#include "types/MapPoint.h"

namespace TRACKING_BENCH
{
    long unsigned int MapPoint::nNextId = 0;
    std::mutex MapPoint::mGlobalMutex;
    MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap):
    mnFirstKFid()
    {

    }

    MapPoint::MapPoint(const cv::Mat &Pos, Frame *pFrame, Map *pMap, const int &idxF)
    {

    }

    void MapPoint::SetWorldPos(const cv::Mat &pos)
    {

    }

    cv::Mat MapPoint::GetWorldPos()
    {
        return cv::Mat();
    }

    cv::Mat MapPoint::GetNormal()
    {
        return cv::Mat();
    }

    KeyFrame *MapPoint::GetReferenceKeyFrame()
    {
        return nullptr;
    }

    std::map<KeyFrame *, size_t> MapPoint::GetObservations()
    {
        return std::map<KeyFrame *, size_t>();
    }

    int MapPoint::Observations()
    {
        return 0;
    }

    void MapPoint::AddObservation()
    {

    }

    void MapPoint::EraseObservation()
    {

    }

    int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
    {
        return 0;
    }

    bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
    {
        return false;
    }

    void MapPoint::SetBadFlag()
    {

    }

    bool MapPoint::isBad()
    {
        return false;
    }

    void MapPoint::Replace(MapPoint *pMP)
    {

    }

    MapPoint *MapPoint::GetReplaced()
    {
        return nullptr;
    }
}
