#ifndef TRACKING_BENCH_MAP_H
#define TRACKING_BENCH_MAP_H
#include <vector>
#include <mutex>
#include <set>
#include <list>
#include <memory>

namespace TRACKING_BENCH
{
    class MapPoint;
    class Frame;
    class Map
    {
    public:
        Map();
        ~Map();

        void AddKeyFrame(std::shared_ptr<Frame> pKF);
        void AddMapPoint(const std::shared_ptr<MapPoint>& pMP);
        void EraseMapPoint(const std::shared_ptr<MapPoint>& pMP);
        void EraseKeyFrame(std::shared_ptr<Frame> pKF);
        void EraseMapPointSafe(const std::shared_ptr<MapPoint>& pMP);
        void EraseKeyFrameSafe(const std::shared_ptr<Frame>& pKF);
        void RemoveOldFrames(int num);
        std::vector<std::shared_ptr<Frame>> GetAllKeyFrames();
        std::vector<std::shared_ptr<MapPoint>> GetAllMapPoints();

        long unsigned int MapPointsInMap();
        long unsigned int KeyFramesInMap();

        void clear();
        long unsigned int GetMaxKFid();
        std::mutex mMutexPointCreation;
        std::mutex mMutexMap;
        std::mutex mMutexMapPoints;
        std::list<std::shared_ptr<MapPoint>> mspCandidatesMapPoints;
    private:
        std::set<std::shared_ptr<MapPoint>> mspMapPoints;
        std::set<std::shared_ptr<Frame>> mspKeyFrames;
        long unsigned int mnMaxKFid;
    };
}
#endif //TRACKING_BENCH_MAP_H
