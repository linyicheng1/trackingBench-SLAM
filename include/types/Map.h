#ifndef TRACKING_BENCH_MAP_H
#define TRACKING_BENCH_MAP_H
#include <vector>
#include <mutex>
#include <set>

namespace TRACKING_BENCH
{
    class MapPoint;
    class KeyFrame;
    class Map
    {
    public:
        Map();
        ~Map();

        void AddKeyFrame(KeyFrame* pKF);
        void AddMapPoint(MapPoint* pMP);
        void EraseMapPoint(MapPoint* pMP);
        void EraseKeyFrame(KeyFrame* pKF);

        std::vector<KeyFrame*> GetAllKeyFrames();
        std::vector<MapPoint*> GetAllMapPoints();

        long unsigned int MapPointsInMap();
        long unsigned int KeyFramesInMap();

        void clear();
        long unsigned int GetMaxKFid();
    private:
        std::set<MapPoint*> mspMapPoints;
        std::set<KeyFrame*> mspKeyFrames;
        std::mutex mMutexMap;
        long unsigned int mnMaxKFid;
    };
}
#endif //TRACKING_BENCH_MAP_H
