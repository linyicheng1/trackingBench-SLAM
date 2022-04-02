#ifndef TRACKING_BENCH_KEYFRAME_H
#define TRACKING_BENCH_KEYFRAME_H
#include "types/Frame.h"
#include <map>

namespace TRACKING_BENCH
{
    class Map;
    class MapPoint;

    class KeyFrame : public Frame
    {
    public:
        KeyFrame(Frame &F, Map* pMap);

        // MapPoints
        void AddMapPoint(MapPoint* pMP, const size_t& idx);
        void EraseMapPointMatch(const size_t &idx);
        void EraseMapPointMatch(MapPoint* pMP);
        void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);

        // connection
        void AddConnection(KeyFrame* pKF, const int &weight);
        void EraseConnection(KeyFrame* pKF);
        void UpdateConnections();
        void UpdateBestCovisibles();
        std::set<KeyFrame*> GetConnectedKeyFrames();
        std::vector<KeyFrame*> GetVectorCovisibleKeyFrames();
        std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
        std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
        int GetWeight(KeyFrame* pKF);
        static bool WeightComp(int a, int b){
            return a > b;
        }
        // frame
        void SetBadFlag();
        bool isBad();
    private:
        long unsigned int mnFrameId;
        static long unsigned int nNextId;
        // connection
        std::map<KeyFrame*, int> mConnectedKeyFrameWeights;
        std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
        std::vector<int> mvOrderedWeights;
        std::mutex mMutexConnections;
        // Map
        Map* mpMap;
    };
}

#endif //TRACKING_BENCH_KEYFRAME_H
