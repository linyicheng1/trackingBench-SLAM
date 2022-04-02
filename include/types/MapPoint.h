#ifndef TRACKING_BENCH_MAPPOINT_H
#define TRACKING_BENCH_MAPPOINT_H
#include <opencv2/core.hpp>
#include <mutex>
#include <map>

namespace TRACKING_BENCH
{
    class KeyFrame;
    class Map;
    class Frame;

    class MapPoint
    {
    public:
        MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
        MapPoint(const cv::Mat &Pos, Frame* pFrame, Map* pMap, const int &idxF);

        void SetWorldPos(const cv::Mat &pos);
        cv::Mat GetWorldPos();

        cv::Mat GetNormal();
        KeyFrame* GetReferenceKeyFrame();

        std::map<KeyFrame*, size_t> GetObservations();
        int Observations();

        void AddObservation();
        void EraseObservation();

        int GetIndexInKeyFrame(KeyFrame* pKF);
        bool IsInKeyFrame(KeyFrame* pKF);

        void SetBadFlag();
        bool isBad();

        void Replace(MapPoint* pMP);
        MapPoint* GetReplaced();


    private:
        long unsigned int mnId;
        static long unsigned int nNextId;
        long int mnFirstKFid;
        long int mnFirstFrame;
        int nObs;

        cv::Mat mWorldPos;

        std::map<KeyFrame*, size_t> mObservations;
        cv::Mat mNormalVector;
        cv::Mat mDescriptor;
        KeyFrame* mpRefKf;

        int mnVisible;
        int mnFound;

        bool mbBad;
        MapPoint* mpReplaced;

        float mfMinDistance;
        float mfMaxDistance;

        Map* mpMap;

        std::mutex mMutexPos;
        std::mutex mMutexFeatures;
        static std::mutex mGlobalMutex;
    };
}

#endif //TRACKING_BENCH_MAPPOINT_H
