#ifndef TRACKING_BENCH_MAPPOINT_H
#define TRACKING_BENCH_MAPPOINT_H
#include <opencv2/core.hpp>
#include <mutex>
#include <map>
#include <Eigen/Core>
#include <memory>
#include <utility>

namespace TRACKING_BENCH
{
    class Map;
    class Frame;
    class MapPoint;
    class Feature;


    class MapPoint
    {
    public:

        MapPoint(const Eigen::Vector3f &Pos, std::shared_ptr<Map>&  pMap,
                 std::shared_ptr<Frame>& pFrame,
                 std::shared_ptr<Feature>&  features, cv::Mat des);
        // pos
        void SetWorldPos(const Eigen::Vector3f& pos);
        Eigen::Vector3f GetWorldPos();

        // normal
        Eigen::Vector3f GetNormal();

        // related frames
        std::shared_ptr<Frame> GetReferenceFrame();
        std::map<std::shared_ptr<Frame>, size_t> GetObservations();
        bool GetCloseViewObs(const Eigen::Vector3f& framePos, std::shared_ptr<Feature>& ftr)const;
        std::vector<std::shared_ptr<Feature>> GetFeatures();
        std::shared_ptr<Feature> GetReferenceFeature();
        int Observations();
        int GetIndexInFrame(const std::shared_ptr<Frame>& pKF);
        bool IsInFrame(const std::shared_ptr<Frame>& pKF);

        // observation operator
        void AddObservation(const std::shared_ptr<Frame>& pKF, size_t idx);
        void EraseObservation(const std::shared_ptr<Frame>& pKF);

        void SetBadFlag();
        bool isBad();

        void IncreaseVisible(int n=1);
        void IncreaseFound(int n=1);
        float GetFoundRatio();
        inline int GetFound() const{return mnFound;}

        void ComputeDistinctiveDescriptors();

        void Replace(const std::shared_ptr<MapPoint>& pMP);
        std::shared_ptr<MapPoint> GetReplaced();
        cv::Mat GetDescriptor(){return mDescriptor;}

        // distance
        void UpdateNormalAndDepth();

        float GetMinDistanceInvariance();
        float GetMaxDistanceInvariance();
        int PredictScale(const float &currentDist, std::shared_ptr<Frame> pF);

        // tracking
//        float mTrackProjX;
//        float mTrackProjY;
//        float mTrackProjXR;
//        bool mbTrackInView;
//        int mnTrackScaleLevel;
//        float mTrackViewCos;
//        long unsigned int mnTrackReferenceForFrame;
//        long unsigned int mnLastFrameSeen;
        long unsigned int last_projected_id;
        int n_failed_reproj = 0;
        cv::Mat_<double> grad;
        int type;
        static std::mutex mGlobalMutex;
    private:
        long unsigned int mnId;
        static long unsigned int nNextId;

        int nObs;

        Eigen::Vector3f mWorldPos;

        std::vector<std::shared_ptr<Feature>> mFeatures;
        std::shared_ptr<Feature> mpRefFeature;
        std::map<std::shared_ptr<Frame>, size_t> mObservations;
        Eigen::Vector3f mNormalVector;
        cv::Mat mDescriptor;
        std::shared_ptr<Frame> mpRefKF;

        int mnVisible;
        int mnFound;

        bool mbBad;
        std::shared_ptr<MapPoint> mpReplaced;

        float mfMinDistance;
        float mfMaxDistance;

        std::shared_ptr<Map> mpMap;

        std::mutex mMutexPos;
        std::mutex mMutexFeatures;
    };
}

#endif //TRACKING_BENCH_MAPPOINT_H
