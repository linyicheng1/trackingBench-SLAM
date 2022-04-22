#include "types/MapPoint.h"

#include <utility>
#include "types/Frame.h"
#include "types/Map.h"
#include "matchers/matcher.h"

namespace TRACKING_BENCH
{
    long unsigned int MapPoint::nNextId = 0;
    std::mutex MapPoint::mGlobalMutex;

    MapPoint::MapPoint(const Eigen::Vector3f &Pos, std::shared_ptr<Map>&  pMap,
                       std::shared_ptr<Frame>& pFrame,
                       std::shared_ptr<Feature>&  features):
            nObs(0),mWorldPos(Pos),mpRefKF(pFrame), mnVisible(1),
            mnFound(1), mbBad(false), mpReplaced(nullptr), mpMap(pMap),
            mpRefFeature(features)
    {
        Eigen::Vector3f Ow = mpRefKF->GetCameraCenter();
        mNormalVector = mWorldPos - Ow;
        mNormalVector = mNormalVector.normalized();

        Eigen::Vector3f PC = Pos - Ow;
        const float dist = PC.norm();
        const int level = mpRefFeature->kp.octave;
        const float levelScaleFactor =  mpRefKF->GetScaleFactors()[level];
        const int nLevels = mpRefKF->GetLevels();

        mfMaxDistance = dist*levelScaleFactor;
        mfMinDistance = mfMaxDistance/mpRefKF->GetScaleFactors()[nLevels-1];

        if(!mpRefKF->GetDescriptors().empty())
            mpRefKF->GetDescriptors().row(mpRefFeature->idxF).copyTo(mDescriptor);

        // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
        std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
        mnId=nNextId++;
        mpRefKF = nullptr;
    }

    /**
     * @brief set map point pos in world
     * @param pos
     */
    void MapPoint::SetWorldPos(const Eigen::Vector3f &pos)
    {
        std::unique_lock<std::mutex> lock2(mGlobalMutex);
        std::unique_lock<std::mutex> lock(mMutexPos);
        mWorldPos = pos;
    }

    /**
     * @brief get map point pos in world
     * @return pos
     */
    Eigen::Vector3f MapPoint::GetWorldPos()
    {
        //std::unique_lock<std::mutex> lock(mMutexPos);
        return mWorldPos;
    }
    /**
     * @brief get normal vector
     * @return normal vector
     */
    Eigen::Vector3f MapPoint::GetNormal()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return mNormalVector;
    }

    std::shared_ptr<Frame> MapPoint::GetReferenceFrame()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    std::map<std::shared_ptr<Frame>, size_t> MapPoint::GetObservations()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mObservations;
    }


    int MapPoint::Observations()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return nObs;
    }

    void MapPoint::AddObservation(const std::shared_ptr<Frame>& pKF, size_t idx)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
            return;
        mObservations[pKF] = idx;
        nObs ++;
    }

    void MapPoint::EraseObservation(const std::shared_ptr<Frame>& pKF)
    {
        bool bBad = false;
        {
            std::unique_lock<std::mutex> lock(mMutexFeatures);
            if(mObservations.count(pKF))
            {
                int idx = (int)mObservations[pKF];
                nObs --;

                mObservations.erase(pKF);
                if(mpRefKF == pKF)
                    mpRefKF = mObservations.begin()->first;
                if(nObs <= 2)
                    bBad = true;
            }
        }
        if(bBad)
            SetBadFlag();
    }

    int MapPoint::GetIndexInFrame(const std::shared_ptr<Frame>& pKF)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
            return (int)mObservations[pKF];
        else
            return -1;
    }

    bool MapPoint::IsInFrame(const std::shared_ptr<Frame>& pKF)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return (mObservations.count(pKF));
    }

    void MapPoint::SetBadFlag()
    {
        std::map<std::shared_ptr<Frame>, size_t> obs;
        {
            std::unique_lock<std::mutex> lock(mMutexFeatures);
            std::unique_lock<std::mutex> lock2(mMutexPos);
            mbBad = true;
            obs = mObservations;
            mObservations.clear();
        }
        for(auto &mit:obs)
        {
            std::shared_ptr<Frame> pKF = mit.first;
            pKF->EraseMapPointMatch(mit.second);
        }
        mpMap->EraseMapPoint(static_cast<std::shared_ptr<MapPoint>>(this));
    }

    bool MapPoint::isBad()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        return mbBad;
    }

    void MapPoint::Replace(const std::shared_ptr<MapPoint>& pMP)
    {
        if(pMP->mnId == this->mnId)
            return;

        int nVisible,nFound;
        std::map<std::shared_ptr<Frame>, size_t> obs;
        {
            std::unique_lock<std::mutex> lock(mMutexFeatures);
            std::unique_lock<std::mutex> lock2(mMutexPos);
            obs = mObservations;
            mObservations.clear();
            mbBad = true;
            nVisible = mnVisible;
            nFound = mnFound;
            mpReplaced = pMP;
        }
        for(auto& mit:obs)
        {
            std::shared_ptr<Frame> pKF = mit.first;
            if(!pMP->IsInFrame(pKF))
            {
                pKF->ReplaceMapPointMatch(mit.second, pMP);
                pMP->AddObservation(pKF, mit.second);
            }
            else
            {
                pKF->EraseMapPointMatch(mit.second);
            }
        }
        pMP->IncreaseFound(nFound);
        pMP->IncreaseVisible(nVisible);
        pMP->ComputeDistinctiveDescriptors();
        mpMap->EraseMapPoint(static_cast<std::shared_ptr<MapPoint>>(this));
    }

    std::shared_ptr<MapPoint> MapPoint::GetReplaced()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    float MapPoint::GetMinDistanceInvariance()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return 0.8f * mfMinDistance;
    }

    float MapPoint::GetMaxDistanceInvariance()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return 1.2f * mfMaxDistance;
    }

    int MapPoint::PredictScale(const float &currentDist, std::shared_ptr<Frame> pF)
    {
        float ratio;
        {
            std::unique_lock<std::mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }
        // TODO
        int nScale = 0;//ceil(log(ratio) / pF->mfLogScaleFactor);
        if(nScale < 0)
            nScale = 0;
        else if(nScale >= pF->GetLevels())
            nScale = pF->GetLevels() - 1;
        return nScale;
    }

    void MapPoint::IncreaseVisible(int n)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mnVisible+=n;
    }

    void MapPoint::IncreaseFound(int n)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mnFound+=n;
    }

    float MapPoint::GetFoundRatio()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound)/mnVisible;
    }

    void MapPoint::ComputeDistinctiveDescriptors()
    {
        // Retrieve all observed descriptors
        std::vector<cv::Mat> vDescriptors;

        std::map<std::shared_ptr<Frame>,size_t> observations;

        {
            std::unique_lock<std::mutex> lock1(mMutexFeatures);
            if(mbBad)
                return;
            observations=mObservations;
        }

        if(observations.empty())
            return;

        vDescriptors.reserve(observations.size());

        for(auto & observation : observations)
        {
            std::shared_ptr<Frame> pKF = observation.first;
            // TODO
//            if(!pKF->isBad())
//                vDescriptors.push_back(pKF->GetDescriptors().row((int)observation.second));
        }

        if(vDescriptors.empty())
            return;

        // Compute distances between them
        const size_t N = vDescriptors.size();

        float Distances[N][N];
        for(size_t i=0;i<N;i++)
        {
            Distances[i][i]=0;
            for(size_t j=i+1;j<N;j++)
            {
                int distij = Matcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
                Distances[i][j] = (float)distij;
                Distances[j][i ]= (float)distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for(size_t i=0;i<N;i++)
        {
            std::vector<int> vDists(Distances[i],Distances[i]+N);
            sort(vDists.begin(),vDists.end());
            int median = vDists[(int)(0.5f*(float)(N-1))];

            if(median<BestMedian)
            {
                BestMedian = median;
                BestIdx = (int)i;
            }
        }

        {
            std::unique_lock<std::mutex> lock(mMutexFeatures);
            mDescriptor = vDescriptors[BestIdx].clone();
        }
    }

    void MapPoint::UpdateNormalAndDepth()
    {
        std::map<std::shared_ptr<Frame>, size_t> observations;
        std::shared_ptr<Frame> pRefKF;
        Eigen::Vector3f Pos;
        {
            std::unique_lock<std::mutex> lock1(mMutexFeatures);
            std::unique_lock<std::mutex> lock2(mMutexPos);
            if(mbBad)
                return;
            observations=mObservations;
            pRefKF = mpRefKF;
            Pos = mWorldPos;
        }

        if(observations.empty())
            return;

        Eigen::Vector3f normal;
        int n=0;
        for(auto & observation : observations)
        {
            std::shared_ptr<Frame> pKF = observation.first;
            Eigen::Vector3f Owi = pKF->GetCameraCenter();
            Eigen::Vector3f normali = mWorldPos - Owi;
            normal = normal + normali.normalized();
            n++;
        }

        Eigen::Vector3f PC = Pos - pRefKF->GetCameraCenter();
        const float dist = PC.norm();
        const int level = mpRefFeature->kp.octave;
        const float levelScaleFactor =  pRefKF->GetScaleFactors()[level];
        const int nLevels = pRefKF->GetLevels();

        {
            std::unique_lock<std::mutex> lock3(mMutexPos);
            mfMaxDistance = dist*levelScaleFactor;
            mfMinDistance = mfMaxDistance/pRefKF->GetScaleFactors()[nLevels-1];
            mNormalVector = normal/n;
        }
    }

    std::vector<std::shared_ptr<Feature>> MapPoint::GetFeatures()
    {
        return mFeatures;
    }

    std::shared_ptr<Feature> MapPoint::GerReferenceFeature()
    {
        return mpRefFeature;
    }
}
