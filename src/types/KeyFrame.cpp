#include "types/KeyFrame.h"
#include "types/MapPoint.h"

namespace TRACKING_BENCH
{

    KeyFrame::KeyFrame(Frame &F, Map *pMap)
    {

    }

    void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = pMP;
    }

    void KeyFrame::EraseMapPointMatch(const size_t &idx)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = static_cast<MapPoint*>(nullptr);
    }

    void KeyFrame::EraseMapPointMatch(MapPoint *pMP)
    {
        int idx = pMP->GetIndexInKeyFrame(this);
        if(idx >= 0)
            mvpMapPoints[idx] = static_cast<MapPoint*>(nullptr);
    }

    void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP)
    {
        mvpMapPoints[idx] = pMP;
    }

    void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
    {
        {
            std::unique_lock<std::mutex> lock(mMutexConnections);
            if(!mConnectedKeyFrameWeights.count(pKF) || mConnectedKeyFrameWeights[pKF]!=weight)
                mConnectedKeyFrameWeights[pKF]=weight;
            else
                return;
        }
        UpdateBestCovisibles();
    }

    void KeyFrame::EraseConnection(KeyFrame *pKF)
    {
        bool bUpdate = false;
        {
            std::unique_lock<std::mutex> lock(mMutexConnections);
            if(mConnectedKeyFrameWeights.count(pKF))
            {
                mConnectedKeyFrameWeights.erase(pKF);
                bUpdate = true;
            }
        }
        if(bUpdate)
            UpdateBestCovisibles();
    }

    void KeyFrame::UpdateConnections()
    {
        std::map<KeyFrame*, int> KFCounter;
        std::vector<MapPoint*> vpMP;

        {
            std::unique_lock<std::mutex> lockMPs(mMutexFeatures);
            vpMP = mvpMapPoints;
        }

        for(auto vit:vpMP)
        {
            if(!vit)
                continue;
            if(vit->isBad())
                continue;

            std::map<KeyFrame*, size_t> observations = vit->GetObservations();
            for(auto mit:observations)
            {
                if(mit.first->mnId == mnId)
                    continue;
                KFCounter[mit.first] ++;
            }
        }

        if(KFCounter.empty())
            return;

        int nMax = 0;
        KeyFrame* pKFMax = nullptr;
        int th = 15;
        std::vector<std::pair<int, KeyFrame*>> vPairs;
        vPairs.reserve(KFCounter.size());

        for(auto mit:KFCounter)
        {
            if(mit.second > nMax)
            {
                nMax = mit.second;
                pKFMax = mit.first;
            }
            if(mit.second>=th)
            {
                vPairs.emplace_back(mit.second, mit.first);
                (mit.first)->AddConnection(this, mit.second);
            }
        }

        if(vPairs.empty())
        {
            vPairs.emplace_back(nMax, pKFMax);
            pKFMax->AddConnection(this, nMax);
        }

        std::sort(vPairs.begin(), vPairs.end());
        std::list<KeyFrame*> lKFs;
        std::list<int> lWs;
        for(auto & vPair : vPairs)
        {
            lKFs.push_front(vPair.second);
            lWs.push_front(vPair.first);
        }

        {
            std::unique_lock<std::mutex> lock(mMutexConnections);

            mConnectedKeyFrameWeights = KFCounter;
            mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
            mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
        }
    }

    void KeyFrame::UpdateBestCovisibles()
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        std::vector<std::pair<int, KeyFrame*>> vPairs;
        vPairs.reserve(mConnectedKeyFrameWeights.size());
        for(auto mit:mConnectedKeyFrameWeights)
        {
            vPairs.emplace_back(mit.second, mit.first);
        }

        std::sort(vPairs.begin(), vPairs.end());

        std::list<KeyFrame*> lKFs;
        std::list<int> lWs;
        for(auto i:vPairs)
        {
            lKFs.push_front(i.second);
            lWs.push_front(i.first);
        }
        mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
        mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
    }

    std::set<KeyFrame *> KeyFrame::GetConnectedKeyFrames()
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        std::set<KeyFrame*> s;
        for(auto mit:mConnectedKeyFrameWeights)
            s.insert(mit.first);
        return s;
    }

    std::vector<KeyFrame *> KeyFrame::GetVectorCovisibleKeyFrames()
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        return mvpOrderedConnectedKeyFrames;
    }

    std::vector<KeyFrame *> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        if((int)mvpOrderedConnectedKeyFrames.size() < N)
            return mvpOrderedConnectedKeyFrames;
        else
            return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);
    }

    std::vector<KeyFrame *> KeyFrame::GetCovisiblesByWeight(const int &w)
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        if(mvpOrderedConnectedKeyFrames.empty())
            return {};
        auto it = std::upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, KeyFrame::WeightComp);

        if(it == mvOrderedWeights.end())
            return {};
        else
        {
            int n = (int)(it-mvOrderedWeights.begin());
            return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + n);
        }
    }

    int KeyFrame::GetWeight(KeyFrame *pKF)
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
            return (int)mConnectedKeyFrameWeights.count(pKF);
        else
            return 0;
    }

    void KeyFrame::SetBadFlag()
    {

    }

    bool KeyFrame::isBad()
    {

    }


}
