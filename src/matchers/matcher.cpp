#include "matchers/matcher.h"
#include "types/Frame.h"
#include "types/Map.h"
#include "types/MapPoint.h"
#include "camera/CameraModel.h"
#include <opencv2/opencv.hpp>
#include "Eigen/Core"
#include "Eigen/Dense"
#include <boost/bind.hpp>

namespace TRACKING_BENCH
{

    Matcher::Matcher():
            m_flann_matcher(new cv::flann::LshIndexParams(20,10,2))
    {
        m_bf_matcher = std::make_shared<cv::BFMatcher>(cv::NORM_HAMMING, true);
    }

    /**
     * @brief Match by OpenCV nearest neighbor
     * @param F1 Current Frame Or Key Frame
     * @param F2 Reference Frame Or Key Frame
     * @param MinLevel Min Level
     * @param MaxLevel Max Level
     * @param MapPointOnly Only Match the features associate with Map Point
     * @return Matches
     */
    std::vector<cv::DMatch> Matcher::searchByNN(
            const std::shared_ptr<Frame>& F1,
            const std::shared_ptr<Frame>& F2,
            int MinLevel, int MaxLevel,
            float ratio, float minTh,
            bool MapPointOnly)
    {
        assert(MinLevel <= MaxLevel);
        cv::Mat d1, d2;
        std::vector<int> id1;
        if(MinLevel == 0 && MaxLevel == F1->GetMaxLevel() && !MapPointOnly)
        {// using all points to match
            d1 = F1->GetDescriptors();
            d2 = F2->GetDescriptors();
        }
        else
        {
            id1.resize(F1->GetKeys().size());

            int cnt = 0, match_num = 0;
            for(auto&f:F1->GetKeys())
            {
                if(f->kp.octave >= MinLevel && f->kp.octave <= MaxLevel)
                {// only the features between [MinLevel, MaxLevel]
                    if(!MapPointOnly || F1->GetMapPoint(cnt) != nullptr)
                    {// all points with MapPointOnly == false , Only Match the features associate with Map Point when MapPointOnly == true
                        id1.at(match_num) = cnt;
                        // descriptor
                        d1.row(match_num) = F1->GetDescriptors().row(cnt);
                        match_num ++;
                    }
                }
                cnt ++;
            }
            d2 = F2->GetDescriptors();
        }

        std::vector<cv::DMatch> matches;
        std::vector<cv::DMatch> good_matches;
        m_flann_matcher.match(d1, d2, matches);

        float min_distance = std::min_element(matches.begin(), matches.end(),[](const cv::DMatch& m1, const cv::DMatch& m2){return m1.distance < m2.distance;})->distance;
        //float max_distance = std::max_element(matches.begin(), matches.end(),[](const cv::DMatch& m1, const cv::DMatch& m2){return m1.distance < m2.distance;})->distance;

        for(const auto &m:matches)
        {
            if(m.distance < std::fmin(ratio*min_distance, minTh))
            {
                good_matches.emplace_back(m);
            }
        }
        // change ids
        if(!id1.empty())
        {
            for(auto m:good_matches)
            {
                m.queryIdx = id1.at(m.queryIdx);
            }
        }
        return good_matches;
    }

    /**
     * @brief Match by OpenCV nearest neighbor
     * @param map Map Points
     * @param F1 Current Frame Or Key Frame
     * @return Matches
     */
    std::vector<cv::DMatch> Matcher::searchByNN(Map *map, Frame *F1, bool Projection)
    {
        std::vector<cv::DMatch> matches;
        std::vector<cv::DMatch> good_matches;
        cv::Mat des2;
        std::vector<int> id1;
        if(!Projection)
        {
            for(int i = 0; i< map->MapPointsInMap();i ++)
            {
                // point descriptor
                des2.row(i) = map->GetAllMapPoints().at(i)->GetDescriptor();
            }
        }
        else
        {// only
            id1.resize(map->MapPointsInMap());
            int match = 0;
            for(int i = 0; i< map->MapPointsInMap();i ++)
            {
                auto p = map->GetAllMapPoints().at(i)->GetWorldPos();
                //TODO
                //if(F1->GetCameraModel()->IsInFrame(F1->GetCameraModel()->World2Cam(p)))
                {// projection in image
                    // point descriptor
                    des2.row(match) = map->GetAllMapPoints().at(i)->GetDescriptor();
                    id1.at(match) = i;
                    match ++;
                }
            }
        }

        m_flann_matcher.match(F1->GetDescriptors(), des2, matches);

        float min_distance = std::min_element(matches.begin(), matches.end(),[](const cv::DMatch& m1, const cv::DMatch& m2){return m1.distance < m2.distance;})->distance;
        //float max_distance = std::max_element(matches.begin(), matches.end(),[](const cv::DMatch& m1, const cv::DMatch& m2){return m1.distance < m2.distance;})->distance;

        for(const auto &m:matches)
        {
            if(m.distance < std::fmin(nRatio*min_distance, TH_HIGH))
            {
                good_matches.emplace_back(m);
            }
        }

        // change ids
        if(!id1.empty())
        {
            for(auto m:good_matches)
            {
                m.queryIdx = id1.at(m.queryIdx);
            }
        }
        return good_matches;
    }

    /**
     * @brief Match by OpenCV Brute Force
     * @param F1 Current Frame Or Key Frame
     * @param F2 Reference Frame Or Key Frame
     * @param MinLevel Min Level
     * @param MaxLevel Max Level
     * @param MapPointOnly Only Match the features associate with Map Point
     * @return Matches
     */
    std::vector<cv::DMatch> Matcher::searchByBF(
            const std::shared_ptr<Frame>& F1,
            const std::shared_ptr<Frame>& F2,
            int MinLevel, int MaxLevel,
            float ratio, float minTh,
            bool MapPointOnly)
    {
        assert(MinLevel <= MaxLevel);
        cv::Mat d1, d2;
        std::vector<int> id1;
        if(MinLevel == 0 && MaxLevel == F1->GetMaxLevel() && !MapPointOnly)
        {// using all points to match
            d1 = F1->GetDescriptors();
            d2 = F2->GetDescriptors();
        }
        else
        {
            id1.resize(F1->GetKeys().size());

            int cnt = 0, match_num = 0;
            for(auto&f:F1->GetKeys())
            {
                if(f->kp.octave >= MinLevel && f->kp.octave <= MaxLevel)
                {// only the features between [MinLevel, MaxLevel]
                    if(!MapPointOnly || F1->GetMapPoint(cnt) != nullptr)
                    {// all points with MapPointOnly == false , Only Match the features associate with Map Point when MapPointOnly == true
                        id1.at(match_num) = cnt;
                        // descriptor
                        d1.row(match_num) = F1->GetDescriptors().row(cnt);
                        match_num ++;
                    }
                }
                cnt ++;
            }
            d2 = F2->GetDescriptors();
        }

        std::vector<cv::DMatch> matches;
        std::vector<cv::DMatch> good_matches;
        m_bf_matcher->match(d1, d2, matches);

        float min_distance = std::min_element(matches.begin(), matches.end(),[](const cv::DMatch& m1, const cv::DMatch& m2){return m1.distance < m2.distance;})->distance;
        //float max_distance = std::max_element(matches.begin(), matches.end(),[](const cv::DMatch& m1, const cv::DMatch& m2){return m1.distance < m2.distance;})->distance;

        for(const auto &m:matches)
        {
            if(m.distance < std::fmin(ratio*min_distance, minTh))
            {
                good_matches.emplace_back(m);
            }
        }
        // change ids
        if(!id1.empty())
        {
            for(auto m:good_matches)
            {
                m.queryIdx = id1.at(m.queryIdx);
            }
        }
        return good_matches;
    }

    /**
     * @brief Match by OpenCV Brute Force
     * @param map Map Points
     * @param F1 Current Frame Or Key Frame
     * @return Matches
     */
    std::vector<cv::DMatch> Matcher::searchByBF(Map *map, Frame *F1, bool Projection)
    {
        std::vector<cv::DMatch> matches;
        std::vector<cv::DMatch> good_matches;
        cv::Mat des2;
        std::vector<int> id1;
        if(!Projection)
        {
            for(int i = 0; i< map->MapPointsInMap();i ++)
            {
                // point descriptor
                des2.row(i) = map->GetAllMapPoints().at(i)->GetDescriptor();
            }
        }
        else
        {// only
            id1.resize(map->MapPointsInMap());
            int match = 0;
            for(int i = 0; i< map->MapPointsInMap();i ++)
            {
                auto p = map->GetAllMapPoints().at(i)->GetWorldPos();
                // TODO
//                if(F1->GetCameraModel()->IsInFrame(F1->GetCameraModel()->World2Cam(p)))
                {// projection in image
                    // point descriptor
                    des2.row(match) = map->GetAllMapPoints().at(i)->GetDescriptor();
                    id1.at(match) = i;
                    match ++;
                }
            }
        }

        m_flann_matcher.match(F1->GetDescriptors(), des2, matches);

        float min_distance = std::min_element(matches.begin(), matches.end(),[](const cv::DMatch& m1, const cv::DMatch& m2){return m1.distance < m2.distance;})->distance;
        //float max_distance = std::max_element(matches.begin(), matches.end(),[](const cv::DMatch& m1, const cv::DMatch& m2){return m1.distance < m2.distance;})->distance;

        for(const auto &m:matches)
        {
            if(m.distance < std::fmin(nRatio*min_distance, TH_HIGH))
            {
                good_matches.emplace_back(m);
            }
        }

        // change ids
        if(!id1.empty())
        {
            for(auto m:good_matches)
            {
                m.queryIdx = id1.at(m.queryIdx);
            }
        }
        return good_matches;
    }

    /**
     * @brief Match by violence search in window
     * @param F1 Current Frame Or Key Frame
     * @param F2 Reference Frame Or Key Frame
     * @param MapPointOnly Only Match the features associate with Map Point
     * @return Matches
     */
    std::vector<cv::DMatch> Matcher::searchByViolence(
            const std::shared_ptr<Frame>& F1,
            const std::shared_ptr<Frame>& F2,
            int min_level,
            int max_level,
            float search_r,
            bool MapPointOnly)
    {
        std::vector<cv::DMatch> matches;

        // rot check
        std::vector<int> rotHist[HISTO_LENGTH];
        for(int i = 0;i < HISTO_LENGTH; i++)
        {
            rotHist[i].reserve(500);
        }
        const float factor = 1.f / (float)HISTO_LENGTH;

        for(size_t i1 = 0; i1 < F1->GetKeys().size();i1 ++)
        {// for all features in F1
            // features in F2 window
            std::vector<size_t> vIndices2 = F2->GetFeaturesInArea(F1->GetKeys().at(i1)->kp.pt.x, F1->GetKeys().at(i1)->kp.pt.y, search_r,min_level,max_level);
            if(vIndices2.empty())
                continue;
            if(!MapPointOnly || F1->GetMapPoint(i1) != nullptr)
            {// exclude non map point features when MapPointOnly == true
                // find best match
                // feature descriptor in F1
                cv::Mat d1 = F1->GetDescriptors().row((int)i1);
                // init it
                int bestDist = INT_MAX;
                int bestDist2 = INT_MAX;
                int bestIdx2 = -1;

                for(auto& i2:vIndices2)
                {// for all window feature to find out best match
                    // feature descriptor in F2
                    cv::Mat d2 = F2->GetDescriptors().row((int)i2);
                    // distance
                    int dist = DescriptorDistance(d1, d2);
                    // record best result and the second best
                    if(dist < bestDist)
                    {
                        bestDist2 = bestDist;
                        bestDist = dist;
                        bestIdx2 = (int)i2;
                    }
                    else if(dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }
                // the match is good enough
                if(bestDist <= TH_LOW)
                {
                    if((float)bestDist <(float)bestDist2*nRatio)
                    {// and much better than the second best
                        // got one match
                        matches.emplace_back((int)i1, bestIdx2, (float)bestDist);
                        // calculate orientation
                        if(checkOrientation)
                        {
                            float rot = F1->GetKeys().at(i1)->kp.angle - F2->GetKeys().at(bestIdx2)->kp.angle;
                            if(rot < 0)
                                rot += 360.f;
                            int bin = (int)roundf(rot*factor);
                            if(bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            // push in rotHist
                            rotHist[bin].push_back((int)matches.size()-1);
                        }
                    }
                }
            }
        }
        std::vector<cv::DMatch> good_matches;
        good_matches.reserve(matches.size());
        if(checkOrientation)
        {
            int ind[3] = {-1, -1, -1};
            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind[0], ind[1], ind[2]);

            for(int i =0;i < HISTO_LENGTH;i ++)
            {
                if(i == ind[0] || i == ind[1] || i == ind[2])
                {
                    for(auto item:rotHist[i])
                    {
                        good_matches.emplace_back(matches[item]);
                    }
                }
            }
            return good_matches;
        }
        return matches;
    }


    /**
     * @brief Match by projection search, From F2 project to F1
     * @param F1 Current Frame Or Key Frame
     * @param F2 Reference Frame Or Key Frame
     * @param MapPointOnly Only Match the features associate with Map Point
     * @return Matches
     */
    std::vector<cv::DMatch> Matcher::searchByProjection(
            const std::shared_ptr<Frame>& F1,
            const std::shared_ptr<Frame>& F2)
    {
        std::vector<cv::DMatch> matches;

        // Rotation Histogram (to check rotation consistency)
        std::vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/(float)HISTO_LENGTH;

        // current frame pose
        const Eigen::Matrix3f Rcw = F1->GetPose().block<3, 3>(0, 0);
        const Eigen::Vector3f tcw = F1->GetPose().block<3, 1>(0, 3);
        const Eigen::Vector3f twc = -Rcw.transpose() * tcw;

        // ref frame pose
        const Eigen::Matrix3f Rlw = F2->GetPose().block<3, 3>(0, 0);
        const Eigen::Vector3f tlw = F2->GetPose().block<3, 1>(0 ,3);
        const Eigen::Vector3f tlc = Rlw*twc+tlw;

        for(int i2=0; i2<F2->GetKeys().size(); i2++)
        {
            auto pMP = F2->GetMapPoint(i2);
            if(pMP)
            {// for all map points in F2
                if(!pMP->isBad())
                {
                    // Project
                    // point pos in world
                    auto x3Dw = pMP->GetWorldPos();
                    // point pos in F1 camera
                    Eigen::Vector3f x3Dc = Rcw*x3Dw+tcw;

                    // x, y, 1/z
                    const float xc = x3Dc[0];
                    const float yc = x3Dc[1];
                    const float invzc = 1.0f / x3Dc[2];
                    // behind F1 camera
                    if(invzc<0)
                        continue;
                    // F1 image (u, v)
                    auto uv = F1->GetCameraModel()->World2Cam(x3Dc);
                    // ensure in F1 camera
                    if(!F1->GetCameraModel()->IsInFrame(uv.cast<int>()))
                        continue;

                    // feature octave
                    int nLastOctave = F2->GetKeys().at(i2)->kp.octave;

                    // Search in a window. Size depends on scale
                    float radius = nRatio * F1->GetScaleFactors()[nLastOctave];

                    // features in window
                    std::vector<size_t> vIndices1;
                    vIndices1 = F1->GetFeaturesInArea((float)uv[0],(float)uv[1], radius, nLastOctave-1, nLastOctave+1);

                    // none
                    if(vIndices1.empty())
                        continue;
                    // pMP's descriptor
                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx1 = -1;
                    // for all features in F1 window
                    for(auto i1 : vIndices1)
                    {
                        // ensure not been matched
                        if(F1->GetMapPoint(i1))
                            if(F1->GetMapPoint(i1)->Observations()>0)
                                  continue;
                        // descriptors
                        const cv::Mat &d = F1->GetDescriptors().row((int)i1);
                        // cal distance
                        const int dist = DescriptorDistance(dMP,d);
                        // record best match
                        if(dist<bestDist)
                        {
                            bestDist = dist;
                            bestIdx1 = (int)i1;
                        }
                    }

                    if(bestDist<=TH_HIGH)
                    {// best match is close enough
                        matches.emplace_back(bestIdx1, i2, bestDist);
                        if(checkOrientation)
                        {
                            float rot = F2->GetKeys()[i2]->kp.angle - F1->GetKeys()[bestIdx1]->kp.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back((int)matches.size()-1);
                        }
                    }
                }
            }
        }

        //Apply rotation consistency
        std::vector<cv::DMatch> good_matches;
        good_matches.reserve(matches.size());
        if(checkOrientation)
        {
            int ind[3] = {-1, -1, -1};

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind[0], ind[1], ind[2]);

            for(int i =0;i < HISTO_LENGTH;i ++)
            {
                if(i == ind[0] || i == ind[1] || i == ind[2])
                {
                    for(auto item:rotHist[i])
                    {
                        good_matches.emplace_back(matches[item]);
                    }
                }
            }
            return good_matches;
        }
        return matches;
    }

    /**
     * @brief Match by projection
     * @param map Map Points
     * @param F1 Current Frame Or Key Frame
     * @return Matches
     */
    std::vector<cv::DMatch> Matcher::searchByProjection(Map *map, Frame *F1)
    {
        const bool bFactor = nRatio!=1.0;

        for(const auto& pMP:map->GetAllMapPoints())
        {
            if(pMP->isBad())
                continue;
            // project
            int nPredictedLevel;
            float nPredictedU, nPredictedV, ViewCos = 0.5;
            if(F1->IsInFrustum(pMP, nPredictedLevel, nPredictedU, nPredictedV, ViewCos))
                continue;

            // The size of the window will depend on the viewing direction
            float r = 4.f;
            if(ViewCos > 0.998)//
                r = 2.5;

            if(bFactor)
                r*=nRatio;

            const std::vector<size_t> vIndices =
                    F1->GetFeaturesInArea(nPredictedU,nPredictedV,r*F1->GetScaleFactors()[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

            if(vIndices.empty())
                continue;

            const cv::Mat MPdescriptor = pMP->GetDescriptor();

            int bestDist=256;
            int bestLevel= -1;
            int bestDist2=256;
            int bestLevel2 = -1;
            int bestIdx =-1 ;

            // Get best and second matches with near keypoints
            for(unsigned long idx : vIndices)
            {
                if(F1->GetMapPointMatches()[idx])
                    if(F1->GetMapPointMatches()[idx]->Observations()>0)
                        continue;

                const cv::Mat &d = F1->GetDescriptors().row((int)idx);

                const int dist = DescriptorDistance(MPdescriptor,d);

                if(dist<bestDist)
                {
                    bestDist2=bestDist;
                    bestDist=dist;
                    bestLevel2 = bestLevel;
                    bestLevel = F1->GetKeys()[idx]->kp.octave;
                    bestIdx=(int)idx;
                }
                else if(dist<bestDist2)
                {
                    bestLevel2 = F1->GetKeys()[idx]->kp.octave;
                    bestDist2=dist;
                }
            }

            // Apply ratio to second match (only if best and second are in the same scale level)
            if(bestDist<=TH_HIGH)
            {
                if(bestLevel==bestLevel2 && bestDist>nRatio*bestDist2)
                    continue;

                F1->ReplaceMapPointMatch(bestIdx, pMP);
            }
        }
    }

    std::vector<cv::DMatch> Matcher::searchByBow(
            const std::shared_ptr<Frame>& F1,
            const std::shared_ptr<Frame>& F2,
            bool MapPointOnly)
    {
        std::vector<cv::DMatch> matches, good_matches;
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f / HISTO_LENGTH;

        auto f1it = F1->GetFeatureVector().cbegin();
        auto f2it = F2->GetFeatureVector().cbegin();
        auto f1end = F1->GetFeatureVector().cend();
        auto f2end = F2->GetFeatureVector().cend();

        while(f1it != f1end && f2it != f2end)
        {
            if(f1it->first == f2it->first)
            {
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];
                    const cv::Mat &d1 = F1->GetDescriptors().row((int)idx1);

                    int bestDist1=256;
                    int bestIdx2 =-1 ;
                    int bestDist2=256;

                    for(unsigned long idx2 : f2it->second)
                    {
                        if (MapPointOnly && F2->GetMapPoint(idx2) == nullptr)
                            continue;
                        const cv::Mat &d2 = F2->GetDescriptors().row((int)idx2);

                        int dist = DescriptorDistance(d1,d2);

                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdx2=idx2;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    }

                    if(bestDist1<TH_LOW)
                    {
                        if(bestDist1<nRatio*bestDist2)
                        {
                            matches.emplace_back(idx1, bestIdx2, bestDist1);
                            if(checkOrientation)
                            {
                                float rot = F1->GetKeys()[idx1]->kp.angle - F2->GetKeys()[bestIdx2]->kp.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back((int)matches.size()-1);
                            }
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = F1->GetFeatureVector().lower_bound(f2it->first);
            }
            else
            {
                f2it = F2->GetFeatureVector().lower_bound(f1it->first);
            }
        }

        if(checkOrientation)
        {
            int ind[3] = {-1, -1, -1};

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind[0], ind[1], ind[2]);

            for(int i =0;i < HISTO_LENGTH;i ++)
            {
                if(i == ind[0] || i == ind[1] || i == ind[2])
                {
                    for(auto item:rotHist[i])
                    {
                        good_matches.emplace_back(matches[item]);
                    }
                }
            }
            return good_matches;
        }
        return matches;
    }


    std::vector<cv::DMatch> Matcher::searchByOPFlow(
            const std::shared_ptr<Frame>& F1,
            const std::shared_ptr<Frame>& F2,
            std::vector<cv::Point2f>& cur_points,
            bool equalized,
            bool reject,
            bool MapPointOnly)
    {
        std::vector<cv::DMatch> matches;
        cv::Mat img1, img2;
        if(equalized)
        {
            img1 = F1->Equalize();
        }
        else
            img1 = F1->GetImage();
        img2 = F2->GetImage();

        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(img2, img1, F2->GetCVKeys(), cur_points, status, err, cv::Size(21, 21), 3);
        // reduce vector
        for (int i = 0;i < int(F2->GetKeys().size()); i++)
            if (status[i] && !F1->GetCameraModel()->IsInFrame(Eigen::Vector2i(cur_points.at(i).x, cur_points.at(i).y)))
                status[i] = 0;

        if(reject)
        {
            // reject
            rejectWithF(cur_points, F2->GetCVKeys(), status);
        }

        for(int i = 0;i < status.size();i ++)
        {
            if (status[i])
            {
                cv::DMatch match;
                match.queryIdx = i;
                match.trainIdx = i;
                matches.emplace_back(match);
            }
        }
        return matches;
    }

    /**
     * @brief SVO like, match features by direct method
     *        image align  --> initial pos
     *        projection + feature align --> match result
     * @param map map
     * @param F1 current frame
     * @return
     */
    std::vector<cv::DMatch> Matcher::searchByDirect(Map *map, Frame *F1, Frame *F2)
    {
        // get initial pose by direct method
        auto T = SparseImageAlign(F1, F2);
        F1->SetPose(T);

        // feature align
        return FeaturesAlign(map, F1);
    }

    int Matcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist=0;

        for(int i=0; i<8; i++, pa++, pb++)
        {
            unsigned  int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += ((((int)v + ((int)v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }
        return dist;
    }

    void  Matcher::ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1=0;
        int max2=0;
        int max3=0;

        for(int i=0; i<L; i++)
        {
            const int s = (int)histo[i].size();
            if(s>max1)
            {
                max3=max2;
                max2=max1;
                max1=s;
                ind3=ind2;
                ind2=ind1;
                ind1=i;
            }
            else if(s>max2)
            {
                max3=max2;
                max2=s;
                ind3=ind2;
                ind2=i;
            }
            else if(s>max3)
            {
                max3=s;
                ind3=i;
            }
        }

        if((float)max2<0.1f*(float)max1)
        {
            ind2=-1;
            ind3=-1;
        }
        else if((float)max3<0.1f*(float)max1)
        {
            ind3=-1;
        }
    }

    void Matcher::rejectWithF(std::vector<cv::Point2f>& cur_pts, const std::vector<cv::Point2f>& last_pts, std::vector<uchar>& status)
    {
        std::vector<int> id;
        std::vector<cv::Point2f> pts1, pts2;
        id.reserve(status.size());

        for (int i = 0; i < status.size();i ++)
        {
            if(status[i])
            {
                id.emplace_back(i);
                pts1.emplace_back(cur_pts.at(i));
                pts2.emplace_back(last_pts.at(i));
            }
        }
        std::vector<uchar> fund_status;
        if(last_pts.size() > 8)
        {
            cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 1.0, 0.99, fund_status);
        }

        for(int i = 0;i < id.size();i ++)
        {
            if(fund_status[i] == 0)
            {
                status[id[i]] = 0;
            }
        }
    }

    Eigen::Matrix<float, 4, 4> Matcher::SparseImageAlign(Frame *F1, Frame *F2)
    {
//        if(F2->GetKeys().empty())
//        {
//            std::cerr<<"can not align with F2 points "<<F2->GetKeys().size()<<std::endl;
//            return F1->GetPose();
//        }
//        Eigen::Matrix<float, 4, 4> T_cur_frame_ref(F1->GetPose() * F2->GetPoseInverse());
//        float mu = 0.1;
//        ref_patch_cache = cv::Mat(F2->GetKeys().size(), patch_area, CV_32F);
//
//        jacobian_cache.resize(Eigen::NoChange, ref_patch_cache.rows*patch_area);
//        std::vector<bool> visible_fts;
//        visible_fts.resize(ref_patch_cache.rows, false);
//
//        for(int level = mnMaxLevel; level >= mnMinLevel; level --)
//        {
//            mu = 0.1;
//            jacobian_cache.setZero();
//            have_ref_patch_cache = false;
//            // start optimize, Levenberg Marquardt
//            // init optimize
//            //double chi2 = ComputeResiduals(T_cur_frame_ref, F1, F2, visible_fts,level, true, false);
//            double rho = 0;
//            int n_trials = 0;
//            bool stop = false;
//            float nu = 2.0;
//            for(int iter = 0; iter < mnIter; ++iter)
//            {
//                rho = 0;
//                n_trials = 0;
//                do
//                {// while rho <= 0 && stop == false
//                    // init variables
//                    auto new_T = T_cur_frame_ref;
//                    double new_chi2 = -1;
//
//                    // compute initial error
//                    ComputeResiduals(new_T, F1, F2, visible_fts,level, true, false);
//                    // add damping term:
//                    mH += (mH.diagonal()*mu).asDiagonal();
//                    // solve problem
//                    // Hx = b
//                    mx = mH.ldlt().solve(mJRes);
//                    if(!(bool) std::isnan((double)mx[0]))
//                    {
//                        cv::Mat_<double> delta = cv::Mat::eye(cv::Size(4,4), CV_32F);
//                        delta.row(0).col(3) = - mx[3];
//                        delta.row(1).col(3) = - mx[4];
//                        delta.row(2).col(3) = - mx[5];
//                        cv::Vec3d rve{-mx[0], -mx[1], -mx[2]};
//                        cv::Rodrigues(delta.rowRange(0, 3).colRange(0, 3), rve);
//                        new_T = T_cur_frame_ref * delta;
//                        new_chi2 = ComputeResiduals(new_T, F1, F2, visible_fts,level, true, false);
//                        //rho = chi2 - new_chi2;
//                    }
//                    else
//                    {
//                        std::cout<<" Matrix is close to singular!"<<std::endl;
//                        rho = -1;
//                    }
//
//                    if(rho > 0)
//                    {// update
//                        T_cur_frame_ref = new_T;
//                        chi2 = new_chi2;
//                        stop = mx.norm() <= mdEps;
//                        mu *= max(1./3, min(1. - pow(2*rho-1, 3), 2./3.));
//                        nu = 2.;
//                    }
//                    else
//                    {
//                        mu *= nu;
//                        nu *= 2.;
//                        ++ n_trials;
//                        if (n_trials >= mnNTrialsMax)
//                            stop = true;
//                    }
//
//                } while (!(rho > 0 || stop));
//                if(stop)
//                    break;
//            }
//        }
//
//        return T_cur_frame_ref * F2->GetPose();
    }

    std::vector<cv::DMatch> Matcher::FeaturesAlign(Map *Map, Frame *F1)
    {
//        // initial
//        std::vector<cv::DMatch> matches;
//        float n_matches = 0;
//        float n_trials = 0;
//        std::for_each(grid_.cells.begin(), grid_.cells.end(),[&](Cell* c){c->clear();});
//
//        // co-view frames
//        std::list<std::pair<Frame*, double>> close_kfs;
//        std::vector<std::pair<Frame*, std::size_t>> overlap_kfs;
//        close_kfs.sort(boost::bind(&std::pair<Frame*, double>::second, _1)
//                           < boost::bind(&std::pair<Frame*, double>::second, _2));
//
//        const int max_n_kfs = 10;
//        overlap_kfs.reserve(max_n_kfs);
//        size_t n = 0;
//        for(auto it_frame = close_kfs.begin(), ite_frame = close_kfs.end(); it_frame != ite_frame && n < max_n_kfs; ++it_frame, ++n)
//        {
//            Frame* ref_frame = it_frame->first;
//            overlap_kfs.emplace_back(it_frame->first, 0);
//
//            size_t ftr_cnt = 0;
//            for(auto it_ftr = ref_frame->GetKeys().begin(), ite_ftr = ref_frame->GetKeys().end();
//            it_ftr != ite_ftr; ++it_ftr)
//            {
//                if(ref_frame->GetMapPoint(ftr_cnt) == nullptr)
//                    continue;
//
//                if(ref_frame->GetMapPoint(ftr_cnt)->last_projected_id == F1->GetId())
//                    continue;
//                ref_frame->GetMapPoint(ftr_cnt)->last_projected_id = F1->GetId();
//
//                cv::Mat p = ref_frame->GetMapPoint(ftr_cnt)->GetWorldPos();
//                cv::Vec2d px = F1->GetCameraModel()->World2Cam(cv::Vec3d{p.at<double>(0), p.at<double>(1), p.at<double>(2)});
//                if(F1->GetCameraModel()->IsInFrame(px))
//                {
//                    const int k = (int)(px[1] / grid_.cell_size) * grid_.grid_n_cols + (int)(px[0] / grid_.cell_size);
//                    grid_.cells.at(k)->push_back(Candidate(ref_frame->GetMapPoint(ftr_cnt),px));
//                    overlap_kfs.back().second ++;
//                }
//                ftr_cnt ++;
//            }
//        }
//
//        {// reproject candidates
//            std::unique_lock<std::mutex> lock1(Map->mMutexMapPoints);
//            auto it = Map->mspCandidatesMapPoints.begin();
//            while (it!=Map->mspCandidatesMapPoints.end())
//            {
//                cv::Mat p = (*it)->GetWorldPos();
//                cv::Vec2d px = F1->GetCameraModel()->World2Cam(cv::Vec3d{p.at<double>(0), p.at<double>(1), p.at<double>(2)});
//
//                if(!F1->GetCameraModel()->IsInFrame(px))
//                {
//                    const int k = (int)(px[1] / grid_.cell_size) * grid_.grid_n_cols + (int)(px[0] / grid_.cell_size);
//
//                    grid_.cells.at(k)->push_back(Candidate(*it,px));
//
//                    (*it)->n_failed_reproj += 3;
//                    if((*it)->n_failed_reproj > 30)
//                    {
//                        it = Map->mspCandidatesMapPoints.erase(it);
//                        continue;
//                    }
//                }
//                ++it;
//            }
//        }
//        const int maxFtx = 1000;
//        for(size_t i = 0;i < grid_.cells.size(); ++i)
//        {
//            if(ReprojectCell(*grid_.cells.at(grid_.cell_order[i]), F1))
//                ++n_matches;
//            if(n_matches > maxFtx)
//                break;
//        }
//    }
//
//    double Matcher::ComputeResiduals(cv::Mat &T_cur_from_ref, Frame* F1, Frame* F2, std::vector<bool>& visible_fts, int level, bool linearize_system, bool compute_weight_scale)
//    {
//        const cv::Mat& cur_img = F1->GetImagePyramid().at(level);
//
//        if(false == have_ref_patch_cache)
//            PreComputeReferencePatches(F1, F2, level);
//
//        std::vector<float> errors;
//        if(compute_weight_scale)
//            errors.resize(visible_fts.size());
//
//        const int stride = F1->GetImage().cols;
//        const int border = patch_half_size + 1;
//        const float scale = F1->GetScaleFactors().at(level);
//        const cv::Mat ref_pos(F2->GetTranslation());
//        float chi2 = 0.0;
//        size_t feature_counter = 0;
//        auto visible_it = visible_fts.begin();
//
//        for (auto it=F2->GetKeys().begin(); it != F2->GetKeys().end();
//        ++it, ++feature_counter, ++visible_it)
//        {
//            if(!*visible_it)
//                continue;
//
//            const double depth = cv::norm(F2->GetMapPoint(feature_counter)->GetWorldPos() - ref_pos);
//            const cv::Mat xyz_ref = F2->GetMapPoint(feature_counter)->GetNormal() * depth;
//            const cv::Mat xyz_cur = T_cur_from_ref * xyz_ref;
//            const auto uv_cur_pyr = F1->GetCameraModel()->World2Cam(
//                    cv::Vec3d{ xyz_cur.at<double>(0) * scale,
//                                    xyz_cur.at<double>(1) * scale,
//                                    xyz_cur.at<double>(2) * scale});
//
//            const auto u_cur = (float)uv_cur_pyr[0];
//            const auto v_cur = (float)uv_cur_pyr[1];
//            const int u_cur_i = (int)floorf(u_cur);
//            const int v_cur_i = (int)floorf(v_cur);
//
//            if(u_cur_i - border < 0 || v_cur_i < 0 ||
//               u_cur_i + border >= cur_img.cols || v_cur_i + border >= cur_img.rows)
//                continue;
//
//            // compute bilateral interpolation weights for the current image
//            const float subpix_u_cur = u_cur-u_cur_i;
//            const float subpix_v_cur = v_cur-v_cur_i;
//            const float w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
//            const float w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
//            const float w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
//            const float w_cur_br = subpix_u_cur * subpix_v_cur;
//            float* ref_patch_cache_ptr = reinterpret_cast<float*>(ref_patch_cache.data) + patch_area*feature_counter;
//            size_t pixel_counter = 0; // is used to compute the index of the cached jacobian
//
//            for(int y = 0; y < patch_size; ++y)
//            {
//                uint8_t* cur_img_ptr = (uint8_t*) cur_img.data + (v_cur_i+y-patch_half_size)*stride + (u_cur_i-patch_half_size);
//                for(int x = 0; x < patch_size; ++x,++pixel_counter, ++ref_patch_cache_ptr)
//                {
//                    const float intensity_cur = w_cur_tl*cur_img_ptr[0] + w_cur_tr*cur_img_ptr[1] + w_cur_bl*cur_img_ptr[stride] + w_cur_br*cur_img_ptr[stride+1];
//                    const float res = intensity_cur - (*ref_patch_cache_ptr);
//                    if(compute_weight_scale)
//                        errors.push_back(fabsf(res));
//
//                    float weight = 1.0f;
//                    // todo
//                    const float huber_k = 1.345f;
//                    if(compute_weight_scale)
//                    {
//                        float abs_res = fabsf(res/scale);
//                        if(abs_res >= huber_k)
//                            weight = huber_k / abs_res;
//                    }
//
//                    chi2 += res * res * weight;
//
//                    if(linearize_system)
//                    {
//                        const Eigen::Matrix<double, 6, 1> J(jacobian_cache.col(feature_counter*patch_area+pixel_counter));
//                        mH.noalias() += J * J.transpose() * weight;
//                        mJRes.noalias() -= J * res * weight;
//                    }
//                }
//            }
//        }

    }

    //
    void Matcher::PreComputeReferencePatches( Frame* F1, Frame* F2, int level)
    {
//        const int border = patch_half_size + 1;
//        const cv::Mat& ref_img = F2->GetImagePyramid().at(level);
//        const int stride = ref_img.cols;
//        const float scale = F2->GetScaleFactors().at(level);
//        const cv::Mat ref_pos = F2->GetTranslation();
//        const double focal_length = F2->GetCameraModel()->ErrorMultiplier();
//        size_t feature_counter = 0;
//        size_t pt_cnt = 0;
//        for(auto& pt:F2->GetKeys())
//        {
//            const float u_ref = pt.pt.x * scale;
//            const float v_ref = pt.pt.y * scale;
//            const float u_ref_i = floorf(u_ref);
//            const float v_ref_i = floorf(v_ref);
//            if (F2->GetMapPoint(pt_cnt) == nullptr ||
//              v_ref_i - border < 0 || u_ref_i - border < 0 ||
//              v_ref_i + border > ref_img.cols || u_ref_i + border > ref_img.rows)
//            {
//                continue;
//            }
//
//            const double depth(cv::norm(F2->GetMapPoint(pt_cnt)->GetWorldPos() - ref_pos));
//            const cv::Mat xyz_ref(F2->GetMapPoint(pt_cnt)->GetNormal() * depth);
//
//            // jacobian
//            Eigen::Matrix<double, 2, 6> frame_jac;
//            Frame::JacobianXYZ2uv(xyz_ref, frame_jac);
//
//            // dx, dy
//            const float subpix_u_ref = u_ref_i - u_ref_i;
//            const float subpix_v_ref = v_ref_i - v_ref_i;
//            const float w_ref_tl = (1.f - subpix_u_ref) * (1.f - subpix_v_ref);
//            const float w_ref_tr = subpix_u_ref * (1.f - subpix_v_ref);
//            const float w_ref_bl = (1.f - subpix_u_ref) * subpix_v_ref;
//            const float w_ref_br = subpix_u_ref * subpix_v_ref;
//            size_t pixel_counter = 0;
//            float* cache_ptr = reinterpret_cast<float*>(ref_patch_cache.data) + patch_area*feature_counter;
//
//            for(int y = 0;y < patch_size; ++y)
//            {
//                uint8_t * ref_img_ptr = (uint8_t*) ref_img.data + (int)((v_ref_i + y - patch_half_size) * stride) + (int)(u_ref_i - patch_half_size);
//                for(int x = 0; x < patch_size; ++x, ++ref_img_ptr, ++cache_ptr, ++pixel_counter)
//                {
//                    *cache_ptr = w_ref_tl * ref_img_ptr[0] + w_ref_tr * ref_img_ptr[1] + w_ref_bl * ref_img_ptr[stride] + w_ref_br * ref_img_ptr[stride + 1];
//                    float dx = 0.5f * ((w_ref_tl*ref_img_ptr[1] + w_ref_tr*ref_img_ptr[2] + w_ref_bl*ref_img_ptr[stride+1] + w_ref_br*ref_img_ptr[stride+2])
//                                       -(w_ref_tl*ref_img_ptr[-1] + w_ref_tr*ref_img_ptr[0] + w_ref_bl*ref_img_ptr[stride-1] + w_ref_br*ref_img_ptr[stride]));
//                    float dy = 0.5f * ((w_ref_tl*ref_img_ptr[stride] + w_ref_tr*ref_img_ptr[1+stride] + w_ref_bl*ref_img_ptr[stride*2] + w_ref_br*ref_img_ptr[stride*2+1])
//                                       -(w_ref_tl*ref_img_ptr[-stride] + w_ref_tr*ref_img_ptr[1-stride] + w_ref_bl*ref_img_ptr[0] + w_ref_br*ref_img_ptr[1]));
//                    jacobian_cache.col(feature_counter*patch_area + pixel_counter) =
//                            (dx * frame_jac.row(0) + dy * frame_jac.row(1)) * (focal_length * F2->GetScaleFactors().at(level));
//                }
//            }
//        }
//
//        have_ref_patch_cache = true;
    }



    bool Matcher::Align1D(
            const cv::Mat& cur_img,
            const Eigen::Vector2f& dir,                  // direction in which the patch is allowed to move
            uint8_t* ref_patch_with_border,
            uint8_t* ref_patch,
            const int n_iter,
            Eigen::Vector2d& cur_px_estimate,
            double& h_inv)
    {
//        const int halfpatch_size_ = 4;
//        const int patch_size = 8;
//        const int patch_area = 64;
//        bool converged=false;
//
//        // compute derivative of template and prepare inverse compositional
//        float __attribute__((__aligned__(16))) ref_patch_dv[patch_area];
//        Eigen::Matrix2f H; H.setZero();
//
//        // compute gradient and hessian
//        const int ref_step = patch_size+2;
//        float* it_dv = ref_patch_dv;
//        for(int y=0; y<patch_size; ++y)
//        {
//            uint8_t* it = ref_patch_with_border + (y+1)*ref_step + 1;
//            for(int x=0; x<patch_size; ++x, ++it, ++it_dv)
//            {
//                Eigen::Vector2f J;
//                J[0] = 0.5*(dir[0]*(it[1] - it[-1]) + dir[1]*(it[ref_step] - it[-ref_step]));
//                J[1] = 1;
//                *it_dv = J[0];
//                H += J*J.transpose();
//            }
//        }
//        h_inv = 1.0/H(0,0)*patch_size*patch_size;
//        Eigen::Matrix2f Hinv = H.inverse();
//        float mean_diff = 0;
//
//        // Compute pixel location in new image:
//        float u = cur_px_estimate.x();
//        float v = cur_px_estimate.y();
//
//        // termination condition
//        const float min_update_squared = 0.03*0.03;
//        const int cur_step = cur_img.step.p[0];
//        float chi2 = 0;
//        Eigen::Vector2f update; update.setZero();
//        for(int iter = 0; iter<n_iter; ++iter)
//        {
//            int u_r = floor(u);
//            int v_r = floor(v);
//            if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_)
//                break;
//
//            if(isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
//                return false;
//
//            // compute interpolation weights
//            float subpix_x = u-u_r;
//            float subpix_y = v-v_r;
//            float wTL = (1.0-subpix_x)*(1.0-subpix_y);
//            float wTR = subpix_x * (1.0-subpix_y);
//            float wBL = (1.0-subpix_x)*subpix_y;
//            float wBR = subpix_x * subpix_y;
//
//            // loop through search_patch, interpolate
//            uint8_t* it_ref = ref_patch;
//            float* it_ref_dv = ref_patch_dv;
//            float new_chi2 = 0.0;
//            Eigen::Vector2f Jres; Jres.setZero();
//            for(int y=0; y<patch_size; ++y)
//            {
//                uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_;
//                for(int x=0; x<patch_size; ++x, ++it, ++it_ref, ++it_ref_dv)
//                {
//                    float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
//                    float res = search_pixel - *it_ref + mean_diff;
//                    Jres[0] -= res*(*it_ref_dv);
//                    Jres[1] -= res;
//                    new_chi2 += res*res;
//                }
//            }
//
//            if(iter > 0 && new_chi2 > chi2)
//            {
//#if SUBPIX_VERBOSE
//                cout << "error increased." << endl;
//#endif
//                u -= update[0];
//                v -= update[1];
//                break;
//            }
//
//            chi2 = new_chi2;
//            update = Hinv * Jres;
//            u += update[0]*dir[0];
//            v += update[0]*dir[1];
//            mean_diff += update[1];
//
//#if SUBPIX_VERBOSE
//            cout << "Iter " << iter << ":"
//         << "\t u=" << u << ", v=" << v
//         << "\t update = " << update[0] << ", " << update[1]
//         << "\t new chi2 = " << new_chi2 << endl;
//#endif
//
//            if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
//            {
//#if SUBPIX_VERBOSE
//                cout << "converged." << endl;
//#endif
//                converged=true;
//                break;
//            }
//        }
//
//        cur_px_estimate << u, v;
//        return converged;
    }

    bool Matcher::Align2D(const cv::Mat &cur_img, uint8_t *ref_patch_with_border, uint8_t *ref_patch, const int n_iter,
                          Eigen::Vector2d &cur_px_estimate, bool no_simd)
    {
//        const int halfpatch_size_ = 4;
//        const int patch_size_ = 8;
//        const int patch_area_ = 64;
//        bool converged=false;
//
//        // compute derivative of template and prepare inverse compositional
//        float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_];
//        float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
//        Eigen::Matrix3f H; H.setZero();
//
//        // compute gradient and hessian
//        const int ref_step = patch_size_+2;
//        float* it_dx = ref_patch_dx;
//        float* it_dy = ref_patch_dy;
//        for(int y=0; y<patch_size_; ++y)
//        {
//            uint8_t* it = ref_patch_with_border + (y+1)*ref_step + 1;
//            for(int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy)
//            {
//                Eigen::Vector3f J;
//                J[0] = 0.5f * (float)(it[1] - it[-1]);
//                J[1] = 0.5f * (float)(it[ref_step] - it[-ref_step]);
//                J[2] = 1;
//                *it_dx = J[0];
//                *it_dy = J[1];
//                H += J*J.transpose();
//            }
//        }
//        Eigen::Matrix3f Hinv = H.inverse();
//        float mean_diff = 0;
//
//        // Compute pixel location in new image:
//        float u = cur_px_estimate.x();
//        float v = cur_px_estimate.y();
//
//        // termination condition
//        const float min_update_squared = 0.03*0.03;
//        const int cur_step = cur_img.step.p[0];
////  float chi2 = 0;
//        Eigen::Vector3f update; update.setZero();
//        for(int iter = 0; iter<n_iter; ++iter)
//        {
//            int u_r = floor(u);
//            int v_r = floor(v);
//            if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_)
//                break;
//
//            if(isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
//                return false;
//
//            // compute interpolation weights
//            float subpix_x = u-u_r;
//            float subpix_y = v-v_r;
//            float wTL = (1.0-subpix_x)*(1.0-subpix_y);
//            float wTR = subpix_x * (1.0-subpix_y);
//            float wBL = (1.0-subpix_x)*subpix_y;
//            float wBR = subpix_x * subpix_y;
//
//            // loop through search_patch, interpolate
//            uint8_t* it_ref = ref_patch;
//            float* it_ref_dx = ref_patch_dx;
//            float* it_ref_dy = ref_patch_dy;
////    float new_chi2 = 0.0;
//            Eigen::Vector3f Jres; Jres.setZero();
//            for(int y=0; y<patch_size_; ++y)
//            {
//                uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_;
//                for(int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
//                {
//                    float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
//                    float res = search_pixel - *it_ref + mean_diff;
//                    Jres[0] -= res*(*it_ref_dx);
//                    Jres[1] -= res*(*it_ref_dy);
//                    Jres[2] -= res;
////        new_chi2 += res*res;
//                }
//            }
//
//
///*
//    if(iter > 0 && new_chi2 > chi2)
//    {
//#if SUBPIX_VERBOSE
//      cout << "error increased." << endl;
//#endif
//      u -= update[0];
//      v -= update[1];
//      break;
//    }
//    chi2 = new_chi2;
//*/
//            update = Hinv * Jres;
//            u += update[0];
//            v += update[1];
//            mean_diff += update[2];
//
//#if SUBPIX_VERBOSE
//            cout << "Iter " << iter << ":"
//         << "\t u=" << u << ", v=" << v
//         << "\t update = " << update[0] << ", " << update[1]
////         << "\t new chi2 = " << new_chi2 << endl;
//#endif
//
//            if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
//            {
//#if SUBPIX_VERBOSE
//                cout << "converged." << endl;
//#endif
//                converged=true;
//                break;
//            }
//        }
//
//        cur_px_estimate << u, v;
//        return converged;
    }

    bool Matcher::ReprojectCell(Matcher::Cell &cell, Frame *frame)
    {

    }

    bool Matcher::FindMatchDirect(const MapPoint &pt, Frame *F1, cv::Vec2d& px_cur)
    {
//        Eigen::Matrix2f A_cur_ref;
//        int search_level;
//        uint8_t* patch;
//        uint8_t* patch_with_border;
//        double h_inv;
//
//        // get A_cur_ref from pose
//        {
//            const int half_size = 5;
//            const Eigen::Vector3d xyz_ref;
//            Eigen::Vector3d xyz_du_ref;
//            Eigen::Vector3d xyz_dv_ref;
//
//            const Eigen::Vector2d pt_cur(F1->GetCameraModel()->World2Cam(cv::Vec3d(0,0,0)));
//            const Eigen::Vector2d pt_du(F1->GetCameraModel()->World2Cam(cv::Vec3d(0,0,0)));
//            const Eigen::Vector2d pt_dv(F1->GetCameraModel()->World2Cam(cv::Vec3d(0,0,0)));
//            A_cur_ref.col(0) = (pt_du - pt_cur) / half_size;
//            A_cur_ref.col(1) = (pt_dv - pt_cur) / half_size;
//        }
//        // get search_level from A_cur_ref
//        double D = A_cur_ref.determinant();
//        while (D > 3.0 && search_level < F1->GetMaxLevel() - 1)
//        {
//            search_level += 1;
//            D *= 0.25;
//        }
//        // get patch with border
//        Eigen::Vector2d px_scaled(px_cur * F1->GetScaleFactors().at(search_level));
//        {
//            const int patch_size_border = patch_size + 2;
//            const int patch_half_border = patch_half_size + 1;
//            Eigen::Matrix2f A_ref_cur(A_cur_ref.inverse());
//            uint8_t* patch_ptr = patch_with_border;
//
//            for (int y = 0; y < patch_size_border; ++y)
//            {
//                for(int x = 0; x < patch_size_border; ++x, ++patch_ptr)
//                {
//                    Eigen::Vector2f px_patch(x - patch_half_border, y - patch_half_border);
//                    px_patch *= F1->GetScaleFactors().at(search_level);
//                    const Eigen::Vector2f px(A_ref_cur * px_patch, px_scaled);
//                    if(px[0] < 0 || px[1] < 0 || px[0] >= (float)F1->GetImage().cols - 1 || px[1] >= (float)F1->GetImage().rows - 1)
//                        *patch_ptr = 0;
//                    else
//                    {
//                        int u = floor(px[0]);
//                        int v = floor(px[1]);
//                        float subpix_u = px[0] - (float)u;
//                        float subpix_v = px[1] - (float)v;
//                        float w00 = (1.f - subpix_u)*(1.f - subpix_v);
//                        float w01 = (1.f - subpix_u)*subpix_v;
//                        float w10 = subpix_u*(1.f - subpix_v);
//                        float w11 = 1.f - w00 - w01 - w10;
//                        const size_t stride = F1->GetImage().step.p[0];
//                        unsigned char* ptr = F1->GetImage().data + y*stride + x;
//                        *patch_ptr = (uint8_t)(w00*(float)ptr[0] + w01*(float)ptr[stride] + w10*(float)ptr[1] + w11*(float)ptr[stride+1]);
//                    }
//                }
//            }
//        }
//        // get patch
//        uint8_t* patch_ptr = patch;
//        for(int y = 1; y < patch_size + 1; ++y, patch_ptr += patch_size)
//        {
//            uint8_t* y_ptr = patch_with_border + y * (patch_size + 2) + 1;
//            for(int x = 0; x < patch_size; ++x)
//                patch_ptr[x] = y_ptr[x];
//        }
//
//
//        bool success = false;
//        const int align_max_iter = 10;
//        if(pt.type == EDGE)
//        {
//            Eigen::Vector2f grad(pt.grad.at<float>(0), pt.grad.at<float>(1));
//            Eigen::Vector2f dir_cur(A_cur_ref * grad);
//
//            dir_cur.normalize();
//
//            success = Align1D(F1->GetImagePyramid().at(search_level),
//                              dir_cur,patch_with_border, patch,align_max_iter,
//                              px_scaled, h_inv);
//        }
//        else
//        {
//            success = Align2D(F1->GetImagePyramid().at(search_level),
//                              patch_with_border, patch,align_max_iter,
//                              px_scaled);
//        }
//        px_cur[0] = px_scaled.x() / F1->GetScaleFactors().at(search_level);
//        px_cur[1] = px_scaled.y() / F1->GetScaleFactors().at(search_level);
    }

}

