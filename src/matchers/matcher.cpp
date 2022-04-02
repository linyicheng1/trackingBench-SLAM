#include "matchers/matcher.h"
#include "types/Frame.h"

namespace TRACKING_BENCH
{

    Matcher::Matcher()
    {
        m_flann_matcher = cv::FlannBasedMatcher::create();
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
    std::vector<cv::DMatch> Matcher::searchByNN(Frame *F1, Frame *F2, int MinLevel, int MaxLevel, bool MapPointOnly)
    {
        assert(MinLevel <= MaxLevel);
        cv::Mat d1, d2;
        std::vector<int> id1;
        if(MinLevel == 0 && MaxLevel == F1->getMaxLevel() && !MapPointOnly)
        {// using all points to match
            d1 = F1->getDescriptors();
            d2 = F2->getDescriptors();
        }
        else
        {
            id1.resize(F1->getKeysUn().size());

            int cnt = 0, match_num = 0;
            for(auto&f:F1->getKeysUn())
            {
                if(f.octave >= MinLevel && f.octave <= MaxLevel)
                {// only the features between [MinLevel, MaxLevel]
                    if(!MapPointOnly || F1->getMapPoint(&f) != nullptr)
                    {// all points with MapPointOnly == false , Only Match the features associate with Map Point when MapPointOnly == true
                        id1.at(match_num) = cnt;
                        // descriptor
                        d1.row(match_num) = F1->getDescriptors().row(cnt);
                        match_num ++;
                    }
                }
                cnt ++;
            }
            d2 = F2->getDescriptors();
        }

        std::vector<cv::DMatch> matches;
        std::vector<cv::DMatch> good_matches;
        m_flann_matcher->match(d1, d2, matches);

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
            id1.resize(map->);
            int cnt = 0, match = 0;
            for(int i = 0; i< map;i ++)
            {
                des2.row(i);
            }
        }
        else
        {// only
        }

        m_flann_matcher->match(F1->getDescriptors(), des2, matches);

        float min_distance = std::min_element(matches.begin(), matches.end(),[](const cv::DMatch& m1, const cv::DMatch& m2){return m1.distance < m2.distance;})->distance;
        //float max_distance = std::max_element(matches.begin(), matches.end(),[](const cv::DMatch& m1, const cv::DMatch& m2){return m1.distance < m2.distance;})->distance;

        for(const auto &m:matches)
        {
            if(m.distance < std::fmin(nRatio*min_distance, TH_HIGH))
            {
                good_matches.emplace_back(m);
            }
        }
        return good_matches;
    }
    std::vector<cv::DMatch> Matcher::searchByBF(Frame *F1, Frame *F2, bool MapPointOnly)
    {
        std::vector<cv::DMatch> matches;
        std::vector<cv::DMatch> good_matches;
        m_bf_matcher->match(F1->getDescriptors(), F2->getDescriptors(), matches);

        float min_distance = std::min_element(matches.begin(), matches.end(),[](const cv::DMatch& m1, const cv::DMatch& m2){return m1.distance < m2.distance;})->distance;
        //float max_distance = std::max_element(matches.begin(), matches.end(),[](const cv::DMatch& m1, const cv::DMatch& m2){return m1.distance < m2.distance;})->distance;

        for(const auto &m:matches)
        {
            if(m.distance < std::fmin(nRatio*min_distance, TH_HIGH))
            {
                good_matches.emplace_back(m);
            }
        }
        return good_matches;
    }
    std::vector<cv::DMatch> Matcher::searchByBF(Map *map, Frame *F1)
    {

    }

    std::vector<cv::DMatch> Matcher::searchByViolence(Frame *F1, Frame *F2, bool MapPointOnly)
    {
        std::vector<cv::DMatch> matches;
        std::vector<int> matches12;
        std::vector<float> distance12;

        matches.reserve((int)fmin((double)F1->getKeysUn().size(), (double)F2->getKeysUn().size()));
        std::vector<int> rotHist[HISTO_LENGTH];
        for(int i = 0;i < HISTO_LENGTH; i++)
        {
            rotHist[i].reserve(500);
        }
        const float factor = 1.f / (float)HISTO_LENGTH;
        for(size_t i1 = 0; i1 < F1->getKeysUn().size();i1 ++)
        {
            std::vector<size_t> vIndices2 = F2->GetFeaturesInArea(F1->getKeysUn().at(i1).pt.x, F1->getKeysUn().at(i1).pt.y, 10,0,1);
            if(vIndices2.empty())
                continue;

            cv::Mat d1 = F1->getDescriptors().row((int)i1);
            int bestDist = INT_MAX;
            int bestDist2 = INT_MAX;
            int bestIdx2 = -1;

            for(auto& i2:vIndices2)
            {
                cv::Mat d2 = F2->getDescriptors().row((int)i2);
                int dist = descriptorDistance(d1, d2);

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

            if(bestDist <= TH_LOW)
            {
                if((float)bestDist <(float)bestDist2*nRatio)
                {
                    if(checkOrientation)
                    {
                        float rot = F1->getKeysUn().at(i1).angle - F2->getKeysUn().at(bestIdx2).angle;
                        if(rot < 0)
                            rot += 360.f;
                        int bin = (int)roundf(rot*factor);
                        if(bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back((int)i1);
                    }
                }
            }
        }
        if(checkOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            computeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for(int i = 0;i < HISTO_LENGTH;i ++)
            {
                if(i==ind1 || i==ind2 || i == ind3)
                {// save matches
                    for(auto &id1:rotHist[i])
                    {
                        cv::DMatch match;
                        match.distance = distance12.at(id1);
                        match.queryIdx = id1;
                        match.trainIdx = matches12.at(id1);
                    }
                }
            }
        }
    }

    std::vector<cv::DMatch> Matcher::searchByViolence(Map *map, Frame *F1)
    {

    }

    std::vector<cv::DMatch> Matcher::searchByProjection(Frame *F1, Frame *F2, bool MapPointOnly)
    {

    }

    std::vector<cv::DMatch> Matcher::searchByProjection(Map *KF1, Frame *F1)
    {

    }

    std::vector<cv::DMatch> Matcher::searchByFeatureAlignment(Frame *F1, Frame *F2, bool MapPointOnly)
    {

    }

    std::vector<cv::DMatch> Matcher::searchByFeatureAlignment(Map *map, Frame *F1)
    {

    }

    std::vector<cv::DMatch> Matcher::searchByBow(Frame *F1, Frame *F2, bool MapPointOnly)
    {

    }

    std::vector<cv::DMatch> Matcher::searchByBow(Map *map, Frame *F2)
    {

    }

    std::vector<cv::DMatch> Matcher::searchByOPFlow(Frame *F1, Frame *F2, bool MapPointOnly)
    {

    }

    std::vector<cv::DMatch> Matcher::searchByOPFlow(Map *map, Frame *F1)
    {

    }

    std::vector<cv::DMatch> Matcher::searchByDirect(Frame *F1, Frame *F2, bool MapPointOnly, bool ph)
    {

    }

    std::vector<cv::DMatch> Matcher::searchByDirect(Map *map, Frame *F1)
    {

    }



    int Matcher::descriptorDistance(const cv::Mat &a, const cv::Mat &b)
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

    void  Matcher::computeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
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
}

