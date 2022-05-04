#ifndef TRACKING_BENCH_MATCHER_H
#define TRACKING_BENCH_MATCHER_H

#include <vector>
#include "opencv2/features2d.hpp"
#include <memory>
#include <Eigen/Core>
#include <list>

namespace TRACKING_BENCH
{
    class Frame;
    class MapPoint;
    class Map;
    class KeyFrame;

    class Matcher
    {
    public:
        Matcher();
        ~Matcher() = default;
        //  match parameter
        int TH_LOW = 50;
        int TH_HIGH = 100;
        int HISTO_LENGTH = 30;
        bool checkOrientation = true;
        float nRatio{};

        // F1 current frame
        // OpenCV NN
        std::vector<cv::DMatch> searchByNN(
                const std::shared_ptr<Frame>& F1,
                const std::shared_ptr<Frame>& F2,
                int MinLevel, int MaxLevel,
                float ratio, float minTh,
                bool MapPointOnly = false);
        std::vector<cv::DMatch> searchByNN(Map* map, Frame* F1, bool Projection = true);
        // OpenCV BF
        std::vector<cv::DMatch> searchByBF(
                const std::shared_ptr<Frame>& F1,
                const std::shared_ptr<Frame>& F2,
                int MinLevel, int MaxLevel,
                float ratio, float minTh,
                bool MapPointOnly = false);
        std::vector<cv::DMatch> searchByBF(Map* map, Frame* F1, bool Projection = true);

        // Violence
        void setViolenceParam(int low, int high, int histo_length, bool check, float ratio)
        {
            TH_LOW = low;
            TH_HIGH = high;
            HISTO_LENGTH = histo_length;
            checkOrientation=check;
            nRatio=ratio;
        }
        std::vector<cv::DMatch> searchByViolence(
                const std::shared_ptr<Frame>& F1,
                const std::shared_ptr<Frame>& F2,
                int min_level = 0,
                int max_level = 1,
                float search_r = 10,
                bool MapPointOnly = false);

        // Projection
        void setProjectionParam(int low, int high, int histo_length, bool check, float ratio)
        {
            TH_LOW = low;
            TH_HIGH = high;
            HISTO_LENGTH = histo_length;
            checkOrientation=check;
            nRatio=ratio;
        }

        std::vector<cv::DMatch> searchByProjection(
                const std::shared_ptr<Frame>& F1,
                const std::shared_ptr<Frame>& F2);

        std::vector<cv::DMatch> searchByProjection(
                const std::shared_ptr<Map>& map,
                const std::shared_ptr<Frame>& F1, float r);

        // Bow accelerate ORB only
        void setBowParam(int low, int high, int histo_length, bool check, float ratio)
        {
            TH_LOW = low;
            TH_HIGH = high;
            HISTO_LENGTH = histo_length;
            checkOrientation=check;
            nRatio=ratio;
        }
        std::vector<cv::DMatch> searchByBow(
                const std::shared_ptr<Frame>& F1,
                const std::shared_ptr<Frame>& F2,
                bool MapPointOnly = false);

        // Optical flow
        std::vector<cv::DMatch> searchByOPFlow(
                const std::shared_ptr<Frame>& F1,
                const std::shared_ptr<Frame>& F2,
                std::vector<cv::Point2f>& cur_points,
                bool equalized,
                bool reject,
                bool MapPointOnly = false);

        // Projection + Feature Alignment
        int mnMaxLevel;
        int mnMinLevel;
        int mnIter;
        int mnNTrialsMax;
        double mdEps;

        void setDirectParam(int maxLevel, int minLevel, int iter, int trials, double eps)
        {
            mnMaxLevel = maxLevel;
            mnMinLevel = minLevel;
            mnIter = iter;
            mnNTrialsMax = trials;
            mdEps = eps;
        }
        Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::ColMajor> jacobian_cache;

        struct Candidate {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            std::shared_ptr<MapPoint> pt;       //!< 3D point.
            Eigen::Vector2f px;     //!< projected 2D pixel location.
            Candidate(const std::shared_ptr<MapPoint>& pt, Eigen::Vector2f& x) : pt(pt), px(x) {}
        };
        typedef std::list<Candidate > Cell;
        typedef std::vector<Cell*> CandidateGrid;
        /// The grid stores a set of candidate matches. For every grid cell we try to find one match.
        struct Grid
        {
            CandidateGrid cells;
            std::vector<int> cell_order;
            int cell_size;
            int grid_n_cols;
            int grid_n_rows;
        };
        Grid grid_;
        Eigen::Matrix<float, 4, 4> SparseImageAlign(const std::shared_ptr<Frame>& F1, const std::shared_ptr<Frame>& F2);
        std::vector<cv::DMatch> FeaturesAlign(
                const std::shared_ptr<Map>& M,
                const std::shared_ptr<Frame>& F1,
                int maxFtx = 1000);
        std::vector<cv::DMatch> searchByDirect(std::shared_ptr<Map> M, const std::shared_ptr<Frame>& F1, const std::shared_ptr<Frame>& F2);
        bool Align2D(const cv::Mat& cur_img, uint8_t* ref_patch_with_border, uint8_t* ref_patch, const int n_iter, Eigen::Vector2f& cur_ps_estimate, bool no_simd = false);
        bool FindMatchDirect(std::shared_ptr<MapPoint>& pt, const std::shared_ptr<Frame>& F1, Eigen::Vector2f& px_cur);

        static int DescriptorDistance(const cv::Mat& a, const cv::Mat& b);
        static void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
    private:
        cv::FlannBasedMatcher m_flann_matcher;
        std::shared_ptr<cv::BFMatcher> m_bf_matcher;

        void rejectWithF(std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, std::vector<uchar>& status);
        const int patch_half_size = 2;
        const int patch_size = 4;
        const int patch_area = 16;
        bool have_ref_patch_cache = false;
        cv::Mat ref_patch_cache;
        Eigen::Matrix<double, 6, 6> mH;
        Eigen::Matrix<double, 6, 1> mJRes;
        Eigen::Matrix<double, 6, 1> mx;


        double ComputeResiduals(Eigen::Matrix4f& T_cur_from_ref,
                const std::shared_ptr<Frame>& F1,
                const std::shared_ptr<Frame>& F2,
                const std::vector<int>& F2_ids,
                std::vector<bool>& visible_fts,
                int level, bool linearize_system);
        void PreComputeReferencePatches(
                const std::shared_ptr<Frame>& F1,
                const std::shared_ptr<Frame>& F2,
                const std::vector<int>& F2_ids,
                std::vector<bool>& visible_fts,
                int level);

        bool ReprojectCell(Cell& cell, const std::shared_ptr<Frame>& frame);
        //bool FindMatchDirect(std::shared_ptr<MapPoint>& pt, const std::shared_ptr<Frame>& F1, Eigen::Vector2f& px_cur);
        //bool Align2D(const cv::Mat& cur_img, uint8_t* ref_patch_with_border, uint8_t* ref_patch, const int n_iter, Eigen::Vector2f& cur_ps_estimate, bool no_simd = false);
        bool Align1D(const cv::Mat& cur_img,const Eigen::Vector2f& dir,uint8_t* ref_patch_with_border,uint8_t* ref_patch,const int n_iter,Eigen::Vector2d& cur_px_estimate,double& h_inv);
    };
}

#endif //TRACKING_BENCH_MATCHER_H
