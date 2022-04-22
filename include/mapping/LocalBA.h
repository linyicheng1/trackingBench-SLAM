#ifndef TRACKING_BENCH_LOCAL_BA_H
#define TRACKING_BENCH_LOCAL_BA_H
#include <Eigen/Core>
#include <memory>
#include <vector>
#include <opencv2/core.hpp>

namespace TRACKING_BENCH
{
    class Frame;
    class LocalBA
    {
    public:
        LocalBA() = default;

        std::vector<float> AddMapPointsByStereo(
                const std::shared_ptr<Frame>& current_frame,
                const std::shared_ptr<Frame>& stereo_frame,
                float bf, float fx);
        Eigen::Vector3f LinearTriangle(const Eigen::Vector2f& p0, const Eigen::Vector2f& p1, const Eigen::Matrix4f& Tcw0, const Eigen::Matrix4f& Tcw1);

        int PoseOptimization(std::shared_ptr<Frame>& F);
    private:

    };
}

#endif //TRACKING_BENCH_LOCAL_BA_H
