#include "camera/CameraModel.h"
#include "opencv2/opencv.hpp"

namespace TRACKING_BENCH
{
    PinholeCamera::PinholeCamera(int width, int height, float fx, float fy, float cx, float cy, float d0,
                                 float d1, float d2, float d3, float d4):
                                 CameraModel(width, height),mFx(fx),mFy(fy),mCx(cx),mCy(cy),
                                 mbDistortion(fabs(d0) > 0.0000001),
                                 mUndistMap1(height, width, CV_16SC2),
                                 mUndistMap2(height, width, CV_16SC2),
                                 mbUseOptimization(false)
    {
        md[0] = d0;md[1]=d1;md[2]=d2;md[3]=d3;md[4]=d4;
        mCVK = (cv::Mat_<float>(3, 3) << mFx, 0.0, mCx, 0.0, mFy, mCy, 0.0, 0.0, 1.0);
        mCVD = (cv::Mat_<float>(1, 5) << md[0], md[1], md[2], md[3], md[4]);
        cv::initUndistortRectifyMap(mCVK, mCVD, cv::Mat(), mCVK,
                                    cv::Size(mnWidth, mnHeight), CV_32F, mUndistMap1, mUndistMap2);
    }

    PinholeCamera::~PinholeCamera()
    = default;

    /**
     * @brief image (u,v) --> camera (x,y)
     * @param u
     * @param v
     * @return camera (x,y)
     */
    Eigen::Vector3f PinholeCamera::Cam2World(const float &u, const float &v, bool distor) const
    {
        Eigen::Vector3f xyz;

        if(mbDistortion && distor)
        {
            cv::Point2f uv((float)u, (float)v), px;
            const cv::Mat src_pt(1, 1, CV_32FC2, &uv.x);
            cv::Mat dst_pt(1, 1, CV_32FC2, &px.x);
            cv::undistortPoints(src_pt, dst_pt, mCVK, mCVD);
            xyz[0] = px.x;
            xyz[1] = px.y;
            xyz[2] = 1.f;
        }
        else
        {
            xyz[0] = (u - mCx) / mFx;
            xyz[1] = (v - mCy) / mFy;
            xyz[2] = 1.f;
        }
        return xyz;
    }

    Eigen::Vector3f PinholeCamera::Cam2World(const Eigen::Vector2f &uv, bool distor) const
    {
        return Cam2World(uv[0], uv[1], distor);
    }

    /**
     * @brief camera world --> (u,v)
     * @param xyz_c xyz coordinate in camera
     * @return (u,v) in image
     */
    Eigen::Vector2f PinholeCamera::World2Cam(const Eigen::Vector3f &xyz_c) const
    {
        return World2Cam(Eigen::Vector2f(xyz_c[0]/xyz_c[2], xyz_c[1]/xyz_c[2]));
    }

    Eigen::Vector2f PinholeCamera::World2Cam(const Eigen::Vector2f &uv) const
    {
        Eigen::Vector2f px;
        if(!mbDistortion)
        {
            px[0] = mFx * uv[0] + mCx;
            px[1] = mFy * uv[1] + mCy;
        }
        else
        {
            float x, y, r2, r4, r6, a1, a2, a3, cdist, xd, yd;
            x = uv[0];
            y = uv[1];
            r2 = x*x + y*y;
            r4 = r2 * r2;
            r6 = r4 * r2;
            a1 = 2*x*y;
            a2 = r2 + 2*x*x;
            a3 = r2 + 2*y*y;
            cdist = 1 + md[0] * r2 + md[1] * r4 + md[4] * r6;
            xd = x * cdist + md[2] * a1 + md[3] * a2;
            yd = y * cdist + md[2] * a3 + md[3] * a1;
            px[0] = xd * mFx + mCx;
            px[1] = yd * mFy + mCy;
        }
        return px;
    }

    void PinholeCamera::UndistortImage(const cv::Mat &raw, cv::Mat &rectified)
    {
        if(mbDistortion)
            cv::remap(raw, rectified, mUndistMap1, mUndistMap2, CV_INTER_LINEAR);
        else
            rectified = raw.clone();
    }

    Eigen::Vector2f PinholeCamera::UndistortPoint(const Eigen::Vector2f &uv_pt) const
    {
        Eigen::Vector2f re;
        if(mbDistortion)
        {
            cv::Point2f uv((float)uv_pt.x(), (float)uv_pt.y()), px;
            const cv::Mat src_pt(1, 1, CV_32FC2, &uv.x);
            cv::Mat dst_pt(1, 1, CV_32FC2, &px.x);
            cv::undistortPoints(src_pt, dst_pt, mCVK, mCVD, cv::Mat(), mCVK);
            re.x() = px.x;
            re.y() = px.y;
        }
        else
        {
            re = uv_pt;
        }
        return re;
    }
}

