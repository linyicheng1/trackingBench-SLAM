#ifndef TRACKING_BENCH_CAMERAMODEL_H
#define TRACKING_BENCH_CAMERAMODEL_H

#include "Eigen/Core"
#include "opencv2/core.hpp"

namespace TRACKING_BENCH
{
    class CameraModel
    {
    protected:
        int mnWidth{};
        int mnHeight{};
    public:
        CameraModel() = default;
        CameraModel(const CameraModel &model)
        {
            mnWidth = model.mnWidth;
            mnHeight = model.mnHeight;
        }
        CameraModel(int width, int height):mnWidth(width),mnHeight(height){}

        virtual Eigen::Vector3f Cam2World(const float& x, const float& y, bool distor = false) const = 0;
        virtual Eigen::Vector3f Cam2World(const Eigen::Vector2f& px, bool distor = false) const = 0;
        virtual Eigen::Vector2f World2Cam(const Eigen::Vector3f& xyz_c) const = 0;
        virtual Eigen::Vector2f  World2Cam(const Eigen::Vector2f& uv)const = 0;
        virtual Eigen::Vector2f UndistortPoint(const Eigen::Vector2f& uv)const = 0;
        virtual double ErrorMultiplier() const = 0;

        inline int Width() const {return mnWidth;}
        inline int Height() const {return mnHeight;}

        inline bool IsInFrame(const Eigen::Vector2i& obs, int boundary = 0, float scale = 1) const
        {
            if(obs[0] >= boundary && obs[0] < (int)((float)Width()  * scale) - boundary &&
               obs[1] >= boundary && obs[1] < (int)((float)Height() * scale) - boundary)
                return true;
            return false;
        }

        virtual void UndistortImage(const cv::Mat& raw, cv::Mat& rectified) = 0;

        virtual Eigen::Vector2f focal_length() const= 0;
    };

    class PinholeCamera:public CameraModel
    {
    public:
        PinholeCamera(int width, int height,
                      float fx, float fy, float cx, float cy,
                      float d0 = 0.0, float d1 = 0.0, float d2 = 0.0, float d3 = 0.0, float d4 = 0.0);
        ~PinholeCamera();

        Eigen::Vector3f Cam2World(const float& x, const float& y, bool distor = false) const override;
        Eigen::Vector3f Cam2World(const Eigen::Vector2f& px, bool distor = false) const override;
        Eigen::Vector2f World2Cam(const Eigen::Vector3f& xyz_c) const override;
        Eigen::Vector2f  World2Cam(const Eigen::Vector2f& uv)const override;
        Eigen::Vector2f UndistortPoint(const Eigen::Vector2f& uv)const override;
        Eigen::Vector2f focal_length() const{return Eigen::Vector2f{mFx, mFy};}

        double ErrorMultiplier() const override
        {
            return fabs(4.0 * mFx * mFx);
        }

        inline cv::Mat GetCVK() const{ return mCVK;}
        inline cv::Mat GetCVKInv() const{ return mCVKInv;}

        inline float fx() const {return mFx;}
        inline float fy() const {return mFy;}
        inline float cx() const {return mCx;}
        inline float cy() const {return mCy;}
        inline float d0() const {return md[0];}
        inline float d1() const {return md[1];}
        inline float d2() const {return md[2];}
        inline float d3() const {return md[3];}
        inline float d4() const {return md[4];}
        void UndistortImage(const cv::Mat& raw, cv::Mat& rectified);
    private:
        const float mFx, mFy;
        const float mCx, mCy;
        bool mbDistortion;
        float md[5]{};

        cv::Mat mCVK, mCVD, mCVKInv;
        cv::Mat mUndistMap1, mUndistMap2;

        bool mbUseOptimization;
    };

}

#endif //TRACKING_BENCH_CAMERAMODEL_H
