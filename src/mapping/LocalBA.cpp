#include "mapping/LocalBA.h"
#include <Eigen/SVD>
#include "types/Frame.h"
#include "types/MapPoint.h"
#include "camera/CameraModel.h"
#include "matchers/matcher.h"

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include<suitesparse/cholmod.h>
#include <opencv2/opencv.hpp>

namespace TRACKING_BENCH
{
    Eigen::Vector3f LocalBA::LinearTriangle(
            const Eigen::Vector2f& p0,
            const Eigen::Vector2f& p1,
            const Eigen::Matrix4f& Tcw0,
            const Eigen::Matrix4f& Tcw1)
    {
        Eigen::Vector3f result;
        Eigen::Matrix4f design_matrix = Eigen::Matrix4f::Zero();
        design_matrix.row(0) = p0[0] * Tcw0.row(2) - Tcw0.row(0);
        design_matrix.row(1) = p0[1] * Tcw0.row(2) - Tcw0.row(1);
        design_matrix.row(2) = p1[0] * Tcw1.row(2) - Tcw1.row(0);
        design_matrix.row(3) = p1[1] * Tcw1.row(2) - Tcw1.row(1);

        Eigen::Vector4f point;
        point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

        result(0) = point(0) / point(3);
        result(1) = point(1) / point(3);
        result(2) = point(2) / point(3);
    }


    std::vector<float> LocalBA::AddMapPointsByStereo(const std::shared_ptr<Frame>& current_frame,
                                                     const std::shared_ptr<Frame>& stereo_frame,
                                                     const float bf, const float fx)
    {
        const size_t N = (int)current_frame->GetKeys().size();
        std::vector<float> Depth(N, -1.0f);
        std::shared_ptr<Matcher> matcher = std::make_shared<Matcher>();
        std::vector<cv::Point2f> pts;
        auto matches = matcher->searchByOPFlow(stereo_frame, current_frame, pts, true, true);

        cv::Mat show = current_frame->GetImagePyramid()[0].clone();
        cv::cvtColor(show, show, CV_GRAY2BGR);
        for (auto& m:matches)
        {
            auto left_id = m.trainIdx;
            auto right_id = m.queryIdx;
            auto left_pt = current_frame->GetKey(left_id);
            auto right_pt = pts.at(right_id);
            cv::line(show, left_pt->kp.pt, right_pt, cv::Scalar(0,0,255),4);
            Depth[left_id] = bf / fabsf(pts.at(right_id).x - current_frame->GetKey(left_id)->kp.pt.x);
        }
        cv::imshow("stereo", show);
        return Depth;
/*
        // old
        std::vector<cv::DMatch> matches;

        const size_t N = (int)current_frame->GetKeys().size();
        std::vector<float> Depth(N, -1.0f);

        const int thORBDist = 75;
        const int nRows = current_frame->GetImagePyramid().at(0).rows;

        std::vector<std::vector<std::size_t>> vRowIndices(nRows, std::vector<size_t>());
        for (int i=0;i < nRows; i++)
            vRowIndices.reserve(200);

        // Assign keyPoints to row table
        const size_t Nr = stereo_frame->GetKeys().size();
        for (size_t iR = 0; iR < Nr; iR ++)
        {
            const float &kpY = stereo_frame->GetKey(iR)->px.y();
            const float r = 2.f * current_frame->GetInverseScaleFactors().at(stereo_frame->GetKey(iR)->kp.octave);
            const int max_r = ceil(kpY + r);
            const int min_r = floor(kpY - r);

            for(int yi = min_r; yi <= max_r; yi ++)
                vRowIndices[yi].push_back(iR);
        }

        // set limits for search
        const float minZ = bf / fx;
        const float minD = 0;
        const float maxD = bf/ minZ;

        std::vector<std::pair<int, int>> vDistIdx;
        vDistIdx.reserve(N);
        cv::Mat show = current_frame->GetImagePyramid()[0].clone();
        cv::cvtColor(show, show, CV_GRAY2BGR);

        for ( size_t iL=0; iL<N; iL ++)
        {
            const int& levelL = current_frame->GetKey(iL)->kp.octave;
            const float& vL = current_frame->GetKey(iL)->px.y();
            const float& uL = current_frame->GetKey(iL)->px.x();


            const std::vector<size_t>& vCandidates = vRowIndices[(size_t)vL];

            if(vCandidates.empty())
                continue;

            const float minU = 0;//uL - maxD;
            const float maxU = 100000;//uL - minD;

            if (maxU < 0)
                continue;

            int bestDist0 = INT_MAX;
            size_t bestIdxR0 = 0;

            const cv::Mat& dL = current_frame->GetDescriptor((int)iL);

            for (unsigned long iR : vCandidates)
            {
                const cv::KeyPoint& kpR = stereo_frame->GetKey(iR)->kp;
                if(kpR.octave < levelL -1 || kpR.octave > levelL + 1)
                    continue;

                const float& uR = kpR.pt.x;

                if (uR >= minU && uR <= maxU)
                {
                    const cv::Mat& dR = stereo_frame->GetDescriptor((int)iR);
                    const int dist = Matcher::DescriptorDistance(dL, dR);
                    if(dist < bestDist0 && dist > 0)
                    {
                        bestDist0 = dist;
                        bestIdxR0 = iR;
                    }
                }
            }
            // subpixel match by correlation
            if (bestDist0 < 30)//thORBDist)
            {
                static int cnt = 0;
                if(cnt == 10)
                {
                    for(auto c:vCandidates)
                    {
                        cv::Point pt(stereo_frame->GetKey(c)->kp.pt);
                        cv::circle(show, pt, 1, cv::Scalar(255, 0,0),4);
                    }
                }
                cnt ++;
                cv::DMatch m;
                m.queryIdx = iL;
                m.trainIdx = bestIdxR0;
                m.distance = bestDist0;
                matches.emplace_back(m);

                auto left_pt = current_frame->GetKey(m.queryIdx);
                auto right_pt = stereo_frame->GetKey(m.trainIdx);
                if (abs(left_pt->kp.pt.x - right_pt->kp.pt.x) > 0)
                    Depth[iL]= bf/abs(left_pt->kp.pt.x - right_pt->kp.pt.x);
                else
                    Depth[iL] = -1;
//                const float uR0 = stereo_frame->GetKey(bestIdxR0)->px.x();
//                const float scaleFactor = current_frame->GetScaleFactors().at(levelL);
//
//                const float scaled_uL = round(current_frame->GetKey(iL)->kp.pt.x * scaleFactor);
//                const float scaled_vL = round(current_frame->GetKey(iL)->kp.pt.y * scaleFactor);
//                const float scaled_uR0 = round(uR0 * scaleFactor);
//
//
//                // sliding window search
//                const int w = 5;
//                if(scaled_vL - w < 0 || scaled_uL-w < 0)
//                    continue;
//                cv::Mat IL = current_frame->GetImagePyramid().at(levelL).rowRange(scaled_vL-w, scaled_vL+w+1).colRange(scaled_uL-w, scaled_uL+w+1);
//                IL.convertTo(IL, CV_32F);
//                IL = IL - IL.at<float>(w, w) * cv::Mat::ones(IL.rows, IL.cols, CV_32F);
//
//                int bestDist = INT_MAX;
//                int bestincR = 0;
//                const int L = 5;
//                std::vector<float> vDists;
//                vDists.resize(2*L + 1);
//
//                const float iniu = scaled_uR0 - L -w;
//                const float endu = scaled_uR0 + L + w + 1;
//                if(iniu < 0 || endu >= stereo_frame->GetImagePyramid().at(levelL).cols)
//                    continue;
//
//                for(int incR=-L; incR<=+L; incR++)
//                {
//                    cv::Mat IR = stereo_frame->GetImagePyramid()[levelL].rowRange(scaled_vL-w,scaled_vL+w+1).colRange(scaled_uR0+incR-w,scaled_uR0+incR+w+1);
//                    IR.convertTo(IR,CV_32F);
//                    IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);
//
//                    float dist = cv::norm(IL,IR,cv::NORM_L1);
//                    if(dist<bestDist)
//                    {
//                        bestDist =  dist;
//                        bestincR = incR;
//                    }
//
//                    vDists[L+incR] = dist;
//                }
//
//                if(bestincR==-L || bestincR==L)
//                    continue;
//
//                // Sub-pixel match (Parabola fitting)
//                const float dist1 = vDists[L+bestincR-1];
//                const float dist2 = vDists[L+bestincR];
//                const float dist3 = vDists[L+bestincR+1];
//
//                const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));
//
//                if(deltaR<-1 || deltaR>1)
//                    continue;
//
//                // Re-scaled coordinate
//                float bestuR = current_frame->GetInverseScaleFactors().at(levelL)*((float)scaled_uR0+(float)bestincR+deltaR);
//
//                float disparity = (uL-bestuR);
//
//                if(disparity>=minD && disparity<maxD)
//                {
//                    if(disparity<=0)
//                    {
//                        disparity=0.01;
//                        bestuR = uL-0.01;
//                    }
//
//
//                    Depth[iL]= bf/disparity;
//                    vDistIdx.push_back(pair<int,int>(bestDist,iL));
//                }
            }
        }
//        sort(vDistIdx.begin(), vDistIdx.end());
//        const auto median = (float)vDistIdx[(int)(vDistIdx.size()/2)].first;
//        const float thDist = 1.5f*1.4f*median;
//
//        for (size_t i=vDistIdx.size()-1; i>=0;i --)
//        {
//            if ((float)vDistIdx[i].first < thDist)
//                break;
//            else
//            {
//                Depth[i] = -1;
//            }
//        }


        std::vector<cv::KeyPoint> kps1, kps2;
        for (auto& kp:current_frame->GetKeys())
        {
            kps1.emplace_back(kp->kp);
        }
        for (auto& kp:stereo_frame->GetKeys())
        {
            kps2.emplace_back(kp->kp);
        }

        for (int i = 10;i < matches.size(); i+= 10000)
        {
            auto match = matches.at(i);
            auto left_pt = current_frame->GetKey(match.queryIdx);
            auto right_pt = stereo_frame->GetKey(match.trainIdx);
            cv::line(show, left_pt->kp.pt, right_pt->kp.pt, cv::Scalar(0,0,255),4);
        }
//        cv::drawMatches(
//                current_frame->GetImagePyramid()[0],
//                kps1,
//                stereo_frame->GetImagePyramid()[0],
//                kps2,
//                matches, show);
        cv::imshow("Stereo", show);
//        cv::waitKey(0);
        return Depth; */
    }

    int LocalBA::PoseOptimization(std::shared_ptr<Frame>& F)
    {
        g2o::SparseOptimizer optimizer;
        optimizer.setVerbose(false);

        std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
        linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();

        // 选择迭代策略，通常还是L-M算法居多
        auto* solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
        );
        solver->setUserLambdaInit(0.0001);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences=0;

        // Set Frame vertex
        auto vSE3 = new g2o::VertexSE3Expmap();
        Eigen::Matrix3d R = F->GetRotation().cast<double>().transpose();
        Eigen::Vector3d t = - R * F->GetTranslation().cast<double>();
        vSE3->setEstimate(g2o::SE3Quat(R, t));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const auto N = (int)F->GetKeys().size();

        vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        const float deltaMono = sqrtf(5.991f);

        cv::Mat show = F->GetImagePyramid().at(0).clone();
        cv::cvtColor(show, show, cv::COLOR_GRAY2BGR);

        {
            std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);

            for(int i=0; i<N; i++)
            {
                auto pMP = F->GetMapPoint(i);
                if(pMP)
                {
                    // Monocular observation
                    nInitialCorrespondences++;

                    Eigen::Matrix<double,2,1> obs;
                    obs << F->GetKey(i)->px.x(), F->GetKey(i)->px.y();

                    auto e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);

                    const float invSigma2 = F->GetInverseScaleSigmaSquares().at(F->GetKey(i)->kp.octave);
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    auto rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->fx = 718.856;
                    e->fy = 718.856;
                    e->cx = 607.1928;
                    e->cy = 185.2157;

                    e->Xw[0] = F->GetMapPoint(i)->GetWorldPos().x();
                    e->Xw[1] = F->GetMapPoint(i)->GetWorldPos().y();
                    e->Xw[2] = F->GetMapPoint(i)->GetWorldPos().z();

                    Eigen::Vector3d pw = e->Xw;
                    Eigen::Vector3d pc = vSE3->estimate().map(pw);
                    auto proj = e->cam_project(pc);
                    double my_e2 = (proj - obs).dot(proj - obs);
                    e->computeError();
                    auto e2 = (float)e->chi2();
                    if(e2 > 3)
                    {
//                        e->setLevel(1);
//                        std::cout<<"here!";
                    }

                    auto T_cw = F->GetPose();
                    const Eigen::Vector3f pw0 = F->GetMapPoint(i)->GetWorldPos();
                    const Eigen::Vector3f pc0 = T_cw.block<3, 3>(0, 0) * pw0 + T_cw.block<3, 1>(0, 3);
                    const Eigen::Vector2f uv0 = F->GetCameraModel()->World2Cam(pc0);

                    cv::Point p0(obs.x(), obs.y());
                    cv::Point p1(proj.x(), proj.y());
                    cv::Point p2(uv0.x(), uv0.y());

                    cv::circle(show, p0, 8,cv::Scalar(0, 0, 255), -1);
                    cv::circle(show, p1, 2,cv::Scalar(0, 255, 0), -1);
                    cv::circle(show, p2, 2,cv::Scalar(255, 0, 0), -1);


                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }
        }

        cv::imshow("sba", show);
//        cv::waitKey(0);

        if(nInitialCorrespondences<3)
            return 0;

        // test error
        float sum_err = 0;
        int cnt = 0;
        for(auto& ee:vpEdgesMono)
        {
            if(ee->level() == 0)
            {
                ee->computeError();
                auto e2 = (float)ee->chi2();
                sum_err += e2;
                cnt ++;
            }
        }
        std::cout<<"sum err "<<sum_err<<" mean err: "<<sum_err / cnt<<std::endl;
        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4]={5.991,5.991,5.991,5.991};
        const int its[4]={10,10,10,10};

        int nBad=0;
        for(size_t it=0; it<4; it++)
        {
            Eigen::Matrix3f R0 = F->GetRotation().transpose();
            Eigen::Vector3f t0 = - R0 * F->GetTranslation();
            vSE3->setEstimate(g2o::SE3Quat(R0.cast<double>(), t0.cast<double>()));

            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad=0;
            for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
            {
                g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if(F->GetOutlier(idx))
                {
                    e->computeError();
                }

                const auto chi2 = (float)e->chi2();

                if(chi2>chi2Mono[it])
                {
                    F->SetOutlier(idx, true);
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    F->SetOutlier(idx, false);
                    e->setLevel(0);
                }

                if(it==2)
                    e->setRobustKernel(nullptr);
            }

            float sum_err = 0;
            int cnt = 0;
            for(auto& ee:vpEdgesMono)
            {
                if(ee->level() == 0)
                {
                    ee->computeError();
                    auto e2 = (float)ee->chi2();
                    sum_err += e2;
                    cnt ++;
                }
            }
            std::cout<<"sum err "<<sum_err<<" mean err: "<<sum_err / cnt<<std::endl;

            if(optimizer.edges().size()<10)
                break;
        }


        // Recover optimized pose and return number of inliers
        auto* vSE3_recov = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        pose.block<3, 3>(0, 0) = SE3quat_recov.rotation().toRotationMatrix().cast<float>();
        pose.block<3, 1>(0, 3) = SE3quat_recov.translation().cast<float>();
        F->SetPose(pose);
        return nInitialCorrespondences-nBad;
    }
}

