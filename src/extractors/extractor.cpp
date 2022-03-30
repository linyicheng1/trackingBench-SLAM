#include <opencv2/imgproc.hpp>
#include "extractors/extractor.h"

namespace TRACKING_BENCH
{
    Extractor::Extractor(int num, float _scaleFactor, int _levels):
    nFeatures(num), nLevels(_levels),scaleFactor(_scaleFactor)
    {
        mvScaleFactor.resize(nLevels);
        mvScaleFactor[0]=1.0f;
        for(int i=1; i<nLevels; i++)
        {
            mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        }
        mvInvScaleFactor.resize(nLevels);
        for(int i=0; i<nLevels; i++)
        {
            mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        }
        mvImagePyramid.resize(nLevels);
    }

    void Extractor::ComputePyramid(cv::Mat image)
    {
        for (int level = 0; level < nLevels; ++level)
        {
            float scale = mvInvScaleFactor[level];
            cv::Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
            cv::Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
            cv::Mat temp(wholeSize, image.type()), masktemp;
            mvImagePyramid[level] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

            // Compute the resized image
            if( level != 0 )
            {
                cv::resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, CV_INTER_LINEAR);

                copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               cv::BORDER_REFLECT_101+cv::BORDER_ISOLATED);
            }
            else
            {
                copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               cv::BORDER_REFLECT_101);
            }
        }
    }
}
