#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include"Map.hpp"
#include"MapPoint.hpp"
#include"KeyFrame.hpp"
#include<pangolin/pangolin.h>

#include<mutex>

namespace ORB_SLAM_CUSTOM
{

class MapDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapDrawer(std::shared_ptr<Map> pMap, const string &strSettingPath);

    std::shared_ptr<Map> mpMap;

    void DrawMapPoints();
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
    void SetCurrentCameraPose(const cv::Mat &Tcw);
    void SetReferenceKeyFrame(KeyFrame *pKF);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

private:

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    cv::Mat mCameraPose;

    std::mutex mMutexCamera;
};

} //namespace ORB_SLAM
