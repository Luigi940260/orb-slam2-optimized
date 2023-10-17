#pragma once

#include<opencv2/core/core.hpp>

#include<Eigen/Dense>
#include"Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include"Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM_CUSTOM
{

class Converter
{
public:
    static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

    static g2o::SE3Quat toSE3Quat(const Eigen::Isometry3f &cvT);
    //static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);

    static Eigen::Isometry3f toIso(const g2o::SE3Quat &SE3);
    static Eigen::Isometry3f toIso(const g2o::Sim3 &Sim3);
    static Eigen::Isometry3f toIso(const Eigen::Matrix3d &R, const Eigen::Vector3d &t);

    static std::vector<float> toQuaternion(const Eigen::Matrix3f &R);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}// namespace ORB_SLAM
