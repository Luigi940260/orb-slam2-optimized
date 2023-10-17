#include "Converter.hpp"

namespace ORB_SLAM_CUSTOM
{

std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

g2o::SE3Quat Converter::toSE3Quat(const Eigen::Isometry3f &T)
{
    Eigen::Matrix3d R = T.rotation().cast<double>();
    Eigen::Vector3d t = T.translation().cast<double>();

    return g2o::SE3Quat(R,t);
}

Eigen::Isometry3f Converter::toIso(const g2o::SE3Quat &SE3)
{
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    T.matrix() = SE3.to_homogeneous_matrix().cast<float>();
    return T;
}

Eigen::Isometry3f Converter::toIso(const g2o::Sim3 &Sim3)
{
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    T.linear() = Sim3.rotation().toRotationMatrix().cast<float>();
    T.translation() = Sim3.translation().cast<float>();
    return T;
}

Eigen::Isometry3f Converter::toIso(const Eigen::Matrix3d &R, const Eigen::Vector3d &t)
{
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    T.linear() = R.cast<float>();
    T.translation() = t.cast<float>();

    return T;
}

std::vector<float> Converter::toQuaternion(const Eigen::Matrix3f &R)
{
    Eigen::Quaternionf q(R);

    std::vector<float> v(4);
    v.at(0) = q.x();
    v.at(1) = q.y();
    v.at(2) = q.z();
    v.at(3) = q.w();

    return v;
}

} //namespace ORB_SLAM
