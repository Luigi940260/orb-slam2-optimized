#pragma once

#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
using namespace cv;
#include <vector>

#include "KeyFrame.hpp"

namespace ORB_SLAM_CUSTOM
{

class Sim3Solver
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Sim3Solver(std::shared_ptr<KeyFrame> pKF1, std::shared_ptr<KeyFrame> pKF2, const std::vector<std::shared_ptr<MapPoint>> &vpMatched12);

    void SetRansacParameters(double probability = 0.99, int minInliers = 6 , int maxIterations = 300);

    bool find(std::vector<bool> &vbInliers12, int &nInliers);

    bool iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);

    Eigen::Matrix3f GetEstimatedRotation();
    Eigen::Vector3f GetEstimatedTranslation();


protected:

    void ComputeCentroid(const Eigen::Matrix3f &P, Eigen::Matrix3f &Pr, Eigen::Vector3f &C);

    void ComputeSim3(const Eigen::Matrix3f &P1, const Eigen::Matrix3f &P2);

    void CheckInliers();

    void Project(const std::vector<Eigen::Vector3f> &vP3Dw, std::vector<Eigen::Vector2f> &vP2D, const Eigen::Isometry3f &Tcw, const Eigen::Matrix3f &K);
    void FromCameraToImage(const std::vector<Eigen::Vector3f> &vP3Dc, std::vector<Eigen::Vector2f> &vP2D, const Eigen::Matrix3f &K);


protected:

    // KeyFrames and matches

    std::vector<Eigen::Vector3f> mvX3Dc1;
    std::vector<Eigen::Vector3f> mvX3Dc2;
    std::vector<std::shared_ptr<MapPoint>> mvpMatches12;
    std::vector<size_t> mvnIndices1;
    std::vector<size_t> mvSigmaSquare1;
    std::vector<size_t> mvSigmaSquare2;
    std::vector<size_t> mvnMaxError1;
    std::vector<size_t> mvnMaxError2;

    int N;
    int mN1;
    int map_points_1_size;

    // Current Estimation
    Eigen::Matrix3f mR12i;
    Eigen::Vector3f mt12i;
    float ms12i;
    Eigen::Isometry3f mT12i;
    Eigen::Isometry3f mT21i;
    std::vector<bool> mvbInliersi;
    int mnInliersi;

    // Current Ransac State
    int mnIterations;
    std::vector<bool> mvbBestInliers;
    int mnBestInliers;
    Eigen::Isometry3f mBestT12;
    Eigen::Matrix3f mBestRotation;
    Eigen::Vector3f mBestTranslation;

    // Indices for random selection
    std::vector<size_t> mvAllIndices;

    // Projections
    std::vector<Eigen::Vector2f> mvP1im1;
    std::vector<Eigen::Vector2f> mvP2im2;

    // RANSAC probability
    double mRansacProb;

    // RANSAC min inliers
    int mRansacMinInliers;

    // RANSAC max iterations
    int mRansacMaxIts;

    // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*mSigma2
    float mTh;
    float mSigma2;

    // Calibration
    Eigen::Matrix3f mK1;
    Eigen::Matrix3f mK2;

};

} //namespace ORB_SLAM
