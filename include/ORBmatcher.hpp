#pragma once

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include"MapPoint.hpp"
#include"KeyFrame.hpp"
#include"Frame.hpp"
#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>


namespace ORB_SLAM_CUSTOM
{

class LoopClosing;
class MapPoint;
class KeyFrame;
class Frame;

class ORBmatcher
{    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ORBmatcher(float nnratio=0.6, bool checkOri=true);

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    int SearchByProjection(Frame &F, const std::vector<std::shared_ptr<MapPoint>> &vpMapPoints, const float th=3);

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th);

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    int SearchByProjection(Frame &CurrentFrame, std::shared_ptr<KeyFrame> pKF, const std::set<std::shared_ptr<MapPoint>> &sAlreadyFound, const float th, const int ORBdist);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
    int SearchByProjection(std::shared_ptr<KeyFrame> pKF, const Eigen::Isometry3f &Scw, const std::vector<std::shared_ptr<MapPoint>> &vpPoints, std::vector<std::shared_ptr<MapPoint>> &vpMatched, int th);

    // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
    // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
    // Used in Relocalisation and Loop Detection
    int SearchByBoW(std::shared_ptr<KeyFrame> pKF, Frame &F, std::vector<std::shared_ptr<MapPoint>> &vpMapPointMatches);
    int SearchByBoW(std::shared_ptr<KeyFrame> pKF1, std::shared_ptr<KeyFrame> pKF2, std::vector<std::shared_ptr<MapPoint>> &vpMatches12);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    int SearchForTriangulation(std::shared_ptr<KeyFrame> pKF1, std::shared_ptr<KeyFrame> pKF2, const Eigen::Matrix3f &F12,
                               std::vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo);

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1
    int SearchBySim3(std::shared_ptr<KeyFrame> pKF1, std::shared_ptr<KeyFrame> pKF2, std::vector<std::shared_ptr<MapPoint>> &vpMatches12, const Eigen::Matrix3f &R12, const Eigen::Vector3f &t12, const float th);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    int Fuse(std::shared_ptr<KeyFrame> pKF, const vector<std::shared_ptr<MapPoint>> &vpMapPoints, const float th=3.0);

    // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
    int Fuse(std::shared_ptr<KeyFrame> pKF, const Eigen::Isometry3f &Scw, const std::vector<std::shared_ptr<MapPoint>> &vpPoints, float th, vector<std::shared_ptr<MapPoint>> &vpReplacePoint);

public:

    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;


protected:

    bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const Eigen::Matrix3f &F12, const std::shared_ptr<KeyFrame> pKF);

    float RadiusByViewingCos(const float &viewCos);

    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

    float mfNNratio;
    bool mbCheckOrientation;
};

}// namespace ORB_SLAM
