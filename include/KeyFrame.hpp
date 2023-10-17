#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include "MapPoint.hpp"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.hpp"
#include "ORBextractor.hpp"
#include "Frame.hpp"
#include "KeyFrameDatabase.hpp"
#include "Converter.hpp"
#include "ORBmatcher.hpp"

#include <mutex>


namespace ORB_SLAM_CUSTOM
{

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

class KeyFrame : public std::enable_shared_from_this<KeyFrame>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KeyFrame(Frame &F, std::shared_ptr<Map> pMap, std::shared_ptr<KeyFrameDatabase> pKFDB);

    // Pose functions
    void SetPose(const Eigen::Isometry3f &Tcw);
    Eigen::Isometry3f GetPose();
    Eigen::Isometry3f GetPoseInverse();
    Eigen::Vector3f GetCameraCenter();
    Eigen::Vector4f GetStereoCenter();
    Eigen::Matrix3f GetRotation();
    Eigen::Vector3f GetTranslation();

    // Bag of Words Representation
    void ComputeBoW();

    // Covisibility graph functions
    void AddConnection(std::shared_ptr<KeyFrame> pKF, const int &weight);
    void EraseConnection(std::shared_ptr<KeyFrame> pKF);
    void UpdateConnections();
    void UpdateBestCovisibles();
    std::set<std::shared_ptr<KeyFrame>> GetConnectedKeyFrames();
    std::vector<std::shared_ptr<KeyFrame> > GetVectorCovisibleKeyFrames();
    std::vector<std::shared_ptr<KeyFrame>> GetBestCovisibilityKeyFrames(const int &N);
    std::vector<std::shared_ptr<KeyFrame>> GetCovisiblesByWeight(const int &w);
    int GetWeight(std::shared_ptr<KeyFrame> pKF);

    // Spanning tree functions
    void AddChild(std::shared_ptr<KeyFrame> pKF);
    void EraseChild(std::shared_ptr<KeyFrame> pKF);
    void ChangeParent(std::shared_ptr<KeyFrame> pKF);
    std::set<std::shared_ptr<KeyFrame>> GetChilds();
    std::shared_ptr<KeyFrame> GetParent();
    bool hasChild(std::shared_ptr<KeyFrame> pKF);

    // Loop Edges
    void AddLoopEdge(std::shared_ptr<KeyFrame> pKF);
    std::set<std::shared_ptr<KeyFrame>> GetLoopEdges();

    // MapPoint observation functions
    void AddMapPoint(std::shared_ptr<MapPoint> pMP, const size_t &idx);
    void EraseMapPointMatch(const size_t &idx);
    void EraseMapPointMatch(std::shared_ptr<MapPoint> pMP);
    void ReplaceMapPointMatch(const size_t &idx, std::shared_ptr<MapPoint> pMP);
    std::set<std::shared_ptr<MapPoint>> GetMapPoints();
    std::vector<std::shared_ptr<MapPoint>> GetMapPointMatches();
    int TrackedMapPoints(const int &minObs);
    std::shared_ptr<MapPoint> GetMapPoint(const size_t &idx);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    Eigen::Vector3f UnprojectStereo(int i);

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    static bool weightComp( int a, int b){
        return a>b;
    }

    static bool lId(std::shared_ptr<KeyFrame> pKF1, std::shared_ptr<KeyFrame> pKF2){
        return pKF1->mnId<pKF2->mnId;
    }


    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

    static long unsigned int nNextId;
    long unsigned int mnId;
    const long unsigned int mnFrameId;

    const double mTimeStamp;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;

    // Variables used by loop closing
    Eigen::Isometry3f mTcwGBA;
    Eigen::Isometry3f mTcwBefGBA;
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth; // negative value for monocular points
    const cv::Mat mDescriptors;

    //BoW
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // Pose relative to parent (this is computed when bad flag is activated)
    Eigen::Isometry3f mTcp;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    static Eigen::Matrix3f mK;
    static Eigen::Matrix3f mKinv;


    // The following variables need to be accessed trough a mutex to be thread safe.
protected:

    // SE3 Pose and camera center
    Eigen::Isometry3f Tcw;
    Eigen::Isometry3f Twc;
    Eigen::Vector3f Ow;

    Eigen::Vector4f Cw; // Stereo middel point. Only for visualization

    // MapPoints associated to keypoints
    std::vector<std::shared_ptr<MapPoint>> mvpMapPoints;

    // BoW
    std::shared_ptr<KeyFrameDatabase> mpKeyFrameDB;
    std::shared_ptr<ORBVocabulary> mpORBvocabulary;

    // Grid over the image to speed up feature matching
    std::vector< std::vector <std::vector<size_t> > > mGrid;

    std::map<std::shared_ptr<KeyFrame>,int> mConnectedKeyFrameWeights;
    std::vector<std::shared_ptr<KeyFrame>> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    std::shared_ptr<KeyFrame> mpParent;
    std::set<std::shared_ptr<KeyFrame>> mspChildrens;
    std::set<std::shared_ptr<KeyFrame>> mspLoopEdges;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;    

    float mHalfBaseline; // Only for visualization

    std::shared_ptr<Map> mpMap;

    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM
