#pragma once

#include"KeyFrame.hpp"
#include"Frame.hpp"
#include"Map.hpp"
#include "ORBmatcher.hpp"

#include<opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include<mutex>

namespace ORB_SLAM_CUSTOM
{

class KeyFrame;
class Map;
class Frame;


class MapPoint : public std::enable_shared_from_this<MapPoint>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapPoint(const Eigen::Vector3f &Pos, std::shared_ptr<KeyFrame> pRefKF, std::shared_ptr<Map> pMap);
    MapPoint(const Eigen::Vector3f &Pos,  std::shared_ptr<Map> pMap, Frame* pFrame, const int &idxF);

    void SetWorldPos(const Eigen::Vector3f &Pos);
    Eigen::Vector3f GetWorldPos();

    Eigen::Vector3f GetNormal();
    std::shared_ptr<KeyFrame> GetReferenceKeyFrame();

    std::map<std::shared_ptr<KeyFrame>,size_t> GetObservations();
    int Observations();

    void AddObservation(std::shared_ptr<KeyFrame> pKF,size_t idx);
    void EraseObservation(std::shared_ptr<KeyFrame> pKF);

    int GetIndexInKeyFrame(std::shared_ptr<KeyFrame> pKF);
    bool IsInKeyFrame(std::shared_ptr<KeyFrame> pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(std::shared_ptr<MapPoint> pMP);    
    std::shared_ptr<MapPoint> GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, std::shared_ptr<KeyFrame>pKF);
    int PredictScale(const float &currentDist, Frame* pF);

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;

    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;    
    Eigen::Vector3f mPosGBA;
    long unsigned int mnBAGlobalForKF;


    static std::mutex mGlobalMutex;

protected:    

     // Position in absolute coordinates
    Eigen::Vector3f mWorldPos;

    // Keyframes observing the point and associated index in keyframe
    std::map<std::shared_ptr<KeyFrame>,size_t> mObservations;

    // Mean viewing direction
    Eigen::Vector3f mNormalVector;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    // Reference KeyFrame
    std::shared_ptr<KeyFrame> mpRefKF;

    // Tracking counters
    int mnVisible;
    int mnFound;

    // Bad flag (we do not currently erase MapPoint from memory)
    bool mbBad;
    std::shared_ptr<MapPoint> mpReplaced;

    // Scale invariance distances
    float mfMinDistance;
    float mfMaxDistance;

    std::shared_ptr<Map> mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM
