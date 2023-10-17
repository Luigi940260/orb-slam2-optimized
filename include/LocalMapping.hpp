#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <opencv2/core/eigen.hpp>

#include "KeyFrame.hpp"
#include "Map.hpp"
#include "LoopClosing.hpp"
#include "Tracking.hpp"
#include "KeyFrameDatabase.hpp"
#include "ORBmatcher.hpp"
#include "Optimizer.hpp"

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include <mutex>


namespace ORB_SLAM_CUSTOM
{

class Tracking;
class LoopClosing;
class Map;

class LocalMapping
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LocalMapping(std::shared_ptr<Map> pMap);

    void SetLoopCloser(std::shared_ptr<LoopClosing> pLoopCloser);

    void SetTracker(std::shared_ptr<Tracking> pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(std::shared_ptr<KeyFrame> pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    Eigen::Matrix3f ComputeF12(std::shared_ptr<KeyFrame> &pKF1, std::shared_ptr<KeyFrame> &pKF2);

    Eigen::Matrix3f SkewSymmetricMatrix(const Eigen::Vector3f &v);

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    std::shared_ptr<Map> mpMap;

    std::shared_ptr<LoopClosing> mpLoopCloser;
    std::shared_ptr<Tracking> mpTracker;

    std::list<std::shared_ptr<KeyFrame>> mlNewKeyFrames;

    std::shared_ptr<KeyFrame> mpCurrentKeyFrame;

    std::list<std::shared_ptr<MapPoint>> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;

    std::vector<long int> keyframe_insertion_time;
    std::vector<long int> map_point_culling_time;
    std::vector<long int> map_point_creation_time;
    std::vector<long int> map_point_fusion_time;
    std::vector<long int> local_ba_time;
    std::vector<long int> keyframe_culling_time;
};

} //namespace ORB_SLAM
