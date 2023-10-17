#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "KeyFrame.hpp"
#include "LocalMapping.hpp"
#include "Map.hpp"
#include "ORBVocabulary.hpp"
#include "Tracking.hpp"
#include "Sim3Solver.hpp"
#include "Converter.hpp"
#include "Optimizer.hpp"
#include "ORBmatcher.hpp"
#include "Utility.hpp"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include "KeyFrameDatabase.hpp"

#include <thread>
#include <mutex>
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

void mean_stddev_time(std::vector<long int> &time, float &mean, float &std_dev);

namespace ORB_SLAM_CUSTOM
{

class Tracking;
class LocalMapping;
class KeyFrameDatabase;


class LoopClosing
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LoopClosing(std::shared_ptr<Map> pMap, std::shared_ptr<KeyFrameDatabase> pDB, std::shared_ptr<ORBVocabulary> pVoc);

    void SetTracker(std::shared_ptr<Tracking> pTracker);

    void SetLocalMapper(std::shared_ptr<LocalMapping> pLocalMapper);

    // Main function
    void Run();

    void InsertKeyFrame(std::shared_ptr<KeyFrame> pKF);

    void RequestReset();

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(unsigned long nLoopKF);

    bool isRunningGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }   

    void RequestFinish();

    bool isFinished();

protected:

    bool CheckNewKeyFrames();

    bool DetectLoop();

    bool ComputeSim3();

    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

    void CorrectLoop();

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    std::shared_ptr<Map> mpMap;
    std::shared_ptr<Tracking> mpTracker;

    std::shared_ptr<KeyFrameDatabase> mpKeyFrameDB;
    std::shared_ptr<ORBVocabulary> mpORBVocabulary;

    std::shared_ptr<LocalMapping> mpLocalMapper;

    std::list<std::shared_ptr<KeyFrame>> mlpLoopKeyFrameQueue;

    std::mutex mMutexLoopQueue;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;

    // Loop detector variables
    std::shared_ptr<KeyFrame> mpCurrentKF;
    std::shared_ptr<KeyFrame> mpMatchedKF;
    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<std::shared_ptr<KeyFrame>> mvpEnoughConsistentCandidates;
    std::vector<std::shared_ptr<KeyFrame>> mvpCurrentConnectedKFs;
    std::vector<std::shared_ptr<MapPoint>> mvpCurrentMatchedPoints;
    std::vector<std::shared_ptr<MapPoint>> mvpLoopMapPoints;
    Eigen::Isometry3f mScw;
    g2o::Sim3 mg2oScw;

    long unsigned int mLastLoopKFid;

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;
    bool mbFinishedGBA;
    bool mbStopGBA;
    std::mutex mMutexGBA;
    std::thread* mpThreadGBA;

    std::vector<long int> detection_time;
    std::vector<long int> loop_fusion_time;
    std::vector<long int> sim3_detection_time;
    std::vector<long int> sim3_computation_time;
    std::vector<long int> essential_graph_time;
    std::vector<long int> global_ba_time;
    std::vector<long int> graph_update_time;

    unsigned int mnFullBAIdx;
};

} //namespace ORB_SLAM
