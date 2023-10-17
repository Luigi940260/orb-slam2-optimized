#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include "FrameDrawer.hpp"
#include "MapDrawer.hpp"
#include "Tracking.hpp"
#include "System.hpp"

#include <mutex>
#include <pangolin/pangolin.h>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

namespace ORB_SLAM_CUSTOM
{

class Tracking;
class FrameDrawer;
class MapDrawer;
class System;

class Viewer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Viewer(System* pSystem, std::shared_ptr<FrameDrawer> pFrameDrawer, std::shared_ptr<MapDrawer> pMapDrawer, 
            std::shared_ptr<Tracking> pTracking, const string &strSettingPath);

    // Main thread function. Draw points, keyframes, the current camera pose and the last processed
    // frame. Drawing is refreshed according to the camera fps. We use Pangolin.
    void Run();

    void RequestFinish();

    void RequestStop();

    bool isFinished();

    bool isStopped();

    void Release();

private:

    bool Stop();

    System* mpSystem;
    std::shared_ptr<FrameDrawer> mpFrameDrawer;
    std::shared_ptr<MapDrawer> mpMapDrawer;
    std::shared_ptr<Tracking> mpTracker;

    // 1/fps in ms
    double mT;
    float mImageWidth, mImageHeight;

    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    bool mbStopped;
    bool mbStopRequested;
    std::mutex mMutexStop;

};

}
	

