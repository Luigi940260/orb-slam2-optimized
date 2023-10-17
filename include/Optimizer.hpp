#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "Map.hpp"
#include "MapPoint.hpp"
#include "KeyFrame.hpp"
#include "LoopClosing.hpp"
#include "Frame.hpp"
#include "Utility.hpp"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.hpp"

#include<mutex>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

namespace ORB_SLAM_CUSTOM
{

class LoopClosing;

class Optimizer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void static BundleAdjustment(const std::vector<std::shared_ptr<KeyFrame>> &vpKF, const std::vector<std::shared_ptr<MapPoint>> &vpMP,
                                 int nIterations = 5, bool *pbStopFlag=NULL, const unsigned long nLoopKF=0,
                                 const bool bRobust = true);
    void static GlobalBundleAdjustemnt(std::shared_ptr<Map> pMap, int nIterations=5, bool *pbStopFlag=NULL,
                                       const unsigned long nLoopKF=0, const bool bRobust = true);
    void static LocalBundleAdjustment(std::shared_ptr<KeyFrame> pKF, bool *pbStopFlag, std::shared_ptr<Map> pMap);
    int static PoseOptimization(Frame* pFrame);

    // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
    void static OptimizeEssentialGraph(std::shared_ptr<Map> pMap, std::shared_ptr<KeyFrame> pLoopKF, std::shared_ptr<KeyFrame> pCurKF,
                                       const KeyFrameAndPose &NonCorrectedSim3,
                                       const KeyFrameAndPose &CorrectedSim3,
                                       const map<std::shared_ptr<KeyFrame>, set<std::shared_ptr<KeyFrame>> > &LoopConnections);

    // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)
    static int OptimizeSim3(std::shared_ptr<KeyFrame> pKF1, std::shared_ptr<KeyFrame> pKF2, std::vector<std::shared_ptr<MapPoint>> &vpMatches1,
                            g2o::Sim3 &g2oS12, const float th2);
};

} //namespace ORB_SLAM
