#pragma once

#include "Tracking.hpp"
#include "MapPoint.hpp"
#include "Map.hpp"

#include <opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include<mutex>


namespace ORB_SLAM_CUSTOM
{

class Tracking;
class Viewer;

class FrameDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FrameDrawer(std::shared_ptr<Map> pMap);

    // Update info from the last processed frame.
    void Update(Tracking* pTracker);

    // Draw last processed frame.
    cv::Mat DrawFrame();

protected:

    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

    // Info of the frame to be drawn
    cv::Mat mIm;
    int N;
    vector<cv::KeyPoint> mvCurrentKeys;
    vector<bool> mvbMap, mvbVO;
    bool mbOnlyTracking;
    int mnTracked, mnTrackedVO;
    vector<cv::KeyPoint> mvIniKeys;
    vector<int> mvIniMatches;
    int mState;

    std::shared_ptr<Map> mpMap;

    std::mutex mMutex;
};

} //namespace ORB_SLAM
