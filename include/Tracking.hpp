#pragma once

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include"Viewer.hpp"
#include"FrameDrawer.hpp"
#include"Map.hpp"
#include"LocalMapping.hpp"
#include"LoopClosing.hpp"
#include"Frame.hpp"
#include "ORBVocabulary.hpp"
#include"KeyFrameDatabase.hpp"
#include"ORBextractor.hpp"
//#include "Initializer.h"
#include "MapDrawer.hpp"
#include "System.hpp"
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include"ORBmatcher.hpp"
#include"Converter.hpp"
#include"Optimizer.hpp"
//#include"MLPnPsolver.hpp"
#include"PnPsolver.hpp"

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include<iostream>

#include<mutex>

#include <mutex>

namespace ORB_SLAM_CUSTOM
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking
{  

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Tracking(System* pSys, std::shared_ptr<ORBVocabulary> pVoc, std::shared_ptr<FrameDrawer> pFrameDrawer, 
                std::shared_ptr<MapDrawer> pMapDrawer, std::shared_ptr<Map> pMap,
             std::shared_ptr<KeyFrameDatabase> pKFDB, const string &strSettingPath, const int sensor);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    Eigen::Isometry3f GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);
    //cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
    //cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

    void SetLocalMapper(std::shared_ptr<LocalMapping> pLocalMapper);
    void SetLoopClosing(std::shared_ptr<LoopClosing> pLoopClosing);
    void SetViewer(std::shared_ptr<Viewer> pViewer);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);


public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<Eigen::Isometry3f> mlRelativeFramePoses;
    list<std::shared_ptr<KeyFrame>> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset();

    std::vector<long int> orb_time;
    std::vector<long int> stereo_time;
    std::vector<long int> pose_prediction_time;
    std::vector<long int> local_map_time;
    std::vector<long int> relocalization_time;
    std::vector<long int> new_kf_decision_time;
    std::vector<long int> new_kf_creation_time;

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    //void MonocularInitialization();
    //void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    std::shared_ptr<LocalMapping> mpLocalMapper;
    std::shared_ptr<LoopClosing> mpLoopClosing;

    //ORB
    std::shared_ptr<ORBextractor> mpORBextractorLeft; 
    std::shared_ptr<ORBextractor> mpORBextractorRight;
    //cv::Ptr<cv::Feature2D> mpORBextractorLeft;
    //cv::Ptr<cv::Feature2D> mpORBextractorRight;
    //std::shared_ptr<ORBextractor* mpIniORBextractor;

    //BoW
    std::shared_ptr<ORBVocabulary> mpORBVocabulary;
    std::shared_ptr<KeyFrameDatabase> mpKeyFrameDB;

    // Initalization (only for monocular)
    //Initializer* mpInitializer;

    //Local Map
    std::shared_ptr<KeyFrame> mpReferenceKF;
    std::vector<std::shared_ptr<KeyFrame>> mvpLocalKeyFrames;
    std::vector<std::shared_ptr<MapPoint>> mvpLocalMapPoints;
    
    // System
    System* mpSystem;
    
    //Drawers
    std::shared_ptr<Viewer> mpViewer;
    std::shared_ptr<FrameDrawer> mpFrameDrawer;
    std::shared_ptr<MapDrawer> mpMapDrawer;

    //Map
    std::shared_ptr<Map> mpMap;

    //Calibration matrix
    Eigen::Matrix3f mK;
    cv::Mat mDistCoef;
    float mbf;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    std::shared_ptr<KeyFrame> mpLastKeyFrame;
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Motion Model
    Eigen::Isometry3f mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<std::shared_ptr<MapPoint>> mlpTemporalPoints;
};

} //namespace ORB_SLAM
