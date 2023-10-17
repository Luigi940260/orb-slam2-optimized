#include "System.hpp"

namespace ORB_SLAM_CUSTOM
{

System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer):mSensor(sensor), mpViewer(nullptr), mbReset(false),mbActivateLocalizationMode(false),
        mbDeactivateLocalizationMode(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }


    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    mpVocabulary = std::make_shared<ORBVocabulary>();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    //Create KeyFrame Database
    mpKeyFrameDatabase = std::make_shared<KeyFrameDatabase>(mpVocabulary);

    //Create the Map
    mpMap = std::make_shared<Map>();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = std::make_shared<FrameDrawer>(mpMap);
    mpMapDrawer = std::make_shared<MapDrawer>(mpMap, strSettingsFile);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = std::make_shared<Tracking>(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = std::make_shared<LocalMapping>(mpMap);
    mptLocalMapping = new thread(&ORB_SLAM_CUSTOM::LocalMapping::Run,mpLocalMapper);

    //Initialize the Loop Closing thread and launch
    mpLoopCloser = std::make_shared<LoopClosing>(mpMap, mpKeyFrameDatabase, mpVocabulary);
    mptLoopClosing = new thread(&ORB_SLAM_CUSTOM::LoopClosing::Run, mpLoopCloser);

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    {
        mpViewer = std::make_shared<Viewer>(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

Eigen::Isometry3f System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
    if(mSensor!=STEREO)
    {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
        exit(-1);
    }   

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    Eigen::Isometry3f Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp);

    list_of_poses.push_back(Tcw);
    list_of_timestamps.push_back(timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}
/*
cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
{
    if(mSensor!=RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}
*/

void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        usleep(5000);
    }

    float mean, std_dev;
    mean_stddev_time(mpTracker->orb_time, mean, std_dev);
    std::cout << "ORB TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(mpTracker->stereo_time, mean, std_dev);
    std::cout << "STEREO TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(mpTracker->pose_prediction_time, mean, std_dev);
    std::cout << "POSE PREDICTION TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(mpTracker->relocalization_time, mean, std_dev);
    std::cout << "RELOCALIZATION TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(mpTracker->local_map_time, mean, std_dev);
    std::cout << "LOCAL MAP TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(mpTracker->new_kf_decision_time, mean, std_dev);
    std::cout << "NEW KF DECISION TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(mpTracker->new_kf_creation_time, mean, std_dev);
    std::cout << "NEW KF CREATION TIME = " << mean << " - STD-DEV : " << std_dev << '\n';

    if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    auto it_time = list_of_timestamps.begin();
    auto it_pose = list_of_poses.begin();

    for (auto it = it_pose, it_end = list_of_poses.end(); it != it_end; it++, it_pose++, it_time++)
    {
        Eigen::Isometry3f Tcw = *it;
        Eigen::Isometry3f Twc = Tcw.inverse(); 
        Eigen::Matrix3f Rwc = Twc.rotation();
        Eigen::Vector3f twc = Twc.translation();

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *it_time << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    } 
    f.close();

    cout << endl << "online trajectory saved!" << endl;
}

void System::SaveOfflineTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    vector<std::shared_ptr<KeyFrame>> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Eigen::Isometry3f Two = vpKFs.at(0)->GetPoseInverse();

    std::cout << "UNTIL HERE\n";

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).

    list<std::shared_ptr<KeyFrame>>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<Eigen::Isometry3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        std::shared_ptr<KeyFrame> pKF = *lRit;

        if (pKF == nullptr)
            continue;

        Eigen::Isometry3f Trw = Eigen::Isometry3f::Identity();

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        Eigen::Isometry3f Tcw = (*lit)*Trw;
        Eigen::Isometry3f Twc = Tcw.inverse(); 
        Eigen::Matrix3f Rwc = Twc.rotation();
        Eigen::Vector3f twc = Twc.translation();

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    
    f.close();
    cout << endl << "trajectory Offline saved!" << endl;
}


void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<std::shared_ptr<KeyFrame>> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        std::shared_ptr<KeyFrame> pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        Eigen::Matrix3f R = pKF->GetRotation().transpose();
        vector<float> q = Converter::toQuaternion(R);
        Eigen::Vector3f t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t(0) << " " << t(1) << " " << t(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<std::shared_ptr<KeyFrame>> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Eigen::Isometry3f Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<std::shared_ptr<KeyFrame>>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<Eigen::Isometry3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        std::shared_ptr<KeyFrame> pKF = *lRit;

        Eigen::Isometry3f Trw = Eigen::Isometry3f::Identity();

        while(pKF->isBad())
        {
          //  cout << "bad parent" << endl;
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        Eigen::Isometry3f Tcw = (*lit)*Trw;
        Eigen::Isometry3f Twc = Tcw.inverse();
        Eigen::Matrix3f Rwc = Twc.rotation();
        Eigen::Vector3f twc = Twc.translation();

        f << setprecision(9) << Rwc(0,0) << " " << Rwc(0,1)  << " " << Rwc(0,2) << " "  << twc(0) << " " <<
             Rwc(1,0) << " " << Rwc(1,1)  << " " << Rwc(1,2) << " "  << twc(1) << " " <<
             Rwc(2,0) << " " << Rwc(2,1)  << " " << Rwc(2,2) << " "  << twc(2) << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}

int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<std::shared_ptr<MapPoint>> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

} //namespace ORB_SLAM
