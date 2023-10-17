#include "LocalMapping.hpp"

namespace ORB_SLAM_CUSTOM
{

LocalMapping::LocalMapping(std::shared_ptr<Map> pMap):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(std::shared_ptr<LoopClosing> pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(std::shared_ptr<Tracking> pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            // BoW conversion and insertion in Map
            auto start = std::chrono::high_resolution_clock::now();
            ProcessNewKeyFrame();
            auto end = std::chrono::high_resolution_clock::now();
            keyframe_insertion_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

            // Check recent MapPoints
            start = std::chrono::high_resolution_clock::now();
            MapPointCulling();
            end = std::chrono::high_resolution_clock::now();
            map_point_culling_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

            // Triangulate new MapPoints
            start = std::chrono::high_resolution_clock::now();
            CreateNewMapPoints();
            end = std::chrono::high_resolution_clock::now();
            map_point_creation_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                start = std::chrono::high_resolution_clock::now();   
                SearchInNeighbors();
                end = std::chrono::high_resolution_clock::now();
                map_point_fusion_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
            }

            mbAbortBA = false;

            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA
                if(mpMap->KeyFramesInMap()>2)
                {
                    start = std::chrono::high_resolution_clock::now();                        
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);
                    end = std::chrono::high_resolution_clock::now();
                    local_ba_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
                }

                // Check redundant local Keyframes
                start = std::chrono::high_resolution_clock::now();                        
                KeyFrameCulling();
                end = std::chrono::high_resolution_clock::now();
                keyframe_culling_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
            }

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(std::shared_ptr<KeyFrame> pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<std::shared_ptr<MapPoint>> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        std::shared_ptr<MapPoint> pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<std::shared_ptr<MapPoint>>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    const int cnThObs = 3;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        std::shared_ptr<MapPoint> pMP = *lit;
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    const vector<std::shared_ptr<KeyFrame>> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(10);

    ORBmatcher matcher(0.6,false);

    Eigen::Matrix3f Rcw1 = mpCurrentKeyFrame->GetRotation();
    Eigen::Matrix3f Rwc1 = Rcw1.transpose();
    Eigen::Vector3f tcw1 = mpCurrentKeyFrame->GetTranslation();
    Eigen::Matrix<float,3,4> Tcw1;
    Tcw1.leftCols<3>() = Rcw1;
    Tcw1.rightCols<1>() = tcw1;
    Eigen::Vector3f Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        std::shared_ptr<KeyFrame> pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
        const float baseline = (Ow2-Ow1).norm();

        if(baseline<pKF2->mb)
            continue;

        // Compute Fundamental Matrix
        Eigen::Matrix3f F12 = ComputeF12(mpCurrentKeyFrame,pKF2);
        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        Eigen::Matrix3f Rcw2 = pKF2->GetRotation();
        Eigen::Matrix3f Rwc2 = Rcw2.transpose();
        Eigen::Vector3f tcw2 = pKF2->GetTranslation();
        Eigen::Matrix<float,3,4> Tcw2;
        Tcw2.leftCols<3>() = Rcw2;
        Tcw2.rightCols<1>() = tcw2;

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            Eigen::Vector3f xn1((kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            Eigen::Vector3f xn2((kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            Eigen::Vector3f ray1 = Rwc1*xn1;
            Eigen::Vector3f ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(ray1.norm()*ray2.norm());

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            Eigen::Vector3f x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                Eigen::Matrix4f A;
                
                A.row(0) = xn1(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2(1)*Tcw2.row(2)-Tcw2.row(1);

                Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);

                const Eigen::Matrix4f &V = svd.matrixV();

                Eigen::Vector4f temp = V.rightCols<1>();

                if (temp(3) == 0)
                    continue;
                
                x3D = Eigen::Vector3f(temp(0)/temp(3), temp(1)/temp(3), temp(2)/temp(3));

                //std::cout << "POSITION EIGEN\n" << x3D << '\n';

                //cv::cv2eigen(x3D_temp, x3D);
            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3D)+tcw1.z();
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3D)+tcw2.z();
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3D)+tcw1.x();
            const float y1 = Rcw1.row(1).dot(x3D)+tcw1.y();
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3D)+tcw2.x();
            const float y2 = Rcw2.row(1).dot(x3D)+tcw2.y();
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            Eigen::Vector3f normal1 = x3D-Ow1;
            float dist1 = normal1.norm();

            Eigen::Vector3f normal2 = x3D-Ow2;
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            std::shared_ptr<MapPoint> pMP = std::make_shared<MapPoint>(x3D,mpCurrentKeyFrame,mpMap);

            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }

    }

    //std::cout << "NEW POINTS = " << nnew << '\n';
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    const vector<std::shared_ptr<KeyFrame>> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(10);
    vector<std::shared_ptr<KeyFrame>> vpTargetKFs;
    for(vector<std::shared_ptr<KeyFrame>>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        std::shared_ptr<KeyFrame> pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<std::shared_ptr<KeyFrame>> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<std::shared_ptr<KeyFrame>>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            std::shared_ptr<KeyFrame> pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<std::shared_ptr<MapPoint>> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<std::shared_ptr<KeyFrame>>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        std::shared_ptr<KeyFrame> pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<std::shared_ptr<MapPoint>> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<std::shared_ptr<KeyFrame>>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        std::shared_ptr<KeyFrame> pKFi = *vitKF;

        vector<std::shared_ptr<MapPoint>> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<std::shared_ptr<MapPoint>>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            std::shared_ptr<MapPoint> pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        std::shared_ptr<MapPoint> pMP=vpMapPointMatches[i];
        if(pMP)
        {\
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

Eigen::Matrix3f LocalMapping::ComputeF12(std::shared_ptr<KeyFrame> &pKF1, std::shared_ptr<KeyFrame> &pKF2)
{
    //Eigen::Matrix3f R1w = pKF1->GetRotation();
    //Eigen::Vector3f t1w = pKF1->GetTranslation();
    //Eigen::Matrix3f R2w = pKF2->GetRotation();
    //Eigen::Vector3f t2w = pKF2->GetTranslation();
    Eigen::Isometry3f T1w = pKF1->GetPose();
    Eigen::Isometry3f T2wi = pKF2->GetPoseInverse();

    Eigen::Isometry3f T12 = T1w*T2wi;
    Eigen::Matrix3f R12 = T12.rotation();
    Eigen::Vector3f t12 = T12.translation();

    Eigen::Matrix3f t12x = SkewSymmetricMatrix(t12);

    const Eigen::Matrix3f &K1inv = pKF1->mKinv;
    const Eigen::Matrix3f &K2inv = pKF2->mKinv;


    return K1inv.transpose()*t12x*R12*K2inv;
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<std::shared_ptr<KeyFrame>> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<std::shared_ptr<KeyFrame>>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        std::shared_ptr<KeyFrame> pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<std::shared_ptr<MapPoint>> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            std::shared_ptr<MapPoint> pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                        continue;

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<std::shared_ptr<KeyFrame>, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<std::shared_ptr<KeyFrame>, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            std::shared_ptr<KeyFrame> pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

Eigen::Matrix3f LocalMapping::SkewSymmetricMatrix(const Eigen::Vector3f &v)
{
    Eigen::Matrix3f R;
    R <<0, -v(2), v(1),
        v(2), 0, -v(0),
        -v(1), v(0), 0;
    return R;
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;

    float mean, std_dev;
    mean_stddev_time(keyframe_insertion_time, mean, std_dev);
    std::cout << "KEYFRAME INSERTION TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(map_point_culling_time, mean, std_dev);
    std::cout << "MAP POINT CULLING TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(map_point_creation_time, mean, std_dev);
    std::cout << "MAP POINT CREATION TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(map_point_fusion_time, mean, std_dev);
    std::cout << "MAP POINTS FUSION TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(local_ba_time, mean, std_dev);
    std::cout << "LOCAL BA TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(keyframe_culling_time, mean, std_dev);
    std::cout << "KEYFRAME CULLING TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
