#include "LoopClosing.hpp"

void mean_stddev_time(std::vector<long int> &time, float &mean, float &std_dev)
{
    float tot_time = 0;
    float tot_squared_time = 0;
    for (int i = 0; i < time.size(); i++)
    {
        tot_time+=time[i];
        tot_squared_time+= time[i]*time[i];
    }
    mean = tot_time/time.size();
    std_dev = sqrt(time.size()*tot_squared_time - tot_time*tot_time)/time.size();
}

namespace ORB_SLAM_CUSTOM
{

LoopClosing::LoopClosing(std::shared_ptr<Map> pMap, std::shared_ptr<KeyFrameDatabase> pDB, std::shared_ptr<ORBVocabulary> pVoc):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(nullptr), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetTracker(std::shared_ptr<Tracking> pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(std::shared_ptr<LocalMapping> pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            // Detect loop candidates and check covisibility consistency
            auto start = std::chrono::high_resolution_clock::now();
            if(DetectLoop())
            {
                auto end = std::chrono::high_resolution_clock::now();
                detection_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               start = std::chrono::high_resolution_clock::now();   
               if(ComputeSim3())
               {
                    end = std::chrono::high_resolution_clock::now();
                    sim3_detection_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
                   // Perform loop fusion and pose graph optimization
                   CorrectLoop();
               }
            }
        }       

        ResetIfRequested();

        if(CheckFinish())
            break;

        usleep(5000);
    }

    SetFinish();
}

void LoopClosing::InsertKeyFrame(std::shared_ptr<KeyFrame> pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

bool LoopClosing::DetectLoop()
{
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if(mpCurrentKF->mnId<mLastLoopKFid+10)
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    const vector<std::shared_ptr<KeyFrame>> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        std::shared_ptr<KeyFrame> pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

        if(score<minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    vector<std::shared_ptr<KeyFrame>> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty())
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
        std::shared_ptr<KeyFrame> pCandidateKF = vpCandidateKFs[i];

        set<std::shared_ptr<KeyFrame>> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            set<std::shared_ptr<KeyFrame>> sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false;
            for(set<std::shared_ptr<KeyFrame>>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                if(sPreviousGroup.count(*sit))
                {
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

            if(bConsistent)
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                if(!vbConsistentGroup[iG])
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                }
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if(!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;


    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF);

    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        return true;
    }

    mpCurrentKF->SetErase();
    return false;
}

bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3
    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75,true);

    vector<std::unique_ptr<Sim3Solver>> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    vector<vector<std::shared_ptr<MapPoint>> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches

    for(int i=0; i<nInitialCandidates; i++)
    {
        std::shared_ptr<KeyFrame> pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase();

        if(pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }

        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);

        if(nmatches<20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            vpSim3Solvers[i] = std::make_unique<Sim3Solver>(mpCurrentKF,pKF,vvpMapPointMatches[i]);
            vpSim3Solvers[i]->SetRansacParameters(0.99,20,300);
        }

        nCandidates++;
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i])
                continue;

            std::shared_ptr<KeyFrame> pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            auto start = std::chrono::high_resolution_clock::now();
            bool sim_ok  = vpSim3Solvers[i]->iterate(5,bNoMore,vbInliers,nInliers);
            auto end = std::chrono::high_resolution_clock::now();
            sim3_computation_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if(sim_ok)
            {
                vector<std::shared_ptr<MapPoint>> vpMapPointMatches(vvpMapPointMatches[i].size(), nullptr);
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }

                Eigen::Matrix3f R = vpSim3Solvers[i]->GetEstimatedRotation();
                Eigen::Vector3f t = vpSim3Solvers[i]->GetEstimatedTranslation();
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,R,t,7.5);
                g2o::Sim3 gScm(R.cast<double>(), t.cast<double>() ,1.0);
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10);

                // If optimization is succesful stop ransacs and continue
                if(nInliers>=20)
                {
                    bMatch = true;
                    mpMatchedKF = pKF;
                    g2o::Sim3 gSmw(pKF->GetRotation().cast<double>(), pKF->GetTranslation().cast<double>(),1.0);
                    mg2oScw = gScm*gSmw;
                    mScw = Converter::toIso(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    vector<std::shared_ptr<KeyFrame>> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    for(vector<std::shared_ptr<KeyFrame>>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        std::shared_ptr<KeyFrame> pKF = *vit;
        vector<std::shared_ptr<MapPoint>> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            std::shared_ptr<MapPoint> pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    // If enough matches accept Loop
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    if(nTotalMatches>=40)
    {
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else
    {
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

}

void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    auto start = std::chrono::high_resolution_clock::now();
    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF]=mg2oScw;
    Eigen::Isometry3f Twc = mpCurrentKF->GetPoseInverse();


    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        for(vector<std::shared_ptr<KeyFrame>>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            std::shared_ptr<KeyFrame> pKFi = *vit;

            Eigen::Isometry3f Tiw = pKFi->GetPose();

            if(pKFi!=mpCurrentKF)
            {
                Eigen::Isometry3f Tic = Tiw*Twc;
                Eigen::Matrix3f Ric = Tic.rotation();
                Eigen::Vector3f tic = Tic.translation();
                g2o::Sim3 g2oSic(Ric.cast<double>(), tic.cast<double>(), 1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;
            }

            Eigen::Matrix3f Riw = Tiw.rotation();
            Eigen::Vector3f tiw = Tiw.translation();
            g2o::Sim3 g2oSiw(Riw.cast<double>(), tiw.cast<double>(), 1.0);
            //Pose without correction
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            std::shared_ptr<KeyFrame> pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            vector<std::shared_ptr<MapPoint>> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                std::shared_ptr<MapPoint> pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                Eigen::Vector3d eigP3Dw = pMPi->GetWorldPos().cast<double>();
                Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                Eigen::Vector3f cvCorrectedP3Dw = eigCorrectedP3Dw.cast<float>();
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]

            Eigen::Isometry3f correctedTiw = Converter::toIso(eigR, eigt);

            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
                std::shared_ptr<MapPoint> pLoopMP = mvpCurrentMatchedPoints[i];
                std::shared_ptr<MapPoint> pCurMP = mpCurrentKF->GetMapPoint(i);
                if(pCurMP)
                    pCurMP->Replace(pLoopMP);
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3);


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<std::shared_ptr<KeyFrame>, set<std::shared_ptr<KeyFrame>> > LoopConnections;

    for(vector<std::shared_ptr<KeyFrame>>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        std::shared_ptr<KeyFrame> pKFi = *vit;
        vector<std::shared_ptr<KeyFrame>> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        pKFi->UpdateConnections();
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
        for(vector<std::shared_ptr<KeyFrame>>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for(vector<std::shared_ptr<KeyFrame>>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    loop_fusion_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    // Optimize graph
    start = std::chrono::high_resolution_clock::now();
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections);
    end = std::chrono::high_resolution_clock::now();
    essential_graph_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    mpMap->InformNewBigChange();

    // Add loop edge
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    

    mLastLoopKFid = mpCurrentKF->mnId;   
}

void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);

    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        std::shared_ptr<KeyFrame> pKF = mit->first;

        g2o::Sim3 g2oScw = mit->second;
        Eigen::Isometry3f cvScw = Converter::toIso(g2oScw);

        vector<std::shared_ptr<MapPoint>> vpReplacePoints(mvpLoopMapPoints.size(), nullptr);
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            std::shared_ptr<MapPoint> pRep = vpReplacePoints[i];
            if(pRep)
            {
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}


void LoopClosing::RequestReset()
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
        usleep(5000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    auto start = std::chrono::high_resolution_clock::now();
    int idx =  mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    auto end = std::chrono::high_resolution_clock::now();
    global_ba_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    start = std::chrono::high_resolution_clock::now();
    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<std::shared_ptr<KeyFrame>> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                std::shared_ptr<KeyFrame> pKF = lpKFtoCheck.front();
                const set<std::shared_ptr<KeyFrame>> sChilds = pKF->GetChilds();
                Eigen::Isometry3f Twc = pKF->GetPoseInverse();
                for(set<std::shared_ptr<KeyFrame>>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    std::shared_ptr<KeyFrame> pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        Eigen::Isometry3f Tchildc = pChild->GetPose()*Twc;
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<std::shared_ptr<MapPoint>> vpMPs = mpMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                std::shared_ptr<MapPoint> pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    std::shared_ptr<KeyFrame> pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    Eigen::Matrix3f Rcw = pRefKF->mTcwBefGBA.rotation();
                    Eigen::Vector3f tcw = pRefKF->mTcwBefGBA.translation();
                    Eigen::Vector3f Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    Eigen::Isometry3f Twc = pRefKF->GetPoseInverse();
                    Eigen::Matrix3f Rwc = Twc.rotation();
                    Eigen::Vector3f twc = Twc.translation();

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }            

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
    end = std::chrono::high_resolution_clock::now();
    graph_update_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;

    float mean, std_dev;
    mean_stddev_time(detection_time, mean, std_dev);
    std::cout << "DETECTION TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(sim3_computation_time, mean, std_dev);
    std::cout << "SIM3 COMPUTATION TIME TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(sim3_detection_time, mean, std_dev);
    std::cout << "SIM3 DETECTION TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(loop_fusion_time, mean, std_dev);
    std::cout << "LOOP FUSION TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(essential_graph_time, mean, std_dev);
    std::cout << "ESSENTIAL GRAPH TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(global_ba_time, mean, std_dev);
    std::cout << "GLOBAL BA TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
    mean_stddev_time(graph_update_time, mean, std_dev);
    std::cout << "GLOBAL GRAPH UPDATE TIME = " << mean << " - STD-DEV : " << std_dev << '\n';
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
