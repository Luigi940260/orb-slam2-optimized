

#include "KeyFrame.hpp"

namespace ORB_SLAM_CUSTOM
{

long unsigned int KeyFrame::nNextId=0;
Eigen::Matrix3f KeyFrame::mK, KeyFrame::mKinv;

KeyFrame::KeyFrame(Frame &F, std::shared_ptr<Map> pMap, std::shared_ptr<KeyFrameDatabase> pKFDB):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(nullptr), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap)
{
    mnId=nNextId++;

    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

    mTcwGBA.setIdentity();
    mTcwBefGBA.setIdentity();
    Tcw.setIdentity();
    Twc.setIdentity();
    Ow.setZero();
    Cw.setZero();

    SetPose(F.mTcw);    
}

void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::SetPose(const Eigen::Isometry3f &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tcw = Tcw_;
    //Eigen::Matrix3f Rcw = Tcw.rotation();
    //Eigen::Vector3f tcw = Tcw.translation();
    //std::cout << Tcw.matrix() << '\n';
    Twc = Tcw.inverse();
    //Eigen::Matrix3f Rwc = Twc.rotation();
    Ow = Twc.translation();
    //std::cout << Twc.matrix() << '\n';

    Eigen::Vector4f center(mHalfBaseline, 0 , 0, 1);
    Cw = Twc*center;
}

Eigen::Isometry3f KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw;
}

Eigen::Isometry3f KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc;
}

Eigen::Vector3f KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow;
}

Eigen::Vector4f KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw;
}


Eigen::Matrix3f KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rotation();
}

Eigen::Vector3f KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.translation();
}

void KeyFrame::AddConnection(std::shared_ptr<KeyFrame>pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,std::shared_ptr<KeyFrame>> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<std::shared_ptr<KeyFrame>,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

    sort(vPairs.begin(),vPairs.end());
    list<std::shared_ptr<KeyFrame>> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    mvpOrderedConnectedKeyFrames = vector<std::shared_ptr<KeyFrame>>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
}

set<std::shared_ptr<KeyFrame>> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<std::shared_ptr<KeyFrame>> s;
    for(map<std::shared_ptr<KeyFrame>,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

vector<std::shared_ptr<KeyFrame>> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

vector<std::shared_ptr<KeyFrame>> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<std::shared_ptr<KeyFrame>>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

vector<std::shared_ptr<KeyFrame>> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<std::shared_ptr<KeyFrame>>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    if(it==mvOrderedWeights.end())
        return vector<std::shared_ptr<KeyFrame>>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<std::shared_ptr<KeyFrame>>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int KeyFrame::GetWeight(std::shared_ptr<KeyFrame>pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

void KeyFrame::AddMapPoint(std::shared_ptr<MapPoint> pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=nullptr;
}

void KeyFrame::EraseMapPointMatch(std::shared_ptr<MapPoint> pMP)
{
    std::shared_ptr<KeyFrame> this_ptr = shared_from_this();
    int idx = pMP->GetIndexInKeyFrame(this_ptr);
    if(idx>=0)
        mvpMapPoints.at(idx)=nullptr;
}


void KeyFrame::ReplaceMapPointMatch(const size_t &idx, std::shared_ptr<MapPoint> pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<std::shared_ptr<MapPoint>> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<std::shared_ptr<MapPoint>> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        std::shared_ptr<MapPoint> pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        std::shared_ptr<MapPoint> pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

vector<std::shared_ptr<MapPoint>> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

std::shared_ptr<MapPoint> KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

void KeyFrame::UpdateConnections()
{
    std::shared_ptr<KeyFrame> this_ptr = shared_from_this();
    map<std::shared_ptr<KeyFrame>,int> KFcounter;

    vector<std::shared_ptr<MapPoint>> vpMP;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for(vector<std::shared_ptr<MapPoint>>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        std::shared_ptr<MapPoint> pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        map<std::shared_ptr<KeyFrame>,size_t> observations = pMP->GetObservations();

        for(map<std::shared_ptr<KeyFrame>,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId)
                continue;
            KFcounter[mit->first]++;
        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    std::shared_ptr<KeyFrame> pKFmax=nullptr;
    int th = 15;

    vector<pair<int,std::shared_ptr<KeyFrame>> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(map<std::shared_ptr<KeyFrame>,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th)
        {
            vPairs.push_back(make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this_ptr, mit->second);
        }
    }

    if(vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this_ptr, nmax);
    }

    sort(vPairs.begin(),vPairs.end());
    list<std::shared_ptr<KeyFrame>> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<std::shared_ptr<KeyFrame>>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        if(mbFirstConnection && mnId!=0)
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this_ptr);
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(std::shared_ptr<KeyFrame>pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(std::shared_ptr<KeyFrame>pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(std::shared_ptr<KeyFrame>pKF)
{
    std::shared_ptr<KeyFrame> this_ptr = shared_from_this();
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this_ptr);
}

set<std::shared_ptr<KeyFrame>> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

std::shared_ptr<KeyFrame> KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(std::shared_ptr<KeyFrame>pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(std::shared_ptr<KeyFrame>pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<std::shared_ptr<KeyFrame>> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{   
    std::shared_ptr<KeyFrame> this_ptr = shared_from_this();
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for(map<std::shared_ptr<KeyFrame>,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this_ptr);

    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this_ptr);
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<std::shared_ptr<KeyFrame>> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            std::shared_ptr<KeyFrame> pC;
            std::shared_ptr<KeyFrame> pP;

            for(set<std::shared_ptr<KeyFrame>>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                std::shared_ptr<KeyFrame> pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<std::shared_ptr<KeyFrame>> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<std::shared_ptr<KeyFrame>>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if(!mspChildrens.empty())
            for(set<std::shared_ptr<KeyFrame>>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this_ptr);
        mTcp = Tcw*mpParent->GetPoseInverse();
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this_ptr);
    mpKeyFrameDB->erase(this_ptr);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(std::shared_ptr<KeyFrame> pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

Eigen::Vector3f KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        Eigen::Vector3f x3Dc(x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        return Twc.rotation()*x3Dc+Twc.translation();
    }
    else
        return Eigen::Vector3f::Zero();
}

} //namespace ORB_SLAM
