#include "Sim3Solver.hpp"

namespace ORB_SLAM_CUSTOM
{

Sim3Solver::Sim3Solver(std::shared_ptr<KeyFrame> pKF1, std::shared_ptr<KeyFrame> pKF2, const vector<std::shared_ptr<MapPoint>> &vpMatched12):
    mnIterations(0), mnBestInliers(0)
{
    vector<std::shared_ptr<MapPoint>> vpKeyFrameMP1 = pKF1->GetMapPointMatches();

    mN1 = vpMatched12.size();

    mvpMatches12 = vpMatched12;
    mvnIndices1.reserve(mN1);
    mvX3Dc1.reserve(mN1);
    mvX3Dc2.reserve(mN1);

    Eigen::Matrix3f Rcw1 = pKF1->GetRotation();
    Eigen::Vector3f tcw1 = pKF1->GetTranslation();
    Eigen::Matrix3f Rcw2 = pKF2->GetRotation();
    Eigen::Vector3f tcw2 = pKF2->GetTranslation();

    mvAllIndices.reserve(mN1);
    map_points_1_size = 0;
    size_t idx=0;
    for(int i1=0; i1<mN1; i1++)
    {
        if(vpMatched12[i1])
        {
            std::shared_ptr<MapPoint> pMP1 = vpKeyFrameMP1[i1];
            std::shared_ptr<MapPoint> pMP2 = vpMatched12[i1];

            if(!pMP1)
                continue;

            if(pMP1->isBad() || pMP2->isBad())
                continue;

            int indexKF1 = pMP1->GetIndexInKeyFrame(pKF1);
            int indexKF2 = pMP2->GetIndexInKeyFrame(pKF2);

            if(indexKF1<0 || indexKF2<0)
                continue;

            const cv::KeyPoint &kp1 = pKF1->mvKeysUn[indexKF1];
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[indexKF2];

            const float sigmaSquare1 = pKF1->mvLevelSigma2[kp1.octave];
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];

            mvnMaxError1.push_back(9.210*sigmaSquare1);
            mvnMaxError2.push_back(9.210*sigmaSquare2);

            map_points_1_size++;
            mvnIndices1.push_back(i1);

            Eigen::Vector3f X3D1w = pMP1->GetWorldPos();
            Eigen::Vector3f X3D1c = Rcw1*X3D1w+tcw1;
            mvX3Dc1.push_back(X3D1c);

            Eigen::Vector3f X3D2w = pMP2->GetWorldPos();
            Eigen::Vector3f X3D2c = Rcw2*X3D2w+tcw2;
            mvX3Dc2.push_back(X3D2c);

            mvAllIndices.push_back(idx);
            idx++;
        }
    }

    mT12i.setIdentity();
    mT21i.setIdentity();
    mR12i.setIdentity();
    mBestT12.setIdentity();
    mBestRotation.setIdentity();
    mt12i.setZero();
    mBestTranslation.setZero();

    mK1 = pKF1->mK;
    mK2 = pKF2->mK;

    FromCameraToImage(mvX3Dc1,mvP1im1,mK1);
    FromCameraToImage(mvX3Dc2,mvP2im2,mK2);

    SetRansacParameters();
}

void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;    

    N = map_points_1_size; // number of correspondences

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    float epsilon = (float)mRansacMinInliers/N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if(mRansacMinInliers==N)
        nIterations=1;
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(epsilon,3)));

    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));

    mnIterations = 0;
}

bool Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers = vector<bool>(mN1,false);
    nInliers=0;

    if(N<mRansacMinInliers)
    {
        bNoMore = true;
        return false;
    }

    vector<size_t> vAvailableIndices;

    Eigen::Matrix3f P3Dc1i;
    Eigen::Matrix3f P3Dc2i;

    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts && nCurrentIterations<nIterations)
    {
        nCurrentIterations++;
        mnIterations++;

        vAvailableIndices = mvAllIndices;

        // Get min set of points
        for(short i = 0; i < 3; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

            int idx = vAvailableIndices[randi];
            P3Dc1i.col(i) = mvX3Dc1[idx];
            P3Dc2i.col(i) = mvX3Dc2[idx];

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        ComputeSim3(P3Dc1i,P3Dc2i);

        CheckInliers();

        if(mnInliersi>=mnBestInliers)
        {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i;
            mBestRotation = mR12i;
            mBestTranslation = mt12i;

            if(mnInliersi>mRansacMinInliers)
            {
                nInliers = mnInliersi;
                for(int i=0; i<N; i++)
                    if(mvbInliersi[i])
                        vbInliers[mvnIndices1[i]] = true;
                return true;
            }
        }
    }

    if(mnIterations>=mRansacMaxIts)
        bNoMore=true;

    return false;
}

bool Sim3Solver::find(vector<bool> &vbInliers12, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts,bFlag,vbInliers12,nInliers);
}

void Sim3Solver::ComputeCentroid(const Eigen::Matrix3f &P, Eigen::Matrix3f &Pr, Eigen::Vector3f &C)
{
    C = P.rowwise().sum();
    C = C/3.f;
    for(int i=0; i<3; i++)
    {
        Pr.col(i)=P.col(i)-C;
    }
}

void Sim3Solver::ComputeSim3(const Eigen::Matrix3f &P1, const Eigen::Matrix3f &P2)
{
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates
    Eigen::Matrix3f Pr1 = Eigen::Matrix3f::Identity(); // Relative coordinates to centroid (set 1)
    Eigen::Matrix3f Pr2 = Eigen::Matrix3f::Identity(); // Relative coordinates to centroid (set 2)
    Eigen::Vector3f O1 = Eigen::Vector3f::Zero(); // Centroid of P1
    Eigen::Vector3f O2 = Eigen::Vector3f::Zero(); // Centroid of P2

    ComputeCentroid(P1,Pr1,O1);
    ComputeCentroid(P2,Pr2,O2);

    // Step 2: Compute M matrix

    Eigen::Matrix3f M = Pr2*Pr1.transpose();

    // Step 3: Compute N matrix

    float N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    Eigen::Matrix4f N;

    N11 = M(0,0)+M(1,1)+M(2,2);
    N12 = M(1,2)-M(2,1);
    N13 = M(2,0)-M(0,2);
    N14 = M(0,1)-M(1,0);
    N22 = M(0,0)-M(1,1)-M(2,2);
    N23 = M(0,1)+M(1,0);
    N24 = M(2,0)+M(0,2);
    N33 = -M(0,0)+M(1,1)-M(2,2);
    N34 = M(1,2)+M(2,1);
    N44 = -M(0,0)-M(1,1)+M(2,2);

    N <<N11, N12, N13, N14,
        N12, N22, N23, N24,
        N13, N23, N33, N34,
        N14, N24, N34, N44;


    // Step 4: Calculating eigen vectors
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4f> solver(4);
    solver.compute(N);
    // The optimal quaternion is the eigenvector associated to the higher eigenvalue
    Eigen::Vector4f eigvec = solver.eigenvectors().col(3); // Maximum eigenvaues is the last
    Eigen::Quaternionf q;
    q.w() = eigvec(0);
    q.x() = eigvec(1);
    q.y() = eigvec(2);
    q.z() = eigvec(3);

    mR12i = q.toRotationMatrix(); // computes the rotation matrix from the quaternion

    // Step 5: Scale -> Fixed to 1.0 for Stereo/RGBD

    // Step 6: Translation
    mt12i = O1 - mR12i*O2;

    // Step 8: Transformation

    // Step 8.1 T12

    mT12i.setIdentity();
    mT12i.linear() = mR12i;
    mT12i.translation() = mt12i;

    // Step 8.2 T21
    mT21i.setIdentity();
    mT21i = mT12i.inverse();
}


void Sim3Solver::CheckInliers()
{
    vector<Eigen::Vector2f> vP1im2, vP2im1;
    Project(mvX3Dc2,vP2im1,mT12i,mK1);
    Project(mvX3Dc1,vP1im2,mT21i,mK2);

    mnInliersi=0;

    for(size_t i=0; i<mvP1im1.size(); i++)
    {
        Eigen::Vector2f dist1 = mvP1im1[i]-vP2im1[i];
        Eigen::Vector2f dist2 = vP1im2[i]-mvP2im2[i];

        const float err1 = dist1.dot(dist1);
        const float err2 = dist2.dot(dist2);

        if(err1<mvnMaxError1[i] && err2<mvnMaxError2[i])
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
            mvbInliersi[i]=false;
    }
}


Eigen::Matrix3f Sim3Solver::GetEstimatedRotation()
{
    return mBestRotation;
}

Eigen::Vector3f Sim3Solver::GetEstimatedTranslation()
{
    return mBestTranslation;
}

void Sim3Solver::Project(const vector<Eigen::Vector3f> &vP3Dw, vector<Eigen::Vector2f> &vP2D, const Eigen::Isometry3f &Tcw, const Eigen::Matrix3f &K)
{
    Eigen::Matrix3f Rcw = Tcw.rotation();
    Eigen::Vector3f tcw = Tcw.translation();
    const float &fx = K(0,0);
    const float &fy = K(1,1);
    const float &cx = K(0,2);
    const float &cy = K(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dw.size());

    for(size_t i=0, iend=vP3Dw.size(); i<iend; i++)
    {
        Eigen::Vector3f P3Dc = Rcw*vP3Dw[i]+tcw;
        const float invz = 1/(P3Dc.z());
        const float x = P3Dc.x()*invz;
        const float y = P3Dc.y()*invz;

        vP2D.push_back(Eigen::Vector2f(fx*x+cx, fy*y+cy));
    }
}

void Sim3Solver::FromCameraToImage(const vector<Eigen::Vector3f> &vP3Dc, vector<Eigen::Vector2f> &vP2D, const Eigen::Matrix3f &K)
{
    const float &fx = K(0,0);
    const float &fy = K(1,1);
    const float &cx = K(0,2);
    const float &cy = K(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dc.size());

    for(size_t i=0, iend=vP3Dc.size(); i<iend; i++)
    {
        const float invz = 1/(vP3Dc.at(i).z());
        const float x = vP3Dc.at(i).x()*invz;
        const float y = vP3Dc.at(i).y()*invz;

        vP2D.push_back(Eigen::Vector2f(fx*x+cx, fy*y+cy));
    }
}

} //namespace ORB_SLAM
