#include <iostream>

#include "PnPsolver.hpp"

using namespace std;

namespace ORB_SLAM_CUSTOM
{


PnPsolver::PnPsolver(const Frame &F, const vector<std::shared_ptr<MapPoint>> &vpMapPointMatches):
		maximum_number_of_correspondences(0), number_of_correspondences(0), mnInliersi(0),
		mnIterations(0), mnBestInliers(0), N(0)
{
	N_points = vpMapPointMatches.size();
	mvP2D.reserve(F.mvpMapPoints.size());
	mvSigma2.reserve(F.mvpMapPoints.size());
	mvP3Dw.reserve(F.mvpMapPoints.size());
	mvKeyPointIndices.reserve(F.mvpMapPoints.size());
	mvAllIndices.reserve(F.mvpMapPoints.size());

	int idx=0;
	for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
	{
		std::shared_ptr<MapPoint> pMP = vpMapPointMatches[i];

		if(pMP)
		{
			if(!pMP->isBad())
			{
				const cv::KeyPoint &kp = F.mvKeysUn[i];

				mvP2D.push_back(Eigen::Vector2f(kp.pt.x, kp.pt.y));
				mvSigma2.push_back(F.mvLevelSigma2[kp.octave]);

				mvP3Dw.push_back(pMP->GetWorldPos());

				mvKeyPointIndices.push_back(i);
				mvAllIndices.push_back(idx);               

				idx++;
			}
		}
	}

	// Set camera calibration parameters
	fx = F.fx;
	fy = F.fy;
	cx = F.cx;
	cy = F.cy;

	mRefinedTcw.setIdentity();
	mBestTcw.setIdentity();
	mTcwi.setIdentity();
}


void PnPsolver::SetRansacParameters(double probability, int minInliers, int maxIterations, int minSet, float epsilon, float th2)
{
	mRansacProb = probability;
	mRansacMinInliers = minInliers;
	mRansacMaxIts = maxIterations;
	mRansacEpsilon = epsilon;
	mRansacMinSet = minSet;

	N = mvP2D.size(); // number of correspondences

	mvbInliersi.resize(N);

	// Adjust Parameters according to number of correspondences
	int nMinInliers = N*mRansacEpsilon;
	if(nMinInliers<mRansacMinInliers)
		nMinInliers=mRansacMinInliers;
	if(nMinInliers<minSet)
		nMinInliers=minSet;
	mRansacMinInliers = nMinInliers;

	if(mRansacEpsilon<(float)mRansacMinInliers/N)
		mRansacEpsilon=(float)mRansacMinInliers/N;

	// Set RANSAC iterations according to probability, epsilon, and max iterations
	int nIterations;

	if(mRansacMinInliers==N)
		nIterations=1;
	else
		nIterations = ceil(log(1-mRansacProb)/log(1-pow(mRansacEpsilon,3)));

	mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));

	mvMaxError.resize(mvSigma2.size());
	for(size_t i=0; i<mvSigma2.size(); i++)
		mvMaxError[i] = mvSigma2[i]*th2;
}

bool PnPsolver::find(vector<bool> &vbInliers, int &nInliers, Eigen::Matrix4f& T)
{
	bool bFlag;
	return iterate(mRansacMaxIts,bFlag,vbInliers,nInliers, T);    
}

bool PnPsolver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers, Eigen::Matrix4f &T)
{
	bNoMore = false;
	vbInliers.clear();
	nInliers=0;

	set_maximum_number_of_correspondences(mRansacMinSet);

	if(N<mRansacMinInliers)
	{
		bNoMore = true;
		return false;
	}

	vector<size_t> vAvailableIndices;

	int nCurrentIterations = 0;
	while(mnIterations<mRansacMaxIts || nCurrentIterations<nIterations)
	{
		nCurrentIterations++;
		mnIterations++;
		reset_correspondences();

		vAvailableIndices = mvAllIndices;

		// Get min set of points
		for(short i = 0; i < mRansacMinSet; ++i)
		{
			int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

			int idx = vAvailableIndices[randi];

			add_correspondence(mvP3Dw[idx],mvP2D[idx]);

			vAvailableIndices[randi] = vAvailableIndices.back();
			vAvailableIndices.pop_back();
		}

		// Compute camera pose
		compute_pose(mRi, mti);

		// Check inliers
		CheckInliers();

		if(mnInliersi>=mRansacMinInliers)
		{
			// If it is the best solution so far, save it
			if(mnInliersi>mnBestInliers)
			{
				mvbBestInliers = mvbInliersi;
				mnBestInliers = mnInliersi;
				mBestTcw.topLeftCorner<3,3>() = mRi;
				mBestTcw.topRightCorner<3,1>() = mti;
			}

			if(Refine())
			{
				nInliers = mnRefinedInliers;
				vbInliers = vector<bool>(N_points,false);
				for(int i=0; i<N; i++)
				{
					if(mvbRefinedInliers[i])
					vbInliers[mvKeyPointIndices[i]] = true;
				}
				T = mRefinedTcw;
				return true;
			}

		}
	}

	if(mnIterations>=mRansacMaxIts)
	{
		bNoMore=true;
		if(mnBestInliers>=mRansacMinInliers)
		{
			nInliers=mnBestInliers;
			vbInliers = vector<bool>(N_points,false);
			for(int i=0; i<N; i++)
			{
				if(mvbBestInliers[i])
					vbInliers[mvKeyPointIndices[i]] = true;
			}
			T = mBestTcw;
			return true;
		}
	}

	return false;
}

bool PnPsolver::Refine()
{
	vector<int> vIndices;
	vIndices.reserve(mvbBestInliers.size());

	for(size_t i=0; i<mvbBestInliers.size(); i++)
	{
		if(mvbBestInliers[i])
		{
			vIndices.push_back(i);
		}
	}

	set_maximum_number_of_correspondences(vIndices.size());

	reset_correspondences();

	for(size_t i=0; i<vIndices.size(); i++)
	{
		int idx = vIndices[i];
		add_correspondence(mvP3Dw[idx],mvP2D[idx]);
	}

	// Compute camera pose
	compute_pose(mRi, mti);

	// Check inliers
	CheckInliers();

	mnRefinedInliers =mnInliersi;
	mvbRefinedInliers = mvbInliersi;

	if(mnInliersi>mRansacMinInliers)
	{
		
		cv::Mat Rcw(3,3,CV_32F);
		cv::Mat tcw(3,1,CV_32F);
		cv::eigen2cv(mRi, Rcw);
		cv::eigen2cv(mti, tcw);
		mRefinedTcw.topLeftCorner<3,3>() = mRi;
		mRefinedTcw.topRightCorner<3,1>() = mti;
		return true;
	}

	return false;
}


void PnPsolver::CheckInliers()
{
		mnInliersi=0;

		for(int i=0; i<N; i++)
		{
				Eigen::Vector3f p3Dw = mvP3Dw[i];
				Eigen::Vector2f p2Dtrue = mvP2D[i];

				Eigen::Vector3f p3Dc = mRi*p3Dw + mti;

				float invZc = 1/p3Dc.z();

				Eigen::Vector2f p2Dest(cx + fx * p3Dc.x() * invZc,cy + fy * p3Dc.y() * invZc);
				
				float error2 = (p2Dest-p2Dtrue).squaredNorm();

				if(error2<mvMaxError[i])
				{
						mvbInliersi[i]=true;
						mnInliersi++;
				}
				else
				{
						mvbInliersi[i]=false;
				}
		}
}


void PnPsolver::set_maximum_number_of_correspondences(int n)
{
	if (maximum_number_of_correspondences < n) {
		maximum_number_of_correspondences = n;

		pws.setZero(maximum_number_of_correspondences, 3);
		us.setZero(maximum_number_of_correspondences, 2);
		alphas.setZero(maximum_number_of_correspondences, 4);
		pcs.setZero(maximum_number_of_correspondences, 3);
	}
}

void PnPsolver::reset_correspondences(void)
{
	number_of_correspondences = 0;
}

void PnPsolver::add_correspondence(const Eigen::Vector3f &p3D, const Eigen::Vector2f &p2D)
{
	pws.row(number_of_correspondences) = p3D.transpose().cast<double>();
	us.row(number_of_correspondences) = p2D.transpose().cast<double>();

	number_of_correspondences++;
}

void PnPsolver::choose_control_points(void)
{
	// Take C0 as the reference points centroid:
	cws.setZero();

	cws.row(0) = pws.colwise().sum();

	cws.row(0) /= number_of_correspondences;

	// Take C1, C2, and C3 along the three principal directions of the reference points
	Eigen::Matrix<double,Eigen::Dynamic,3> PW0(number_of_correspondences,3);
	for (int i = 0; i < number_of_correspondences; i++)
		PW0.row(i) = pws.row(i) - cws.row(0);

	Eigen::Matrix3d PW0tPW0 = PW0.transpose()*PW0;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(PW0tPW0);

	const Eigen::Matrix3d &UCt = solver.eigenvectors();
	const Eigen::Vector3d &DC = solver.eigenvalues(); 

	for (int i = 0; i < 3; i++)
	{
		double k = sqrt(DC(i)/number_of_correspondences);
		cws.row(i+1) = cws.row(0)+k*UCt.col(i).transpose();
	}
}

void PnPsolver::compute_barycentric_coordinates(void)
{
	Eigen::Matrix3d CC, CC_inv;

	for(int i = 0; i < 3; i++)
    	for(int j = 1; j < 4; j++)
    		CC(i,j-1) = cws(j,i) - cws(0,i);

	CC_inv = CC.inverse();
	//CC_inv.transposeInPlace();

	for(int i = 0; i < number_of_correspondences; i++) 
	{
		//alphas(i,0) = 0.0;
		for (int j = 0; j < 3; j++)
			alphas(i,j+1) = CC_inv.row(j).dot(pws.row(i) - cws.row(0));

		alphas(i,0)=1.0-alphas(i,1)-alphas(i,2)-alphas(i,3);
		//alphas(i,0) = 1.0 - alphas.row(i).sum();
	}
}

void PnPsolver::compute_ccs(const Eigen::Vector4d &betas, const Eigen::Matrix<double,12,12> &U)
{
	ccs.setZero();

	for (int i = 0; i < 4; i++)
		for (int j =0; j < 4; j++)
			ccs.row(i).noalias() += betas(j)*U.block<3,1>(3*i,j).transpose();
}

void PnPsolver::compute_pcs()
{
	pcs.noalias() = alphas*ccs;
}

double PnPsolver::compute_pose(Eigen::Matrix3f &R, Eigen::Vector3f &t)
{
	choose_control_points();
	compute_barycentric_coordinates();

	// Create the matrix M (2n x 12)
	Eigen::Matrix<double, Eigen::Dynamic, 12> M(2*number_of_correspondences, 12);

	for (int i = 0; i < number_of_correspondences; i++) // Number of correspondeces
		for (int j = 0; j < 4; j++) // Number of control points
		{
			M(2*i,3*j)=alphas(i,j)*fx;
			M(2*i,3*j+1)=0.0;
			M(2*i,3*j+2)=alphas(i,j)*(cx-us(i,0));
			
			M(2*i+1,3*j)=0.0;
			M(2*i+1,3*j+1)=alphas(i,j)*fy;
			M(2*i+1,3*j+2)=alphas(i,j)*(cy-us(i,1));
		}

	Eigen::Matrix<double,12,12> MtM = M.transpose()*M;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,12,12>> solver(MtM);

	const Eigen::Matrix<double, 12, 12> &U  = solver.eigenvectors(); 

	Eigen::Matrix<double,6,10> L_6x10;
	Eigen::Matrix<double,6,1> rho;

	compute_L_6x10(U, L_6x10);
	compute_rho(rho);

	vector<Eigen::Vector4d> Betas(4,Eigen::Vector4d::Zero());
	double rep_errors[4];
	vector<Eigen::Matrix3d> Rs(4,Eigen::Matrix3d::Zero());
	vector<Eigen::Vector3d> ts(4,Eigen::Vector3d::Zero());

	find_betas_approx_1(L_6x10, rho, Betas[1]);
	gauss_newton(L_6x10, rho, Betas[1]);
	rep_errors[1] = compute_R_and_t(U, Betas[1], Rs[1], ts[1]);

	find_betas_approx_2(L_6x10, rho, Betas[2]);
	gauss_newton(L_6x10, rho, Betas[2]);
	rep_errors[2] = compute_R_and_t(U, Betas[2], Rs[2], ts[2]);

	find_betas_approx_3(L_6x10, rho, Betas[3]);
	gauss_newton(L_6x10, rho, Betas[3]);
	rep_errors[3] = compute_R_and_t(U, Betas[3], Rs[3], ts[3]);

	int N = 1;
	if (rep_errors[2] < rep_errors[1]) N = 2;
	if (rep_errors[3] < rep_errors[N]) N = 3;

	R = Rs[N].cast<float>();
	t = ts[N].cast<float>();
	
	return rep_errors[N];
}

double PnPsolver::reprojection_error(const Eigen::Matrix3d &R, const Eigen::Vector3d &t)
{
	double sum2 = 0.0;

	for(int i = 0; i < number_of_correspondences; i++) {

		Eigen::Vector3d p3DC = R*pws.row(i).transpose() + t;
		double inv_Zc = 1.0 / p3DC.z();
		Eigen::Matrix<double,1,2> p2DC(cx + fx * p3DC.x() * inv_Zc, cy + fy * p3DC.y() * inv_Zc);

		sum2 += (us.row(i) - p2DC).norm();
	}

	return sum2 / number_of_correspondences;
}

void PnPsolver::estimate_R_and_t(Eigen::Matrix3d &R, Eigen::Vector3d &t)
{
	Eigen::Matrix<double,1,3> pc0 = pcs.colwise().sum();
	Eigen::Matrix<double,1,3> pw0 = pws.colwise().sum();

	pc0 /= number_of_correspondences;
	pw0 /= number_of_correspondences;

	Eigen::Matrix3d M = Eigen::Matrix3d::Zero();

	for(int i = 0; i < number_of_correspondences; i++) {
		
		M.noalias() += (pcs.row(i)-pc0).transpose()*((pws.row(i)-pw0));

	}
	
	float N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

	Eigen::Matrix4d N;

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

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(N);

	Eigen::Vector4d eigvec = solver.eigenvectors().col(3); // Maximum eigenvaues is the last
	Eigen::Quaterniond q; // The needed rotation is actually the inverse (from world to cam)
	q.w() = eigvec(0);
	q.x() = -eigvec(1);
	q.y() = -eigvec(2);
	q.z() = -eigvec(3);

	R = q.toRotationMatrix(); // computes the rotation matrix from the quaternion

	//Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

	//const Eigen::Matrix3d &U = svd.matrixU();
	//const Eigen::Matrix3d &V = svd.matrixV();

	//R = U*V.transpose();

	const double det = R.determinant();

	if (det < 0) {
		R.row(2) = -R.row(2);
	}
	t = pc0.transpose() - R*pw0.transpose();
}

void PnPsolver::solve_for_sign(void)
{
	if (pcs(0,2) < 0.0) 
	{
		ccs = -ccs;
		pcs = -pcs;
	}
}

double PnPsolver::compute_R_and_t(const Eigen::Matrix<double,12,12> &U, const Eigen::Vector4d &betas,
					 Eigen::Matrix3d &R, Eigen::Vector3d &t)
{
	compute_ccs(betas, U);
	compute_pcs();

	solve_for_sign();

	estimate_R_and_t(R, t);

	return reprojection_error(R, t);
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

void PnPsolver::find_betas_approx_1(const Eigen::Matrix<double,6,10> &L_6x10, const Eigen::Matrix<double,6,1> &Rho,
						 Eigen::Vector4d &betas)
{
	Eigen::MatrixXd L_6x4(6,4);
	Eigen::Vector4d b4;

	L_6x4.col(0) = L_6x10.col(0);
	L_6x4.col(1) = L_6x10.col(1);
	L_6x4.col(2) = L_6x10.col(3);
	L_6x4.col(3) = L_6x10.col(6);

	b4 = L_6x4.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Rho);

	if (b4[0] < 0) {
		betas[0] = sqrt(-b4[0]);
		betas[1] = -b4[1] / betas[0];
		betas[2] = -b4[2] / betas[0];
		betas[3] = -b4[3] / betas[0];
	} else {
		betas[0] = sqrt(b4[0]);
		betas[1] = b4[1] / betas[0];
		betas[2] = b4[2] / betas[0];
		betas[3] = b4[3] / betas[0];
	}
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

void PnPsolver::find_betas_approx_2(const Eigen::Matrix<double,6,10> &L_6x10, const Eigen::Matrix<double,6,1> &Rho,
						 Eigen::Vector4d &betas)
{
	Eigen::MatrixXd L_6x3(6,3);
	Eigen::Vector3d b3;
	
	L_6x3.col(0)=L_6x10.col(0);
	L_6x3.col(1)=L_6x10.col(1);
	L_6x3.col(2)=L_6x10.col(2);

	b3 = L_6x3.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Rho);

	if (b3[0] < 0) {
		betas[0] = sqrt(-b3[0]);
		betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
	} else {
		betas[0] = sqrt(b3[0]);
		betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
	}

	if (b3[1] < 0) betas[0] = -betas[0];

	betas[2] = 0.0;
	betas[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

void PnPsolver::find_betas_approx_3(const Eigen::Matrix<double,6,10> &L_6x10, const Eigen::Matrix<double,6,1> &Rho,
						 Eigen::Vector4d &betas)
{
	Eigen::MatrixXd L_6x5(6,5);
	Eigen::Matrix<double,5,1> b5;
	
	L_6x5.col(0)=L_6x10.col(0);
	L_6x5.col(1)=L_6x10.col(1);
	L_6x5.col(2)=L_6x10.col(2);
	L_6x5.col(3)=L_6x10.col(3);
	L_6x5.col(4)=L_6x10.col(4);

	b5 = L_6x5.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Rho);

	if (b5[0] < 0) {
		betas[0] = sqrt(-b5[0]);
		betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
	} else {
		betas[0] = sqrt(b5[0]);
		betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
	}
	if (b5[1] < 0) betas[0] = -betas[0];
	betas[2] = b5[3] / betas[0];
	betas[3] = 0.0;
}

void PnPsolver::compute_L_6x10(const Eigen::Matrix<double,12,12> &U, Eigen::Matrix<double,6,10> &l_6x10)
{
	vector<Eigen::Matrix<double,6,3>> dv(4,Eigen::Matrix<double,6,3>::Zero());

	for(int i = 0; i < 4; i++) 
	{
		int a = 0, b = 1;
		for(int j = 0; j < 6; j++) 
		{
			dv[i].row(j) = (U.block<3,1>(3*a,i) - U.block<3,1>(3*b,i)).transpose();

			b++;
			if (b > 3) 
			{
				a++;
				b = a + 1;
			}
		}
	}

	for(int i = 0; i < 6; i++)
	{
		l_6x10(i,0) = 	dv[0].row(i).dot(dv[0].row(i));
		l_6x10(i,1) = 2.0*dv[0].row(i).dot(dv[1].row(i));
		l_6x10(i,2) = 	dv[1].row(i).dot(dv[1].row(i));
		l_6x10(i,3) = 2.0*dv[0].row(i).dot(dv[2].row(i));
		l_6x10(i,4) = 2.0*dv[1].row(i).dot(dv[2].row(i));
		l_6x10(i,5) = 	dv[2].row(i).dot(dv[2].row(i));
		l_6x10(i,6) = 2.0*dv[0].row(i).dot(dv[3].row(i));
		l_6x10(i,7) = 2.0*dv[1].row(i).dot(dv[3].row(i));
		l_6x10(i,8) = 2.0*dv[2].row(i).dot(dv[3].row(i));
		l_6x10(i,9) = 	dv[3].row(i).dot(dv[3].row(i));
	}
}

void PnPsolver::compute_rho(Eigen::Matrix<double,6,1> &rho)
{
	rho[0]=(cws.row(0)-cws.row(1)).squaredNorm();
	rho[1]=(cws.row(0)-cws.row(2)).squaredNorm();
	rho[2]=(cws.row(0)-cws.row(3)).squaredNorm();
	rho[3]=(cws.row(1)-cws.row(2)).squaredNorm();
	rho[4]=(cws.row(1)-cws.row(3)).squaredNorm();
	rho[5]=(cws.row(2)-cws.row(3)).squaredNorm();
}

void PnPsolver::compute_A_and_b_gauss_newton(const Eigen::Matrix<double,6,10> &l_6x10, const Eigen::Matrix<double,6,1> &rho,
					Eigen::Vector4d &betas, Eigen::Matrix<double,6,4,Eigen::RowMajor> &A, Eigen::Matrix<double,6,1> &b)
{
	Eigen::Matrix4d L_temp;
	for(int i = 0; i < 6; i++) {
		L_temp << 2*l_6x10(i,0),  l_6x10(i,1),  l_6x10(i,3),  l_6x10(i,6),
					l_6x10(i,1),2*l_6x10(i,2),  l_6x10(i,4),  l_6x10(i,7),
					l_6x10(i,3),  l_6x10(i,4),2*l_6x10(i,5),  l_6x10(i,8),
					l_6x10(i,6),  l_6x10(i,7),  l_6x10(i,8),2*l_6x10(i,9);
		
		A.row(i) = (L_temp*betas).transpose();

		b(i) = rho(i) - (
			l_6x10(i,0) * betas[0] * betas[0] +
			l_6x10(i,1) * betas[0] * betas[1] +
			l_6x10(i,2) * betas[1] * betas[1] +
			l_6x10(i,3) * betas[0] * betas[2] +
			l_6x10(i,4) * betas[1] * betas[2] +
			l_6x10(i,5) * betas[2] * betas[2] +
			l_6x10(i,6) * betas[0] * betas[3] +
			l_6x10(i,7) * betas[1] * betas[3] +
			l_6x10(i,8) * betas[2] * betas[3] +
			l_6x10(i,9) * betas[3] * betas[3]);
	}
}

void PnPsolver::gauss_newton(const Eigen::Matrix<double,6,10> &L_6x10, const Eigen::Matrix<double,6,1> &Rho,
			Eigen::Vector4d &betas)
{
	const int iterations_number = 5;

	Eigen::Matrix<double,6,4,Eigen::RowMajor> A;
	Eigen::Matrix<double,6,1> B;
	Eigen::Matrix<double,4,1> X;

	for(int k = 0; k < iterations_number; k++) {
		compute_A_and_b_gauss_newton(L_6x10, Rho,
				 betas, A, B);
		qr_solve(A, B, X);

		betas += X;
	}
}

void PnPsolver::qr_solve(Eigen::Matrix<double,6,4,Eigen::RowMajor> &A, Eigen::Matrix<double,6,1> &b, 
				Eigen::Matrix<double,4,1> &X)
{
	static int max_nr = 0;
	static double * A1, * A2;

	const int nr = A.rows();
	const int nc = A.cols();

	if (max_nr != 0 && max_nr < nr) {
		delete [] A1;
		delete [] A2;
	}
	if (max_nr < nr) {
		max_nr = nr;
		A1 = new double[nr];
		A2 = new double[nr];
	}

	double * pA = A.data(), * ppAkk = pA;
	for(int k = 0; k < nc; k++) {
		double * ppAik = ppAkk, eta = fabs(*ppAik);
		for(int i = k + 1; i < nr; i++) 
		{
			double elt = fabs(*ppAik);
			if (eta < elt) eta = elt;
			ppAik += nc;
		}

		if (eta == 0) 
		{
			A1[k] = A2[k] = 0.0;
			cerr << "God damnit, A is singular, this shouldn't happen." << endl;
			return;
		} else 
		{
			double * ppAik = ppAkk, sum = 0.0, inv_eta = 1. / eta;
			for(int i = k; i < nr; i++) 
			{
				*ppAik *= inv_eta;
				sum += *ppAik * *ppAik;
				ppAik += nc;
			}
			double sigma = sqrt(sum);
			if (*ppAkk < 0)
				sigma = -sigma;
			*ppAkk += sigma;
			A1[k] = sigma * *ppAkk;
			A2[k] = -eta * sigma;
			for(int j = k + 1; j < nc; j++) 
			{
				double * ppAik = ppAkk, sum = 0;
				for(int i = k; i < nr; i++) 
				{
					sum += *ppAik * ppAik[j - k];
					ppAik += nc;
				}
				double tau = sum / A1[k];
				ppAik = ppAkk;
				for(int i = k; i < nr; i++) 
				{
					ppAik[j - k] -= tau * *ppAik;
					ppAik += nc;
				}
			}
		}
		ppAkk += nc + 1;
	}

	// b <- Qt b
	double * ppAjj = pA, * pb = b.data();
	for(int j = 0; j < nc; j++) 
	{
		double * ppAij = ppAjj, tau = 0;
		for(int i = j; i < nr; i++)	
		{
			tau += *ppAij * pb[i];
			ppAij += nc;
		}
		tau /= A1[j];
		ppAij = ppAjj;
		for(int i = j; i < nr; i++) 
		{
			pb[i] -= tau * *ppAij;
			ppAij += nc;
		}
		ppAjj += nc + 1;
	}

	// X = R-1 b
	double * pX = X.data();
	pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
	for(int i = nc - 2; i >= 0; i--) 
	{
		double * ppAij = pA + i * nc + (i + 1), sum = 0;

		for(int j = i + 1; j < nc; j++) 
		{
			sum += *ppAij * pX[j];
			ppAij++;
		}
		pX[i] = (pb[i] - sum) / A2[i];
	}
}

} //namespace ORB_SLAM
