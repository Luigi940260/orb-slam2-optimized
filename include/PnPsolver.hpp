#pragma once

#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
using namespace cv;

#include <opencv2/core/core.hpp>
#include "MapPoint.hpp"
#include "Frame.hpp"

#include <vector>
#include <cmath>
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include <algorithm>

namespace ORB_SLAM_CUSTOM
{

class PnPsolver {
 public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	PnPsolver(const Frame &F, const vector<std::shared_ptr<MapPoint>> &vpMapPointMatches);

	void SetRansacParameters(double probability = 0.99, int minInliers = 8 , int maxIterations = 300, int minSet = 4, float epsilon = 0.4,
							 float th2 = 5.991);

	bool find(vector<bool> &vbInliers, int &nInliers, Eigen::Matrix4f &T);

	bool iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers, Eigen::Matrix4f &T);

 private:

	void CheckInliers();
	bool Refine();

	// Functions from the original EPnP code
	void set_maximum_number_of_correspondences(const int n);
	void reset_correspondences(void);
	void add_correspondence(const Eigen::Vector3f &p3D, const Eigen::Vector2f &p2D);

	double compute_pose(Eigen::Matrix3f &R, Eigen::Vector3f &t);

	double reprojection_error(const Eigen::Matrix3d &R, const Eigen::Vector3d &t);

	void choose_control_points();
	void compute_barycentric_coordinates();
	void compute_ccs(const Eigen::Vector4d &betas, const Eigen::Matrix<double,12,12> &U);
	void compute_pcs();

	void solve_for_sign();

	void find_betas_approx_1(const Eigen::Matrix<double,6,10> &L_6x10, const Eigen::Matrix<double,6,1> &Rho, Eigen::Vector4d &betas);
	void find_betas_approx_2(const Eigen::Matrix<double,6,10> &L_6x10, const Eigen::Matrix<double,6,1> &Rho, Eigen::Vector4d &betas);
	void find_betas_approx_3(const Eigen::Matrix<double,6,10> &L_6x10, const Eigen::Matrix<double,6,1> &Rho, Eigen::Vector4d &betas);
	void qr_solve(Eigen::Matrix<double,6,4,Eigen::RowMajor> &A, Eigen::Matrix<double,6,1> &b, Eigen::Matrix<double,4,1> &X);

	void compute_rho(Eigen::Matrix<double,6,1> &rho);
	void compute_L_6x10(const Eigen::Matrix<double,12,12> &Ut, Eigen::Matrix<double,6,10> &l_6x10);

	void gauss_newton(const Eigen::Matrix<double,6,10> &L_6x10, const Eigen::Matrix<double,6,1> &Rho, Eigen::Vector4d &current_betas);
	void compute_A_and_b_gauss_newton(const Eigen::Matrix<double,6,10> &l_6x10, const Eigen::Matrix<double,6,1> &rho,
					Eigen::Vector4d &cb, Eigen::Matrix<double,6,4,Eigen::RowMajor> &A, Eigen::Matrix<double,6,1> &b);

	double compute_R_and_t(const Eigen::Matrix<double,12,12> &U, const Eigen::Vector4d &betas,
			 Eigen::Matrix3d &R, Eigen::Vector3d &t);

	void estimate_R_and_t(Eigen::Matrix3d &R, Eigen::Vector3d &t);

	double cx, cy, fx, fy;

	Eigen::Matrix<double,4,3> cws, ccs;
	Eigen::Matrix<double,Eigen::Dynamic,3> pws, pcs;
	Eigen::Matrix<double,Eigen::Dynamic,4> alphas;
	Eigen::Matrix<double,Eigen::Dynamic,2> us;

	int maximum_number_of_correspondences;
	int number_of_correspondences;

	int N_points;

	// 2D Points
	vector<Eigen::Vector2f> mvP2D;
	vector<float> mvSigma2;

	// 3D Points
	vector<Eigen::Vector3f> mvP3Dw;

	// Index in Frame
	vector<size_t> mvKeyPointIndices;

	// Current Estimation
	Eigen::Matrix3f mRi;
	Eigen::Vector3f mti;
	Eigen::Matrix4f mTcwi;
	vector<bool> mvbInliersi;
	int mnInliersi;

	// Current Ransac State
	int mnIterations;
	vector<bool> mvbBestInliers;
	int mnBestInliers;
	Eigen::Matrix4f mBestTcw;

	// Refined
	Eigen::Matrix4f mRefinedTcw;
	vector<bool> mvbRefinedInliers;
	int mnRefinedInliers;

	// Number of Correspondences
	int N;

	// Indices for random selection [0 .. N-1]
	vector<size_t> mvAllIndices;

	// RANSAC probability
	double mRansacProb;

	// RANSAC min inliers
	int mRansacMinInliers;

	// RANSAC max iterations
	int mRansacMaxIts;

	// RANSAC expected inliers/total ratio
	float mRansacEpsilon;

	// RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2
	float mRansacTh;

	// RANSAC Minimun Set used at each iteration
	int mRansacMinSet;

	// Max square error associated with scale level. Max error = th*th*sigma(level)*sigma(level)
	vector<float> mvMaxError;

};

} //namespace ORB_SLAM