#pragma once

#include "parameters.h"

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;
using namespace DVision;

class KeyFrame
{
public:

	double time_stamp; // 关键帧时间戳
	int index; // 关键帧索引

	// Pose in vins_world
	Eigen::Vector3d origin_vio_T;		
	Eigen::Matrix3d origin_vio_R;

	cv::Mat image; // 拷贝关键帧图像
	cv::Mat thumbnail; // 关键帧图像缩略图，间keyframe.cc的使用

	vector<cv::Point3f> point_3d; // 特征点空间坐标 数组
	vector<cv::Point2f> point_2d_uv; // 特征点像素坐标 数组
	vector<cv::Point2f> point_2d_norm; // 特征点归一化平面坐标 数组
	vector<double> point_id; // 特征点id 数组

	vector<cv::KeyPoint> keypoints; // 重新提取的FAST关键点数组
	vector<cv::KeyPoint> keypoints_norm; // 重新提取的关键点归一化平面坐标数组
	vector<cv::KeyPoint> window_keypoints; // 原本的特征点数组转换成的cv关键点数组

	vector<BRIEF::bitset> brief_descriptors; // 关键帧的描述子数组（重新提取的关键点）
	vector<BRIEF::bitset> window_brief_descriptors; // 关键帧的描述子数组（原始特征点）

	KeyFrame(double _time_stamp, int _index, 
			 Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, 
			 cv::Mat &_image,
			 vector<cv::Point3f> &_point_3d, 
			 vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal, 
			 vector<double> &_point_id);
    // 初始化传参：时间戳，索引值，图，缩略图，特征点三维坐标数组，归一化坐标数组，像素坐标数组，特征点索引数组。提取描述子

	bool findConnection(KeyFrame* old_kf); // 通过描述子距离筛选+PnPRANSAC筛选，两帧之间的内点数量足够多，则返回true（回环帧）
	void computeWindowBRIEFPoint(); // 计算原始特征点的描述子
	void computeBRIEFPoint(); // 计算重新提取关键点的描述子，并保存关键点归一化平面坐标

	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b); // 计算汉明距离

	bool searchInAera(const BRIEF::bitset window_descriptor,
	                  const std::vector<BRIEF::bitset> &descriptors_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old_norm,
	                  cv::Point2f &best_match,
	                  cv::Point2f &best_match_norm);
    // 计算某个原始特征点描述子与另一帧所有描述子的汉明距离，最小汉明距离足够小则返回true并保存匹配点

	void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm);
    // 对每个原始特征点与另一帧所有描述子执行searchInAera，记录结果并保存最小汉明的两个点坐标


	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
	               const std::vector<cv::Point3f> &matched_3d,
	               std::vector<uchar> &status);
    // RANSAC求解两帧PnP后的内点数量
};