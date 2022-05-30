#pragma once

#include <ros/ros.h>
#include <ros/package.h>

#include <eigen3/Eigen/Dense>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <cassert>

#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
#include "ThirdParty/DBoW/TemplatedDatabase.h"
#include "ThirdParty/DBoW/TemplatedVocabulary.h"

#include "../visual_feature/camera_models/CameraFactory.h"
#include "../visual_feature/camera_models/CataCamera.h"
#include "../visual_feature/camera_models/PinholeCamera.h"

using namespace std;

extern camodocal::CameraPtr m_camera;

extern Eigen::Vector3d tic;
extern Eigen::Matrix3d qic;

extern string PROJECT_NAME;
extern string IMAGE_TOPIC;

extern int DEBUG_IMAGE;
extern int LOOP_CLOSURE;
extern double MATCH_IMAGE_SCALE;

extern ros::Publisher pub_match_img;
extern ros::Publisher pub_match_msg;
extern ros::Publisher pub_key_pose;






class BriefExtractor
{
public:

    DVision::BRIEF m_brief; // 复合BRIEF，用于调用方法

    virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<DVision::BRIEF::bitset> &descriptors) const
    // 重载()操作符，用来计算图片上指定关键点的描述子
    {
        m_brief.compute(im, keys, descriptors);
    }

    BriefExtractor(){};

    BriefExtractor(const std::string &pattern_file) // 载入BRIEF描述子计算pattern
    {
        // The DVision::BRIEF extractor computes a random pattern by default when
        // the object is created.
        // We load the pattern that we used to build the vocabulary, to make
        // the descriptors compatible with the predefined vocabulary

        // loads the pattern
        cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
        if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

        vector<int> x1, y1, x2, y2;
        fs["x1"] >> x1;
        fs["x2"] >> x2;
        fs["y1"] >> y1;
        fs["y2"] >> y2;

        m_brief.importPairs(x1, y1, x2, y2);
    }
};

extern BriefExtractor briefExtractor;