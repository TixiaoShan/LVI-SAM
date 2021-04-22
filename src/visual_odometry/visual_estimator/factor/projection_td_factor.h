#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"

//对imu-camera 时间戳不完全同步和 Rolling shutter 相机的支持
class ProjectionTdFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1>
{
  public:
    ProjectionTdFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
                       const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
                       const double _td_i, const double _td_j, const double _row_i, const double _row_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j;//角点在归一化平面的坐标
    Eigen::Vector3d velocity_i, velocity_j;//角点在归一化平面的速度
    double td_i, td_j;//处理IMU数据时用到的时间同步误差
    Eigen::Matrix<double, 2, 3> tangent_base;
    double row_i, row_j;//角点图像坐标的纵坐标
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};
