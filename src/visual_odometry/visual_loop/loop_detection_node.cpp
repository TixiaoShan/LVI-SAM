#include "parameters.h"
#include "keyframe.h"
#include "loop_detection.h"

#define SKIP_FIRST_CNT 10

queue<sensor_msgs::ImageConstPtr>      image_buf; // 存储IMAGE原始图像
queue<sensor_msgs::PointCloudConstPtr> point_buf; // 存关键帧对应点云
queue<nav_msgs::Odometry::ConstPtr>    pose_buf; //存当前帧位姿，都是ROS里的格式

std::mutex m_buf; // 互斥锁
std::mutex m_process; // 互斥锁

LoopDetector loopDetector; // .h文件里定义的类

double SKIP_TIME = 0; // 回环频率控制，隔一定时间回环一次，减少计算量，此处每一帧关键帧都回环
double SKIP_DIST = 0; 

camodocal::CameraPtr m_camera; // 相机模型

Eigen::Vector3d tic; // 外参平移，没用到
Eigen::Matrix3d qic; // 外参旋转，没用到

std::string PROJECT_NAME; // 后续要读入的全局参数
std::string IMAGE_TOPIC;

int DEBUG_IMAGE; // For Debug, 如果要调试IMAGE设置为1，显示回环的两幅图像
int LOOP_CLOSURE; // 执行回环的标志位
double MATCH_IMAGE_SCALE; // 图像中匹配点的大小，0.5像素


ros::Publisher pub_match_img; // 发布匹配图像，标注匹配分数及匹配点对应关系
ros::Publisher pub_match_msg; // 发布匹配帧时间戳对
ros::Publisher pub_key_pose; // 发布位姿点



BriefExtractor briefExtractor;

void new_sequence()
// 开始一个新的图像序列（地图合并功能）
{
    m_buf.lock();
    while(!image_buf.empty())
        image_buf.pop();
    while(!point_buf.empty())
        point_buf.pop();
    while(!pose_buf.empty())
        pose_buf.pop();
    m_buf.unlock();
}

void image_callback(const sensor_msgs::ImageConstPtr &image_msg)
// 图像数据回调函数，将image_msg放入image_buf，同时根据时间戳检测是否是新的图像序列
{
    if(!LOOP_CLOSURE)
        return;

    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();

    // detect unstable camera stream
    static double last_image_time = -1;
    if (last_image_time == -1)
        last_image_time = image_msg->header.stamp.toSec();
    else if (image_msg->header.stamp.toSec() - last_image_time > 1.0 || image_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! detect a new sequence!");
        new_sequence();
    }
    last_image_time = image_msg->header.stamp.toSec();
}

void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg)
// 地图点云回调函数，把point_msg放入point_buf
{
    if(!LOOP_CLOSURE)
        return;

    m_buf.lock();
    point_buf.push(point_msg);
    m_buf.unlock();
}

void pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
// 图像帧位姿回调函数，把pose_msg放入pose_buf
{
    if(!LOOP_CLOSURE)
        return;

    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
}

void extrinsic_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
// 相机IMU的外参回调函数，得到相机到IMU的外参tic和qic
{
    m_process.lock();
    tic = Vector3d(pose_msg->pose.pose.position.x,
                   pose_msg->pose.pose.position.y,
                   pose_msg->pose.pose.position.z);
    qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                      pose_msg->pose.pose.orientation.x,
                      pose_msg->pose.pose.orientation.y,
                      pose_msg->pose.pose.orientation.z).toRotationMatrix();
    m_process.unlock();
}

//回环处理函数，对齐输入的pose、image、point，创建关键帧并进行回环检测
//1.得到具有相同时间戳的pose_msg、image_msg、point_msg
//2.构建pose_graph中用到的关键帧并回环：先剔除最开始的SKIP_FIRST_CNT帧，然后每隔SKIP_CNT，将距上一关键帧距离
//（平移向量的模）超过SKIP_DIS的图像创建为关键帧。
// 1)剔除最开始的11帧 
// 2)每隔SKIP_FIRST_CNT一次
// 3)得到关键帧位姿
// 4)位置距离超过SKIP_DIS采用，控制频率
// 5)创建为关键帧
// 6)回环主要函数addKeyFrame detect loop
void process()
{
    if (!LOOP_CLOSURE)
        return;

    while (ros::ok())
    {
        sensor_msgs::ImageConstPtr image_msg = NULL;
        sensor_msgs::PointCloudConstPtr point_msg = NULL;
        nav_msgs::Odometry::ConstPtr pose_msg = NULL;

        // find out the messages with same time stamp 
        // 1.得到具有相同时间戳的pose_msg、image_msg、point_msg
        m_buf.lock();
        if(!image_buf.empty() && !point_buf.empty() && !pose_buf.empty())
        {
            if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
            {
                pose_buf.pop();
                printf("throw pose at beginning\n");
            }
            else if (image_buf.front()->header.stamp.toSec() > point_buf.front()->header.stamp.toSec())
            {
                point_buf.pop();
                printf("throw point at beginning\n");
            }
            else if (image_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec() 
                && point_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec())
            {
                pose_msg = pose_buf.front();
                pose_buf.pop();
                while (!pose_buf.empty())
                    pose_buf.pop();
                while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    image_buf.pop();
                image_msg = image_buf.front();
                image_buf.pop();

                while (point_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    point_buf.pop();
                point_msg = point_buf.front();
                point_buf.pop();
            }
        }
        m_buf.unlock();

        //2.构建pose_graph中用到的关键帧并回环：先剔除最开始的SKIP_FIRST_CNT帧，
        //然后每隔SKIP_CNT，将距上一关键帧距离（平移向量的模）超过SKIP_DIS的图像创建为关键帧。
        if (pose_msg != NULL)
        {
            // skip fisrt few 剔除最开始的11帧
            static int skip_first_cnt = 0;
            if (skip_first_cnt < SKIP_FIRST_CNT)
            {
                skip_first_cnt++;
                continue;
            }

            // limit frequency 每隔SKIP_FIRST_CNT一次
            static double last_skip_time = -1;
            if (pose_msg->header.stamp.toSec() - last_skip_time < SKIP_TIME)
                continue;
            else
                last_skip_time = pose_msg->header.stamp.toSec();

            // get keyframe pose 得到关键帧位姿
            static Eigen::Vector3d last_t(-1e6, -1e6, -1e6);
            Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  pose_msg->pose.pose.position.z);
            Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                     pose_msg->pose.pose.orientation.x,
                                     pose_msg->pose.pose.orientation.y,
                                     pose_msg->pose.pose.orientation.z).toRotationMatrix();

            // add keyframe 位置距离超过SKIP_DIS采用，控制频率
            if((T - last_t).norm() > SKIP_DIST)
            {
                // convert image
                cv_bridge::CvImageConstPtr ptr;
                if (image_msg->encoding == "8UC1")
                {
                    sensor_msgs::Image img;
                    img.header = image_msg->header;
                    img.height = image_msg->height;
                    img.width = image_msg->width;
                    img.is_bigendian = image_msg->is_bigendian;
                    img.step = image_msg->step;
                    img.data = image_msg->data;
                    img.encoding = "mono8";
                    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
                }
                else
                    ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
                
                cv::Mat image = ptr->image;

                vector<cv::Point3f> point_3d; 
                vector<cv::Point2f> point_2d_uv; 
                vector<cv::Point2f> point_2d_normal;
                vector<double> point_id;

                for (unsigned int i = 0; i < point_msg->points.size(); i++)
                {
                    cv::Point3f p_3d;
                    p_3d.x = point_msg->points[i].x;
                    p_3d.y = point_msg->points[i].y;
                    p_3d.z = point_msg->points[i].z;
                    point_3d.push_back(p_3d);

                    cv::Point2f p_2d_uv, p_2d_normal;
                    double p_id;
                    p_2d_normal.x = point_msg->channels[i].values[0];
                    p_2d_normal.y = point_msg->channels[i].values[1];
                    p_2d_uv.x = point_msg->channels[i].values[2];
                    p_2d_uv.y = point_msg->channels[i].values[3];
                    p_id = point_msg->channels[i].values[4];
                    point_2d_normal.push_back(p_2d_normal);
                    point_2d_uv.push_back(p_2d_uv);
                    point_id.push_back(p_id);
                }

                // new keyframe 创建为关键帧
                static int global_frame_index = 0;
                KeyFrame* keyframe = new KeyFrame(pose_msg->header.stamp.toSec(), global_frame_index, 
                                                  T, R, 
                                                  image,
                                                  point_3d, point_2d_uv, point_2d_normal, point_id);   

                // detect loop 回环主要函数
                m_process.lock();
                loopDetector.addKeyFrame(keyframe, 1);
                m_process.unlock();

                loopDetector.visualizeKeyPoses(pose_msg->header.stamp.toSec());

                global_frame_index++;
                last_t = T;
            }
        }

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
} 

// 主函数&&节点程序入口
// 一、加载配置参数
// 二、初始化全局参数
// 三、回环初始
// 四、订阅各topic执行回调函数和发布topic
// 五、process主线程
int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins");
    ros::NodeHandle n;
    ROS_INFO("\033[1;32m----> Visual Loop Detection Started.\033[0m");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);

    // 一、加载配置参数，等待100微秒，载入
    // Load params
    std::string config_file;
    n.getParam("vins_config_file", config_file); // 在module_sam.launch里，读取config/params_camera.yaml,94行,这个yaml也是大体照搬的vins_mono
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    usleep(100);

    // 二、初始化全局参数
    // Initialize global params
    fsSettings["project_name"] >> PROJECT_NAME;  
    fsSettings["image_topic"]  >> IMAGE_TOPIC;  
    fsSettings["loop_closure"] >> LOOP_CLOSURE;
    fsSettings["skip_time"]    >> SKIP_TIME;
    fsSettings["skip_dist"]    >> SKIP_DIST;
    fsSettings["debug_image"]  >> DEBUG_IMAGE;
    fsSettings["match_image_scale"] >> MATCH_IMAGE_SCALE;
    
    // 三、回环初始化
    if (LOOP_CLOSURE)
    {
        // 读取字典路径
        string pkg_path = ros::package::getPath(PROJECT_NAME);

        // initialize vocabulary 初始化词袋
        string vocabulary_file;
        fsSettings["vocabulary_file"] >> vocabulary_file;  
        vocabulary_file = pkg_path + vocabulary_file;
        loopDetector.loadVocabulary(vocabulary_file);

        // initialize brief extractor 初始化brief特征提取
        string brief_pattern_file;
        fsSettings["brief_pattern_file"] >> brief_pattern_file;  
        brief_pattern_file = pkg_path + brief_pattern_file;
        briefExtractor = BriefExtractor(brief_pattern_file);

        // initialize camera model 初始化相机模型
        m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(config_file.c_str());
    }

    // 四、订阅各topic执行回调函数和发布topic
    ros::Subscriber sub_image     = n.subscribe(IMAGE_TOPIC, 30, image_callback);
    ros::Subscriber sub_pose      = n.subscribe(PROJECT_NAME + "/vins/odometry/keyframe_pose",  3, pose_callback);
    ros::Subscriber sub_point     = n.subscribe(PROJECT_NAME + "/vins/odometry/keyframe_point", 3, point_callback);
    ros::Subscriber sub_extrinsic = n.subscribe(PROJECT_NAME + "/vins/odometry/extrinsic",      3, extrinsic_callback);
    // 订阅原始图片话题image_raw，VIS窗口关键帧位姿keyframe_pose，最新关键帧位姿话题keyframe_point,相机IMU外参tf话题odometry/extrinsic。

    // 发布视觉回环帧match_frame、发布回环帧图片和位姿用于可视化
    pub_match_img = n.advertise<sensor_msgs::Image>             (PROJECT_NAME + "/vins/loop/match_image", 3);
    pub_match_msg = n.advertise<std_msgs::Float64MultiArray>    (PROJECT_NAME + "/vins/loop/match_frame", 3);
    pub_key_pose  = n.advertise<visualization_msgs::MarkerArray>(PROJECT_NAME + "/vins/loop/keyframe_pose", 3);

    // 根据标志位，可以选择关闭回环检测
    if (!LOOP_CLOSURE)
    {
        sub_image.shutdown();
        sub_pose.shutdown();
        sub_point.shutdown();
        sub_extrinsic.shutdown();

        pub_match_img.shutdown();
        pub_match_msg.shutdown();
        pub_key_pose.shutdown();
    }

    // 五、process回环检测主线程
    std::thread measurement_process;
    measurement_process = std::thread(process);

    ros::spin();

    return 0;
}