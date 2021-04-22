# Cloud Info
Header header 

int32[] startRingIndex
int32[] endRingIndex

int32[]  pointColInd # point column index in range image
float32[] pointRange # point range 

int64 imuAvailable
int64 odomAvailable

# Attitude for lidar odometry initialization
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Odometry 
float32 odomX
float32 odomY
float32 odomZ
float32 odomRoll
float32 odomPitch
float32 odomYaw

# Odometry reset ID
int64 odomResetId

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature