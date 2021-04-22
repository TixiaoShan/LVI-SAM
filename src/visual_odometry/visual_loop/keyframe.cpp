#include "keyframe.h"

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// create keyframe online
KeyFrame::KeyFrame(double _time_stamp, int _index, 
                   Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, 
                   cv::Mat &_image,
		           vector<cv::Point3f> &_point_3d, 
                   vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
		           vector<double> &_point_id)
{
	time_stamp = _time_stamp;
	index = _index;

	origin_vio_T = _vio_T_w_i;		
	origin_vio_R = _vio_R_w_i;

	image = _image.clone();
    // cv::resize(image, thumbnail, cv::Size(80, 60));
	cv::resize(image, thumbnail, cv::Size(), MATCH_IMAGE_SCALE, MATCH_IMAGE_SCALE);

	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;

	computeWindowBRIEFPoint();
	computeBRIEFPoint();
	if(!DEBUG_IMAGE)
		image.release();
}

void KeyFrame::computeWindowBRIEFPoint()
{
	for(int i = 0; i < (int)point_2d_uv.size(); i++)
	{
	    cv::KeyPoint key;
	    key.pt = point_2d_uv[i];
	    window_keypoints.push_back(key);
	}
	briefExtractor(image, window_keypoints, window_brief_descriptors);
}

void KeyFrame::computeBRIEFPoint()
{
	const int fast_th = 20; // corner detector response threshold
	if(1)
		cv::FAST(image, keypoints, fast_th, true);
	else
	{
		vector<cv::Point2f> tmp_pts;
		cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
		for(int i = 0; i < (int)tmp_pts.size(); i++)
		{
		    cv::KeyPoint key;
		    key.pt = tmp_pts[i];
		    keypoints.push_back(key);
		}
	}
	briefExtractor(image, keypoints, brief_descriptors);
    
	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}
}

bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
    for(int i = 0; i < (int)descriptors_old.size(); i++)
    {

        int dis = HammingDis(window_descriptor, descriptors_old[i]);
        if(dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }
    //printf("best dist %d", bestDist);
    if (bestIndex != -1 && bestDist < 80)
    {
      best_match = keypoints_old[bestIndex].pt;
      best_match_norm = keypoints_old_norm[bestIndex].pt;
      return true;
    }
    else
      return false;
}

void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
								std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm)
{
    for(int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
          status.push_back(1);
        else
          status.push_back(0);
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }
}

void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status)
{
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    Matrix3d R_inital;
    Vector3d P_inital;
    Matrix3d R_w_c = origin_vio_R * qic;
    Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;

    if (CV_MAJOR_VERSION < 3)
        solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
    else
    {
        if (CV_MINOR_VERSION < 2)
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
        else
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);
    }

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);

    for( int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }
}


bool KeyFrame::findConnection(KeyFrame* old_kf)
{
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;
	vector<uchar> status;

	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;

	searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);

	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
		status.clear();
	    PnPRANSAC(matched_2d_old_norm, matched_3d, status);
	    reduceVector(matched_2d_cur, status);
	    reduceVector(matched_2d_old, status);
	    reduceVector(matched_2d_cur_norm, status);
	    reduceVector(matched_2d_old_norm, status);
	    reduceVector(matched_3d, status);
	    reduceVector(matched_id, status);

        if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
        {
        	if (pub_match_img.getNumSubscribers() != 0)
            {
            	int gap = 10;
            	cv::Mat gap_image(thumbnail.size().height, gap, CV_8UC1, cv::Scalar(255, 255, 255));
                cv::Mat gray_img, loop_match_img;
                cv::Mat old_img = old_kf->thumbnail;
                cv::hconcat(thumbnail, gap_image, gap_image);
                cv::hconcat(gap_image, old_img, gray_img);
                cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
                // plot features in current frame
                for(int i = 0; i< (int)matched_2d_cur.size(); i++)
                {
                    cv::Point2f cur_pt = matched_2d_cur[i] * MATCH_IMAGE_SCALE;
                    cv::circle(loop_match_img, cur_pt, 5*MATCH_IMAGE_SCALE, cv::Scalar(0, 255, 0));
                }
                // plot features in previous frame
                for(int i = 0; i< (int)matched_2d_old.size(); i++)
                {
                    cv::Point2f old_pt = matched_2d_old[i] * MATCH_IMAGE_SCALE;
                    old_pt.x += thumbnail.size().width + gap;
                    cv::circle(loop_match_img, old_pt, 5*MATCH_IMAGE_SCALE, cv::Scalar(0, 255, 0));
                }
                // plot lines connecting features
                for (int i = 0; i< (int)matched_2d_cur.size(); i++)
                {
                    cv::Point2f old_pt = matched_2d_old[i] * MATCH_IMAGE_SCALE;
                    old_pt.x += thumbnail.size().width + gap;
                    cv::line(loop_match_img, matched_2d_cur[i] * MATCH_IMAGE_SCALE, old_pt, cv::Scalar(0, 255, 0), 2*MATCH_IMAGE_SCALE, 8, 0);
                }
                // plot text
                int banner_height = (double)100 * MATCH_IMAGE_SCALE;
                cv::Mat notation(banner_height, thumbnail.size().width + gap + thumbnail.size().width, CV_8UC3, cv::Scalar(255, 255, 255));
                putText(notation, "current frame: " + to_string(index), 
                        cv::Point2f(5, banner_height - 5), CV_FONT_HERSHEY_SIMPLEX, 
                        MATCH_IMAGE_SCALE*2, cv::Scalar(255), 2);
                putText(notation, "previous frame: " + to_string(old_kf->index), 
                        cv::Point2f(5 + thumbnail.size().width + gap, banner_height - 5), CV_FONT_HERSHEY_SIMPLEX, 
                        MATCH_IMAGE_SCALE*2, cv::Scalar(255), 2);
                cv::vconcat(notation, loop_match_img, loop_match_img);
                // publish matched image
    	    	sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", loop_match_img).toImageMsg();
                msg->header.stamp = ros::Time(time_stamp);
    	    	pub_match_img.publish(msg);
            }

            return true;
        }
	}

	return false;
}


int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}