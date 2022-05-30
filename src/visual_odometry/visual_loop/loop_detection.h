#pragma once

#include "parameters.h"
#include "keyframe.h"

using namespace DVision;
using namespace DBoW2;

class LoopDetector
{
public:

	BriefDatabase db; // 描述子数据库
	BriefVocabulary* voc; // 字典

	map<int, cv::Mat> image_pool; // 保存回环帧用于可视化和debug

	list<KeyFrame*> keyframelist; // 关键帧链表，可以看下关键帧的结构定义，在keyfrmae.h

	LoopDetector(); // 默认构造函数

    // 下面是成员函数
	void loadVocabulary(std::string voc_path); // 按照路径加载字典，初始化描述子数据库db
	
	void addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop); // 添加新的关键帧，搜索回环、验证、并发布（可选）
	void addKeyFrameIntoVoc(KeyFrame* keyframe); // 添加关键帧描述子到数据库
	KeyFrame* getKeyFrame(int index); // 按照索引搜索keyframelist中的关键帧

	void visualizeKeyPoses(double time_cur); // 回环帧可视化函数

	int detectLoop(KeyFrame* keyframe, int frame_index); // 搜索当前关键帧的回环帧
};
