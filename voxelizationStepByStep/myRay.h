#pragma once

#include "structSet.h"

#include <vector>

#ifndef MY_RAY
#define MY_RAY

class myRay {

public:

	//�ֱ���
	uint32_t width;
	uint32_t height;

	myRay(uint32_t width, uint32_t hegiht);
	//��ײ���ĺ���Ҫ�ŵ�������ɫ����ȥ
	void rayStart(VkImageView targetTextureView, std::vector<VkBuffer> sceneBuffers);

};

#endif // !MY_RAY
