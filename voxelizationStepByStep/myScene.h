#pragma once

#include "structSet.h"
#include <vector>
#include <memory>

#ifndef MY_SCENE
#define MY_SCENE

class myScene {

public:

	std::vector<Mesh>* sceneMeshs;
	BvhTreeNode* bvhTree;
	std::vector<BvhArrayNode> bvhArray;

	myScene(std::vector<Mesh>* meshs);
	bool splitSpace(int k, float splitPoint, std::vector<uint32_t> meshsIndex, std::vector<uint32_t>& meshsLeftIndex, std::vector<uint32_t>& meshsRightIndex);
	void makeBvhTree(BvhTreeNode* bvhTree, std::vector<uint32_t> meshsIndex);
	//����Ҫ����buffer��������Ҫ�̶���С����������
	//��ô���ǿ���ͨ������һ�����ݽṹ���ݣ�ÿ��Ԫ�ش���ӳ�����bvhNode����������mesh���������Լ����������������е�����
	void bvhTreeToArray(BvhTreeNode* node, uint32_t& childNum, int32_t nodeIndex);

	~myScene();
	void deleteBvhTree(BvhTreeNode* bvhTreeNode);

};

#endif // !MY_SCENE

