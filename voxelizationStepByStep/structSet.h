#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include<glm/glm.hpp>

#include <array>
#include <optional>
#include <string>
#include <vector>

#ifndef STRUCT_SET
#define STRUCT_SET
//�������Ҫ���ύ��������ִ�У���ÿ���������ڲ�ͬ�Ķ����壬��ͬ�Ķ�����֧�ֲ�ͬ��ָ�������Щ������ֻ������ֵ���㣬��Щ������ֻ�����ڴ������
//���������������豸�ṩ�ģ���һ�������豸�ṩĳЩ�����壬���Կ����ṩ���㡢��Ⱦ�Ĺ��ܣ��������ṩ������ɴ��Ĺ���
struct QueueFamilyIndices {

	//std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;
	std::optional<uint32_t> graphicsAndComputeFamily;

	bool isComplete() {
		return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
	}

};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
	glm::vec3 pos;
	glm::vec2 texCoord;
	glm::vec3 normal;
	glm::vec3 tangent;

	Vertex() {};

	Vertex(glm::vec3 pos) {
		this->pos = pos;
		this->texCoord = glm::vec2(0.0f);
		this->normal = glm::vec3(0.0f);
		this->tangent = glm::vec3(0.0f);
	}


	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;

	}

	static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {

		//VAO
		std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);	//��pos��Vertex�е�ƫ��

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, texCoord);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, normal);

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Vertex, tangent);

		return attributeDescriptions;
	}

	bool operator==(const Vertex& other) const {
		return pos == other.pos && texCoord == other.texCoord && normal == other.normal && tangent == other.tangent;
	}

};

struct ComputeVertex {

	glm::vec4 pos;
	glm::vec4 normal;

	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(ComputeVertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;

	}

	static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {

		//VAO
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(ComputeVertex, pos);	//��pos��Vertex�е�ƫ��

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(ComputeVertex, normal);

		return attributeDescriptions;
	}

	bool operator==(const ComputeVertex& other) const {
		return pos == other.pos;
	}

};

struct Material {
	glm::vec4 bxdfPara;	//���ﱾ����ka�ģ��������޸�mtl�ļ������ֲڶȷŵ���ka.x������ֵ�ŵ���ka.y���������ʷŵ���ka.z���ֲڶȷŵ�
	glm::vec4 kd;
	glm::vec4 ks;
	glm::vec4 ke;
};

struct Texture {
	//uint32_t id;
	std::string type;
	std::string path;
};

struct AABBBox {

	float leftX;
	float rightX;
	float leftY;
	float rightY;
	float leftZ;
	float rightZ;

	float getAxis(int k) {
		if (k == 0) {
			return leftX;
		}
		else if (k == 1) {
			return rightX;
		}
		else if (k == 2) {
			return leftY;
		}
		else if (k == 3) {
			return rightY;
		}
		else if (k == 4) {
			return leftZ;
		}
		else if (k == 5) {
			return rightZ;
		}
	}

};

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	std::vector<Texture> textures;
	Material material;
	AABBBox AABB;

	Mesh(std::vector<Vertex> vertices, std::vector<uint32_t> indices, std::vector<Texture> textures, Material material) {
		this->vertices = vertices;
		this->indices = indices;
		this->textures = textures;
		this->material = material;
	}

};

struct UniformBufferObjectVoxel {
	glm::mat4 model;
	glm::mat4 VP[3];
	glm::vec4 voxelSize_Num;
	glm::vec4 voxelStartPos;
};

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec4 cameraPos;
	glm::vec4 randomNumber;
	glm::vec4 swapChainExtent;
};

struct UniformLightBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec4 lightPos_strength;
	glm::vec4 normal;
	glm::vec4 size;
};

struct DescriptorObject {

	VkDescriptorSetLayout discriptorLayout;
	uint32_t uniformBufferNum;
	uint32_t textureNum;
	std::vector<VkDescriptorSet> descriptorSets;
};

struct BvhTreeNode {

	BvhTreeNode* leftNode;
	BvhTreeNode* rightNode;
	AABBBox AABB;
	std::vector<uint32_t> meshIndex;

};

struct BvhArrayNode {

	int32_t leftNodeIndex;
	int32_t rightNodeIndex;
	AABBBox AABB;
	//����ֻ��עҶ�ӽڵ��mesh����Ϊ���ǹ�������scene��ײ��ֱ��bvhTree��Ҷ�ӽڵ㣬����ڵ��е�mesh��ײ����������ֻ��Ҫ��¼Ҷ�ӽڵ��meshIndex����
	//��Ҷ�ӽڵ����ֻ��2��mesh
	//���������
	int32_t meshIndex;
	//int32_t meshIndex2;

};

struct ComputeInputMesh {

	Material material;
	glm::ivec2 indexInIndicesArray;
	AABBBox AABB;
	//char padding[4];	//��Ϊmaterial����vec4����16�ֽڣ�������Ҫ��ǰComputeInputMesh��16�ֽڵı�����������һ��ComputeInputMesh���ܵõ���ȷ��ֵ

};

struct GMMConstant {

	glm::vec4 voxelStartPos;
	glm::ivec4 voxelNum;
	float voxelSize;
	uint32_t photonTracingNum;

};

struct GaussianPara {

	glm::vec2 mean;
	float mixWeight;
	float padding;
	glm::mat2 covarianceMatrix;

	GaussianPara() {
		mixWeight = 0.125f;
		mean = glm::vec2(0.0f);
		covarianceMatrix = glm::mat2(1.0f);
		padding = 0.0;
	}

};

struct Sufficient_Statistic {

	glm::vec2 ss2;	//��˹�ֲ��Ը������Ĺ��ױ�������������λ�õĳ˻�֮��
	float ss1;		//��˹�ֲ���ÿ�������Ĺ��ױ���֮��
	float padding;
	glm::mat2 ss3;	//������λ����������ת�õĳ˻����Ը�˹�ֲ��Ը������Ĺ��ױ����ĳ˻�֮��

	Sufficient_Statistic() {
		ss1 = 0.0f;
		ss2 = glm::vec2(0.0f);
		ss3 = glm::mat2(0.0f);
		padding = 0.0;
	}

};

//#define Use3DTexture
const int gmmNum = 6;
struct GMMPara {

	float photonAvgWeight;
	uint32_t photonNum;
	float padding1;
	float padding2;
	glm::vec4 pos;		//��Ϊstd430��vec3�������������ݵ�Ԫ�أ�����ʱ������߽�Ϊvec4���������������ﴫvec4������shader����vec3����
#ifndef Use3DTexture
	GaussianPara gaussianParas[gmmNum];
#endif
	Sufficient_Statistic SSs[gmmNum];


	GMMPara() {
		pos = glm::vec4(0.0f);
		for (int i = 0; i < gmmNum; i++) {
#ifndef Use3DTexture
			gaussianParas[i] = GaussianPara();
#endif
			SSs[i] = Sufficient_Statistic();
		}
		photonAvgWeight = 0.0f;
		photonNum = 0.0f;
		padding1 = 0.0f;
		padding2 = 0.0f;
	}

};

struct Photon {

	glm::vec2 direction[gmmNum];	//����ά����תΪ2ά����ͬ��Բӳ��
	float weight;	//���� = 0.299f * R + 0.587f * G + 0.114f * B
	float padding1;
	float padding2;
	float padding3;
	glm::vec4 direction_3D;
	glm::vec4 hitPos;
	glm::vec4 startPos;

	Photon() {
		padding1 = 0.0f;
		padding2 = 0.0f;
		padding3 = 0.0f;
		hitPos = glm::vec4(0.0f);
		startPos = glm::vec4(0.0f);
		direction_3D = glm::vec4(0.0f);
		for (int i = 0; i < gmmNum; i++) {
			direction[i] = glm::vec2(0.0f);
		}
		weight = 0.0f;
	}

};

//��ʵ����Ҫ���Ĳ��֣���Ϊ���Ǵ��Ķ���0��������������shader���޸ĵ�
struct PhotonTracingResult {
	uint32_t photonNum;
	float padding1;
	float padding2;
	float padding3;
	Photon photons[10];

	PhotonTracingResult() {
		this->photonNum = 0;
		for (int i = 0; i < 10; i++) {
			photons[0] = Photon();
		}
		padding1 = 0.0f;
		padding2 = 0.0f;
		padding3 = 0.0f;
	}

};

#endif