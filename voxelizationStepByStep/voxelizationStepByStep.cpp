#include <chrono>
#include<stdexcept>
#include<functional>
#include<cstdlib>
#include<cstdint>
#include<limits>
#include<fstream>
#include <random>

#include "structSet.h"
#include "myDevice.h"
#include "myBuffer.h"
#include "myImage.h"
#include "mySwapChain.h"
#include "myModel.h"
#include "myCamera.h"
#include "myDescriptor.h"
#include "myScene.h"
#include "myRay.h"

const uint32_t WIDTH = 512;
const uint32_t HEIGHT = 512;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

//如果不调试，则关闭校验层
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

#define Voxelization_Block

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebygMessenger) {
	//由于是扩展函数，所以需要通过vkGetInstanceProcAddr获得该函数指针
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebygMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

myCamera camera(glm::vec3(0.0f, 6.0f, 15.0f));
float lastTime = 0.0f;
float deltaTime = 0.0f;
bool firstMouse = true;
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
	float xpos = static_cast<float>(xposIn);
	float ypos = static_cast<float>(yposIn);

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

class Tessellation {

public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:

	GLFWwindow* window;		//窗口

	VkInstance instance;	//vulkan实例
	VkDebugUtilsMessengerEXT debugMessenger;	//消息传递者
	VkSurfaceKHR surface;

	//Device
	std::unique_ptr<myDevice> my_device;

	//SwapChain
	std::unique_ptr<mySwapChain> my_swapChain;

	std::unique_ptr<myModel> my_model;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
#ifdef Voxelization_Block
	std::vector<Vertex> vertices_cubes;
	std::vector<uint32_t> indices_cubes;
#endif

	std::unique_ptr<myScene> my_scene;

	std::unique_ptr<myImage> voxelMap;
	const uint32_t voxelNum = 64;
	std::unique_ptr<myImage> depthMap;

	std::unique_ptr<myBuffer> my_buffer;

	std::unique_ptr<myDescriptor> my_descriptor;

	VkRenderPass renderPass;
	VkPipeline voxelGraphicsPipeline;
	VkPipelineLayout voxelGraphicsPipelineLayout;
	VkPipeline finalGraphicsPipeline;
	VkPipelineLayout finalGraphicsPipelineLayout;
	VkPipeline computePipeline;
	VkPipelineLayout computePipelineLayout;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	uint32_t currentFrame = 0;
	int frameNum = 0;

	bool framebufferResized = false;

	void initWindow() {

		glfwInit();

		//阻止GLFW自动创建OpenGL上下文
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		//是否禁止改变窗口大小
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		//glfwSetFramebufferSizeCallback函数在回调时，需要为我们设置framebufferResized，但他不知道我是谁
		//所以通过对window设置我是谁，从而让回调函数知道我是谁
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwSetCursorPosCallback(window, mouse_callback);
		glfwSetScrollCallback(window, scroll_callback);

	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {

		auto app = reinterpret_cast<Tessellation*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;

	}

	void initVulkan() {

		createInstance();
		setupDebugMessenger();
		createSurface();
		createMyDevice();
		createMySwapChain();
		createMyBuffer();
		loadModel();
		createTextureResources();
		createBuffers();
		createRenderPass();
		createFramebuffers();
		createMyDescriptor();
		createGraphicsPipeline();
		createSyncObjects();

	}

	void createInstance() {

		//检测layer
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		//扩展就是Vulkan本身没有实现，但被程序员封装后的功能函数，如跨平台的各种函数，把它当成普通函数即可，别被名字唬到了
		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();	//将扩展的具体信息的指针存储在该结构体中

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();	//将校验层的具体信息的指针存储在该结构体中

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;

		}
		else {
			createInfo.enabledLayerCount = 0;
			createInfo.pNext = nullptr;
		}


		//VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}

	}

	bool checkValidationLayerSupport() {

		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);	//返回可用的层数
		std::vector<VkLayerProperties> availableLayers(layerCount);	//VkLayerProperties是一个结构体，记录层的名字、描述等
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {

			bool layerFound = false;
			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}

		}

		return true;
	}

	std::vector<const char*> getRequiredExtensions() {

		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);	//得到glfw所需的扩展数
		//参数1是指针起始位置，参数2是指针终止位置
		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);	//这个扩展是为了打印校验层反映的错误，所以需要知道是否需要校验层
			extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
		}

		return extensions;
	}

	void setupDebugMessenger() {

		if (!enableValidationLayers)
			return;
		VkDebugUtilsMessengerCreateInfoEXT  createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		//通过func的构造函数给debugMessenger赋值
		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}

	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
		return VK_FALSE;
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
		createInfo.pUserData = nullptr;
	}

	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface");
		}
	}

	void createMyDevice() {
		my_device = std::make_unique<myDevice>(instance, surface);
		my_device->pickPhysicalDevice();
		my_device->createLogicalDevice(enableValidationLayers, validationLayers);

	}

	void createMySwapChain() {
		my_swapChain = std::make_unique<mySwapChain>(window, surface, my_device->logicalDevice, my_device->swapChainSupportDetails, my_device->queueFamilyIndices);
	}

	void createMyBuffer() {
		my_buffer = std::make_unique<myBuffer>();
		my_buffer->createCommandPool(my_device->logicalDevice, my_device->queueFamilyIndices);
		my_buffer->createCommandBuffers(my_device->logicalDevice, MAX_FRAMES_IN_FLIGHT);
	}

	void loadModel() {
		//使用assimp
		uint32_t index = 0;
		//my_model = std::make_unique<myModel>("models/CornellBox-Original.obj");	//给绝对路径读不到，给相对路径能读到
		my_model = std::make_unique<myModel>("models/dragon.obj");
		if (my_model->meshs.size() == 0) {
			throw std::runtime_error("failed to create model!");
		}

		//将所有mesh的顶点合并
		for (uint32_t i = 0; i < my_model->meshs.size(); i++) {

			//this->materials.push_back(my_model->meshs[i].material);

			this->vertices.insert(this->vertices.end(), my_model->meshs[i].vertices.begin(), my_model->meshs[i].vertices.end());

			//因为assimp是按一个mesh一个mesh的存，所以每个indices都是相对一个mesh的，当我们将每个mesh的顶点存到一起时，indices就会出错，我们需要增加索引
			for (uint32_t j = 0; j < my_model->meshs[i].indices.size(); j++) {
				my_model->meshs[i].indices[j] += index;
			}
			index += my_model->meshs[i].vertices.size();
			this->indices.insert(this->indices.end(), my_model->meshs[i].indices.begin(), my_model->meshs[i].indices.end());
		}

		my_scene = std::make_unique<myScene>(&my_model->meshs);

#ifdef Voxelization_Block

		glm::vec3 voxelStartPos = glm::vec3(my_scene->bvhArray[0].AABB.leftX, my_scene->bvhArray[0].AABB.leftY, my_scene->bvhArray[0].AABB.leftZ);
		glm::vec3 cubeVertexOffset[8] = { glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f),
										  glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.0f, 1.0f, 1.0f) };
		glm::uint32_t cubeIndices[36] = {
			1, 0, 3, 1, 3, 2,
			4, 5, 6, 4, 6, 7,
			5, 1, 2, 5, 2, 6,
			0, 4, 7, 0, 7, 3,
			7, 6, 2, 7, 2, 3,
			0, 1, 5, 0, 5, 4
		};


		float distanceX = my_scene->bvhArray[0].AABB.rightX - my_scene->bvhArray[0].AABB.leftX;
		float distanceY = my_scene->bvhArray[0].AABB.rightY - my_scene->bvhArray[0].AABB.leftY;
		float distanceZ = my_scene->bvhArray[0].AABB.rightZ - my_scene->bvhArray[0].AABB.leftZ;
		glm::vec3 distanceXYZ = glm::vec3(distanceX, distanceY, distanceZ);
		float maxDistance = std::max(distanceX, std::max(distanceY, distanceZ));
		float voxelSize = maxDistance / voxelNum;

		for (int z = 0; z < voxelNum; z++) {
			for (int y = 0; y < voxelNum; y++) {
				for (int x = 0; x < voxelNum; x++) {
					
					for (int i = 0; i < 8; i++) {
						Vertex vertex = Vertex(voxelStartPos + glm::vec3(x, y, z) * voxelSize + voxelSize * cubeVertexOffset[i]);
						this->vertices_cubes.push_back(vertex);
					}

					uint32_t voxelIndex = (z * voxelNum * voxelNum + y * voxelNum + x) * 8;
					for (int i = 0; i < 36; i++) {
						uint32_t index = voxelIndex + cubeIndices[i];
						this->indices_cubes.push_back(index);
					}

				}
			}
		}

		std::unordered_map<Vertex, uint32_t> uniqueVerticesMap{};
		std::vector<Vertex> uniqueVertices;
		std::vector<uint32_t> uniqueIndices;
		for (uint32_t j = 0; j < this->indices_cubes.size(); j++) {
			Vertex vertex = this->vertices_cubes[indices_cubes[j]];
			if (uniqueVerticesMap.count(vertex) == 0) {
				uniqueVerticesMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
				uniqueVertices.push_back(vertex);
			}
			uniqueIndices.push_back(uniqueVerticesMap[vertex]);
		}
		this->vertices_cubes = uniqueVertices;
		this->indices_cubes = uniqueIndices;

		//this->vertices_cubes.resize(8);
		//this->indices_cubes.resize(36);
		//for (int i = 0; i < 8; i++) {
		//	this->vertices_cubes.push_back(Vertex(voxelStartPos + cubeVertexOffset[i] * distanceXYZ));
		//}
		//for (int i = 0; i < 36; i++) {
		//	this->indices_cubes.push_back(cubeIndices[i]);
		//}


#endif

	}

	void createTextureResources() {

		//heightMap = std::make_unique<myImage>("../../images/iceland_heightmap.png", my_device->physicalDevice, my_device->logicalDevice, my_device->graphicsQueue, my_buffer->commandPool, false);
		voxelMap = std::make_unique<myImage>(my_device->physicalDevice, my_device->logicalDevice, voxelNum, voxelNum, voxelNum,
			1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R32_UINT, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
		//voxelMap->transitionImageLayout(my_device->computeQueue, my_buffer->commandPool, voxelMap->image, VK_FORMAT_R32_UINT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 1);

		depthMap = std::make_unique<myImage>(my_device->physicalDevice, my_device->logicalDevice, my_swapChain->swapChainExtent.width, my_swapChain->swapChainExtent.height,
			1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R32_UINT, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
		//depthMap->transitionImageLayout(my_device->computeQueue, my_buffer->commandPool, depthMap->image, VK_FORMAT_R32_UINT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 1);

	}

	//简单一点，静态场景
	void createBuffers() {

		my_buffer->createStaticBuffer(my_device->physicalDevice, my_device->logicalDevice, my_device->graphicsQueue, sizeof(Vertex) * this->vertices.size(), &this->vertices);
		my_buffer->createStaticBuffer(my_device->physicalDevice, my_device->logicalDevice, my_device->graphicsQueue, sizeof(uint32_t) * this->indices.size(), &this->indices);

#ifdef Voxelization_Block
		my_buffer->createStaticBuffer(my_device->physicalDevice, my_device->logicalDevice, my_device->graphicsQueue, sizeof(Vertex) * this->vertices_cubes.size(), &this->vertices_cubes);
		my_buffer->createStaticBuffer(my_device->physicalDevice, my_device->logicalDevice, my_device->graphicsQueue, sizeof(uint32_t) * this->indices_cubes.size(), &this->indices_cubes);
#endif

		//相机MVP
		my_buffer->createUniformBuffers(my_device->physicalDevice, my_device->logicalDevice, MAX_FRAMES_IN_FLIGHT, sizeof(UniformBufferObject), false);

		//voxel多个面的投影
		my_buffer->createUniformBuffers(my_device->physicalDevice, my_device->logicalDevice, MAX_FRAMES_IN_FLIGHT, sizeof(UniformBufferObjectVoxel), true);
		UniformBufferObjectVoxel voxelUniformBufferObject{};
		voxelUniformBufferObject.model = glm::mat4(1.0f);

		float centerX = (my_scene->bvhArray[0].AABB.rightX + my_scene->bvhArray[0].AABB.leftX) * 0.5f;
		float centerY = (my_scene->bvhArray[0].AABB.rightY + my_scene->bvhArray[0].AABB.leftY) * 0.5f;
		float centerZ = (my_scene->bvhArray[0].AABB.rightZ + my_scene->bvhArray[0].AABB.leftZ) * 0.5f;

		float distanceX = my_scene->bvhArray[0].AABB.rightX - my_scene->bvhArray[0].AABB.leftX;
		float distanceY = my_scene->bvhArray[0].AABB.rightY - my_scene->bvhArray[0].AABB.leftY;
		float distanceZ = my_scene->bvhArray[0].AABB.rightZ - my_scene->bvhArray[0].AABB.leftZ;

		glm::vec3 viewPoint = glm::vec3(my_scene->bvhArray[0].AABB.rightX + 0.1f, centerY, centerZ);
		voxelUniformBufferObject.VP[0] = glm::lookAt(viewPoint, viewPoint + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		voxelUniformBufferObject.VP[0] = glm::ortho(-0.51f * distanceZ, 0.51f * distanceZ, -0.51f * distanceY, 0.51f * distanceY, 0.1f, 100.0f) * voxelUniformBufferObject.VP[0];

		viewPoint = glm::vec3(centerX, my_scene->bvhArray[0].AABB.rightY + 0.1f, centerZ);
		voxelUniformBufferObject.VP[1] = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
		voxelUniformBufferObject.VP[1] = glm::ortho(-0.51f * distanceX, 0.51f * distanceX, -0.51f * distanceZ, 0.51f * distanceZ, 0.1f, 100.0f) * voxelUniformBufferObject.VP[1];

		viewPoint = glm::vec3(centerX, centerY, my_scene->bvhArray[0].AABB.rightZ + 0.1f);
		voxelUniformBufferObject.VP[2] = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		voxelUniformBufferObject.VP[2] = glm::ortho(-0.51f * distanceX, 0.51f * distanceX, -0.51f * distanceY, 0.51f * distanceY, 0.1f, 100.0f) * voxelUniformBufferObject.VP[2];

		float maxDistance = std::max(distanceX, std::max(distanceY, distanceZ));
		voxelUniformBufferObject.voxelSize_Num = glm::vec4(maxDistance / voxelNum, voxelNum, 0.0f, 0.0f);
		voxelUniformBufferObject.voxelStartPos = glm::vec4(my_scene->bvhArray[0].AABB.leftX, my_scene->bvhArray[0].AABB.leftY, my_scene->bvhArray[0].AABB.leftZ, 0.0f);

		memcpy(my_buffer->uniformBuffersMappedsStatic[0], &voxelUniformBufferObject, sizeof(UniformBufferObjectVoxel));

	}

	void createRenderPass() {

		VkAttachmentDescription colorAttachmentResolve{};
		colorAttachmentResolve.format = my_swapChain->swapChainImageFormat;
		colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentResolveRef{};
		colorAttachmentResolveRef.attachment = 0;
		colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		/*
#ifdef Voxelization_Block

		VkAttachmentDescription depthAttachment{};
		depthAttachment.format = VK_FORMAT_R32_UINT;
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		std::array< VkAttachmentDescription, 2> attachements = { colorAttachmentResolve,  depthAttachment };

		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription final_subpass{};
		final_subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		final_subpass.colorAttachmentCount = 1;
		final_subpass.pColorAttachments = &colorAttachmentResolveRef;
		final_subpass.pDepthStencilAttachment = &depthAttachmentRef;

#else
		std::array< VkAttachmentDescription, 1> attachements = { colorAttachmentResolve };

		VkSubpassDescription final_subpass{};
		final_subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		final_subpass.colorAttachmentCount = 1;
		final_subpass.pColorAttachments = &colorAttachmentResolveRef;

#endif
*/
		std::array< VkAttachmentDescription, 1> attachements = { colorAttachmentResolve };

		VkSubpassDescription final_subpass{};
		final_subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		final_subpass.colorAttachmentCount = 1;
		final_subpass.pColorAttachments = &colorAttachmentResolveRef;

		VkSubpassDescription voxel_subpass{};
		voxel_subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		
		std::array< VkSubpassDescription, 2 > subpasses = { voxel_subpass, final_subpass };

		VkSubpassDependency dependency{};
		dependency.srcSubpass = 0;
		dependency.dstSubpass = 1;
		dependency.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependency.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		dependency.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependency.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = attachements.size();
		renderPassInfo.pAttachments = attachements.data();
		renderPassInfo.subpassCount = subpasses.size();
		renderPassInfo.pSubpasses = subpasses.data();
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(my_device->logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

	}

	void createFramebuffers() {
		std::vector<VkImageView> imageViews;
		my_buffer->createFramebuffers(my_swapChain->swapChainImageViews.size(), my_swapChain->swapChainImageViews, my_swapChain->extent, imageViews, nullptr, renderPass, my_device->logicalDevice);
		/*
#ifdef Voxelization_Block
		my_buffer->createFramebuffers(my_swapChain->swapChainImageViews.size(), my_swapChain->swapChainImageViews, my_swapChain->extent, imageViews, depthMap->imageView, renderPass, my_device->logicalDevice);
#else
		my_buffer->createFramebuffers(my_swapChain->swapChainImageViews.size(), my_swapChain->swapChainImageViews, my_swapChain->extent, imageViews, nullptr, renderPass, my_device->logicalDevice);
#endif
		*/
	}

	void createMyDescriptor() {

		my_descriptor = std::make_unique<myDescriptor>(my_device->logicalDevice, MAX_FRAMES_IN_FLIGHT);

		uint32_t uniformBufferNumAllLayout = 2;
		uint32_t storageBufferNumAllLayout = 0;
		std::vector<uint32_t> textureNumAllLayout = { 4 };
		std::vector<VkDescriptorType> types = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE };
		my_descriptor->createDescriptorPool(uniformBufferNumAllLayout, storageBufferNumAllLayout, types, textureNumAllLayout);

		std::vector<VkShaderStageFlagBits> usages = { VK_SHADER_STAGE_ALL };
		// 创造相机uniformDescriptorObject
		std::vector<VkBuffer> uniformBuffers = { my_buffer->uniformBuffers[0] };
		std::vector<std::vector<VkBuffer>> uniformBuffersAllSet = { uniformBuffers };
		std::vector<VkDeviceSize> bufferSize = { sizeof(UniformBufferObject) };
		my_descriptor->descriptorObjects.push_back(my_descriptor->createDescriptorObject(1, 0, &usages, nullptr, 1, &uniformBuffersAllSet, bufferSize, nullptr, nullptr));

		//创建voxel多个面的uniformBuffer
		uniformBuffers = { my_buffer->uniformBuffersStatic[0] };
		uniformBuffersAllSet = { uniformBuffers };
		bufferSize = { sizeof(UniformBufferObjectVoxel) };
		my_descriptor->descriptorObjects.push_back(my_descriptor->createDescriptorObject(1, 0, &usages, nullptr, 1, &uniformBuffersAllSet, bufferSize, nullptr, nullptr));

		//---------------------------------------------voxelization-------------------------------------------------

		VkDescriptorSetLayout voxelTextureDescriptorSetLayout;
		std::array<VkDescriptorSetLayoutBinding, 2> voxel_layoutBindings{};
		voxel_layoutBindings[0].binding = 0;
		voxel_layoutBindings[0].descriptorCount = 1;
		voxel_layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		voxel_layoutBindings[0].pImmutableSamplers = nullptr;
		voxel_layoutBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		voxel_layoutBindings[1].binding = 1;
		voxel_layoutBindings[1].descriptorCount = 1;
		voxel_layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		voxel_layoutBindings[1].pImmutableSamplers = nullptr;
		voxel_layoutBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo voxel_layoutInfo{};
		voxel_layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		voxel_layoutInfo.bindingCount = voxel_layoutBindings.size();
		voxel_layoutInfo.pBindings = voxel_layoutBindings.data();
		if (vkCreateDescriptorSetLayout(my_device->logicalDevice, &voxel_layoutInfo, nullptr, &voxelTextureDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}

		VkDescriptorSetAllocateInfo voxel_allocInfo{};
		voxel_allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		voxel_allocInfo.descriptorPool = my_descriptor->discriptorPool;
		voxel_allocInfo.descriptorSetCount = 1;
		voxel_allocInfo.pSetLayouts = &voxelTextureDescriptorSetLayout;

		VkDescriptorSet voxel_descriptorSet;
		if (vkAllocateDescriptorSets(my_device->logicalDevice, &voxel_allocInfo, &voxel_descriptorSet) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		std::array<VkWriteDescriptorSet, 2> voxel_descriptorWrites{};
		VkDescriptorImageInfo depthMapInfo{};
		depthMapInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		depthMapInfo.imageView = depthMap->imageView;
		depthMapInfo.sampler = depthMap->textureSampler;
		voxel_descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		voxel_descriptorWrites[0].dstSet = voxel_descriptorSet;
		voxel_descriptorWrites[0].dstBinding = 0;
		voxel_descriptorWrites[0].dstArrayElement = 0;
		voxel_descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		voxel_descriptorWrites[0].descriptorCount = 1;
		voxel_descriptorWrites[0].pImageInfo = &depthMapInfo;

		VkDescriptorImageInfo voxelMapInfo{};
		voxelMapInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		voxelMapInfo.imageView = voxelMap->imageView;
		voxelMapInfo.sampler = voxelMap->textureSampler;
		voxel_descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		voxel_descriptorWrites[1].dstSet = voxel_descriptorSet;
		voxel_descriptorWrites[1].dstBinding = 1;
		voxel_descriptorWrites[1].dstArrayElement = 0;
		voxel_descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		voxel_descriptorWrites[1].descriptorCount = 1;
		voxel_descriptorWrites[1].pImageInfo = &voxelMapInfo;

		vkUpdateDescriptorSets(my_device->logicalDevice, voxel_descriptorWrites.size(), voxel_descriptorWrites.data(), 0, nullptr);

		DescriptorObject voxelDescriptorObject{};
		voxelDescriptorObject.discriptorLayout = voxelTextureDescriptorSetLayout;
		voxelDescriptorObject.descriptorSets.push_back(voxel_descriptorSet);
		my_descriptor->descriptorObjects.push_back(voxelDescriptorObject);

		//------------------------------------------present---------------------------------------

		VkDescriptorSetLayout presentTextureDescriptorSetLayout;
		/*
#ifdef Voxelization_Block
		std::array<VkDescriptorSetLayoutBinding, 1> present_layoutBindings{};
		present_layoutBindings[0].binding = 0;
		present_layoutBindings[0].descriptorCount = 1;
		present_layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		present_layoutBindings[0].pImmutableSamplers = nullptr;
		present_layoutBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

#else
		std::array<VkDescriptorSetLayoutBinding, 2> present_layoutBindings{};
		present_layoutBindings[0].binding = 0;
		present_layoutBindings[0].descriptorCount = 1;
		present_layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		present_layoutBindings[0].pImmutableSamplers = nullptr;
		present_layoutBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		present_layoutBindings[1].binding = 1;
		present_layoutBindings[1].descriptorCount = 1;
		present_layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		present_layoutBindings[1].pImmutableSamplers = nullptr;
		present_layoutBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
#endif
		*/
		std::array<VkDescriptorSetLayoutBinding, 2> present_layoutBindings{};
		present_layoutBindings[0].binding = 0;
		present_layoutBindings[0].descriptorCount = 1;
		present_layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		present_layoutBindings[0].pImmutableSamplers = nullptr;
		present_layoutBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		present_layoutBindings[1].binding = 1;
		present_layoutBindings[1].descriptorCount = 1;
		present_layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		present_layoutBindings[1].pImmutableSamplers = nullptr;
		present_layoutBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		
		VkDescriptorSetLayoutCreateInfo present_layoutInfo{};
		present_layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		present_layoutInfo.bindingCount = present_layoutBindings.size();
		present_layoutInfo.pBindings = present_layoutBindings.data();
		if (vkCreateDescriptorSetLayout(my_device->logicalDevice, &present_layoutInfo, nullptr, &presentTextureDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}

		VkDescriptorSetAllocateInfo present_allocInfo{};
		present_allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		present_allocInfo.descriptorPool = my_descriptor->discriptorPool;
		present_allocInfo.descriptorSetCount = 1;
		present_allocInfo.pSetLayouts = &presentTextureDescriptorSetLayout;

		VkDescriptorSet present_descriptorSet;
		if (vkAllocateDescriptorSets(my_device->logicalDevice, &present_allocInfo, &present_descriptorSet) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		/*
#ifdef Voxelization_Block
		std::array<VkWriteDescriptorSet, 1> present_descriptorWrites{};
		VkDescriptorImageInfo present_depthMapInfo{};
		present_depthMapInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		present_depthMapInfo.imageView = voxelMap->imageView;
		present_depthMapInfo.sampler = voxelMap->textureSampler;
		present_descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		present_descriptorWrites[0].dstSet = present_descriptorSet;
		present_descriptorWrites[0].dstBinding = 0;
		present_descriptorWrites[0].dstArrayElement = 0;
		present_descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		present_descriptorWrites[0].descriptorCount = 1;
		present_descriptorWrites[0].pImageInfo = &present_depthMapInfo;
#else
		std::array<VkWriteDescriptorSet, 2> present_descriptorWrites{};
		VkDescriptorImageInfo present_depthMapInfo{};
		present_depthMapInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		present_depthMapInfo.imageView = depthMap->imageView;
		present_depthMapInfo.sampler = depthMap->textureSampler;
		present_descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		present_descriptorWrites[0].dstSet = present_descriptorSet;
		present_descriptorWrites[0].dstBinding = 0;
		present_descriptorWrites[0].dstArrayElement = 0;
		present_descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		present_descriptorWrites[0].descriptorCount = 1;
		present_descriptorWrites[0].pImageInfo = &present_depthMapInfo;

		VkDescriptorImageInfo present_voxelMapInfo{};
		present_voxelMapInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		present_voxelMapInfo.imageView = voxelMap->imageView;
		present_voxelMapInfo.sampler = voxelMap->textureSampler;
		present_descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		present_descriptorWrites[1].dstSet = present_descriptorSet;
		present_descriptorWrites[1].dstBinding = 1;
		present_descriptorWrites[1].dstArrayElement = 0;
		present_descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		present_descriptorWrites[1].descriptorCount = 1;
		present_descriptorWrites[1].pImageInfo = &present_voxelMapInfo;
#endif
		*/
		std::array<VkWriteDescriptorSet, 2> present_descriptorWrites{};
		VkDescriptorImageInfo present_depthMapInfo{};
		present_depthMapInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		present_depthMapInfo.imageView = depthMap->imageView;
		present_depthMapInfo.sampler = depthMap->textureSampler;
		present_descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		present_descriptorWrites[0].dstSet = present_descriptorSet;
		present_descriptorWrites[0].dstBinding = 0;
		present_descriptorWrites[0].dstArrayElement = 0;
		present_descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		present_descriptorWrites[0].descriptorCount = 1;
		present_descriptorWrites[0].pImageInfo = &present_depthMapInfo;

		VkDescriptorImageInfo present_voxelMapInfo{};
		present_voxelMapInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		present_voxelMapInfo.imageView = voxelMap->imageView;
		present_voxelMapInfo.sampler = voxelMap->textureSampler;
		present_descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		present_descriptorWrites[1].dstSet = present_descriptorSet;
		present_descriptorWrites[1].dstBinding = 1;
		present_descriptorWrites[1].dstArrayElement = 0;
		present_descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		present_descriptorWrites[1].descriptorCount = 1;
		present_descriptorWrites[1].pImageInfo = &present_voxelMapInfo;

		vkUpdateDescriptorSets(my_device->logicalDevice, present_descriptorWrites.size(), present_descriptorWrites.data(), 0, nullptr);

		DescriptorObject presentDescriptorObject{};
		presentDescriptorObject.discriptorLayout = presentTextureDescriptorSetLayout;
		presentDescriptorObject.descriptorSets.push_back(present_descriptorSet);
		my_descriptor->descriptorObjects.push_back(presentDescriptorObject);


	}

	void createGraphicsPipeline() {

		auto voxelVertShaderCode = readFile("C:/Users/fangzanbo/Desktop/渲染/GPU/新功能/voxelizationStepByStep/voxelizationStepByStep/shaders/voxelVert.spv");
		auto voxelGeomShaderCode = readFile("C:/Users/fangzanbo/Desktop/渲染/GPU/新功能/voxelizationStepByStep/voxelizationStepByStep/shaders/voxelGeom.spv");
		auto voxelFragShaderCode = readFile("C:/Users/fangzanbo/Desktop/渲染/GPU/新功能/voxelizationStepByStep/voxelizationStepByStep/shaders/voxelFrag.spv");

		VkShaderModule voxelVertShaderModule = createShaderModule(voxelVertShaderCode);
		VkShaderModule voxelGeomShaderModule = createShaderModule(voxelGeomShaderCode);
		VkShaderModule voxelFragShaderModule = createShaderModule(voxelFragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = voxelVertShaderModule;
		vertShaderStageInfo.pName = "main";
		vertShaderStageInfo.pSpecializationInfo = nullptr;

		VkPipelineShaderStageCreateInfo geomShaderStageInfo{};
		geomShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		geomShaderStageInfo.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
		geomShaderStageInfo.module = voxelGeomShaderModule;
		geomShaderStageInfo.pName = "main";
		geomShaderStageInfo.pSpecializationInfo = nullptr;

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = voxelFragShaderModule;
		fragShaderStageInfo.pName = "main";
		fragShaderStageInfo.pSpecializationInfo = nullptr;

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, geomShaderStageInfo, fragShaderStageInfo };

		//VAO
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		//设置渲染图元方式
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		//设置视口
		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		//设置光栅化器，主要是深度测试等的开关、面剔除等
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_NONE;	//VK_CULL_MODE_FRONT_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeState{};
		conservativeState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
		conservativeState.pNext = NULL;
		conservativeState.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
		conservativeState.extraPrimitiveOverestimationSize = 0.0f; // 根据需要设置
		
		rasterizer.pNext = &conservativeState;

		//多采样
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;// .2f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = nullptr;	// &colorBlendAttachment;

		//即使没有需要输出的texture，也要写一个VkPipelineColorBlendAttachmentState放入colorBlending.pAttachments，否则release版本会在创建管线时报出空指针异常
		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optiona
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_FALSE;
		depthStencil.depthWriteEnable = VK_FALSE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.minDepthBounds = 0.0f; // Optional
		depthStencil.maxDepthBounds = 1.0f; // Optional
		depthStencil.stencilTestEnable = VK_FALSE;
		depthStencil.front = {}; // Optional
		depthStencil.back = {}; // Optional

		//一般渲染管道状态都是固定的，不能渲染循环中修改，但是某些状态可以，如视口，长宽和混合常数
		//同样通过宏来确定可动态修改的状态
		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		//pipeline布局
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		std::array<VkDescriptorSetLayout, 3> voxelDescriptorLayouts = { my_descriptor->descriptorObjects[0].discriptorLayout, my_descriptor->descriptorObjects[1].discriptorLayout,  my_descriptor->descriptorObjects[2].discriptorLayout };
		pipelineLayoutInfo.setLayoutCount = voxelDescriptorLayouts.size();
		pipelineLayoutInfo.pSetLayouts = voxelDescriptorLayouts.data();
		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr;

		if (vkCreatePipelineLayout(my_device->logicalDevice, &pipelineLayoutInfo, nullptr, &voxelGraphicsPipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 3;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = voxelGraphicsPipelineLayout;
		pipelineInfo.renderPass = renderPass;	//先建立连接，获得索引
		pipelineInfo.subpass = 0;	//对应renderpass的哪个子部分
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	//可以直接使用现有pipeline
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(my_device->logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &voxelGraphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(my_device->logicalDevice, voxelVertShaderModule, nullptr);
		vkDestroyShaderModule(my_device->logicalDevice, voxelGeomShaderModule, nullptr);
		vkDestroyShaderModule(my_device->logicalDevice, voxelFragShaderModule, nullptr);

		//final图形管线
#ifdef Voxelization_Block
		auto finalVertShaderCode = readFile("C:/Users/fangzanbo/Desktop/渲染/GPU/新功能/voxelizationStepByStep/voxelizationStepByStep/shaders/finalVert_Block.spv");
		auto finalFragShaderCode = readFile("C:/Users/fangzanbo/Desktop/渲染/GPU/新功能/voxelizationStepByStep/voxelizationStepByStep/shaders/finalFrag_Block.spv");
#else
		auto finalVertShaderCode = readFile("C:/Users/fangzanbo/Desktop/渲染/GPU/新功能/voxelizationStepByStep/voxelizationStepByStep/shaders/finalVert.spv");
		auto finalFragShaderCode = readFile("C:/Users/fangzanbo/Desktop/渲染/GPU/新功能/voxelizationStepByStep/voxelizationStepByStep/shaders/finalFrag.spv");
#endif
		VkShaderModule finalVertShaderModule = createShaderModule(finalVertShaderCode);
		VkShaderModule finalFragShaderModule = createShaderModule(finalFragShaderCode);

		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = finalVertShaderModule;
		vertShaderStageInfo.pName = "main";
		vertShaderStageInfo.pSpecializationInfo = nullptr;

		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = finalFragShaderModule;
		fragShaderStageInfo.pName = "main";
		fragShaderStageInfo.pSpecializationInfo = nullptr;

		VkPipelineShaderStageCreateInfo final_shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optiona
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;

		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		/*
#ifdef Voxelization_Block
		depthStencil.depthTestEnable = VK_TRUE;
#else
		depthStencil.depthTestEnable = VK_FALSE;
#endif
*/
		depthStencil.depthTestEnable = VK_FALSE;
		depthStencil.depthWriteEnable = VK_FALSE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.minDepthBounds = 0.0f; // Optional
		depthStencil.maxDepthBounds = 1.0f; // Optional
		depthStencil.stencilTestEnable = VK_FALSE;
		depthStencil.front = {}; // Optional
		depthStencil.back = {}; // Optional

		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;	//顶点序列是逆时针的是正面
		rasterizer.pNext = nullptr;

		VkPipelineLayoutCreateInfo finalPipelineLayoutInfo{};
		finalPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		std::array<VkDescriptorSetLayout, 3> finalDescriptorLayouts = { my_descriptor->descriptorObjects[0].discriptorLayout, my_descriptor->descriptorObjects[1].discriptorLayout, my_descriptor->descriptorObjects[3].discriptorLayout };
		finalPipelineLayoutInfo.setLayoutCount = finalDescriptorLayouts.size();
		finalPipelineLayoutInfo.pSetLayouts = finalDescriptorLayouts.data();
		finalPipelineLayoutInfo.pushConstantRangeCount = 0;
		finalPipelineLayoutInfo.pPushConstantRanges = nullptr;
		if (vkCreatePipelineLayout(my_device->logicalDevice, &finalPipelineLayoutInfo, nullptr, &finalGraphicsPipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = final_shaderStages;
		pipelineInfo.layout = finalGraphicsPipelineLayout;
		pipelineInfo.subpass = 1;
		if (vkCreateGraphicsPipelines(my_device->logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &finalGraphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(my_device->logicalDevice, finalVertShaderModule, nullptr);
		vkDestroyShaderModule(my_device->logicalDevice, finalFragShaderModule, nullptr);

	}

	/*
	void createComputePipeline() {

		auto computeShaderCode = readFile("C:/Users/fangzanbo/Desktop/渲染/GPU/新功能/voxelizationStepByStep/voxelizationStepByStep/shaders/voxelization.spv");
		VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

		VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
		computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		computeShaderStageInfo.module = computeShaderModule;
		computeShaderStageInfo.pName = "main";

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 3;
		std::array<VkDescriptorSetLayout, 3> computeDescriptorSetLayouts = { my_descriptor->descriptorObjects[0].discriptorLayout, my_descriptor->descriptorObjects[1].discriptorLayout, my_descriptor->descriptorObjects[3].discriptorLayout };
		pipelineLayoutInfo.pSetLayouts = computeDescriptorSetLayouts.data();
		if (vkCreatePipelineLayout(my_device->logicalDevice, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute pipeline layout!");
		}

		VkComputePipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineInfo.layout = computePipelineLayout;
		pipelineInfo.stage = computeShaderStageInfo;
		if (vkCreateComputePipelines(my_device->logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute pipeline!");
		}

		vkDestroyShaderModule(my_device->logicalDevice, computeShaderModule, nullptr);

	}
	*/

	VkShaderModule createShaderModule(const std::vector<char>& code) {

		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(my_device->logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;

	}

	static std::vector<char> readFile(const std::string& filename) {

		std::ifstream file(filename, std::ios::ate | std::ios::binary);
		if (!file.is_open()) {
			throw std::runtime_error("failed to open file!");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();
		return buffer;

	}

	void createSyncObjects() {

		//信号量主要用于Queue之间的同步
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		//用于CPU和GPU之间的同步
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		//第一帧可以直接获得信号，而不会阻塞
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		//每一帧都需要一定的信号量和栏栅
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(my_device->logicalDevice, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(my_device->logicalDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(my_device->logicalDevice, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create graphics semaphores!");
			}
		}

	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			processInput(window);
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(my_device->logicalDevice);

	}

	void processInput(GLFWwindow* window)
	{
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);

		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			camera.ProcessKeyboard(FORWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			camera.ProcessKeyboard(BACKWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			camera.ProcessKeyboard(LEFT, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			camera.ProcessKeyboard(RIGHT, deltaTime);
	}

	//从这个函数中看出来，fence主要在循环中进行阻塞，而semaphore主要在每个循环中的各个阶段进行阻塞，实现串行
	void drawFrame() {

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		vkWaitForFences(my_device->logicalDevice, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(my_device->logicalDevice, my_swapChain->swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
		//VK_ERROR_OUT_OF_DATE_KHR：交换链与表面不兼容，无法再用于渲染。通常在调整窗口大小后发生。
		//VK_SUBOPTIMAL_KHR：交换链仍可用于成功呈现到表面，但表面属性不再完全匹配。
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		updateUniformBuffer(currentFrame);

		vkResetFences(my_device->logicalDevice, 1, &inFlightFences[currentFrame]);
		vkResetCommandBuffer(my_buffer->commandBuffers[0][currentFrame], 0);
		recordCommandBuffer(my_buffer->commandBuffers[0][currentFrame], imageIndex);

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &my_buffer->commandBuffers[0][currentFrame];
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &renderFinishedSemaphores[currentFrame];

		if (vkQueueSubmit(my_device->graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit shadow command buffer!");
		};

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderFinishedSemaphores[currentFrame];

		VkSwapchainKHR swapChains[] = { my_swapChain->swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		result = vkQueuePresentKHR(my_device->presentQueue, &presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}

		//currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
		frameNum = (frameNum + 1) % 100000;

	}

	void clearTexture(VkImage image) {

		VkCommandBuffer commandBuffer = myBuffer::beginSingleTimeCommands(my_device->logicalDevice, my_buffer->commandPool);

		VkClearColorValue clearColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = 1;
		subresourceRange.baseArrayLayer = 0;
		subresourceRange.layerCount = 1;

		vkCmdClearColorImage(commandBuffer, image, VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &subresourceRange);

		myBuffer::endSingleTimeCommands(my_device->logicalDevice, my_device->graphicsQueue, commandBuffer, my_buffer->commandPool);

	}

	void updateUniformBuffer(uint32_t currentImage) {

		float currentTime = static_cast<float>(glfwGetTime());;
		deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
		lastTime = currentTime;

		UniformBufferObject ubo{};
		ubo.model = glm::mat4(1.0f);
		ubo.view = camera.GetViewMatrix();
		ubo.proj = glm::perspective(glm::radians(45.0f), my_swapChain->swapChainExtent.width / (float)my_swapChain->swapChainExtent.height, 0.1f, 100.0f);
		//怪不得，我从obj文件中看到场景的顶点是顺时针的，但是在shader中得是逆时针才对，原来是这里proj[1][1]1 *= -1搞的鬼
		//那我们在计算着色器中处理顶点数据似乎不需要这个啊
		ubo.proj[1][1] *= -1;
		ubo.cameraPos = glm::vec4(camera.Position, 1.0f);
		ubo.randomNumber = glm::vec4(float(frameNum % 1000));
		ubo.swapChainExtent = glm::vec4(my_swapChain->swapChainExtent.width, my_swapChain->swapChainExtent.height, 0.0f, 0.0f);

		memcpy(my_buffer->uniformBuffersMappeds[0][currentFrame], &ubo, sizeof(ubo));

	}

	void recreateSwapChain() {

		int width = 0, height = 0;
		//获得当前window的大小
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(my_device->logicalDevice);
		cleanupSwapChain();
		createMySwapChain();
		createTextureResources();
		createFramebuffers();
	}

	//这个函数记录渲染的命令，并指定渲染结果所在的纹理索引
	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		clearTexture(commandBuffer);

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(my_swapChain->swapChainExtent.width);
		viewport.height = static_cast<float>(my_swapChain->swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = my_swapChain->swapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = my_buffer->swapChainFramebuffers[0][imageIndex];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = my_swapChain->swapChainExtent;

		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		/*
#ifdef Voxelization_Block
		clearValues[1].depthStencil = { 65536, 0 };
#endif
*/
		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkBuffer vertexBuffers[] = { my_buffer->buffersStatic[0] };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffer, my_buffer->buffersStatic[1], 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelGraphicsPipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelGraphicsPipelineLayout, 0, 1, &my_descriptor->descriptorObjects[0].descriptorSets[0], 0, nullptr);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelGraphicsPipelineLayout, 1, 1, &my_descriptor->descriptorObjects[1].descriptorSets[0], 0, nullptr);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelGraphicsPipelineLayout, 2, 1, &my_descriptor->descriptorObjects[2].descriptorSets[0], 0, nullptr);
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(this->indices.size()), 1, 0, 0, 0);
		//uint32_t index = 0;
		//for (uint32_t i = 0; i < my_model->meshs.size(); i++) {
		//	//这个场景中的模型都没有纹理，所以不需要换绑描述符
		//	vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(my_model->meshs[i].indices.size()), 1, index, 0, 0);
		//	index += my_model->meshs[i].indices.size();
		//}

		//将voxelMap转为VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL布局
		//voxel_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		//voxel_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		//voxel_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		//voxel_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		//voxel_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		//
		////image含有很多的subsource，比如颜色数据、mipmap
		//voxel_barrier.image = voxelMap->image;
		//voxel_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		//voxel_barrier.subresourceRange.baseMipLevel = 0;
		//voxel_barrier.subresourceRange.levelCount = 1;
		//voxel_barrier.subresourceRange.baseArrayLayer = 0;
		//voxel_barrier.subresourceRange.layerCount = 1;
		//voxel_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		//voxel_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		//
		//vkCmdPipelineBarrier(
		//	commandBuffer,
		//	VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		//	VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		//	0,
		//	0, nullptr,
		//	0, nullptr,
		//	1, &voxel_barrier
		//);

		vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, finalGraphicsPipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, finalGraphicsPipelineLayout, 0, 1, &my_descriptor->descriptorObjects[0].descriptorSets[0], 0, nullptr);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, finalGraphicsPipelineLayout, 1, 1, &my_descriptor->descriptorObjects[1].descriptorSets[0], 0, nullptr);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, finalGraphicsPipelineLayout, 2, 1, &my_descriptor->descriptorObjects[3].descriptorSets[0], 0, nullptr);
#ifdef Voxelization_Block
		VkBuffer vertex_cubesBuffers[] = { my_buffer->buffersStatic[2] };
		VkDeviceSize offsets_cubes[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertex_cubesBuffers, offsets_cubes);
		vkCmdBindIndexBuffer(commandBuffer, my_buffer->buffersStatic[3], 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(this->indices_cubes.size()), 1, 0, 0, 0);
#else
		vkCmdDraw(commandBuffer, 3, 1, 0, 0);
#endif

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}

	}

	void clearTexture(VkCommandBuffer commandBuffer) {

//#ifndef Voxelization_Block
		VkImageMemoryBarrier depth_barrier{};
		depth_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		depth_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depth_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		depth_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		depth_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		//image含有很多的subsource，比如颜色数据、mipmap
		depth_barrier.image = depthMap->image;
		depth_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		depth_barrier.subresourceRange.baseMipLevel = 0;
		depth_barrier.subresourceRange.levelCount = 1;
		depth_barrier.subresourceRange.baseArrayLayer = 0;
		depth_barrier.subresourceRange.layerCount = 1;
		depth_barrier.srcAccessMask = 0;
		depth_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &depth_barrier
		);

		VkClearColorValue depth_clearColor;
		depth_clearColor.uint32[0] = 65536;	//4294967295
		depth_clearColor.uint32[1] = 0;
		depth_clearColor.uint32[2] = 0;
		depth_clearColor.uint32[3] = 0;
		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = 1;
		subresourceRange.baseArrayLayer = 0;
		subresourceRange.layerCount = 1;
		vkCmdClearColorImage(commandBuffer, depthMap->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &depth_clearColor, 1, &subresourceRange);

		depth_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		depth_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		depth_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		depth_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		depth_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		//image含有很多的subsource，比如颜色数据、mipmap
		depth_barrier.image = depthMap->image;
		depth_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		depth_barrier.subresourceRange.baseMipLevel = 0;
		depth_barrier.subresourceRange.levelCount = 1;
		depth_barrier.subresourceRange.baseArrayLayer = 0;
		depth_barrier.subresourceRange.layerCount = 1;
		depth_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		depth_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &depth_barrier
		);
//#endif

		//-----------------------------------------------------------------------

		VkImageMemoryBarrier voxel_barrier{};
		voxel_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		voxel_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		voxel_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		voxel_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		voxel_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		//image含有很多的subsource，比如颜色数据、mipmap
		voxel_barrier.image = voxelMap->image;
		voxel_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		voxel_barrier.subresourceRange.baseMipLevel = 0;
		voxel_barrier.subresourceRange.levelCount = 1;
		voxel_barrier.subresourceRange.baseArrayLayer = 0;
		voxel_barrier.subresourceRange.layerCount = 1;
		voxel_barrier.srcAccessMask = 0;
		voxel_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &voxel_barrier
		);

		VkClearColorValue voxel_clearColor;
		voxel_clearColor.uint32[0] = 0;	//4294967295
		voxel_clearColor.uint32[1] = 0;
		voxel_clearColor.uint32[2] = 0;
		voxel_clearColor.uint32[3] = 0;
		VkImageSubresourceRange voxel_subresourceRange = {};
		voxel_subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		voxel_subresourceRange.baseMipLevel = 0;
		voxel_subresourceRange.levelCount = 1;
		voxel_subresourceRange.baseArrayLayer = 0;
		voxel_subresourceRange.layerCount = 1;
		vkCmdClearColorImage(commandBuffer, voxelMap->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &voxel_clearColor, 1, &voxel_subresourceRange);

		voxel_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		voxel_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		voxel_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		voxel_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		voxel_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		//image含有很多的subsource，比如颜色数据、mipmap
		voxel_barrier.image = voxelMap->image;
		voxel_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		voxel_barrier.subresourceRange.baseMipLevel = 0;
		voxel_barrier.subresourceRange.levelCount = 1;
		voxel_barrier.subresourceRange.baseArrayLayer = 0;
		voxel_barrier.subresourceRange.layerCount = 1;
		voxel_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		voxel_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &voxel_barrier
		);

	}

	void cleanup() {

		cleanupSwapChain();

		vkDestroyPipeline(my_device->logicalDevice, voxelGraphicsPipeline, nullptr);
		vkDestroyPipelineLayout(my_device->logicalDevice, voxelGraphicsPipelineLayout, nullptr);
		vkDestroyPipeline(my_device->logicalDevice, finalGraphicsPipeline, nullptr);
		vkDestroyPipelineLayout(my_device->logicalDevice, finalGraphicsPipelineLayout, nullptr);
		vkDestroyRenderPass(my_device->logicalDevice, renderPass, nullptr);

		vkDestroyDescriptorPool(my_device->logicalDevice, my_descriptor->discriptorPool, nullptr);

		my_descriptor->clean();

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(my_device->logicalDevice, renderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(my_device->logicalDevice, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(my_device->logicalDevice, inFlightFences[i], nullptr);
		}
		my_buffer->clean(my_device->logicalDevice, MAX_FRAMES_IN_FLIGHT);

		my_device->clean();

		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();

	}

	void cleanupSwapChain() {

		voxelMap->clean();
		depthMap->clean();
		for (size_t i = 0; i < my_buffer->swapChainFramebuffers.size(); i++) {
			for (int j = 0; j < my_buffer->swapChainFramebuffers[i].size(); j++) {
				vkDestroyFramebuffer(my_device->logicalDevice, my_buffer->swapChainFramebuffers[i][j], nullptr);
			}
		}
		for (size_t i = 0; i < my_swapChain->swapChainImageViews.size(); i++) {
			vkDestroyImageView(my_device->logicalDevice, my_swapChain->swapChainImageViews[i], nullptr);
		}
		vkDestroySwapchainKHR(my_device->logicalDevice, my_swapChain->swapChain, nullptr);
	}

};

int main() {

	Tessellation app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		system("pause");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

}