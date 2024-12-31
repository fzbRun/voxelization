#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include<iostream>
#include<vector>
#include<map>
#include<set>

#include "structSet.h"

#ifndef MY_DEVICE
#define MY_DEVICE

class myDevice {

public:

	VkInstance instance;
	VkSurfaceKHR surface;

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;

	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkQueue computeQueue;

	SwapChainSupportDetails swapChainSupportDetails;
	QueueFamilyIndices queueFamilyIndices;

	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

	const std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME,
		//VK_NV_VIEWPORT_ARRAY2_EXTENSION_NAME,
		//VK_KHR_MULTIVIEW_EXTENSION_NAME,
		//VK_NV_VIEWPORT_SWIZZLE_EXTENSION_NAME,
		//VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME
	};

	//���캯��
	myDevice(VkInstance instance, VkSurfaceKHR surface);

	//vulkan����
	void pickPhysicalDevice();
	void createLogicalDevice(bool enableValidationLayers, std::vector<const char*> validationLayers);

	//��������
	int rateDeviceSuitability(VkSurfaceKHR surface, VkPhysicalDevice device);
	SwapChainSupportDetails querySwapChainSupport(VkSurfaceKHR surface, VkPhysicalDevice device);
	QueueFamilyIndices findQueueFamilies(VkSurfaceKHR surface, VkPhysicalDevice device);
	bool checkDeviceExtensionSupport(VkPhysicalDevice device);
	VkSampleCountFlagBits getMaxUsableSampleCount();
	//uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

	void clean();

};


#endif