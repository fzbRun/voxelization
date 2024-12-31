#include "myDevice.h"

myDevice::myDevice(VkInstance instance, VkSurfaceKHR surface) {
	this->instance = instance;
	this->surface = surface;
}


//������Ҫ�ҵ�һ����Ҳ���Զ���������豸����Ҫ���Կ������¶������������棩������������Ⱦ
//��ͬ���Կ��в�ͬ�Ĺ��ܣ������е��Կ����Լ����������Ⱦ
//����Ĺ���ָ���Ƕ����壬�Ǵ�Ĺ��ܣ���feature�Ǹ�ϸ��
void myDevice::pickPhysicalDevice() {

	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
	if (deviceCount == 0) {
		throw std::runtime_error("failed to find GPUS with Vulkan support");
	}

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());	//���ǰ����ȼ��ŵ�

	//���Կ���������������ģ�������Ĭ���õ���intel�ļ����Կ����ҵ�3070ֻ�ܳԻ�
	std::multimap<int, VkPhysicalDevice> candidates;
	for (const auto& device : devices) {
		int score = rateDeviceSuitability(surface, device);
		candidates.insert(std::make_pair(score, device));
	}

	if (candidates.rbegin()->first > 0) {
		this->physicalDevice = candidates.rbegin()->second;
		this->msaaSamples = getMaxUsableSampleCount();
		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
		
		uint32_t major = VK_VERSION_MAJOR(deviceProperties.apiVersion);
		uint32_t minor = VK_VERSION_MINOR(deviceProperties.apiVersion);
		uint32_t patch = VK_VERSION_PATCH(deviceProperties.apiVersion);

		std::cout << "Vulkan Version: " << major << "." << minor << "." << patch << std::endl;

		std::cout << deviceProperties.deviceName << std::endl;

	}
	else {
		throw std::runtime_error("failed to find a suitable GPU!");
	}

}

//����ϵͳ�о����߼��豸�������豸֮�֣�����Ϊ��ȥ���������豸�Ĳ����ԣ����鷳��������ϵͳ�������û�
//������������ǲ�һ���ģ�һ�������豸��Ӧ����߼��豸��������Ϊ�����������߼��ĸ��ͬ
//����ϵͳ��������߼���ʵ������󣬶�����������ǹ��ܵļ��ϣ����߼��ǶԹ��ܵĻ��ֵ��Ӽ�
//���߼��豸��ĳһЩ����ʵ�ֵľ�����������豸�����ж�����ܣ�����Щ���ܻ���Ϊһ��һ�����߼��豸���Ӽ���
//ʵ�����߼��豸�������ٴλ��֣�������Queue����һ��ʵ�ֵľ�����
void myDevice::createLogicalDevice(bool enableValidationLayers, std::vector<const char*> validationLayers) {

	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::set<uint32_t> uniqueQueueFamilies = { queueFamilyIndices.graphicsAndComputeFamily.value(), queueFamilyIndices.presentFamily.value() };

	//����ѡȡ�������豸ӵ��һ���Ķ����壨���ܣ�����û�д�����������Ҫ��֮��������
	//����������豸��Ӧһ���߼��豸����һ���߼��豸��Ӧ��������
	float queuePriority = 1.0f;
	for (uint32_t queueFamily : uniqueQueueFamilies) {
		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamily;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueCreateInfo);
	}

	VkPhysicalDeviceFeatures deviceFeatures{};
	deviceFeatures.samplerAnisotropy = VK_TRUE;
	//deviceFeatures.sampleRateShading = VK_TRUE;
	deviceFeatures.geometryShader = VK_TRUE;
	deviceFeatures.fragmentStoresAndAtomics = VK_TRUE;
	deviceFeatures.multiViewport = VK_TRUE;

	VkDeviceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.pQueueCreateInfos = queueCreateInfos.data();
	createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	createInfo.pEnabledFeatures = &deviceFeatures;
	createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
	createInfo.ppEnabledExtensionNames = deviceExtensions.data();

	// Ϊ�豸ָ����ʵ����ͬ��У���
	// ʵ���ϣ��°汾��Vulkan�Ѿ��������ֶ��ߵ�У��㣬
	// ���Զ������豸�й���У�����ֶΡ���������һ�µĻ������Ծɰ汾����
	if (enableValidationLayers) {
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}
	else {
		createInfo.enabledLayerCount = 0;
	}

	if (vkCreateDevice(this->physicalDevice, &createInfo, nullptr, &this->logicalDevice) != VK_SUCCESS) {
		throw std::runtime_error("failed to create logical device!");
	}

	vkGetDeviceQueue(this->logicalDevice, queueFamilyIndices.graphicsAndComputeFamily.value(), 0, &this->graphicsQueue);
	vkGetDeviceQueue(this->logicalDevice, queueFamilyIndices.graphicsAndComputeFamily.value(), 0, &this->computeQueue);
	vkGetDeviceQueue(this->logicalDevice, queueFamilyIndices.presentFamily.value(), 0, &this->presentQueue);

}

int myDevice::rateDeviceSuitability(VkSurfaceKHR surface, VkPhysicalDevice device) {

	//VkPhysicalDeviceProperties deviceProperties;
	//VkPhysicalDeviceFeatures deviceFeatures;
	//vkGetPhysicalDeviceProperties(device, &deviceProperties);	//�豸��Ϣ
	//vkGetPhysicalDeviceFeatures(device, &deviceFeatures);		//�豸����
		
	this->queueFamilyIndices = findQueueFamilies(surface, device);
	VkPhysicalDeviceProperties deviceProperties;
	vkGetPhysicalDeviceProperties(device, &deviceProperties);
	//std::cout << deviceProperties.limits.maxPerStageDescriptorStorageImages << std::endl;

	//����豸�Ƿ�֧�ֽ�������չ
	bool extensionsSupport = checkDeviceExtensionSupport(device);
	bool swapChainAdequate = false;
	if (extensionsSupport) {
		//�ж������豸��ͼ���չʾ�����Ƿ�֧��
		this->swapChainSupportDetails = querySwapChainSupport(surface, device);
		swapChainAdequate = !swapChainSupportDetails.formats.empty() && !swapChainSupportDetails.presentModes.empty();
	}

	VkPhysicalDeviceFeatures supportedFeatures;
	vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
	if (queueFamilyIndices.isComplete() && extensionsSupport && swapChainAdequate && supportedFeatures.samplerAnisotropy) {
		int score = 0;
		if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			score += 1000;
		}
		score += deviceProperties.limits.maxImageDimension2D;
		VkPhysicalDeviceFeatures deviceFeatures;
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
		if (!deviceFeatures.geometryShader) {	//�ҿ���ֻҪ����֧�ּ�����ɫ�����Կ�
			return -1;
		}
		return score;
	}

	return -1;

}

SwapChainSupportDetails myDevice::querySwapChainSupport(VkSurfaceKHR surface, VkPhysicalDevice device) {

	SwapChainSupportDetails details;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
	if (formatCount != 0) {
		details.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
	}

	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
	if (presentModeCount != 0) {
		details.presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
	}

	return details;

}

QueueFamilyIndices myDevice::findQueueFamilies(VkSurfaceKHR surface, VkPhysicalDevice device) {

	QueueFamilyIndices indices;
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());	//��ö���ϵ�е���ϸ��Ϣ

	int i = 0;
	for (const auto& queueFamily : queueFamilies) {
		//�����ͼ������ǲ���˵�Կ���ר�Ŷ���Ⱦ���Ż�
		//��ΪVK_QUEUE_COMPUTE_BIT��˵�Կ�����ͨ�ü���(������ɫ��)������Ⱦʵ����Ҳ��һ�ּ��㣬��ô�ֿ����ߵ�ԭ��Ӧ�þ����Ƿ���ר���Ż�
		//ע��֧��VK_QUEUE_GRAPHICS_BIT��VK_QUEUE_COMPUTE_BIT���豸Ĭ��֧��VK_QUEUE_TRANSFER_BIT���������ݻ��������ݣ�
		if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
			indices.graphicsAndComputeFamily = i;
		}

		VkBool32 presentSupport = false;
		//�ж�i��Ⱥ�Ƿ�Ҳ֧��չʾ������չʾ����˼���ܷ�GPU��Ⱦ�����Ļ��洫����ʾ���ϣ���Щ�Կ����ܲ�δ���ӵ���ʾ��
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

		if (presentSupport) {
			indices.presentFamily = i;
		}

		if (indices.isComplete()) {
			break;
		}
		i++;
	}

	return indices;

}

bool myDevice::checkDeviceExtensionSupport(VkPhysicalDevice device) {

	uint32_t extensionCount;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

	std::vector<VkExtensionProperties> availableExtensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

	std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
	for (const auto& extension : availableExtensions) {
		requiredExtensions.erase(extension.extensionName);
	}

	//��requiredExtensions���ˣ�˵����Ҫ����չȫ��
	return requiredExtensions.empty();

}

VkSampleCountFlagBits myDevice::getMaxUsableSampleCount() {
	VkPhysicalDeviceProperties physicalDeviceProperties;
	vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

	VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
	if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
	if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
	if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
	if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
	if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
	if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

	return VK_SAMPLE_COUNT_1_BIT;
}

/*
uint32_t myDevice::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {

	VkPhysicalDeviceMemoryProperties memProperties;
	//�ҵ���ǰ�����豸���Կ�����֧�ֵ��Դ����ͣ��Ƿ�֧������洢���������ݴ洢��ͨ�����ݴ洢�ȣ�
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		//�����type�˳���ͨ��one-hot����ģ���0100��ʾ����,1000��ʾ�ĺ�
		if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");

}
*/

void myDevice::clean() {
	vkDestroyDevice(logicalDevice, nullptr);
}
