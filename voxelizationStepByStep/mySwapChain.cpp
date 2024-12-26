#include "mySwapChain.h"

mySwapChain::mySwapChain(GLFWwindow* window, VkSurfaceKHR surface, VkDevice logicalDevice, SwapChainSupportDetails swapChainSupport, QueueFamilyIndices indices) {
	this->window = window;
	this->surface = surface;
	this->logicalDevice = logicalDevice;
	createSwapChain(swapChainSupport, indices);
}

void mySwapChain::createSwapChain(SwapChainSupportDetails swapChainSupport, QueueFamilyIndices indices) {

	this->surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);	//��Ҫ��surface��չʾ�������ͨ�����������Լ�ɫ�ʿռ�
	VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
	this->extent = chooseSwapExtent(swapChainSupport.capabilities);

	//�����������С������ͼ������ȣ���ȷ����֧�ֵ�ͼ������������֧�ֵ�ͼ��������������Сͼ����+1
	//���maxImageCount=0�����ʾû�����ƣ������������ط������ƣ��޷�������
	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
	if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = surface;	//��Ҫ������
	createInfo.minImageCount = imageCount;	//�涨�˽������������������������2����˫����
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent;
	createInfo.imageArrayLayers = 1;	//���������z��1�ͱ�ʾ2D����
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	uint32_t queueFamilyIndices[] = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() };

	//ͼ�ζ����帺����Ⱦ���ܣ�Ȼ�󽻸����������������ٽ���չʾ��������ֵ�surface��
	if (indices.graphicsAndComputeFamily != indices.presentFamily) {
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else {
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		//createInfo.queueFamilyIndexCount = 0;
		//createInfo.pQueueFamilyIndices = nullptr;
	}

	createInfo.preTransform = swapChainSupport.capabilities.currentTransform;	//ָ���Ƿ���Ҫ��ǰ��ת��ת
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	createInfo.presentMode = presentMode;
	createInfo.clipped = VK_TRUE;
	createInfo.oldSwapchain = VK_NULL_HANDLE;

	if (vkCreateSwapchainKHR(logicalDevice, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
		throw std::runtime_error("failed to create swap chain!");
	}

	//std::vector<VkImage> swapChainImagesTemp;
	vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, nullptr);
	this->swapChainImages.resize(imageCount);
	vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, this->swapChainImages.data());

	swapChainImageFormat = surfaceFormat.format;
	swapChainExtent = extent;

	createSwapChainImageViews();

}

//���ڽ������е���Ⱦ��������������ǿ���ͨ��imageView��Ϊ���������
void mySwapChain::createSwapChainImageViews() {

	//imageViews�ͽ������е�image������ͬ
	this->swapChainImageViews.resize(this->swapChainImages.size());
	for (size_t i = 0; i < this->swapChainImages.size(); i++) {

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = swapChainImages[i];
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = this->swapChainImageFormat;
		viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		if (vkCreateImageView(logicalDevice, &viewInfo, nullptr, &this->swapChainImageViews[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image views!");
		}

	}

}


VkSurfaceFormatKHR mySwapChain::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {

	for (const auto& availableFormat : availableFormats) {
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			return availableFormat;
		}
	}
	return availableFormats[0];

}

VkPresentModeKHR mySwapChain::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
	for (const auto& availablePresentMode : availablePresentModes) {
		//��������γ��ֻ��棬������ֱ��չʾ����˫���壬�ȵ�
		//VK_PRESENT_MODE_IMMEDIATE_KHR ��Ⱦ��ɺ�����չʾ��ÿ֡���ֺ���Ҫ�ȴ���һ֡��Ⱦ��ɲ����滻�������һ֡��Ⱦ��ʱ��ʱ�����ͻ���ֿ���
		//VK_PRESENT_MODE_FIFO_KHR �໺�壬��Ⱦ��ɺ��ύ���浽����Ļ��壬�̶�ʱ�䣨��ʾ��ˢ��ʱ�䣩����ֵ���ʾ���ϡ������������ˣ���Ⱦ�ͻ�ֹͣ��������
		//VK_PRESENT_MODE_FIFO_RELAXED_KHR ��Ⱦ��ɺ��ύ���浽����Ļ��壬���������һ֡��Ⱦ�Ľ�����������һ֡��ˢ�º��Դ��ڣ���ǰ֡�ύ�����̳��֣���ô�Ϳ��ܵ��¸���
		//VK_PRESENT_MODE_MAILBOX_KHR ��Ⱦ��ɺ��ύ���浽����Ļ��壬�̶�ʱ�䣨��ʾ��ˢ��ʱ�䣩����ֵ���ʾ���ϡ������������ˣ�������滻���Ļ���������������
		if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
			return availablePresentMode;
		}
	}
	return VK_PRESENT_MODE_FIFO_KHR;
}

//һ����˵��Vulkan�����ý�������ͼ��ֱ��ʸ����ڷֱ�����ȡ����ǣ���Щ���ڹ�����������辶���ѳ��Ϳ�ķֱ��ʾ�����Ϊuint32_t�������������֧�ֵ����ֵ
	//��������Ŀ�ģ���ϣ�����ǿ����Լ�����һЩ�����ķֱ��ʣ��������ڷֱ���������������˵���ǲ����û���ʹ��ڵȴ󣬱�����Խ������Ⱦ����ŵ�һ������
	//������������������������Ҫ�ֶ�����һ�½�����ͼ��ֱ��ʡ�
VkExtent2D mySwapChain::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {

	if (capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)()) {
		return capabilities.currentExtent;
	}
	else {		//ĳЩ���ڹ��������������ڴ˴�ʹ�ò�ͬ��ֵ����ͨ����currentExtent�Ŀ�Ⱥ͸߶�����Ϊ���ֵ����ʾ�����ǲ���Ҫ���������Խ�֮��������Ϊ���ڴ�С
		int width, height;
		//��ѯ���ڷֱ���
		glfwGetFramebufferSize(window, &width, &height);
		VkExtent2D actualExtent = {
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent;

	}
}


