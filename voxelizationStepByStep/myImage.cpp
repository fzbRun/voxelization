#include "myImage.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

myImage::myImage(std::string path, VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, bool mipmapEnable) {

	this->physicalDevice = physicalDevice;
	this->logicalDevice = logicalDevice;

	createTextureImage(path.c_str(), queue, commandPool, mipmapEnable);
	this->imageView = createTextureImageView(this->image, this->mipLevels);
	this->textureSampler = createTextureSampler();

}

myImage::myImage(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, uint32_t width, uint32_t height, uint32_t mipLevel, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImageAspectFlags aspectFlags) {

	this->physicalDevice = physicalDevice;
	this->logicalDevice = logicalDevice;

	createImage(width, height, mipLevel, numSamples, format, tiling, usage, properties, this->image, this->imageMemory);
	this->imageView = createImageView(this->image, format, aspectFlags, mipLevel);
	this->textureSampler = createTextureSampler();
}

//3D����
myImage::myImage(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, uint32_t width, uint32_t height, uint32_t depth, uint32_t mipLevel, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImageAspectFlags aspectFlags) {

	this->physicalDevice = physicalDevice;
	this->logicalDevice = logicalDevice;

	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_3D;
	imageInfo.extent.width = static_cast<uint32_t>(width);
	imageInfo.extent.height = static_cast<uint32_t>(height);
	imageInfo.extent.depth = static_cast<uint32_t>(depth);
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.format = format;
	imageInfo.tiling = tiling;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = usage;
	imageInfo.samples = numSamples;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.flags = 0;
	if (vkCreateImage(logicalDevice, &imageInfo, nullptr, &image) != VK_SUCCESS) {
		throw std::runtime_error("failed to create image!");
	}

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(logicalDevice, image, &memRequirements);
	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = myBuffer::findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);
	if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate image memory!");
	}

	vkBindImageMemory(logicalDevice, image, imageMemory, 0);

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
	viewInfo.format = format;
	viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.subresourceRange.aspectMask = aspectFlags;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = mipLevels;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	if (vkCreateImageView(logicalDevice, &viewInfo, nullptr, &this->imageView) != VK_SUCCESS) {
		throw std::runtime_error("failed to create image views!");
	}

	this->textureSampler = createTextureSampler();

}

void myImage::createTextureImage(const char*  texturePath, VkQueue queue, VkCommandPool commandPool, bool mipmapEnable) {

	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load(texturePath, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	VkDeviceSize imageSize = texWidth * texHeight * 4;
	if (!pixels) {
		throw std::runtime_error("failed to load texture image!");
	}

	//��ɫ���в���������ͬ��ֻ��ҪGPU�ɼ������ԺͶ��㻺����һ���������Ƚ����ݴ浽�ݴ滺�����Ŵ浽GPU����������
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	myBuffer::createBuffer(physicalDevice, logicalDevice, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

	void* data;
	vkMapMemory(logicalDevice, stagingBufferMemory, 0, imageSize, 0, &data);
	memcpy(data, pixels, static_cast<size_t>(imageSize));
	vkUnmapMemory(logicalDevice, stagingBufferMemory);

	stbi_image_free(pixels);

	this->mipLevels = mipmapEnable ? static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1 : 1;
	createImage(texWidth, texHeight, this->mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, this->image, this->imageMemory);


	//���Ǵ�����image��֪��ԭ����ʲô���֣�����Ҳ�����ģ�����ֻ��Ҫ��copyǰ�޸����Ĳ���,��������Ҳ������ǰ���command����ϣ��������(��Դmash��stage��Ϊ�����ģ�
	transitionImageLayout(queue, commandPool, this->image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
	copyBufferToImage(queue, commandPool, stagingBuffer, this->image, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

	vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
	vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);

	if (mipmapEnable) {
		generateMipmaps(queue, commandPool, this->image, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, this->mipLevels);
	}
	else {
		//�������˼�������ƬԪ��ɫ����Ҫ������������ȴ���������ɲ��ܿ�ʼ�������ڲ���ǰ�޸Ĳ��֣�����Ҫ��copy command����
		transitionImageLayout(queue, commandPool, this->image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
	}

	//this->imageView = createTextureImageView(logicalDevice, this->image, this->mipLevels);
	//this->textureSampler = createTextureSampler(physicalDevice, logicalDevice, this->mipLevels);

}

VkImageView myImage::createTextureImageView(VkImage textureImage, uint32_t mipLevels) {
	return createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
}

VkSampler myImage::createTextureSampler() {

	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(physicalDevice, &properties);

	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;	//�����Ϊrepeat����ӰMap�߽���������
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.anisotropyEnable = VK_TRUE;
	samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.mipLodBias = 0.0f;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = static_cast<float>(this->mipLevels);

	VkSampler textureSampler;
	if (vkCreateSampler(logicalDevice, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
		throw std::runtime_error("failed to create texture sampler!");
	}

	return textureSampler;

}


//����target texture�����Բ���Ҫmipmapʲô��
void myImage::createImage(uint32_t width, uint32_t height, uint32_t mipLevel, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
	
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = static_cast<uint32_t>(width);
	imageInfo.extent.height = static_cast<uint32_t>(height);
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = mipLevels;
	imageInfo.arrayLayers = 1;
	//findSupportedFormat(physicalDevice, { VK_FORMAT_R8G8B8A8_UINT }, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
	imageInfo.format = format;
	//tiling��һ�����ݴ�Ÿ�ʽ��linear�ǰ��д�ţ���optimal���ǣ����岻֪�����������С��з��ʶ��Ѻã������Ը�˹ģ������ǰ�߷ǳ���
	//VK_IMAGE_TILING_LINEAR: Texels are laid out in row-major order like our pixels array�����������Ҫ����image�ͱ��������ַ�ʽ
	//VK_IMAGE_TILING_OPTIMAL: Texels are laid out in an implementation defined order for optimal access ������ǲ�����ʣ������ַ�ʽ���Ч
	//��˼����������Ƿ�������������Ҫ�õ�һ�֣�������ǲ����ʣ�ֻ����������Ӳ��ȥ�㣬��ڶ����ڲ����Ż������Ч
	imageInfo.tiling = tiling;
	//������һ������ѹ����ʽ
	//VK_IMAGE_LAYOUT_UNDEFINED: Not usable by the GPU and the very first transition will discard the texels.
	//VK_IMAGE_LAYOUT_PREINITIALIZED: Not usable by the GPU, but the first transition will preserve the texels.
	//�������˼�����Ƿ����imageԭ�е����ݲ��ֽ����޸ģ�������������ԭ�в������޸�һ���֣���ʹ��ǰ�ߣ������VK_IMAGE_LAYOUT_UNDEFINED����֮����
	//�������������ǽ��ⲿ��������ݷŵ���image�У����Բ�����imageԭ������
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = usage;
	//�����ʲôϡ��洢���
	imageInfo.samples = numSamples;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.flags = 0;
	if (vkCreateImage(logicalDevice, &imageInfo, nullptr, &image) != VK_SUCCESS) {
		throw std::runtime_error("failed to create image!");
	}

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(logicalDevice, image, &memRequirements);
	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = myBuffer::findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);
	if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate image memory!");
	}

	vkBindImageMemory(logicalDevice, image, imageMemory, 0);

}

VkImageView myImage::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.subresourceRange.aspectMask = aspectFlags;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = mipLevels;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	VkImageView imageView;
	if (vkCreateImageView(logicalDevice, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
		throw std::runtime_error("failed to create image views!");
	}

	return imageView;

}



//������Ҫ��ȷ�ľ��������ύ��Queu�󲢲���˳��ִ�У���������ִ�У���ô���Ǿͱ��뱣֤���ǲ����õ�����Ĳ����Ѿ�����Ҫ���ˣ�������ѹ��Ҫ��
//memory barrier����ʹ��a��b�����׶��ڴ�������ڴ�ɼ�������ʵ����һ������д���Ĺ��̣���ô��������������ǾͿ�����дʱ�޸����ݵĴ洢����
void myImage::transitionImageLayout(VkQueue queue, VkCommandPool commandPool, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {

	VkCommandBuffer commandBuffer = myBuffer::beginSingleTimeCommands(logicalDevice, commandPool);

	//1. ����execution barrier��һ��ͬ�����ֶΣ�ʹ��command��ͬ��������������command������һ�����ϣ���ȷ���������ڵĽ׶Σ���a��b���Ϳ���ʹ��dst�߳���b�׶�������ֱ��src�߳�ִ����a�׶�
	//2. ����execution barrierֻ�ܱ�ִ֤��˳���ͬ���������ܱ�֤�ڴ��ͬ������a�׶������������Ҫ��b�׶�ʹ�ã�����b�׶�ȥ��ȡʱ������û�и��»��Ѿ�����cache���ˣ���ô����Ҫ�õ�memory barrier
	//	ͨ��srcAccessMask��֤a�׶�������������ڴ���õģ��������Ѿ����µ�L2 Cache�У�ͨ��dstAccessMask��֤b�׶λ�õ������Ѿ������ˣ������ݴ�L2 Cache���µ�L1 Cache
	//3. VkBufferMemoryBarrier���ڱ�֤buffer֮����ڴ�����.��VkBufferMemoryBarrier��˵���ṩ��srcQueueFamilyIndex��dstQueueFamilyIndex��ת����Buffer������Ȩ������VkMemoryBarrierû�еĹ��ܡ�
	//	ʵ�����������Ϊ��ת��Buffer������Ȩ�Ļ��ģ��������й�Buffer��ͬ��������ȫ����VkMemoryBarrier����ɣ�һ����˵VkBufferMemoryBarrier�õĺ��١�
	//4. VkImageMemoryBarrier���ڱ�֤Image���ڴ�����������ͨ��ָ��subresourceRange������Image��ĳ������Դ��֤���ڴ�������ͨ��ָ�� oldLayout��newLayout��������ִ��Layout Transition��
	//	���Һ�VkBufferMemoryBarrier����ͬ���ܹ����ת����Image������Ȩ�Ĺ��ܡ�
	//5. �������ǽ�����a, barrier, b, c���������Queue��c��a�Ĳ���û�������������Ի���Ϊbarrier���ȴ�a������ɲ��ܼ�����Ⱦ�����������˷ѣ����Կ���ʹ��event��ֻͬ��a��b����c��������ִ��
	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	//���Խ���������Ϊ���ض���������Ч
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

	//image���кܶ��subsource��������ɫ���ݡ�mipmap
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = mipLevels;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;
	barrier.srcAccessMask = 0;
	barrier.dstAccessMask = 0;

	//https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineStageFlagBits.html
	VkPipelineStageFlags sourceStage;	//a�׶�
	VkPipelineStageFlags destinationStage;	//b�׶�
	//������һ����Ҫע��ĵ㣬�Ǿ���AccessMask��Ҫ��VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT/VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT����ʹ�ã���Щ�׶β�ִ���ڴ���ʣ������κ�srcAccessMask��dstAccessMask���������׶ε���϶���û�������
	//	����Vulkan��������������TOP_OF_PIPE��BOTTOM_OF_PIPE������Ϊ��Execution Barrier�����ڣ�������Memory Barrier��
	//����һ����srcAccessMask���ܻᱻ����ΪXXX_READ��flag��������ȫ����ġ��ö�������һ���ڴ���õĹ�����û�������(��Ϊû��������Ҫ����)��Ҳ����˵�Ƕ�ȡ���ݵ�ʱ��û��ˢ��Cache����Ҫ������ܻ��˷�һ���ֵ����ܡ�
	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		//������ʵէһ��ͦ��ֵģ���Ϊbarrier����������command֮��ģ�����copyimage����ֻ��һ��command���������ʹ��barrier�أ�
		//�������ǿ�������������ֻ��Ҫ���ڴ�ɼ�����ʱ�޸�imageLayout����ô���ǲ�����ǰһ��command��ʲô�����Բ�ϣ��ǰ���command�������ǵ�copy command�����Կ��Խ�srcAccessMask��Ϊ0��sourceStage��ΪVK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
		//ͬʱ���ǿ���copy������command���Ǿ���Ĳ�����������VK_PIPELINE_STAGE_TRANSFER_BIT�����Խ�destinationStage��Ϊ���������ʵ������˿��������
		//ͬʱ���ƵĲ���Ҫ��VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
		barrier.srcAccessMask = 0;	//ֻ��д��������Ҫ�ڴ���ã�����ֻ��Ҫ��������Ҫ�˷�����
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;//������д����ν����Ϊд֮ǰҪ��
		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;	//��Ⱦ��ʼ���ͷ�Ľ׶�
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;	//��ͬ�������в�ͬ�Ľ׶Σ�����copy�������VK_PIPELINE_STAGE_TRANSFER_BIT�������ݸ��ƽ׶�
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		//ͬ��������Ҫ�õ�image��command����ȴ�copy command��VK_PIPELINE_STAGE_TRANSFER_BIT��������ܿ�ʼ����
		//���������ڲ���ǰ���Խ����ָĳ�VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL���ַ�������Ĳ��֣���Ȼ�Ҳ�֪��ѹ������ѹ�Բ�����ʲôӰ�죩
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	}
	else{
		throw std::invalid_argument("unsupported layout transition!");
	}

	vkCmdPipelineBarrier(
		commandBuffer,
		sourceStage, destinationStage,
		0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);

	myBuffer::endSingleTimeCommands(logicalDevice, queue, commandBuffer, commandPool);

}

void myImage::copyBufferToImage(VkQueue queue, VkCommandPool commandPool, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {

	VkCommandBuffer commandBuffer = myBuffer::beginSingleTimeCommands(logicalDevice, commandPool);

	VkBufferImageCopy region{};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;
	region.imageOffset = { 0, 0, 0 };
	region.imageExtent = { width, height, 1 };

	vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	myBuffer::endSingleTimeCommands(logicalDevice, queue, commandBuffer, commandPool);

}

void myImage::generateMipmaps(VkQueue queue, VkCommandPool commandPool, VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {

	VkFormatProperties formatProperties;
	vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);
	if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
		throw std::runtime_error("texture image format does not support linear blitting!");
	}

	VkCommandBuffer commandBuffer = myBuffer::beginSingleTimeCommands(logicalDevice, commandPool);
	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.image = image;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;
	barrier.subresourceRange.levelCount = 1;

	int32_t mipWidth = texWidth;
	int32_t mipHeight = texHeight;

	for (uint32_t i = 1; i < mipLevels; i++) {

		barrier.subresourceRange.baseMipLevel = i - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		//ÿ��blit command����һ��barrier,ʹ��һ����dst��תΪsrc
		vkCmdPipelineBarrier(commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		VkImageBlit blit{};
		blit.srcOffsets[0] = { 0, 0, 0 };	//���
		blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };	//�ֱ���
		blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.srcSubresource.mipLevel = i - 1;
		blit.srcSubresource.baseArrayLayer = 0;
		blit.srcSubresource.layerCount = 1;
		blit.dstOffsets[0] = { 0, 0, 0 };
		blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
		blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.dstSubresource.mipLevel = i;
		blit.dstSubresource.baseArrayLayer = 0;
		blit.dstSubresource.layerCount = 1;

		vkCmdBlitImage(commandBuffer,
			image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &blit,
			VK_FILTER_LINEAR);

		//����Ҫ�����ֶ�ȥת���֣����Ƕ����뱻������mipmap���ڱ�blitǰ������������blit�����ǰ��ͨ��barrier�Զ���srcתΪshader read
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		if (mipWidth > 1) mipWidth /= 2;
		if (mipHeight > 1) mipHeight /= 2;

	}

	//Ϊ���һ��mipmap����barrier
	barrier.subresourceRange.baseMipLevel = mipLevels - 1;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

	vkCmdPipelineBarrier(commandBuffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
		0, nullptr,
		0, nullptr,
		1, &barrier);

	myBuffer::endSingleTimeCommands(logicalDevice, queue, commandBuffer, commandPool);

}


VkFormat myImage::findDepthFormat(VkPhysicalDevice physicalDevice) {
	return findSupportedFormat(physicalDevice, { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

VkFormat myImage::findSupportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {

	for (VkFormat format : candidates) {

		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
		if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
			return format;
		}
		else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
			return format;
		}

	}

	throw std::runtime_error("failed to find supported format!");

}

bool myImage::hasStencilComponent(VkFormat format) {
	return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

void myImage::clean() {

	if (textureSampler) {
		vkDestroySampler(logicalDevice, textureSampler, nullptr);
	}
	vkDestroyImageView(logicalDevice, this->imageView, nullptr);
	vkDestroyImage(logicalDevice, this->image, nullptr);
	vkFreeMemory(logicalDevice, this->imageMemory, nullptr);

}