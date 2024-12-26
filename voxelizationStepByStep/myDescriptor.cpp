#include "myDescriptor.h"

myDescriptor::myDescriptor(VkDevice logicalDevice, uint32_t frameSize) {
	this->logicalDevice = logicalDevice;
	this->frameSize = frameSize;
	//createDescriptorPool();
}

//������ʵͦ��ֵģ���Ϊ���������ǽ���ͬ���������ֿ�����¼�ģ��������������ǽ���ͬ��������Լ�¼��
//���������ؼ�¼�м�����������ÿ���������м����������������ϼ�¼ÿ����������Щ��������ɣ�һ���м�������
//�൱���������ش��Լ��ĸ����ֳ����ó����������һ������������
void myDescriptor::createDescriptorPool(uint32_t uniformBufferNumAllLayout, uint32_t storageBufferNumAllLayout, std::vector<VkDescriptorType> types, std::vector<uint32_t> textureNumAllLayout) {

	std::vector<VkDescriptorPoolSize> poolSizes{};
	VkDescriptorPoolSize poolSize;
	if (uniformBufferNumAllLayout > 0) {
		poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSize.descriptorCount = static_cast<uint32_t>(this->frameSize * uniformBufferNumAllLayout);
		poolSizes.push_back(poolSize);
	}
	if (storageBufferNumAllLayout > 0) {
		poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSize.descriptorCount = static_cast<uint32_t>(storageBufferNumAllLayout);
		poolSizes.push_back(poolSize);
	}
	for (int i = 0; i < textureNumAllLayout.size(); i++) {

		poolSize.type = types[i];
		poolSize.descriptorCount = static_cast<uint32_t>(this->frameSize * textureNumAllLayout[i]);
		poolSizes.push_back(poolSize);

	}

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = static_cast<uint32_t>(this->frameSize * 16);

	if (vkCreateDescriptorPool(logicalDevice, &poolInfo, nullptr, &this->discriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor pool!");
	}

}

//ͨ�û���������������������ͨ�û�������ɫ���׶Ρ����������������͡���������������������ͨ�û���������������ͼ�������������
//ΪɶҪvector����vector��ԭ���������vector�Ƕ�Ӧ����set���������vector��ʾһ��set���м���
DescriptorObject myDescriptor::createDescriptorObject(uint32_t uniformBufferNum, uint32_t textureNum, std::vector<VkShaderStageFlagBits>* uniformBufferUsages, std::vector<VkDescriptorType>* textureDescriptorType,
	uint32_t descriptorSetSize, std::vector<std::vector<VkBuffer>>* uniformBuffers, std::vector<VkDeviceSize> bufferSize, std::vector<std::vector<VkImageView>>* textureViews, std::vector<std::vector<VkSampler>>* textureSamplers) {

	DescriptorObject descriptorObject;

	VkDescriptorSetLayout discriptorLayout = createDescriptorSetLayout(uniformBufferNum, textureNum, uniformBufferUsages, textureDescriptorType);

	descriptorObject.discriptorLayout = discriptorLayout;
	descriptorObject.uniformBufferNum = uniformBufferNum;
	descriptorObject.textureNum = textureNum;

	for (int i = 0; i < frameSize; i++) {

		if (uniformBuffers->at(0).size() == 1 && i == 1) {
			break;
		}

		//��������uniformBuffer����������֡����һ����Ϊһ�飬���Ǵ���createSet��Ҫ��һ����
		std::vector<VkBuffer> bufferComb;
		for (int j = 0; j < uniformBuffers->size(); j++) {
			bufferComb.push_back(uniformBuffers->at(j)[i]);
		}

		for (int u = 0; u < descriptorSetSize; u++) {
			descriptorObject.descriptorSets.push_back(createDescriptorSet(descriptorObject, uniformBuffers == nullptr ? nullptr : &bufferComb, bufferSize, textureDescriptorType,
				textureViews == nullptr ? nullptr : &(textureViews->at(u)),
				textureSamplers == nullptr ? nullptr : &(textureSamplers->at(u))));
		}
	}


	return descriptorObject;
}

VkDescriptorSetLayout myDescriptor::createDescriptorSetLayout(uint32_t uniformBufferNum, uint32_t textureNum, std::vector<VkShaderStageFlagBits>* uniformBufferUsages, std::vector<VkDescriptorType>* textureDescriptorType) {

	std::vector<VkDescriptorSetLayoutBinding> bindings;

	for (int i = 0; i < uniformBufferNum; i++) {

		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = i;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		//�����count��ʾ�ð󶨵��м���������������Ҫ����һ�������shader����ô�������������٣�����һ�����һ��
		//��Ϊ���ǵ�mvp������ͨ��һ���ṹ��һ�𴫵ģ���������Ϊ1
		uboLayoutBinding.descriptorCount = 1;
		//VK_SHADER_STAGE_ALL_GRAPHICS���Է���������ɫ��ʹ��
		uboLayoutBinding.stageFlags = uniformBufferUsages->at(i);

		bindings.push_back(uboLayoutBinding);

	}

	for (int i = 0; i < textureNum; i++) {

		VkDescriptorSetLayoutBinding samplerLayoutBinding{};
		samplerLayoutBinding.binding = uniformBufferNum + i;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = textureDescriptorType->at(i);
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings.push_back(samplerLayoutBinding);

	}

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings = bindings.data();

	VkDescriptorSetLayout layout;
	if (vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor set layout!");
	}

	return layout;

}

VkDescriptorSet myDescriptor::createDescriptorSet(DescriptorObject descriptorObject, std::vector<VkBuffer>* uniformBuffers, std::vector<VkDeviceSize> bufferSize, std::vector<VkDescriptorType>* textureDescriptorType, std::vector<VkImageView>* textureViews, std::vector<VkSampler>* textureSamplers) {

	VkDescriptorSetLayout layout = descriptorObject.discriptorLayout;
	uint32_t uniformBufferNum = descriptorObject.uniformBufferNum;
	uint32_t textureNum = descriptorObject.textureNum;

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = this->discriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &layout;

	VkDescriptorSet descriptorSet;
	//���ؿյ�����������
	if (vkAllocateDescriptorSets(logicalDevice, &allocInfo, &descriptorSet) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	std::vector<VkWriteDescriptorSet> descriptorWrites;
	descriptorWrites.resize(uniformBufferNum + textureNum);

	std::vector<VkDescriptorBufferInfo> bufferInfos;
	bufferInfos.resize(uniformBufferNum);
	for (int j = 0; j < uniformBufferNum; j++) {

		bufferInfos[j].buffer = uniformBuffers->at(j);
		bufferInfos[j].offset = 0;
		bufferInfos[j].range = bufferSize[j];

		descriptorWrites[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[j].dstSet = descriptorSet;
		descriptorWrites[j].dstBinding = j;
		descriptorWrites[j].dstArrayElement = 0;
		descriptorWrites[j].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrites[j].descriptorCount = 1;
		descriptorWrites[j].pBufferInfo = &bufferInfos[j];

	}

	std::vector<VkDescriptorImageInfo> imageInfos;
	imageInfos.resize(textureNum);
	for (int j = 0; j < textureNum; j++) {

		imageInfos[j].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfos[j].imageView = textureViews->at(j);
		if (textureDescriptorType->at(j) != VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT) {
			imageInfos[j].sampler = textureSamplers->at(j);
		}

		descriptorWrites[j + uniformBufferNum].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[j + uniformBufferNum].dstSet = descriptorSet;
		descriptorWrites[j + uniformBufferNum].dstBinding = uniformBufferNum + j;
		descriptorWrites[j + uniformBufferNum].dstArrayElement = 0;
		descriptorWrites[j + uniformBufferNum].descriptorType = textureDescriptorType->at(j);
		descriptorWrites[j + uniformBufferNum].descriptorCount = 1;
		descriptorWrites[j + uniformBufferNum].pImageInfo = &imageInfos[j];	//imageInfo.data();

	}

	vkUpdateDescriptorSets(logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

	return descriptorSet;

}




void myDescriptor::clean() {
	for (int i = 0; i < this->descriptorObjects.size(); i++) {
		vkDestroyDescriptorSetLayout(logicalDevice, this->descriptorObjects[i].discriptorLayout, nullptr);
	}
}