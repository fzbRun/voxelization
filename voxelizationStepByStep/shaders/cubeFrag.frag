#version 450

const float depthDigit = 65536.0f;	//4294967295 = 2的32次方-1

layout(location = 0) in vec3 worldPos;

layout(set = 0, binding = 0) uniform cameraUniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
	vec4 randomNumber;	//xyz是随机数，而w是帧数
	vec4 swapChainExtent;
} cubo;

layout(set = 1, binding = 0) uniform voxelBufferObject{
	mat4 model;
	mat4 VP[3];
	vec4 voxelSize_Num;
	vec4 voxelStartPos;
}vbo;

layout(set = 2, binding = 0, r32ui) uniform coherent volatile uimage2D depthMap;
layout(set = 2, binding = 1, r32ui) uniform coherent volatile uimage3D voxelMap;

layout(location = 0) out vec4 fragColor;

void mian(){
	
	ivec3 voxelIndex = ivec3((worldPos.xyz - vbo.voxelStartPos.xyz) / vbo.voxelSize_Num.x);


}
