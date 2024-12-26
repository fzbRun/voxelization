#version 450

layout(set = 0, binding = 0) uniform cameraUniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
	vec4 randomNumber;	//xyz是随机数，而w是帧数
	vec4 swapChainExtent;
}cubo;

layout(set = 1, binding = 0) uniform voxelBufferObject{
	mat4 model;
	mat4 VP[3];
	vec4 voxelSize_Num;
	vec4 voxelStartPos;
}vbo;

layout(set = 2, binding = 0, r32ui) uniform uimage2D depthMap;
layout(set = 2, binding = 1, r32ui) uniform uimage3D voxelMap;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 FragColor;

const float depthDigit = 65536.0f;

vec3 getWorldPos(vec2 texCoords){

	ivec2 utexCoords = ivec2(texCoords * cubo.swapChainExtent.xy);
	float depth = float(imageLoad(depthMap, utexCoords).r) / depthDigit;
	depth = depth * 2.0f - 1.0f;

	if(depth == 1.0f){
		return vec3(0.0f);
	}

	vec4 clipPos = vec4(texCoords * 2.0f - 1.0f, depth, 1.0f);
	vec4 worldPos = inverse(cubo.proj * cubo.view) * clipPos;
	worldPos /= worldPos.w;

	return worldPos.xyz;

}

ivec3 getVoxelIndex(vec2 texCoord){
	
	vec3 worldPos = getWorldPos(texCoord);
	ivec3 voxelIndex = ivec3((worldPos.xyz - vbo.voxelStartPos.xyz) / vbo.voxelSize_Num.x);
	return voxelIndex;

}

vec4 convRGBA8ToVec4( uint val) {
	return vec4(float((val&0x000000FF)), float((val&0x0000FF00) >> 8U), float((val&0x00FF0000) >> 16U), float((val&0xFF000000) >> 24U));
}

void main(){

	ivec3 voxelIndex = getVoxelIndex(texCoord);
	uint voxelValue = imageLoad(voxelMap, voxelIndex).r;

	//vec3 voxelWorldPos = (vec3(voxelIndex) + 0.5f) * vbo.voxelSize_Num.x + vbo.voxelStartPos.xyz;
	//FragColor = vec4(abs(normalize(voxelWorldPos)), 1.0f);
	//return;

	//FragColor = vec4(convRGBA8ToVec4(voxelValue).rgb / 255.0f, 1.0f);
	FragColor = vec4(unpackUnorm4x8(voxelValue).rgb, 1.0f);

}

/*
layout(location = 0) out vec4 fragColor;

void mian(){
	
	ivec3 voxelIndex = ivec3((worldPos.xyz - vbo.voxelStartPos.xyz) / vbo.voxelSize_Num.x);
	fragColor = 

}
*/