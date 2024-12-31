#version 450

const float depthDigit = 65536.0f;	//4294967295 = 2的32次方-1

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

layout(location = 0) in vec3 worldPos;
layout(location = 1) in flat ivec3 voxelIndex;

layout(location = 0) out vec4 FragColor;

ivec3 getTexelVoxelIndex(vec2 texCoords){
	
	ivec2 texCoordsU = ivec2(texCoords * cubo.swapChainExtent.xy);
	uint texelDepthU = imageLoad(depthMap, texCoordsU).r;
	float texelDepth = float(texelDepthU) / depthDigit;
	texelDepth = texelDepth * 2.0f - 1.0f;

	if(texelDepth == 1.0f){
		discard;
	}

	vec4 texelWorldPos = inverse(cubo.proj * cubo.view) * vec4(vec3(texCoords, texelDepth), 1.0f);
	texelWorldPos /= texelWorldPos.w;

	ivec3 texelVoxelIndex = ivec3((texelWorldPos.xyz - vbo.voxelStartPos.xyz) / vbo.voxelSize_Num.x);
	return texelVoxelIndex;

}

ivec3 getTexelVoxelIndex(vec3 worldPos){
	
	vec4 clipPos = cubo.proj * cubo.view * cubo.model * vec4(worldPos, 1.0f);
	vec4 ndcPos = clipPos / clipPos.w;
	vec2 texCoords = ndcPos.xy * 0.5f + 0.5f;
	ivec2 texCoordsU = ivec2(texCoords * cubo.swapChainExtent.xy);
	uint texelDepthU = imageLoad(depthMap, texCoordsU).r;
	float texelDepth = float(texelDepthU) / depthDigit;
	texelDepth = texelDepth * 2.0f - 1.0f;

	vec4 texelWorldPos = inverse(cubo.proj * cubo.view) * vec4(vec3(ndcPos.xy, texelDepth), 1.0f);
	texelWorldPos /= texelWorldPos.w;
	ivec3 texelVoxelIndex = ivec3((texelWorldPos.xyz - vbo.voxelStartPos.xyz) / vbo.voxelSize_Num.x);
	return texelVoxelIndex;

}

void main(){

	//通过cube得到的片元可能会出现在两个体素之间，这时我们应该选取远离相机的体素
	//vec3 tangent = dFdx(worldPos);
	//vec3 bitangent = dFdy(worldPos);
	//vec3 normal = normalize(cross(tangent, bitangent));
	//ivec3 voxelIndex = ivec3((worldPos.xyz - vbo.voxelStartPos.xyz) / vbo.voxelSize_Num.x);
	//ivec3 texelVoxelIndex = getTexelVoxelIndex(worldPos);
	
	uint voxelValueU = imageLoad(voxelMap, ivec3(voxelIndex)).r;
	vec4 voxelValue = unpackUnorm4x8(voxelValueU);
	if(voxelValue.w == 0){
		discard;
	}
	FragColor = vec4(voxelValue.rgb, 1.0f);

}