#version 450

layout(location = 0) in vec3 worldPos;

layout(set = 0, binding = 0) uniform cameraUniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
	vec4 randomNumber;	//xyz�����������w��֡��
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

const float depthDigit = 65536.0f;	//4294967295 = 2��32�η�-1

vec4 convRGBA8ToVec4( uint val) {
	return vec4(float((val&0x000000FF)), float((val&0x0000FF00) >> 8U), float((val&0x00FF0000) >> 16U), float((val&0xFF000000) >> 24U));
}

uint convVec4ToRGBA8(vec4 val) {
	return (uint(val.w)&0x000000FF) << 24U | (uint(val.z)&0x000000FF) << 16U | (uint(val.y)&0x000000FF) << 8U | (uint(val.x)&0x000000FF);
}

void imageAtomicRGBA8Avg(ivec3 coords, vec4 val) {

	//val.rgb *= 255.0f; // Optimise following calculations
	//uint newVal = convVec4ToRGBA8(val);
	uint newVal = packUnorm4x8(val);
	uint prevStoredVal = 0;
	uint curStoredVal;

	//��Ҫ˼·���ǣ��ж�image�е�ֵ��ղ��õ���image�е�ֵ�Ƿ���ȣ����˵��û�������߳��޸ģ���ô���Խ���ֵ���룻
	//��֮��˵����whileѭ���������������߳��޸ģ������¼�����ֵ
	while((curStoredVal = imageAtomicCompSwap(voxelMap, coords, prevStoredVal, newVal)) != prevStoredVal) {
		prevStoredVal = curStoredVal;
		//vec4 rval = convRGBA8ToVec4(curStoredVal);
		vec4 rval = unpackUnorm4x8(curStoredVal);
		rval.xyz = rval.xyz * rval.w;	// Denormalize
		vec4 curValF = rval + val;		// Add new value
		curValF.xyz /= curValF.w;		 // Renormalize
		//newVal = convVec4ToRGBA8(curValF);
		newVal = packUnorm4x8(curValF);
	}
}

void imageAtomicDepthTest(ivec2 texCoords, float depth){
	
	uint udepth = uint(depth * depthDigit);
	uint prevStoredVal = uint(depthDigit);
	uint curStoredVal = imageAtomicCompSwap(depthMap, texCoords, prevStoredVal, udepth);

	while(curStoredVal != prevStoredVal && udepth < curStoredVal){
		prevStoredVal = curStoredVal;
		curStoredVal = imageAtomicCompSwap(depthMap, texCoords, prevStoredVal, udepth);
	}

}

void main(){

	ivec3 voxelIndex = ivec3((worldPos - vbo.voxelStartPos.xyz) / vbo.voxelSize_Num.x);
	imageAtomicRGBA8Avg(voxelIndex, vec4(normalize(abs(worldPos)), 1.0f));
	//imageAtomicAdd(voxelMap, voxelIndex, 1);

	vec4 ndcPos = cubo.proj * cubo.view * cubo.model * vec4(worldPos, 1.0f);
	ndcPos /= ndcPos.w;
	float depth = ndcPos.z * 0.5f + 0.5f;
	ivec2 texCoords = ivec2((ndcPos.xy * 0.5f + 0.5f) * cubo.swapChainExtent.xy);
	imageAtomicDepthTest(texCoords, depth);

}