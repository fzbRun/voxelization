#version 450

layout(location = 0) in vec3 pos_in;
layout(location = 0) out vec3 worldPos;

layout(set = 0, binding = 0) uniform cameraUniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
	vec4 randomNumber;	//xyz是随机数，而w是帧数
	vec4 swapChainExtent;
} cubo;

void main(){
	vec4 clipPos = cubo.proj * cubo.view * cubo.model * vec4(pos_in, 1.0f);
	gl_Position = clipPos / clipPos.w;
	worldPos = (cubo.model * vec4(pos_in, 1.0f)).xyz;
}