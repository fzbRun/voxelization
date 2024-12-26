#version 450

layout(location = 0) in vec3 pos_in;

layout(set = 0, binding = 0) uniform cameraUniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
	vec4 randomNumber;	//xyz是随机数，而w是帧数
	vec4 swapChainExtent;
} cubo;

//layout(location = 0) out vec3 worldPos;

void main(){
	//gl_Position = cubo.proj * cubo.view * cubo.model * vec4(pos_in, 1.0f);
	//worldPos = (cubo.model * vec4(pos_in, 1.0f)).xyz;
	gl_Position = cubo.model * vec4(pos_in, 1.0f);
}