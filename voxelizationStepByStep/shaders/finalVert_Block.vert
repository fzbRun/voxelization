#version 450

layout(location = 0) in vec3 pos_in;
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

layout(location = 0) out vec3 worldPos;
layout(location = 1) out ivec3 voxelIndex;

void main(){
	//gl_Position = cubo.proj * cubo.view * cubo.model * vec4(pos_in, 1.0f);
	//worldPos = (cubo.model * vec4(pos_in, 1.0f)).xyz;
	
	int instanceIndex = gl_InstanceIndex;
	int Z = instanceIndex / int(vbo.voxelSize_Num.y * vbo.voxelSize_Num.y);
	instanceIndex -= Z * int(vbo.voxelSize_Num.y * vbo.voxelSize_Num.y);
	int Y = instanceIndex / int(vbo.voxelSize_Num.y);
	int X = instanceIndex - Y * int(vbo.voxelSize_Num.y);
	vec3 offset = vec3(X,Y,Z);
	
	vec3 pos = offset * vbo.voxelSize_Num.x + pos_in;
	gl_Position = cubo.proj * cubo.view * cubo.model * vec4(pos, 1.0f);
	worldPos = (cubo.model * vec4(pos, 1.0f)).xyz;
	voxelIndex = ivec3(offset);

}

//layout(location = 0) out vec2 texCoord;
//
//void main(){
//	texCoord = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
//    gl_Position = vec4(texCoord * 2.0f + -1.0f, 0.0f, 1.0f);
//}