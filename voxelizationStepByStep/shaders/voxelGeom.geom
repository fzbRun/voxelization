#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(set = 1, binding = 0) uniform voxelBufferObject{
	mat4 model;
	mat4 VP[3];
	vec4 voxelSize_Num;
	vec4 voxelStartPos;
}vbo;

layout(location = 0) out vec3 worldPos;

void main(){

	//计算面法线
	vec3 p0 = gl_in[0].gl_Position.xyz;
	vec3 p1 = gl_in[1].gl_Position.xyz;
	vec3 p2 = gl_in[2].gl_Position.xyz;

	vec3 e1 = p1 - p0;
	vec3 e2 = p2 - p0;
	vec3 normal = abs(normalize(cross(e1, e2)));

	int viewIndex = normal.x > normal.y ? (normal.x > normal.z ? 0 : 2) : (normal.y > normal.z ? 1 : 2);
	
	worldPos = p0;
	gl_Position = vbo.VP[viewIndex] * gl_in[0].gl_Position;
	EmitVertex();

	worldPos = p1;
	gl_Position = vbo.VP[viewIndex] * gl_in[1].gl_Position;
	EmitVertex();

	worldPos = p2;
	gl_Position = vbo.VP[viewIndex] * gl_in[2].gl_Position;
	EmitVertex();

	EndPrimitive();

}