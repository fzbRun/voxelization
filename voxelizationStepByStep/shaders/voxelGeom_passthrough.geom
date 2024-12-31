#version 450
#extension GL_NV_geometry_shader_passthrough : require
#extension GL_NV_viewport_array2 : require

layout(triangles) in;
layout(location = 0, passthrough) in vec3 worldPos;
layout(passthrough) in gl_PerVertex {
    vec4 gl_Position;
} gl_in[];
layout(viewport_relative) out int gl_Layer;

void main()
{
    gl_Layer = 0;
}