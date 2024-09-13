#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 vertColor;

layout(location = 0) uniform mat4 u_Affine;

out vec4 fragColor;

void main()
{
    gl_Position = vec4(position.x, position.y, position.z, 1.0f) * u_Affine;
    fragColor = vertColor;
}