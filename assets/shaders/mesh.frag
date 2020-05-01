#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec4 target0;

void main() {
    float light;
    if (normal == vec3(0.)) {
        float r = length(position);
        light = .5/(r*r);
    } else {
        light = dot(normal, normalize(-position));
    }
    light = max(.1, light);
    target0 = vec4(vec3(light), 1.);
}