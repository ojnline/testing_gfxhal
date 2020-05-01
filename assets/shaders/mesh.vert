#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 norm;

layout(location = 0) out vec3 position;
layout(location = 1) out vec3 normal;

layout(push_constant) uniform PushConstants {
    vec4 quaternion;
    vec3 translation;
    float scale;
    mat4 projection;
} push_constants;

vec3 saturate(vec3 col) {
    float maximum = max(col.r, max(col.g, col.b));
    return col * (1/maximum);
}

vec3 quaternion_rotate(vec4 q, vec3 v) {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

vec3 apply_model(vec3 v) {
  return push_constants.translation + quaternion_rotate(push_constants.quaternion, v*push_constants.scale);
}

void main() {
  vec3 model = apply_model(pos);
  
  // apply projection matrix
  gl_Position = push_constants.projection * vec4(model, 1.);

  normal = quaternion_rotate(push_constants.quaternion, norm);
  position = model;
}