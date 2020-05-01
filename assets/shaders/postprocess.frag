#version 450 
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec2 uv;

layout(location = 0) out vec4 target0;

layout(set = 0, binding = 0) uniform sampler u_sampler;
layout(set = 1, binding = 0) uniform texture2D u_texture;


float linearize_depth(float d,float zNear,float zFar)
{
    return zNear * zFar / (zFar + d * (zNear - zFar));
}

float sample_linear(vec2 uv) {
    float d = texture(sampler2D(u_texture, u_sampler),uv).r;
    return linearize_depth(d, .1, 100.);
}

float rand(vec2 co){
    return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
}

// vec2 rota(vec2 p, a) {
//     float s = sin(a);
//     float c = cos(a);
//     return vec2(
//         p.x * c - p.y * s,
//         p.x * s + p.y * c
//     );
// }

#define SAMPLES 16
#define RADIUS 0.002

void main() {

    float pd_ = texture(sampler2D(u_texture, u_sampler), uv).r;
    float pd = linearize_depth(pd_, .1, 100); 

    float acc = 0.;
    for(int i = 0;i<SAMPLES;i++) {
        // //offset = lerp(0.1, 1.0, offset);
        // // offset = log(abs(offset-1.));
        // offset = (offset*2-1)*RADIUS/pd;
        // offset = vec3(uv, pd) + offset;
        //offset = sqrt(abs(offset))*2-1;
        float ii = float(i)*6.28/SAMPLES;
        vec2 r = vec2(ii*.4978, ii*.9966);
        vec3 rande = vec3(rand(r), rand(2*r), rand(3*r))*rand(4*r);
        rande = normalize(rande)*rand(uv);
        rande = abs(rande)*2-1;
        vec3 offset = vec3(sin(ii), cos(ii), cos(ii))*RADIUS/pd;
        offset = vec3(uv, pd) + offset;

        float d = sample_linear(offset.xy);

        float rangeCheck = smoothstep(0.0, 1.0, RADIUS / abs(offset.z - pd));
        
        if(d < offset.z && pd_ != 1) {
            acc += rangeCheck;
        }
    }
    acc /= SAMPLES;

    target0 = vec4(vec3(1-acc), 1.);
}