pub use nalgebra as na;
pub use nphysics3d as np;

pub type Pos3 = na::Point3<f32>;
pub type Vec3 = na::Vector3<f32>;
pub type Mat4 = na::Matrix4<f32>;

pub fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

pub fn pos3(x: f32, y: f32, z: f32) -> Pos3 {
    Pos3::new(x, y, z)
}