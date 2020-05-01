use crate::common::*;
use crate::physics::Aabb3;

pub trait Vertex {
    fn get_position<'a>(&'a self) -> &'a Vec3;
}

pub fn calculate_aabb<V: Vertex>(vertices: &[V]) -> Aabb3 {
    let mut min = vec3(0.,0.,0.);
    let mut max = vec3(0.,0.,0.);
    
    for vertex in vertices {
        let pos = vertex.get_position();

        if pos.x < min.x
        {min.x = pos.x}
        if pos.x > max.x
        {max.x = pos.x}
        
        if pos.y < min.y
        {min.y = pos.y}
        if pos.y > max.y
        {max.y = pos.y}
        
        if pos.z < min.z
        {min.z = pos.z}
        if pos.z > max.z
        {max.z = pos.z}
    }

    Aabb3 {
        min,
        max
    }
}