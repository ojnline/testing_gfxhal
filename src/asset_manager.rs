use crate::util::{DeviceRc};
use crate::resource::{self as res, MapResource};
use crate::common::*;

use gfx_hal::{
    self as hal,
    Backend,
    device::Device,
    memory as m,
    buffer as b,
    pso,
    pool::{self, CommandPool},
    command as c,
    queue::CommandQueue
};


use std::default::Default;
use std::path::Path;

use gfx_memory::{self as gfxm, Heaps, Block};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct SerializeMeshData {
    name: String,
    vertices: Vec<PosNorm>,
    indices: Vec<u32>,
    aabb: Aabb3
}

#[repr(C)]
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct PosNorm {
    position: Vec3,
    normal: Vec3
}

// is this neccessary?
pub trait Vertex {
    fn get_position<'a>(&'a self) -> &'a Vec3;
    fn get_position_mut<'a>(&'a mut self) -> &'a mut Vec3;
}
impl Vertex for PosNorm {
    fn get_position<'a>(&'a self) -> &'a Vec3 {
        &self.position
    }
    fn get_position_mut<'a>(&'a mut self) -> &'a mut Vec3 {
        &mut self.position
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Aabb3 {
    pub min: Vec3,
    pub max: Vec3
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

/// This function first translates the mesh so that the center of its 
/// aabb is on the origin and then it scales it so that the longest 
/// aabb side is 1. This way all models scale similarly.
pub fn normalize_mesh<V: Vertex>(vertices: &mut [V], aabb: Aabb3) {
    let extents: Vec3 = aabb.min+aabb.max;

    let corrective_scale = 1. / extents.max();
    let corrective_translation: Vec3 = extents / 2.;

    for v in vertices.iter_mut() {
        let position = v.get_position_mut();
        *position = (*position-corrective_translation)*corrective_scale;
    };
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct AssetId {
    index: usize,
    generation: u32
}

pub enum Asset<B: Backend> {
    Mesh {
        name: String,
        vertices: res::Buffer<B>,
        vertex_size: u32,
        indices: Option<res::Buffer<B>>,
        index_size: u32,
        aabb: Aabb3
    },
    Deleted
}

#[derive(Clone, Copy)]
pub struct AssetManagerConfig {
    staging_buffer_size: u64,
    // resources are created on demand and immediately destroyed
    keep_staging_mapped: bool,
    // gpu copies are divided into multiple transfers if neccessary
    // the transfers are flushed if space in the staging buffer
    // reaches below this value
    min_transfer_size: u64,
    // whether to flush to local memory after each transfer and not when explicitly called or full
    always_flush: bool
}
impl Default for AssetManagerConfig {
    fn default() -> Self {
        Self {
            staging_buffer_size: 8_388_608, // 8 MiB
            min_transfer_size: 4096, // 4 KiB
            keep_staging_mapped: true,
            always_flush: true,
        }
    }
}
pub struct ResourceManager<B: Backend> {
    device: DeviceRc<B>,
    heaps: Heaps<B>,
    staging_buffer: res::Buffer<B>,
    staging_cursor: u64,
    staging_dirty: bool,

    mapped_memory: Option<*mut u8>,
    command_pool: B::CommandPool,
    command_buffer: B::CommandBuffer,

    config: AssetManagerConfig,

    // I basically reinvented Slab but with generation checking
    // it also sucks but is only like 30 lines as opposed to 1300 
    assets: Vec<(u32, Asset<B>)>,
    deleted_assets: Vec<usize>
}

impl<'a, B: Backend> ResourceManager<B> {
    pub fn new(device: DeviceRc<B>, mut heaps: gfxm::Heaps<B>, config: AssetManagerConfig) -> Self {
        
        let device_ref = device.borrow();
        
        use b::Usage as U;
        let mut staging_buffer = res::Buffer::new(
            &device,
            &mut heaps,
            config.staging_buffer_size,
            U::TRANSFER_SRC,
            gfxm::MemoryUsage::Staging{read_back: false}
        );
        let mut command_pool = unsafe {
            device_ref.device.create_command_pool(
                    device_ref.queue_group.family,
                    pool::CommandPoolCreateFlags::empty(),
            ).unwrap()
        };
        use c::CommandBuffer;
        let command_buffer = unsafe {
            let mut command_buffer = command_pool.allocate_one(c::Level::Primary);
            command_buffer.begin_primary(c::CommandBufferFlags::ONE_TIME_SUBMIT);
            command_buffer
        };
        let mapped_memory = if config.keep_staging_mapped {
            let raw = staging_buffer.map_raw(&device).unwrap();
            Some(raw.0)
        } else {None};
        drop(device_ref);

        Self {
            device,
            heaps,
            staging_buffer,
            staging_cursor: 0,
            staging_dirty: false,
            command_pool,
            command_buffer,
            mapped_memory,
            config,
            assets: Vec::new(),
            deleted_assets: Vec::new()
        }
    }
    pub fn get_asset(&self, id: &AssetId) -> Option<&Asset<B>> {
        if let Some((gen, asset)) = self.assets.get(id.index) {
            if *gen != id.generation {
                return None;
            }
            return match asset {
                Asset::Deleted => None,
                _ => Some(asset)
            };
        } else {None}
    }
    pub fn delete_asset(&mut self, id: &AssetId) -> Result<(), &'static str> {
        if let Some((gen, asset)) = self.assets.get_mut(id.index) {
            if *gen != id.generation {
                return Err("Asset doesn't exist anymore.");
            }
            use std::ptr::read;
            match asset {
                Asset::Deleted => return Err("Asset already deleted."),
                // TODO replace with traits
                Asset::Mesh{vertices, indices, ..} => {
                    let device = self.device.borrow();
                    unsafe {
                        read(vertices).dispose(&device, &mut self.heaps);
                        if let Some(indices) = indices {
                            read(indices).dispose(&device, &mut self.heaps);
                        }
                    }
                }
            }
            
            *gen += 1;
            *asset = Asset::Deleted;

            Ok(())
        } else {Err("Asset doesn't exist.")}
    }
    
    pub fn load_obj(&mut self, path: &Path) -> Result<AssetId, &'static str> {
        use std::fs::File;
        use std::io::BufReader;

        let obj_file = Path::new(path);
        let mut cache_directory = std::env::current_dir().unwrap();
        cache_directory.push(".asset_cache");
        let cache_file = cache_directory.join(Path::new("").with_file_name(obj_file.file_name().unwrap()).with_extension("mesh"));

        let mesh = if cache_file.is_file() {
            let raw = std::fs::read(cache_file).unwrap();
            let mesh: SerializeMeshData = bincode::deserialize(&raw[..]).unwrap();
            log::info!("Loaded cached mesh '{}' -- {} triangles", mesh.name, mesh.indices.len()/3);
            mesh
        } else {
            let reader = BufReader::new(File::open(obj_file).unwrap());
            log::trace!("Started the caching of {} @ {} kb", 
                obj_file.file_name().unwrap().to_string_lossy(), 
                reader.buffer().len() as f32 / 1000. // size in megabytes
            );
            let obj = tobj::load_obj(&obj_file);

            let mut vertices = Vec::new();
            let mut indices = Vec::new();

            // iterate over all the meshes and put their data in a single mesh
            obj.unwrap().0.iter().for_each(|model| {
                let mesh = &model.mesh;
                log::trace!("Caching mesh '{}' -- {} triangles" , model.name, mesh.indices.len()/3);
                
                vertices.reserve(mesh.positions.len());
                indices.reserve(mesh.indices.len());

                use std::slice::from_raw_parts;
                
                let positions = unsafe { from_raw_parts(mesh.positions.as_ptr() as *const Vec3, mesh.positions.len() / 3) };
                let normals = unsafe { from_raw_parts(mesh.normals.as_ptr() as *const Vec3, mesh.normals.len() / 3) };

                let offset = vertices.len() as u32;
                indices.extend(
                    mesh.indices.clone().into_iter()
                    .map(|i| i+offset)
                );
                
                for i in 0..(positions.len()) {
                    vertices.push(
                        PosNorm {
                            position: positions[i],
                            normal: *normals.get(i).unwrap_or(&vec3(0.,1.,0.)),
                        }
                    )
                }
            });

            let aabb = calculate_aabb(vertices.as_slice());

            normalize_mesh(&mut vertices.as_mut_slice(), aabb);

            let data = SerializeMeshData {
                name: String::from(obj_file.file_name().unwrap().to_string_lossy()),
                vertices,
                indices,
                aabb
            };

            if !cache_directory.is_dir() {
                std::fs::create_dir(&cache_directory).unwrap();
            }

            log::info!("Writing cached mesh '{}'", data.name);
            std::fs::write(cache_file, bincode::serialize(&data).unwrap()).unwrap();

            data
        };

        let indices = unsafe {std::slice::from_raw_parts(mesh.indices.as_ptr() as *const u8, mesh.indices.len() * std::mem::size_of::<u32>())};
        self.mesh_from_memory(Some(mesh.name), mesh.vertices.as_slice(), Some(indices), Some(mesh.aabb), false, gfxm::MemoryUsage::Private)
    }
    pub fn mesh_from_memory<T: Vertex + Clone>(&mut self, name: Option<String>, vertices: &[T], indices: Option<&[u8]>, aabb: Option<Aabb3>, normalize: bool, usage: gfxm::MemoryUsage) -> Result<AssetId, &'static str> {
        let name = match name {
            Some(name) => name,
            None => format!("unnamed {}", self.assets.len()-1)
        };
        let aabb = match aabb {
            Some(aabb) => aabb,
            None => calculate_aabb(vertices)
        };

        let mut owned_vertices;
        let mut vertices = vertices;

        if normalize {
            owned_vertices = vertices.to_vec();
            normalize_mesh(&mut owned_vertices.as_mut_slice(), aabb);
            vertices = owned_vertices.as_slice();
        }

        use b::Usage as U;
        let vertices = unsafe {std::slice::from_raw_parts(vertices.as_ptr() as *const u8, vertices.len() * std::mem::size_of::<T>())};
        let vertex_buffer = self.new_buffer_with_data(vertices, U::VERTEX | U::TRANSFER_DST, usage);
        let index_buffer = match indices {
            Some(indices) => Some(
                self.new_buffer_with_data(indices, U::INDEX | U::TRANSFER_DST, usage)
            ),
            None => None
        };

        if self.config.always_flush {
            self.flush_transfers();
        }

        Ok(self.push_asset(Asset::Mesh {
            name,
            vertices: vertex_buffer,
            indices: index_buffer,
            vertex_size: vertices.len() as u32,
            index_size: indices.map(|i| i.len()).unwrap_or(0) as u32,
            aabb
        }))
    }
    fn push_asset(&mut self, asset: Asset<B>) -> AssetId {
        if let Some(empty) = self.deleted_assets.pop() {
            let slot = &mut self.assets[empty];
            slot.0 += 1;
            slot.1 = asset;
            return AssetId {
                generation: slot.0,
                index: empty
            };
        } else {
            self.assets.push((0, asset));
            return AssetId {
                generation: 0,
                index: self.assets.len()-1
            };
        }
    }
    fn new_buffer_with_data(&mut self, data: &[u8], usage: hal::buffer::Usage, memory_usage: gfxm::MemoryUsage) -> res::Buffer<B> {
        let data_len = data.len() as u64;

        let buffer = res::Buffer::new(&self.device, &mut self.heaps, data_len, usage, memory_usage);

        let ptr= if let Some(ptr) = self.mapped_memory {
            ptr
        } else {
            let raw = self.staging_buffer.map_raw(&self.device).unwrap();
            raw.0 
        };
        let segment = self.staging_buffer.block.segment().clone();
        let staging_size = segment.size.unwrap();

        let mut cursor = 0;
        while cursor < data_len {
            
            let mut chunk_size = (data_len - cursor).min(staging_size-self.staging_cursor);
            if chunk_size < self.config.min_transfer_size && (cursor + chunk_size) < data_len {
                unsafe {
                    self.device.borrow().device.flush_mapped_memory_ranges(
                        std::iter::once((self.staging_buffer.block.memory(), segment.clone()))
                    ).unwrap();
                }
                self.staging_dirty = true;
                self.flush_transfers();
                chunk_size = (data_len - cursor).min(staging_size-self.staging_cursor);
            }
            self.staging_dirty = true;
            
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr().offset(cursor as isize), ptr.offset(self.staging_cursor as isize), chunk_size as usize);
            }

            let buffer_copy = c::BufferCopy {
                size: chunk_size,
                dst: cursor,
                src: self.staging_cursor
            };
            
            unsafe {
                use c::CommandBuffer;
                let barrier = m::Barrier::Buffer {
                  states: b::State::HOST_WRITE..b::State::MEMORY_READ,
                  families: None,
                  range: b::SubRange::WHOLE,
                  target: &buffer.buffer
                };
                self.command_buffer.pipeline_barrier(
                    pso::PipelineStage::BOTTOM_OF_PIPE..pso::PipelineStage::TOP_OF_PIPE,
                    m::Dependencies::all(),
                    std::iter::once(&barrier)
                );
                self.command_buffer.copy_buffer(&self.staging_buffer.buffer, &buffer.buffer, std::iter::once(&buffer_copy));
            }
            
            cursor += chunk_size;
            self.staging_cursor += chunk_size;
        }

        unsafe {
            self.device.borrow().device.flush_mapped_memory_ranges(
            std::iter::once((self.staging_buffer.block.memory(), segment.clone()))
            ).unwrap();
        }
        if !self.config.keep_staging_mapped {
            unsafe {
                self.device.borrow().device.unmap_memory(self.staging_buffer.block.memory());
            }
        }

        buffer
    }
    pub fn flush_transfers(&mut self) {
        if self.staging_dirty {
            self.staging_dirty = false;
        } else {
            return;
        }
        
        use c::CommandBuffer;
        let cmd_buffer = &mut self.command_buffer;
        
        unsafe {
            cmd_buffer.finish();

            self.device.borrow_mut().queue_group.queues[0]
            .submit_without_semaphores(std::iter::once(&cmd_buffer), None);

            self.device.borrow().queue_group.queues[0].wait_idle().unwrap();
            
            self.command_pool.reset(false);
            cmd_buffer.begin_primary(c::CommandBufferFlags::ONE_TIME_SUBMIT);
        }

        self.staging_cursor = 0;
    }
}

impl<B: Backend> Drop for ResourceManager<B> {
    fn drop(&mut self) {
        use std::ptr::read;

        let device = self.device.borrow();
        
        unsafe {
            device.device.destroy_command_pool(read(&self.command_pool));
            
            read(&self.staging_buffer).dispose(&device, &mut self.heaps);
        }
        
        for asset in &self.assets {
            match asset.1 {
                Asset::Mesh{ref vertices, ref indices, ..} => {
                    unsafe {
                        read(vertices).dispose(&device, &mut self.heaps);
                        if let Some(indices) = indices {
                            read(indices).dispose(&device, &mut self.heaps);
                        }
                    }
                },
                _ => {}
            }
        }

        self.heaps.clear(&device.device);
    }
}