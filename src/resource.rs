use gfx_hal::{
    self as hal,
    device::Device,
    format as f,
    format::Format,
    image as i,
    Backend,
    memory as m,
    window::Extent2D
};

use gfx_memory as gfxm;

use std::{
    mem::ManuallyDrop
};


use super::util::*;

pub unsafe fn allocate_memory<B: Backend>(device: &DeviceRc<B>, requirements: &m::Requirements, properties: m::Properties) -> Result<B::Memory, &'static str> {
    let memory_type_id = device.borrow()
        .memory_types
        .iter()
        .enumerate()
        .find(|&(id, memory_type)| {
            requirements.type_mask & (1 << id) != 0
                && memory_type.properties.contains(properties)
        })
        .map(|(id, _)| hal::MemoryTypeId(id))
        .ok_or("Couldn't find a memory type to support requested requirements!")?;
    device.borrow().device
        .allocate_memory(memory_type_id, requirements.size)
        .map_err(|_| "Couldn't allocate memory!")
}

// pub struct MappedMemory<'a, B: Backend> {
//     pub ptr: *mut u8,
//     pub size: Option<u64>,
//     // things needed to unmap
//     memory: &'a B::Memory,
//     device: DeviceRc<B>,
//     // the actual mapped range
//     // it can be bigger due to aligment needs
//     segment: m::Segment,
// }

// impl<'a, B: Backend> MappedMemory<'a, B> {
//     unsafe fn map(&self, device: DeviceRc<B>, segment: m::Segment, memory: &'a B::Memory) -> MappedMemory<'a, B> {
//         let device_ref = device.borrow();

//         let aligned_segment = {
//             let (aligned_size, aligned_offset) = {
//                 if let Some(seg_size) = segment.size {
//                     // following the vulkan spec
//                     // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkMappedMemoryRange.html
//                     let atom_size = device_ref.limits.non_coherent_atom_size as u64;
//                     // the offset must be a multiple of atom size
//                     // it is rounded to a lower multiple
//                     let aligned_offset = (segment.offset / atom_size)*atom_size;
//                     // the segment size is recalculated
//                     let total_size = segment.offset - aligned_offset + seg_size;
//                     // the total size must be a multiple as well
//                     // rounded to a higher multiple 
//                     let aligned_size = (((total_size + atom_size) - 1) / atom_size) * atom_size;
                    
//                     (Some(aligned_offset), aligned_size)
//                 } else {
//                     // the size being None means that the whole memory object is mapped
//                     // in that case we don't know its size  
//                     (None, 0)
//                 }
//             };

//             m::Segment {
//                 offset: aligned_offset,
//                 size: aligned_size
//             }
//         };
//         let ptr = device_ref.device.map_memory(memory, segment).unwrap();
        
//         Self {
//             // offset the pointer so it is aligned how it was requested
//             ptr: ptr.offset((segment.offset-aligned_segment.offset) as isize),
//             size: segment.size,
//             memory: memory,
//             device: device,
//             segment: aligned_segment
//         }
//     }
//     pub fn flush(&self) {
//         unsafe {
//             self.device.borrow().device.flush_mapped_memory_ranges(
//                 std::iter::once((self.memory, self.segment))
//             );
//         }
//     }
//     pub fn get_segment(&self) -> m::Segment {
//         self.segment.clone()
//     }
//     fn unmap(self) {
//         unsafe {
//             self.device.borrow().device.unmap_memory(self.memory)
//         }
//     }
// }

// impl<'a, B: Backend> Drop for MappedMemory<'a, B> {
//     fn drop(&mut self) {
//         unsafe {
//             std::ptr::read(self).unmap()
//         }   
//     }
// }

use gfxm::Block;
pub trait MapResource<B: Backend> {
    fn get_block<'a>(&'a mut self) -> &'a mut gfxm::MemoryBlock<B>;
    fn map<'a>(&'a mut self, device: &DeviceRc<B>) -> Result<gfxm::MappedRange<'a, B>, hal::device::MapError> {
        let block = self.get_block();
        block.map(&device.borrow().device, block.segment())
    }
    fn map_raw<'a>(&'a mut self, device: &DeviceRc<B>) -> Result<(*mut u8, Option<m::Segment>), hal::device::MapError> {
        let segment = self.get_block().segment().clone();
        let mut mapper = self.map(device)?;
        let (ptr, segment) = unsafe {
            mapper.write(&device.borrow().device, segment)?.forget()
        };
        Ok((ptr as *mut u8, segment))
    }
}

pub struct Buffer<B: Backend> {
    pub buffer: B::Buffer,
    pub block: gfxm::MemoryBlock<B>
}

impl<B: Backend> Buffer<B> {
    pub fn new(device: &DeviceRc<B>, heaps: &mut gfxm::Heaps<B>, size: u64, usage: hal::buffer::Usage, memory_usage: gfxm::MemoryUsage) -> Self {

        let device_ref = device.borrow();
        let mut buffer = unsafe { 
            device_ref.device
                .create_buffer(size, usage)
                .unwrap()
        };
        let req = unsafe {
            device_ref.device.get_buffer_requirements(&buffer)
        };
        //dbg!(&size, &usage);
        let block = heaps.allocate(
            &device_ref.device,
            req.type_mask as u32,
            memory_usage,
            gfxm::Kind::General,
            req.size,
            req.alignment
        ).unwrap();
        unsafe {
            device_ref.device
                .bind_buffer_memory(&block.memory(), block.segment().offset, &mut buffer)
                .unwrap();
        }
        Self {
            buffer,
            block
        }
    }

    pub fn dispose(self, device: &DeviceRef<B>, heaps: &mut gfxm::Heaps<B>) {
        unsafe {
            device.device.destroy_buffer(self.buffer);
            heaps.free(&device.device, self.block);
        }
    }
}

impl<B: Backend> MapResource<B> for Buffer<B> {
    fn get_block<'a>(&'a mut self) -> &'a mut gfxm::MemoryBlock<B> {
        &mut self.block
    }
}

pub struct ImageD2<B: Backend> {
    pub image: ManuallyDrop<B::Image>,
    pub requirements: hal::memory::Requirements,
    pub memory: ManuallyDrop<B::Memory>,
    pub image_view: ManuallyDrop<B::ImageView>,

    device: DeviceRc<B>
}

impl<B: Backend> ImageD2<B> {
    pub fn new(device: DeviceRc<B>, extent: Extent2D, format: f::Format, usage: i::Usage) -> Result<Self, &'static str> {
        unsafe {
            let device_ref = device.borrow();
            let mut the_image = device_ref.device
                .create_image(
                    hal::image::Kind::D2(extent.width, extent.height, 1, 1),
                    1,
                    format,
                    hal::image::Tiling::Optimal,
                    usage,
                    hal::image::ViewCapabilities::empty(),
                )
                .map_err(|_| "Couldn't crate the image!")?;
            let requirements = device_ref.device.get_image_requirements(&the_image);
            
            let memory = allocate_memory(&device, &requirements, m::Properties::DEVICE_LOCAL)?;
            device_ref.device
                .bind_image_memory(&memory, 0, &mut the_image)
                .map_err(|_| "Couldn't bind the image memory!")?;

            let aspects = {
                use {i::Usage as U, f::Aspects as A, f::Format as F};
                let mut a = A::empty();
                if usage.contains(U::COLOR_ATTACHMENT) {a.insert(A::COLOR)}
                if usage.contains(U::DEPTH_STENCIL_ATTACHMENT) {
                    a.insert(A::DEPTH);
                    // figure out if image is used as stencil
                    match format {
                        F::D16UnormS8Uint  |
                        F::D24UnormS8Uint  |
                        F::D32SfloatS8Uint => a.insert(A::STENCIL),
                        _ => {}
                    } 
                }
                a
            };
            let image_view = device_ref.device
                .create_image_view(
                    &the_image,
                    hal::image::ViewKind::D2,
                    format,
                    hal::format::Swizzle::NO,
                    hal::image::SubresourceRange {
                        aspects,
                        levels: 0..1,
                        layers: 0..1,
                    },
                )
                .map_err(|_| "Couldn't create the image view!")?;

            drop(device_ref);
            Ok(Self {
                image: ManuallyDrop::new(the_image),
                requirements,
                memory: ManuallyDrop::new(memory),
                image_view: ManuallyDrop::new(image_view),
                device: device
            })
        }
    }
}

impl<B: Backend> Drop for ImageD2<B> {
    fn drop(&mut self) {
        use core::ptr::read;
        let device = &self.device.borrow().device;
        unsafe {
            device.destroy_image_view(ManuallyDrop::into_inner(read(&self.image_view)));
            device.destroy_image(ManuallyDrop::into_inner(read(&self.image)));
            device.free_memory(ManuallyDrop::into_inner(read(&self.memory)));
        }
    }
}

pub struct CubemapImage<B: Backend> {
    pub image: ManuallyDrop<B::Image>,
    pub requirements: hal::memory::Requirements,
    pub memory: ManuallyDrop<B::Memory>,
    pub image_view: ManuallyDrop<B::ImageView>,

    device: DeviceRc<B>
}

impl<B: Backend> CubemapImage<B> {
    pub fn new(device: DeviceRc<B>, extent: hal::window::Extent2D) -> Result<Self, &'static str> {
        unsafe {
            let device_ref = device.borrow();
            let mut image = device_ref.device
                .create_image(
                    hal::image::Kind::D2(extent.width, extent.height, 1, 1),
                    1,
                    hal::format::Format::D32Sfloat,
                    hal::image::Tiling::Optimal,
                    hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
                    hal::image::ViewCapabilities::empty(),
                )
                .map_err(|_| "Couldn't crate the image!")?;
            let requirements = device_ref.device.get_image_requirements(&image);
            
            let memory = allocate_memory(&device, &requirements, m::Properties::DEVICE_LOCAL)?;
            device_ref.device
                .bind_image_memory(&memory, 0, &mut image)
                .map_err(|_| "Couldn't bind the image memory!")?;
            let image_view = device_ref.device
                .create_image_view(
                    &image,
                    hal::image::ViewKind::D2,
                    hal::format::Format::D32Sfloat,
                    hal::format::Swizzle::NO,
                    hal::image::SubresourceRange {
                        aspects: hal::format::Aspects::DEPTH,
                        levels: 0..1,
                        layers: 0..1,
                    },
                )
                .map_err(|_| "Couldn't create the image view!")?;

            drop(device_ref);
            Ok(Self {
                image: ManuallyDrop::new(image),
                requirements,
                memory: ManuallyDrop::new(memory),
                image_view: ManuallyDrop::new(image_view),
                device
            })
        }
    }
    /// construct a cubemap from a single image
    /// the faces are laid out as:
    ///    -Y
    /// -X -Z +X +Z
    ///    +Y
    /// the extent is the size of the whole image and tile_size is size of the individual tiles
    pub unsafe fn from_image(device: DeviceRc<B>, data: &Vec<u8>, format: Format, extent: Extent2D, size: u32) -> Result<Self, &'static str>{
        let device_ref = device.borrow();
        let mut staging_image = device_ref.device
            .create_image(
                hal::image::Kind::D2(extent.width, extent.height, 1, 1),
                1,
                format,
                hal::image::Tiling::Optimal,
                hal::image::Usage::TRANSFER_SRC,
                hal::image::ViewCapabilities::empty(),
            )
            .map_err(|_| "Couldn't create staging image!")?;

        let requirements = device_ref.device.get_image_requirements(&staging_image);
        assert!(requirements.size >= data.len() as u64);
        let staging_memory = allocate_memory(&device, &requirements, m::Properties::DEVICE_LOCAL)?;
        device.borrow().device
            .bind_image_memory(&staging_memory, 0, &mut staging_image)
            .map_err(|_| "Couldn't bind staging image memory!")?;

        let _cubemap = device_ref.device
        .create_image(
            hal::image::Kind::D2(size, size, 1, 1),
            1,
            format,
            hal::image::Tiling::Optimal,
            hal::image::Usage::TRANSFER_SRC,
            hal::image::ViewCapabilities::empty(),
        )
        .map_err(|_| "Couldn't create staging image!")?;

        unimplemented!()
    }
}

impl<B: Backend> Drop for CubemapImage<B> {
    fn drop(&mut self) {
        use core::ptr::read;
        let device = &self.device.borrow().device;
        unsafe {
            device.destroy_image_view(ManuallyDrop::into_inner(read(&self.image_view)));
            device.destroy_image(ManuallyDrop::into_inner(read(&self.image)));
            device.free_memory(ManuallyDrop::into_inner(read(&self.memory)));
        }
    }
}