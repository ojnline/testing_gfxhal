use super::resource;

use std::{cell::RefCell, mem::ManuallyDrop, ptr, rc::Rc};

use gfx_hal::{
    self as hal,
    adapter::{Adapter, MemoryType, PhysicalDevice},
    device::Device,
    format as f,
    window as w,
    image as i, 
    pool,
    prelude::*,
    queue::{QueueGroup},
    window::Surface,
    Backend, Instance,
};

use winit::window::Window;

pub struct BackendState<B: Backend> {
    pub instance: Option<B::Instance>,
    pub surface: ManuallyDrop<B::Surface>,
    pub adapter: Adapter<B>,
    /// Needs to be kept alive even if its not used directly
    #[allow(dead_code)]
    pub window: Window,
}

impl<B: Backend> BackendState<B> {
    // #[cfg(feature = "gl")]
    // pub fn new(
    //     wb: winit::window::WindowBuilder,
    //     event_loop: &winit::event_loop::EventLoop<()>,
    //     name_hint: &'static str,
    // ) -> Self {
    //     let (context, window) = {
    //         let builder =
    //             back::config_context(back::glutin::ContextBuilder::new(), back::ColorFormat::SELF, None)
    //                 .with_vsync(true);
    //         let windowed_context = builder.build_windowed(wb, event_loop).unwrap();
    //         unsafe {
    //             windowed_context
    //                 .make_current()
    //                 .expect("Unable to make context current")
    //                 .split()
    //         }
    //     };

    //     let surface = back::Surface::from_context(context);
    //     let mut adapters = surface.enumerate_adapters();
    //     println!("{:#?}", adapters);

    //     Self {
    //         surface,
    //         window,
    //         adapter: adapters.pop().unwrap(),
    //         instance: None,
    //     }
    // }

    #[cfg(any(
        feature = "vulkan",
        feature = "dx11",
        feature = "dx12",
        feature = "metal",
        feature = "gl"
    ))]
    pub fn new(
        wb: winit::window::WindowBuilder,
        event_loop: &winit::event_loop::EventLoop<()>,
        name_hint: &'static str,
    ) -> Self {
        let window = wb.build(event_loop).unwrap();
        let instance = B::Instance::create(name_hint, 1).expect("Failed to create an instance!");
        let surface = unsafe {
            instance
                .create_surface(&window)
                .expect("Failed to create a surface!")
        };
        let mut adapters = instance.enumerate_adapters();
        println!("{:?}", adapters);
        let adapter = adapters.remove(0);

        Self {
            surface: ManuallyDrop::new(surface),
            window,
            adapter,
            instance: Some(instance),
        }
    }

    #[allow(unused_variables)]
    #[cfg(not(any(
        feature = "vulkan",
        feature = "dx11",
        feature = "dx12",
        feature = "metal",
        feature = "gl",
    )))]
    pub fn new(
        wb: winit::window::WindowBuilder,
        event_loop: &winit::event_loop::EventLoop<()>,
        name_hint: &'static str,
    ) -> Self {
        unimplemented!("No backend selected.")
    }
}

impl<B: Backend> Drop for BackendState<B> {
    fn drop(&mut self) {
        if let Some(instance) = &self.instance {
            unsafe {
                let surface = ManuallyDrop::into_inner(ptr::read(&self.surface));
                instance.destroy_surface(surface);
            }
        }
    }
}
use std::cell::Ref;
pub type DeviceRc<B> = Rc<RefCell<DeviceState<B>>>;
pub type DeviceRef<'a, B> = Ref<'a, DeviceState<B>>;
pub struct DeviceState<B: Backend> {
    pub device: B::Device,
    pub queue_group: QueueGroup<B>,
    pub memory_types: Vec<MemoryType>,
    pub memory_properties: hal::adapter::MemoryProperties,
    pub limits: hal::Limits,
}

impl<B: Backend> DeviceState<B> {
    pub fn new(backend: &BackendState<B>) -> Self {
        let physical_device = &backend.adapter.physical_device;
        let memory_properties = physical_device.memory_properties();
        let memory_types = physical_device.memory_properties().memory_types;
        let limits = physical_device.limits();

        let (device, queue_group) = {
            let family = backend
                .adapter
                .queue_families
                .iter()
                .find(|family| {
                    backend.surface.supports_queue_family(family)
                        && family.queue_type().supports_graphics()
                })
                .unwrap();
            let mut gpu = unsafe {
                physical_device
                    .open(&[(family, &[1.0])], hal::Features::empty())
                    .unwrap()
            };

            (gpu.device, gpu.queue_groups.pop().unwrap())
        };

        Self {
            device,
            memory_types,
            memory_properties,
            limits,
            queue_group,
        }
    }
}

/// minimal information about a gpu image for the swapchain
pub struct DepthBundle<B: Backend> {
    pub image: B::Image,
    pub memory: B::Memory,
    pub view: B::ImageView,
}

pub struct SwapchainState<B: Backend> {
    pub swapchain: Option<B::Swapchain>,
    pub backbuffer: Option<Vec<B::Image>>, // the optional image is a depth buffer
    pub depth: Option<Vec<DepthBundle<B>>>,
    pub device: DeviceRc<B>,
    pub extent: i::Extent,
    pub format: f::Format,
}

impl<B: Backend> SwapchainState<B> {
    pub unsafe fn new(
        backend: &mut BackendState<B>,
        device: DeviceRc<B>,
        extent: w::Extent2D,
        depth_format_usage: Option<(f::Format, i::Usage)>,
    ) -> Self {
        let physical_device = &backend.adapter.physical_device;
        let caps = backend.surface.capabilities(physical_device);
        let formats = backend.surface.supported_formats(physical_device);
        let format = formats.map_or(f::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == f::ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let mut swap_config = w::SwapchainConfig::from_caps(&caps, format, extent);
        swap_config.present_mode = w::PresentMode::FIFO;
        let extent_2d = swap_config.extent;
        let extent = swap_config.extent.to_extent();
        let (swapchain, backbuffer) = device
            .borrow()
            .device
            .create_swapchain(&mut backend.surface, swap_config, None)
            .expect("Can't create swapchain");

        let depth = if depth_format_usage.is_some() {
            let (format, usage) = depth_format_usage.unwrap();
            Some(backbuffer
                .iter()
                .map(|_| {
                    let depth = resource::ImageD2::new(Rc::clone(&device), extent_2d, format, usage | i::Usage::DEPTH_STENCIL_ATTACHMENT).unwrap();
                    let bundle = DepthBundle {
                        // im sorry
                        image: ManuallyDrop::into_inner(ptr::read(&depth.image)),
                        memory: ManuallyDrop::into_inner(ptr::read(&depth.memory)),
                        view: ManuallyDrop::into_inner(ptr::read(&depth.image_view))
                    };
                    std::mem::forget(depth);
                    bundle
                })
                .collect()) 
            } else {
                None
            };

        let swapchain = SwapchainState {
            swapchain: Some(swapchain),
            backbuffer: Some(backbuffer),
            depth: depth,
            device,
            extent,
            format,
        };
        swapchain
    }
}

impl<B: Backend> Drop for SwapchainState<B> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .borrow()
                .device
                .destroy_swapchain(self.swapchain.take().unwrap());
        }
    }
}

pub struct FramebufferState<B: Backend> {
    pub framebuffers: Option<Vec<B::Framebuffer>>,
    pub framebuffer_fences: Option<Vec<B::Fence>>,
    pub command_pools: Option<Vec<B::CommandPool>>,
    pub command_buffer_lists: Vec<Vec<B::CommandBuffer>>,
    pub frame_count: usize,
    pub frame_images: Option<Vec<(B::Image, B::ImageView)>>,
    pub depth_images: Option<Vec<DepthBundle<B>>>,
    pub acquire_semaphores: Option<Vec<B::Semaphore>>,
    pub present_semaphores: Option<Vec<B::Semaphore>>,
    pub last_ref: usize,
    pub device: DeviceRc<B>,
}

impl<B: Backend> FramebufferState<B> {
    pub unsafe fn new(
        device: DeviceRc<B>,
        render_pass: &B::RenderPass,
        swapchain: &mut SwapchainState<B>,
    ) -> Self {
        let (frame_images, framebuffers) = {
            let extent = i::Extent {
                width: swapchain.extent.width as _,
                height: swapchain.extent.height as _,
                depth: 1,
            };
            let pairs = swapchain
                .backbuffer
                .take()
                .unwrap()
                .into_iter()
                .map(|color| {
                    let color_view = device
                        .borrow()
                        .device
                        .create_image_view(
                            &color,
                            i::ViewKind::D2,
                            swapchain.format,
                            f::Swizzle::NO,
                            i::SubresourceRange {
                                aspects: f::Aspects::COLOR,
                                levels: 0..1,
                                layers: 0..1,
                            },
                        )
                        .unwrap();
                    (color, color_view)
                })
                .collect::<Vec<_>>();
            let fbos = pairs
                .iter()
                .enumerate()
                .map(|(i, &(_, ref rtv))| {
                    let attachments = if swapchain.depth.is_some() {
                        vec![rtv, &swapchain.depth.as_ref().unwrap()[i].view]
                    } else {
                        vec![rtv]
                    };
                    device
                        .borrow()
                        .device
                        .create_framebuffer(render_pass, attachments, extent)
                        .unwrap()
                })
                .collect();
            (pairs, fbos)
        };

        // GL can have zero
        let iter_count = frame_images.len().max(1);

        let mut fences: Vec<B::Fence> = vec![];
        let mut command_pools: Vec<_> = vec![];
        let mut command_buffer_lists = Vec::new();
        let mut acquire_semaphores: Vec<B::Semaphore> = vec![];
        let mut present_semaphores: Vec<B::Semaphore> = vec![];

        for _ in 0..iter_count {
            fences.push(device.borrow().device.create_fence(true).unwrap());
            command_pools.push(
                device
                    .borrow()
                    .device
                    .create_command_pool(
                        device.borrow().queue_group.family,
                        pool::CommandPoolCreateFlags::empty(),
                    )
                    .expect("Can't create command pool"),
            );
            command_buffer_lists.push(Vec::new());

            acquire_semaphores.push(device.borrow().device.create_semaphore().unwrap());
            present_semaphores.push(device.borrow().device.create_semaphore().unwrap());
        }

        let depth_images = if swapchain.depth.is_some() {
            Some(swapchain.depth.take().unwrap())
        } else { None };

        FramebufferState {
            frame_count: iter_count,
            frame_images: Some(frame_images),
            depth_images,
            framebuffers: Some(framebuffers),
            framebuffer_fences: Some(fences),
            command_pools: Some(command_pools),
            command_buffer_lists,
            present_semaphores: Some(present_semaphores),
            acquire_semaphores: Some(acquire_semaphores),
            device,
            last_ref: 0,
        }
    }

    pub fn next_acq_pre_pair_index(&mut self) -> usize {
        if self.last_ref >= self.acquire_semaphores.as_ref().unwrap().len() {
            self.last_ref = 0
        }

        let ret = self.last_ref;
        self.last_ref += 1;
        ret
    }

    pub fn get_frame_data(
        &mut self,
        frame_id: Option<usize>,
        sem_index: Option<usize>,
    ) -> (
        Option<(
            &mut B::Fence,
            &mut B::Framebuffer,
            Option<&mut DepthBundle<B>>,
            &mut B::CommandPool,
            &mut Vec<B::CommandBuffer>,
        )>,
        Option<(&mut B::Semaphore, &mut B::Semaphore)>,
    ) {
        (
            if let Some(fid) = frame_id {
                Some((
                    &mut self.framebuffer_fences.as_mut().unwrap()[fid],
                    &mut self.framebuffers.as_mut().unwrap()[fid],
                    if self.depth_images.is_some() {
                        Some(&mut self.depth_images.as_mut().unwrap()[fid])
                    } else { None },
                    &mut self.command_pools.as_mut().unwrap()[fid],
                    &mut self.command_buffer_lists[fid],
                ))
            } else {
                None
            },
            if let Some(sid) = sem_index {
                Some((
                    &mut self.acquire_semaphores.as_mut().unwrap()[sid],
                    &mut self.present_semaphores.as_mut().unwrap()[sid],
                ))
            } else {
                None
            },
        )
    }
}

impl<B: Backend> Drop for FramebufferState<B> {
    fn drop(&mut self) {
        let device = &self.device.borrow().device;

        unsafe {
            for fence in self.framebuffer_fences.take().unwrap() {
                device.wait_for_fence(&fence, !0).unwrap();
                device.destroy_fence(fence);
            }

            for (mut command_pool, command_buffer_list) in self
                .command_pools
                .take()
                .unwrap()
                .into_iter()
                .zip(self.command_buffer_lists.drain(..))
            {
                command_pool.free(command_buffer_list);
                device.destroy_command_pool(command_pool);
            }

            for acquire_semaphore in self.acquire_semaphores.take().unwrap() {
                device.destroy_semaphore(acquire_semaphore);
            }

            for present_semaphore in self.present_semaphores.take().unwrap() {
                device.destroy_semaphore(present_semaphore);
            }

            for framebuffer in self.framebuffers.take().unwrap() {
                device.destroy_framebuffer(framebuffer);
            }
            
            for (_, rtv) in self.frame_images.take().unwrap() {
                device.destroy_image_view(rtv);
            }

            if self.depth_images.is_some() {
                for depth in self.depth_images.take().unwrap() {
                    device.destroy_image_view(depth.view);
                    device.destroy_image(depth.image);
                    device.free_memory(depth.memory);
                }
            }
        }
    }
}
