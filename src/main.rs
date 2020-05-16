mod resource;
mod util;
mod asset_manager;
use asset_manager as mngr;
mod common;
//mod render_graph;

use gfx_memory as gfxm;
use gfx_hal::{
    self as hal,
    device::Device,
    pso::{self, DescriptorPool},
    window::{Extent2D, Surface},
    Backend,
    image as i,
    format as f,
    buffer as b,
};


use util::*;
use common::*;

use legion as leg;

use leg::prelude::*;
use std::{cell::RefCell, rc::Rc};

use ncollide3d::shape::{ShapeHandle, Ball};
use np::object::{
    ColliderDesc, Ground, BodyPartHandle, RigidBodyDesc
}; 
use np::material::{MaterialHandle, BasicMaterial};

#[macro_use] extern crate microprofile;

#[cfg(feature = "dx11")]
extern crate gfx_backend_dx11 as back;
#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(any(feature = "gl"))]
extern crate gfx_backend_gl as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;
#[cfg(not(any(
    feature = "dx11",
    feature = "dx12",
    feature = "gl",
    feature = "metal",
    feature = "vulkan"
)))]
extern crate gfx_backend_empty as back;

#[derive(Clone, Copy)]
pub struct Model {
    handle: mngr::AssetId,
    scale: f32
}

enum TranformationSource {
    Static(na::Isometry3<f32>),
    PhysicsBody(np::object::DefaultBodyHandle)
}
pub struct Transformation(TranformationSource);

#[derive(Clone)]
struct BodyData {
    is_attracted: bool
}

struct Attractor {
    mass: f32
}

enum Direction {
    Left = 0,
    Right = 1,
    Up = 2,
    Down = 3,
    Forward = 4,
    Backward = 5,
}

#[derive(Clone, Copy)]
pub struct Camera {
    pitch: f32,
    yaw: f32,
    pos: Pos3,
    aspect: f32,
    fov: f32,
    near: f32,
    far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            pitch: 0.,
            yaw: 0.,
            pos: Pos3::origin(),
            aspect: 1.,
            fov: f32::to_radians(50.0),
            near: 0.1,
            far: 1.,
        }
    }
}

impl Camera {
    pub fn get_view_direction(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        )
    }
    pub fn get_view(&self) -> Mat4 {
        Mat4::look_at_lh(
            &(self.pos + self.get_view_direction()),
            &self.pos,
            &Vec3::new(0., -1., 0.),
        )
    }
    pub fn get_projection(&self) -> Mat4 {
        Mat4::new_perspective(self.aspect, self.fov, self.near, self.far)
    }
    pub fn get_transform(&self) -> Mat4 {
        self.get_projection() * self.get_view()
    }
    pub fn update(&mut self, delta: f32, keys: &[bool; 6]) {
        let speed = delta * 1.5;
        let forward = self.get_view_direction();
        let up = Vec3::new(0.,1.,0.);
        let sideways = forward.cross(&up).normalize();

        if keys[Direction::Forward as usize] {
            self.pos.coords += forward * speed;
        }
        if keys[Direction::Backward as usize] {
            self.pos.coords -= forward * speed;
        }
        if keys[Direction::Right as usize] {
            self.pos.coords -= sideways * speed;
        }
        if keys[Direction::Left as usize] {
            self.pos += sideways * speed;
        }
        if keys[Direction::Up as usize] {
            self.pos.coords += up * speed;
        }
        if keys[Direction::Down as usize] {
            self.pos.coords -= up * speed;
        }
    }
}

#[repr(C)]
struct PushConstants {
    isometry: na::Isometry3<f32>,
    scale: f32,
    view: Mat4
}

const APP_NAME: &'static str = "haelp";
const WINDOW_SIZE: [u32; 2] = [512, 512];

use std::mem::ManuallyDrop;
#[allow(unused_variables)]
struct Application<B: Backend> {
    backend: BackendState<B>,
    device: Rc<RefCell<DeviceState<B>>>,
    framebuffer: ManuallyDrop<FramebufferState<B>>,
    extent: Extent2D,
    swapchain: ManuallyDrop<SwapchainState<B>>,
    recreate_swapchain: bool,
    descriptor_pool: ManuallyDrop<B::DescriptorPool>,
    descriptor_sets: Vec<B::DescriptorSet>,
    mesh_pipeline: ManuallyDrop<B::GraphicsPipeline>,
    mesh_pipeline_layout: ManuallyDrop<B::PipelineLayout>,
    post_pipeline: ManuallyDrop<B::GraphicsPipeline>,
    post_pipeline_layout: ManuallyDrop<B::PipelineLayout>,
    mesh_render_pass: ManuallyDrop<B::RenderPass>,
    post_render_pass: ManuallyDrop<B::RenderPass>,
    viewport: pso::Viewport,

    is_focused: bool,
    last_update: std::time::Instant,
    frame_samples: u32,
    frametime_acc: f32,
    camera: Camera,
    keys: [bool; 6],
    resource_manager: mngr::ResourceManager<B>,
    universe: Universe,
    world: World,
    fallback_model: Model,

    last_step: std::time::Instant,
    geometrical_world: np::world::DefaultGeometricalWorld<f32>,
    mechanical_world: np::world::DefaultMechanicalWorld<f32>,
    body_set: np::object::DefaultBodySet<f32>,
    collider_set: np::object::DefaultColliderSet<f32>,
    constraint_set: np::joint::DefaultJointConstraintSet<f32>,
    force_generator_set: np::force_generator::DefaultForceGeneratorSet<f32>
}

impl<B: Backend> Application<B> {
    fn new(mut backend: util::BackendState<B>) -> Self {
        let device = Rc::new(RefCell::new(util::DeviceState::new(&backend)));
        let device_ref = device.borrow();

        let surface_color_format = {
            use hal::format::{ChannelType, Format};

            let supported_formats = backend
                .surface
                .supported_formats(&backend.adapter.physical_device);
            supported_formats.map_or(Format::Rgba8Srgb, |formats| {
                formats
                    .iter()
                    .find(|format| format.base_format().1 == ChannelType::Srgb)
                    .map(|format| *format)
                    .unwrap_or(formats[0])
            })
        };

        let extent = {
            let size = backend
                .window
                .inner_size()
                .to_logical(backend.window.scale_factor());
            hal::window::Extent2D {
                width: size.width,
                height: size.height,
            }
        };

        let mesh_render_pass = {
            use hal::format::Format;
            use hal::image::Layout;
            use hal::pass::{
                Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, SubpassDesc,
            };

            let color_attachment = Attachment {
                format: Some(surface_color_format),
                samples: 1,
                ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
                stencil_ops: AttachmentOps::DONT_CARE,
                layouts: Layout::Undefined..Layout::Present,
            };

            let depth_attachment = Attachment {
                format: Some(Format::D32Sfloat),
                samples: 1,
                ops: AttachmentOps {
                    load: AttachmentLoadOp::Clear,
                    store: AttachmentStoreOp::Store,
                },
                stencil_ops: AttachmentOps::DONT_CARE,
                layouts: Layout::Undefined..Layout::DepthStencilAttachmentOptimal,
            };

            let subpass = SubpassDesc {
                colors: &[(0, Layout::ColorAttachmentOptimal)],
                depth_stencil: Some(&(1, Layout::DepthStencilAttachmentOptimal)),
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            unsafe {
                device_ref
                    .device
                    .create_render_pass(&[color_attachment, depth_attachment], &[subpass], &[])
                    .expect("TODO")
            }
        };

        let post_render_pass = {
            use hal::image::Layout;
            use hal::pass::{
                Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, SubpassDesc,
            };

            let color_attachment = Attachment {
                format: Some(surface_color_format),
                samples: 1,
                ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
                stencil_ops: AttachmentOps::DONT_CARE,
                layouts: Layout::Undefined..Layout::Present,
            };

            let subpass = SubpassDesc {
                colors: &[(0, Layout::Present)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            unsafe {
                device_ref
                    .device
                    .create_render_pass(&[color_attachment], &[subpass], &[])
                    .expect("TODO")
            }
        };

        let mut swapchain = unsafe { 
            SwapchainState::new(&mut backend, Rc::clone(&device), extent, Some((f::Format::D32Sfloat, i::Usage::SAMPLED))) 
        };

        let framebuffer = unsafe {
            FramebufferState::new(Rc::clone(&device), &mesh_render_pass, &mut swapchain) 
        };

        let image_desc_layout = {
            let bindings = vec![pso::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: pso::DescriptorType::Image {
                        ty: pso::ImageDescriptorType::Sampled {
                            with_sampler: false,
                        },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::FRAGMENT,
                    immutable_samplers: false,
                }];
            unsafe {
                device_ref.device.create_descriptor_set_layout(bindings, &[]).ok()
            }
        };

        let sampler_desc_layout = unsafe {
            device_ref.device.create_descriptor_set_layout(
                vec![pso::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: pso::DescriptorType::Sampler,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::FRAGMENT,
                    immutable_samplers: false,
                }], 
                &[]
            ).ok()
        };

        let mut desc_pool = unsafe { 
        device_ref
            .device
            .create_descriptor_pool(
                framebuffer.frame_count+1, // # of sets
                &[
                    pso::DescriptorRangeDesc {
                        ty: pso::DescriptorType::Sampler,
                        count: 1,
                    },
                    pso::DescriptorRangeDesc {
                        ty: pso::DescriptorType::Image {
                            ty: pso::ImageDescriptorType::Sampled {
                                with_sampler: false,
                            },
                        },
                        count: framebuffer.frame_count,
                    },
                ],
                pso::DescriptorPoolCreateFlags::empty(),
            )
            .unwrap()
        };

        let mut sets = Vec::new();

        let sampler = unsafe {
            device_ref.device
                .create_sampler(&i::SamplerDesc::new(i::Filter::Linear, i::WrapMode::Clamp))
                .expect("Can't create sampler")
        };

        // alocate sampler set
        unsafe {
            let set = desc_pool.allocate_set(sampler_desc_layout.as_ref().unwrap()).unwrap();
            device_ref.device.write_descriptor_sets(std::iter::once(pso::DescriptorSetWrite {
                set: &set,
                binding: 0,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Sampler(&sampler)),
            }));
            sets.push(set);
        };

        framebuffer.depth_images.as_ref().unwrap().iter().for_each(|depth_image| {
            unsafe {
                let set = desc_pool.allocate_set(image_desc_layout.as_ref().unwrap()).unwrap();
                device_ref.device.write_descriptor_sets(std::iter::once(pso::DescriptorSetWrite {
                    set: &set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Image(
                        &depth_image.view,
                        i::Layout::ShaderReadOnlyOptimal,
                    ))
                }));
                sets.push(set);
            }
        });
    
        let post_pipeline_layout = unsafe {
            device_ref
                .device
                .create_pipeline_layout(&[sampler_desc_layout.unwrap(), image_desc_layout.unwrap()], &[])
                .expect("TODO")
        };

        let post_pipeline = {
            use glsl_to_spirv::ShaderType;
            use hal::pass::Subpass;
            use hal::pso::{
                BlendState, ColorBlendDesc, ColorMask, EntryPoint, Face,
                GraphicsPipelineDesc, GraphicsShaderSet, Primitive, Rasterizer, Specialization,
            };

            let compile_shader = |glsl, shader_type| {
                use std::io::{Cursor, Read};

                let mut spirv_bytes = vec![];
                let mut compiled_file = glsl_to_spirv::compile(glsl, shader_type).expect("TODO");
                compiled_file.read_to_end(&mut spirv_bytes).expect("TODO");
                let spirv = pso::read_spirv(Cursor::new(&spirv_bytes)).expect("TODO");
                unsafe {
                    device_ref
                        .device
                        .create_shader_module(&spirv)
                        .expect("TODO")
                }
            };

            let vertex_shader_module =
                compile_shader(include_str!("../assets/shaders/fullscreen_triangle.vert"), ShaderType::Vertex);

            let fragment_shader_module =
                compile_shader(include_str!("../assets/shaders/postprocess.frag"), ShaderType::Fragment);

            let (vs_entry, fs_entry) = (
                EntryPoint {
                    entry: "main",
                    module: &vertex_shader_module,
                    specialization: Specialization::default(),
                },
                EntryPoint {
                    entry: "main",
                    module: &fragment_shader_module,
                    specialization: Specialization::default(),
                },
            );

            let shader_entries = GraphicsShaderSet {
                vertex: vs_entry,
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(fs_entry),
            };

            let mut pipeline_desc = GraphicsPipelineDesc::new(
                shader_entries,
                Primitive::TriangleList,
                Rasterizer {
                    //polygon_mode: pso::PolygonMode::Line(hal::pso::State::Static(1.)),
                    cull_face: Face::BACK,
                    front_face: pso::FrontFace::CounterClockwise,
                    ..Rasterizer::FILL
                },
                &post_pipeline_layout,
                Subpass {
                    index: 0,
                    main_pass: &post_render_pass,
                },
            );

            pipeline_desc.blender.targets.push(ColorBlendDesc {
                mask: ColorMask::ALL,
                blend: Some(BlendState::ALPHA),
            });

            unsafe {
                let pipeline = device_ref
                    .device
                    .create_graphics_pipeline(&pipeline_desc, None)
                    .unwrap();

                device_ref
                    .device
                    .destroy_shader_module(vertex_shader_module);
                device_ref
                    .device
                    .destroy_shader_module(fragment_shader_module);

                pipeline
            }
        };

        let mesh_pipeline_layout = unsafe {
            device_ref
                .device
                .create_pipeline_layout(&[], &[(pso::ShaderStageFlags::VERTEX, 0..std::mem::size_of::<PushConstants>() as u32)])
                .expect("TODO")
        };

        let mesh_pipeline = {
            use glsl_to_spirv::ShaderType;
            use hal::pass::Subpass;
            use hal::pso::{
                BlendState, ColorBlendDesc, ColorMask, EntryPoint, Face,
                GraphicsPipelineDesc, GraphicsShaderSet, Primitive, Rasterizer, Specialization,
            };

            let compile_shader = |glsl, shader_type| {
                use std::io::{Cursor, Read};

                let mut spirv_bytes = vec![];
                let mut compiled_file = glsl_to_spirv::compile(glsl, shader_type).expect("TODO");
                compiled_file.read_to_end(&mut spirv_bytes).expect("TODO");
                let spirv = pso::read_spirv(Cursor::new(&spirv_bytes)).expect("TODO");
                unsafe {
                    device_ref
                        .device
                        .create_shader_module(&spirv)
                        .expect("TODO")
                }
            };
            let vertex_shader_module =
                compile_shader(include_str!("../assets/shaders/mesh.vert"), ShaderType::Vertex);

            let fragment_shader_module =
                compile_shader(include_str!("../assets/shaders/mesh.frag"), ShaderType::Fragment);

            let (vs_entry, fs_entry) = (
                EntryPoint {
                    entry: "main",
                    module: &vertex_shader_module,
                    specialization: Specialization::default(),
                },
                EntryPoint {
                    entry: "main",
                    module: &fragment_shader_module,
                    specialization: Specialization::default(),
                },
            );

            let shader_entries = GraphicsShaderSet {
                vertex: vs_entry,
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(fs_entry),
            };

            let mut pipeline_desc = GraphicsPipelineDesc::new(
                shader_entries,
                Primitive::TriangleList,
                Rasterizer {
                    //polygon_mode: pso::PolygonMode::Line(hal::pso::State::Static(1.)),
                    cull_face: Face::NONE,
                    front_face: pso::FrontFace::CounterClockwise,
                    ..Rasterizer::FILL
                },
                &mesh_pipeline_layout,
                Subpass {
                    index: 0,
                    main_pass: &mesh_render_pass,
                },
            );

            pipeline_desc.blender.targets.push(ColorBlendDesc {
                mask: ColorMask::ALL,
                blend: Some(BlendState::ALPHA),
            });

            {
                use hal::format::Format;
                use hal::pso::{AttributeDesc, Element, VertexBufferDesc, VertexInputRate};

                pipeline_desc.depth_stencil = hal::pso::DepthStencilDesc {
                    depth: Some(hal::pso::DepthTest {
                        fun: gfx_hal::pso::Comparison::LessEqual,
                        write: true,
                    }),
                    depth_bounds: false,
                    stencil: None,
                };

                pipeline_desc.vertex_buffers.push(VertexBufferDesc {
                    binding: 0,
                    stride: std::mem::size_of::<mngr::PosNorm>() as u32,
                    rate: VertexInputRate::Vertex,
                });

                // position
                pipeline_desc.attributes.push(AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: Element {
                        format: Format::Rgb32Sfloat,
                        offset: 0,
                    },
                });
                // normal
                pipeline_desc.attributes.push(AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: Element {
                        format: Format::Rgb32Sfloat,
                        offset: 12,
                    },
                });
            }

            unsafe {
                let pipeline = device_ref
                    .device
                    .create_graphics_pipeline(&pipeline_desc, None)
                    .expect("TODO");

                device_ref
                    .device
                    .destroy_shader_module(vertex_shader_module);
                device_ref
                    .device
                    .destroy_shader_module(fragment_shader_module);

                pipeline
            }
        };

        let viewport = pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as i16,
                h: extent.height as i16,
            },
            depth: 0.0..1.0,
        };

        let general_config = gfxm::GeneralConfig {
            block_size_granularity: 524_288, // .5 mib
            max_chunk_size: 33_554_432, // 16 mib
            min_device_allocation: 33_554_432 // 32 mib
        };
        let linear_config = gfxm::LinearConfig {
            linear_size: 134_217_728
        };
        let heaps = unsafe {gfxm::Heaps::new(&device_ref.memory_properties, general_config, linear_config, device_ref.limits.non_coherent_atom_size as u64)};
        drop(device_ref);
        let mut manager = mngr::ResourceManager::new(Rc::clone(&device), heaps, Default::default());
        let lucy = manager.load_obj(std::path::Path::new("assets/models/lucy.obj")).unwrap();
        let bunny = manager.load_obj(std::path::Path::new("assets/models/bunny.obj")).unwrap();
        // let _ball = manager.load_obj(std::path::Path::new("assets/models/sphere.obj")).unwrap();
        manager.flush_transfers();
        
        let mut geometrical_world = np::world::DefaultGeometricalWorld::new();
        let mut mechanical_world = np::world::DefaultMechanicalWorld::new(vec3(0.,0.,0.));
        let mut body_set = np::object::DefaultBodySet::new();
        let mut collider_set = np::object::DefaultColliderSet::new();
        let mut constraint_set = np::joint::DefaultJointConstraintSet::new();
        let mut force_generator_set = np::force_generator::DefaultForceGeneratorSet::new();

        let universe = Universe::new();
        let mut world = universe.create_world();

        let sphere = ShapeHandle::new(Ball::new(1.));

        let ground_handle = body_set.insert(Ground::new());
        let co = ColliderDesc::new(sphere)
            .material(MaterialHandle::new(BasicMaterial::new(0.1, 1.)))
            .build(BodyPartHandle(ground_handle, 0));
        collider_set.insert(co);

        let model = |handle: mngr::AssetId| {
            Model{handle: handle.clone(), scale: 1.}
        };

        let bunny_model = model(bunny);
        let lucy_model = model(lucy);

        world.insert(
            (),
            body_set.iter().map(|body| {
                (TranformationSource::PhysicsBody(body.0), lucy_model, Attractor{mass: 20.})
            }
            )
        );

        Self {
            backend,
            device,
            swapchain: ManuallyDrop::new(swapchain),
            recreate_swapchain: false,
            descriptor_pool: ManuallyDrop::new(desc_pool),
            descriptor_sets: sets,
            framebuffer: ManuallyDrop::new(framebuffer),
            mesh_render_pass: ManuallyDrop::new(mesh_render_pass),
            mesh_pipeline_layout: ManuallyDrop::new(mesh_pipeline_layout),
            mesh_pipeline: ManuallyDrop::new(mesh_pipeline),
            post_render_pass: ManuallyDrop::new(post_render_pass),
            post_pipeline_layout: ManuallyDrop::new(post_pipeline_layout),
            post_pipeline: ManuallyDrop::new(post_pipeline),
            viewport,
            extent,

            is_focused: true,
            last_update: std::time::Instant::now(),
            frame_samples: 0,
            frametime_acc: 0.,
            camera: Camera {
                far: 20.,
                ..Default::default()
            },
            keys: [false; 6],
            fallback_model: bunny_model,
            resource_manager: manager,
            universe,
            world,

            last_step: std::time::Instant::now(),
            geometrical_world,
            mechanical_world,
            body_set,
            collider_set,
            constraint_set,
            force_generator_set
        }
    }

    fn recreate_swapchain(&mut self) {
        self.device.borrow().device.wait_idle().unwrap();

        unsafe {
            drop(ManuallyDrop::into_inner(std::ptr::read(&self.swapchain)));
            self.swapchain = ManuallyDrop::new(SwapchainState::new(
                &mut self.backend,
                Rc::clone(&self.device),
                self.extent,
                Some((f::Format::D32Sfloat, i::Usage::SAMPLED)),
            ));
            drop(ManuallyDrop::into_inner(std::ptr::read(&self.framebuffer)));
            self.framebuffer = ManuallyDrop::new(FramebufferState::new(
                Rc::clone(&self.device),
                &self.mesh_render_pass,
                &mut self.swapchain,
            ));
            self.viewport = pso::Viewport {
                rect: pso::Rect {
                    x: 0,
                    y: 0,
                    w: self.extent.width as i16,
                    h: self.extent.height as i16,
                },
                depth: 0.0..1.0,
            };
        }
    }
    fn draw(&mut self) {
        if self.extent.width == 0 || self.extent.height == 0 {
            return;
        }

        use hal::{
            command::{
                ClearColor, ClearDepthStencil, ClearValue, CommandBuffer, CommandBufferFlags, Level,
            },
            pool::CommandPool,
            queue::{CommandQueue, Submission},
            window::Swapchain,
        };
        use std::iter;

        if self.recreate_swapchain {
            self.recreate_swapchain();
            self.recreate_swapchain = false;
        }

        let sem_index = self.framebuffer.next_acq_pre_pair_index();

        let frame: hal::window::SwapImageIndex = unsafe {
            let (acquire_semaphore, _) = self
                .framebuffer
                .get_frame_data(None, Some(sem_index))
                .1
                .unwrap();
            match self.swapchain.swapchain.as_mut().unwrap().acquire_image(
                !0,
                Some(acquire_semaphore),
                None,
            ) {
                Ok((i, _)) => i,
                Err(_) => {
                    self.recreate_swapchain = true;
                    return;
                }
            }
        };

        let (fid, sid) = self
            .framebuffer
            .get_frame_data(Some(frame as usize), Some(sem_index));

        let (framebuffer_fence, framebuffer, _depth_buffer, command_pool, command_buffers) = fid.unwrap();
        let (image_acquired, image_present) = sid.unwrap();

        unsafe {
            self.device
                .borrow()
                .device
                .wait_for_fence(framebuffer_fence, !0)
                .unwrap();
            self.device
                .borrow()
                .device
                .reset_fence(framebuffer_fence)
                .unwrap();
            command_pool.reset(false);

            // time to finally render
            let mut cmd_buffer = match command_buffers.pop() {
                Some(cmd_buffer) => cmd_buffer,
                None => command_pool.allocate_one(Level::Primary),
            };

            cmd_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);

            cmd_buffer.set_viewports(0, &[self.viewport.clone()]);
            cmd_buffer.set_scissors(0, &[self.viewport.rect]);
            cmd_buffer.bind_graphics_pipeline(&self.mesh_pipeline);
            cmd_buffer.begin_render_pass(
                &self.mesh_render_pass,
                &framebuffer,
                self.viewport.rect,
                &[
                    ClearValue {
                        color: ClearColor {
                            float32: [0.1, 0.5, 0.5, 1.0],
                        },
                    },
                    ClearValue {
                        depth_stencil: ClearDepthStencil {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                ],
                hal::command::SubpassContents::Inline,
            );
            
            let draw_query = <(Read<TranformationSource>, Read<Model>)>::query();

            for (transformation, model) in draw_query.iter(&mut self.world) {
                let mesh = self.resource_manager.get_asset(&model.handle).unwrap();
                    
                if let mngr::Asset::Mesh{vertices, indices, vertex_size, index_size, ..} = mesh {
                    
                    let mut push_constants: PushConstants = std::mem::MaybeUninit::zeroed().assume_init();
                    push_constants.scale = model.scale;
                    push_constants.isometry = match *transformation {
                        TranformationSource::Static(isometry) => isometry,
                        TranformationSource::PhysicsBody(handle) => self.body_set.get(handle).unwrap().part(0).unwrap().position(),
                    };
                    push_constants.view = self.camera.get_transform();

                    cmd_buffer.push_graphics_constants(
                        &self.mesh_pipeline_layout,
                        hal::pso::ShaderStageFlags::VERTEX,
                        0,
                        std::slice::from_raw_parts(std::mem::transmute(&push_constants), std::mem::size_of::<PushConstants>() / 4)
                    );
                    cmd_buffer.bind_vertex_buffers(0, std::iter::once((&vertices.buffer, b::SubRange::WHOLE)));
                    
                    if let Some(indices) = indices {
                        let index_view = b::IndexBufferView {
                            buffer: &indices.buffer,
                            range: b::SubRange::WHOLE,
                            index_type: hal::IndexType::U32
                        };
                        cmd_buffer.bind_index_buffer(index_view);
                        cmd_buffer.draw_indexed(0..(index_size/4), 0, 0..1);
                    } else {
                        cmd_buffer.draw(0..(vertex_size/std::mem::size_of::<mngr::PosNorm>() as u32), 0..1);
                    }
                }
            }
            cmd_buffer.end_render_pass();
            // let image_barrier = hal::memory::Barrier::Image {
            //     states: (i::Access::empty(), i::Layout::General)
            //         ..(
            //             i::Access::SHADER_READ | i::Access::SHADER_WRITE,
            //             i::Layout::General
            //         ),
            //     target: &depth_buffer.as_ref().unwrap().image,
            //     families: None,
            //     range: i::SubresourceRange {
            //         aspects: f::Aspects::DEPTH,
            //         levels: 0..1,
            //         layers: 0..1,
            //     },
            // };
            // cmd_buffer.pipeline_barrier(
            //     pso::PipelineStage::TOP_OF_PIPE..pso::PipelineStage::FRAGMENT_SHADER,
            //     hal::memory::Dependencies::DEVICE_GROUP,
            //     &[image_barrier],
            // );
            // cmd_buffer.bind_graphics_pipeline(&self.post_pipeline);
            // cmd_buffer.begin_render_pass(
            //     &self.post_render_pass,
            //     &framebuffer,
            //     self.viewport.rect,
            //     &[
            //         ClearValue {
            //             color: ClearColor {
            //                 float32: [0.0, 0.0, 0.0, 0.0],
            //             },
            //         },
            //         ],
            //         hal::command::SubpassContents::Inline,
            //     );
            // cmd_buffer.bind_graphics_descriptor_sets(
            //     &self.post_pipeline_layout, 
            //     0,
            //     vec![&self.descriptor_sets[0], &self.descriptor_sets[frame as usize+1]],
            //     Vec::<u32>::new(),
            // );
            // cmd_buffer.draw(0..3, 0..1);
            // cmd_buffer.end_render_pass();
            cmd_buffer.finish();

            let submission = Submission {
                command_buffers: iter::once(&cmd_buffer),
                wait_semaphores: iter::once((&*image_acquired, pso::PipelineStage::BOTTOM_OF_PIPE)),
                signal_semaphores: iter::once(&*image_present),
            };

            self.device.borrow_mut().queue_group.queues[0]
                .submit(submission, Some(framebuffer_fence));
            command_buffers.push(cmd_buffer);

            // present frame
            if let Err(_) = self.swapchain.swapchain.as_ref().unwrap().present(
                &mut self.device.borrow_mut().queue_group.queues[0],
                frame,
                Some(&*image_present),
            ) {
                self.recreate_swapchain = true;
                return;
            }
        }
    }
    fn update(&mut self) {
        microprofile::scope!("main", "update");

        let delta = self.last_update.elapsed().as_secs_f32();
        self.frametime_acc += delta;
        self.frame_samples += 1;

        if self.frametime_acc > 0.1 {
            self.frametime_acc /= self.frame_samples as f32; 
            self.backend.window.set_title(format!("{:.5} {:.3}", self.frametime_acc, 1./self.frametime_acc).as_str());
            self.frametime_acc = 0.;
            self.frame_samples = 0;
        }

        self.last_update = std::time::Instant::now();
        self.camera.update(delta, &self.keys);

        let timestep = self.mechanical_world.timestep();
        
        if self.last_step.elapsed().as_secs_f32() > timestep {
        
            microprofile::scope!("main", "attractors");
            
            self.last_step += std::time::Duration::from_secs_f32(timestep);
            
            let attractor_query = <(Read<TranformationSource>, Read<Attractor>)>::query();
            // some processing is needed so this is done outside the big loop
            // FIXME is this necessary or worth it? Possible compiler magic
            let attractors = attractor_query.iter_entities(&mut self.world);
            // iterate through the attractors and if one's body has been removed remove it too
            
            // why is the borrow checker so mean
            let body_set = &self.body_set;

            let mut attractor_data: Vec<(Vec3, f32)> = Vec::new();
            let mut delete_attractors = Vec::new();

            attractors.for_each(|(entity, (transformation, attractor))|
                match *transformation {
                    TranformationSource::Static(pos) => {
                        attractor_data.push((pos.translation.vector, attractor.mass));
                    },
                    TranformationSource::PhysicsBody(handle) => {
                        if let Some(body) = body_set.get(handle) {
                            attractor_data.push((body.part(0).unwrap().position().translation.vector, attractor.mass));
                        } else {
                            delete_attractors.push(entity);
                        }
                    }
                }
            );
            
            delete_attractors.iter().for_each(|attractor_entity| {
                self.world.delete(*attractor_entity);
            });

            self.body_set.iter_mut().for_each(|(_, body)| {
                let body_position = body.part(0).unwrap().position().translation.vector;
                
                let force: Vec3 = attractor_data.iter().map(|(position, mass)|{
                    if body_position == *position {
                        Vec3::zeros()
                    } else {
                        let direction = position-body_position;
                        let length = (position-body_position).magnitude();
                        let acceleration = mass/length*length*length;
                        timestep*direction*acceleration
                    }
                }).sum();

                use np::math::{ForceType, Force};
                body.apply_force(0, &Force::linear(force), ForceType::AccelerationChange, true);
            });
            microprofile::scope!("main", "physics_update");

            self.mechanical_world.step(
                &mut self.geometrical_world,
                &mut self.body_set,
                &mut self.collider_set,
                &mut self.constraint_set,
                &mut self.force_generator_set
            );
            
        }

        microprofile::flip!();
    }
}

impl<B: Backend> Drop for Application<B> {
    fn drop(&mut self) {
        use std::ptr;
        let device = &self.device.borrow().device;
        device.wait_idle().unwrap();
        unsafe {
            // assures that the swapchain is droppen before the frame buffer
            drop(ManuallyDrop::into_inner(ptr::read(&self.swapchain)));
            drop(ManuallyDrop::into_inner(ptr::read(&self.framebuffer)));

            device.destroy_graphics_pipeline(ManuallyDrop::into_inner(ptr::read(&self.mesh_pipeline)));
            device.destroy_pipeline_layout(ManuallyDrop::into_inner(ptr::read(
                &self.mesh_pipeline_layout,
            )));

            device.destroy_descriptor_pool(ManuallyDrop::into_inner(ptr::read(&self.descriptor_pool)));
            device.destroy_render_pass(ManuallyDrop::into_inner(ptr::read(&self.mesh_render_pass)));
        }
    }
}
fn main() {
    microprofile::init();
	microprofile::set_enable_all_groups(true);

    simple_logger::init_with_level(log::Level::Warn).unwrap();

    let event_loop = winit::event_loop::EventLoop::new();

    let logical_window_size: winit::dpi::LogicalSize<u32> = WINDOW_SIZE.into();

    let wb = winit::window::WindowBuilder::new()
        .with_title(APP_NAME)
        .with_inner_size(logical_window_size);

    let backend: BackendState<back::Backend> = BackendState::new(wb, &event_loop, "healp");
    backend.window.set_cursor_visible(false);
    backend.window.set_cursor_grab(true);
    let mut app = Application::new(backend);

    event_loop.run(move |event, _, control_flow| {
        use winit::event::{Event, WindowEvent, ElementState, MouseButton};
        use winit::event_loop::ControlFlow;

        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(dims) => {
                    app.extent = Extent2D {
                        width: dims.width,
                        height: dims.height,
                    };
                    app.camera.aspect = dims.width as f32 / dims.height as f32;
                    app.recreate_swapchain = true;
                },
                WindowEvent::Focused(focus) => app.is_focused = focus,
                WindowEvent::MouseInput {
                    state,
                    button,
                    ..
                } => {
                    if state == ElementState::Pressed && button == MouseButton::Left {
                        let sphere = ShapeHandle::new(Ball::new(0.2));
                        let rb = RigidBodyDesc::new()
                            .translation(app.camera.pos.coords)
                            .velocity(np::algebra::Velocity3::new(5.*app.camera.get_view_direction(), Vec3::zeros()))
                            .user_data(BodyData{is_attracted: true})
                            .build();
                        let rb_handle = app.body_set.insert(rb);

                        let co = ColliderDesc::new(sphere.clone())
                            .density(1.0)
                            .material(MaterialHandle::new(BasicMaterial::new(0.3, 1.)))
                            .build(BodyPartHandle(rb_handle, 0));
                        app.collider_set.insert(co);

                        app.world.insert(
                            (), 
                            std::iter::once((TranformationSource::PhysicsBody(rb_handle), app.fallback_model, Attractor{mass: 0.}))
                        );
                    }
                }

                WindowEvent::KeyboardInput { input, .. } => {
                    use winit::event::VirtualKeyCode::*;
                    let pressed = input.state == winit::event::ElementState::Pressed;
                    match input.virtual_keycode {
                        Some(W) => app.keys[Direction::Forward as usize] = pressed,
                        Some(S) => app.keys[Direction::Backward as usize] = pressed,
                        Some(A) => app.keys[Direction::Left as usize] = pressed,
                        Some(D) => app.keys[Direction::Right as usize] = pressed,
                        Some(Space) => app.keys[Direction::Up as usize] = pressed,
                        Some(LShift) => app.keys[Direction::Down as usize] = pressed,
                        Some(R) => {
                            app.camera.pos = Pos3::origin();
                            app.camera.pitch = 0.;
                            app.camera.yaw = 0.;
                        },
                        Some(Escape) => *control_flow = winit::event_loop::ControlFlow::Exit,
                        Some(T) => microprofile::dump_file_immediately("profiling_dump.html", ""),
                        _ => {}
                    }
                }
                _ => (),
            },
            Event::DeviceEvent { event, .. } => match event {
                winit::event::DeviceEvent::MouseMotion { delta } => {
                    if app.is_focused {
                        app.camera.yaw += (delta.0 * 0.0015) as f32;
                        app.camera.pitch -= (delta.1 * 0.0015) as f32;
                        app.camera.pitch = app.camera.pitch.min(1.57).max(-1.57);
                    }
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                app.update();
                app.backend.window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                app.draw();
            }
            _ => (),
        }

    });

    microprofile::shutdown();
}
