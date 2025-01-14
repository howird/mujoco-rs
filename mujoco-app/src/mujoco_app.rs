use glium::{
    glutin::surface::WindowSurface,
    winit::{
        application::ApplicationHandler,
        event::{ElementState, KeyEvent},
        event_loop::EventLoop,
        keyboard::{Key, KeyCode, SmolStr},
        window::Window,
    },
    Display,
};

use mujoco_rs_sys::{
    mjr_makeContext, mjr_overlay, mjr_readPixels, mjtCatBit, mjtFont, mjtGridPos,
    mjvCamera, mjvOption, mjvPerturb, mjvScene, mjv_defaultFreeCamera,
    mjv_defaultOption, mjv_defaultPerturb, mjv_makeScene, mjv_updateScene,
    render::{mjrContext, mjrRect, mjr_render},
};
use std::{
    num::NonZeroU32,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use glium::{
    backend::glutin::simple_window_builder::GliumEventLoop,
    glutin::{
        config::ConfigTemplateBuilder,
        context::{ContextApi, ContextAttributesBuilder, NotCurrentGlContext, Version},
        display::GetGlDisplay,
        prelude::GlDisplay,
        surface::SurfaceAttributesBuilder,
    },
    texture::RawImage2d,
    winit::{
        event::WindowEvent, keyboard::NamedKey, raw_window_handle::HasWindowHandle,
    },
    Surface,
};

use glutin_winit::DisplayBuilder;

type CtrlFun = Box<dyn FnMut(&mujoco_rust::Simulation) -> Vec<f64> + Send + 'static>;
type RenderFun = Box<dyn FnMut()>;

// The simulation itself and model are handled by mj-rust's Simulation object
struct RenderState {
    opt: mjvOption,
    cam: mjvCamera,
    pert: mjvPerturb,
    scn: mjvScene,
    con: mjrContext,
}

impl Default for RenderState {
    fn default() -> Self {
        RenderState {
            opt: Default::default(),
            cam: Default::default(),
            pert: Default::default(),
            scn: Default::default(),
            con: Default::default(),
        }
    }
}

struct Rendering {
    state: RenderState,
    event_loop: Option<EventLoop<()>>,
    window: Window,
    display: Display<WindowSurface>,
    render_cb: Option<RenderFun>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhysicsRunningState {
    Paused,
    RateLimited,
    Uncapped,
}

#[derive(Debug, Clone, Copy)]
struct PhysicsState {
    running_state: PhysicsRunningState,
    frame_rate: f32,
}

fn loop_physics_threaded(
    sim: Arc<Mutex<mujoco_rust::Simulation>>,
    state: Arc<Mutex<PhysicsState>>,
    mut ctrl_cb: Option<CtrlFun>,
) {
    let mut last_updated = Instant::now();
    // We should have taken one step, so we know the timestep
    let timestep = sim.lock().unwrap().state.time();
    let mut step_sim = || {
        let locked_sim = sim.lock().unwrap();
        // Apply control law here
        if let Some(fun) = ctrl_cb.as_mut() {
            let control = fun(&locked_sim);
            locked_sim.control(&control);
        }
        locked_sim.step();
    };
    loop {
        let current_state: PhysicsState;
        {
            let locked = state.lock().unwrap();
            current_state = (*locked).clone()
        }
        match current_state.running_state {
            PhysicsRunningState::Paused => {
                last_updated = Instant::now();
                thread::sleep(Duration::from_millis(1));
            }
            PhysicsRunningState::RateLimited => {
                let elapsed_real = last_updated.elapsed();
                let num_steps = (elapsed_real.as_secs_f64() / timestep).floor() as u32;
                for _ in 0..num_steps {
                    step_sim();
                }
                last_updated += Duration::from_secs_f64(num_steps as f64 * timestep);
                thread::sleep(Duration::from_secs_f64(timestep));
            }
            PhysicsRunningState::Uncapped => {
                step_sim();
                last_updated = Instant::now();
            }
        }
    }
}

pub struct MujocoApp {
    ctrl_cb: Option<CtrlFun>,
    rendering: Option<Rendering>,
    sim: Arc<Mutex<mujoco_rust::Simulation>>,
    physics_state: Arc<Mutex<PhysicsState>>,
    frame_rate_limited: bool,
    last_render: Instant,
}

impl MujocoApp {
    pub fn run_app(mut self) {
        if self.rendering.is_some() {
            let event_loop =
                self.rendering.as_mut().unwrap().event_loop.take().unwrap();
            self.launch_physics_thread();
            event_loop.run_app(&mut self).unwrap();
        }
        // Headless
        else {
            println!("Running without Rendering Context");
            self.loop_physics();
        }
    }

    fn launch_physics_thread(&mut self) {
        let sim_clone = self.sim.clone();
        let physics_state_clone = self.physics_state.clone();
        let ctrl_cb = self.ctrl_cb.take();
        thread::spawn(move || {
            loop_physics_threaded(sim_clone, physics_state_clone, ctrl_cb);
        });
    }

    fn loop_physics(&mut self) {
        loop {
            let sim = self.sim.lock().unwrap();
            if let Some(fun) = self.ctrl_cb.as_mut() {
                let control = fun(&sim);
                sim.control(&control);
            }
            sim.step();
        }
    }

    fn render(&mut self) {
        if let Some(rendering) = self.rendering.as_mut() {
            let fps = 1. / self.last_render.elapsed().as_secs_f32();
            self.last_render = Instant::now();
            {
                self.physics_state.lock().unwrap().frame_rate = fps;
            }
            let target = rendering.display.draw();

            let window_size = rendering.window.inner_size();
            let viewport = mjrRect {
                left: 0,
                bottom: 0,
                width: window_size.width as i32,
                height: window_size.height as i32,
            };

            let num_pixels: usize = (window_size.width * window_size.height) as usize;

            let mut rgb = vec![0u8; num_pixels * 3];
            let mut depth = vec![0.0f32; num_pixels];

            {
                let locked_sim = self.sim.lock().unwrap();
                let timestamp = format!("Time = {:.3}", locked_sim.state.time());
                let fps_str = format!("FPS = {:.3}", fps);
                // Need to include \0 end stric character to send this to a raw C string
                let _null_str = "\0";
                let m = locked_sim.model.ptr();
                let d = locked_sim.state.ptr();
                unsafe {
                    // Update camera / scene etc.
                    mjv_updateScene(
                        m,
                        d,
                        &mut rendering.state.opt,
                        &rendering.state.pert,
                        &mut rendering.state.cam,
                        mjtCatBit::ALL as i32,
                        &mut rendering.state.scn,
                    );
                    // Render to a frame buffer
                    mjr_render(
                        viewport,
                        &mut rendering.state.scn,
                        &rendering.state.con,
                    );
                    // Overlay text
                    mjr_overlay(
                        mjtFont::NORMAL as i32,
                        mjtGridPos::TOPLEFT as i32,
                        viewport,
                        timestamp.as_ptr() as *const i8,
                        fps_str.as_ptr() as *const i8,
                        &rendering.state.con,
                    );
                    // Copy framebuffer data to our own RGB array
                    mjr_readPixels(
                        rgb.as_mut_ptr(),
                        depth.as_mut_ptr(),
                        viewport,
                        &rendering.state.con,
                    );
                }
            }
            let color_image =
                RawImage2d::from_raw_rgb(rgb, (window_size.width, window_size.height));
            let color_texture: glium::Texture2d =
                glium::Texture2d::new(&rendering.display, color_image).unwrap();
            color_texture
                .as_surface()
                .fill(&target, glium::uniforms::MagnifySamplerFilter::Linear);

            // Call user specific rendering
            // todo
            if let Some(fun) = rendering.render_cb.as_mut() {
                fun();
            }

            target.finish().unwrap();
        }
    }
}

impl ApplicationHandler for MujocoApp {
    fn resumed(&mut self, _event_loop: &glium::winit::event_loop::ActiveEventLoop) {}

    fn window_event(
        &mut self,
        event_loop: &glium::winit::event_loop::ActiveEventLoop,
        _window_id: glium::winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.render();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Space),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                let mut locked_value = self.physics_state.lock().unwrap();
                match locked_value.running_state {
                    PhysicsRunningState::Paused => {
                        if self.frame_rate_limited {
                            locked_value.running_state =
                                PhysicsRunningState::RateLimited;
                        } else {
                            locked_value.running_state = PhysicsRunningState::Uncapped;
                        }
                    }
                    _ => {
                        locked_value.running_state = PhysicsRunningState::Paused;
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(c),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                if c == "u" {
                    self.frame_rate_limited = !self.frame_rate_limited;
                    let mut locked_value = self.physics_state.lock().unwrap();
                    if self.frame_rate_limited {
                        locked_value.running_state = PhysicsRunningState::RateLimited;
                    } else {
                        locked_value.running_state = PhysicsRunningState::Uncapped;
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(
        &mut self,
        _event_loop: &glium::winit::event_loop::ActiveEventLoop,
    ) {
        if let Some(rendering) = self.rendering.as_ref() {
            rendering.window.request_redraw();
        }
    }
}

pub struct AppBuilder {
    ctrl_cb: Option<CtrlFun>,
    render_data: Option<Rendering>,
    model: mujoco_rust::Model,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct AppBuilderErr(String);

impl AppBuilder {
    pub fn from_model(model: mujoco_rust::Model) -> Self {
        AppBuilder {
            ctrl_cb: None,
            render_data: None,
            model,
        }
    }

    pub fn from_xml(xml: String) -> Result<Self, AppBuilderErr> {
        let model_result = mujoco_rust::Model::from_xml(xml.clone());
        if let Ok(model) = model_result {
            Ok(AppBuilder::from_model(model))
        } else {
            Err(AppBuilderErr(
                format!("Failed to load xml file: '{}'", xml).to_string(),
            ))
        }
    }

    pub fn build(mut self) -> MujocoApp {
        let sim = mujoco_rust::Simulation::new(self.model);

        // If we're setting up rendering
        if let Some(rendering) = self.render_data.as_mut() {
            let state = &mut rendering.state;
            unsafe {
                let m = sim.model.ptr();
                mjv_makeScene(m, &mut state.scn, 1000);
                mjr_makeContext(m, &mut state.con, 11);
                mjv_defaultOption(&mut state.opt);
                mjv_defaultFreeCamera(m, &mut state.cam);
                mjv_defaultPerturb(&mut state.pert);
            }
        }

        // Propagate data by stepping simulation once
        sim.step();

        MujocoApp {
            ctrl_cb: self.ctrl_cb,
            rendering: self.render_data,
            sim: Arc::new(Mutex::new(sim)),
            physics_state: Arc::new(Mutex::new(PhysicsState {
                running_state: PhysicsRunningState::Paused,
                frame_rate: 60.,
            })),
            frame_rate_limited: true,
            last_render: Instant::now(),
        }
    }

    pub fn with_control(mut self, ctrl_function: CtrlFun) -> Self {
        self.ctrl_cb = Some(ctrl_function);
        self
    }

    pub fn with_default_rendering(mut self) -> Self {
        let event_loop = EventLoop::builder()
            .build()
            .expect("Failed to build event loop");
        let (window, display) = build_mujoco_gl_context(&event_loop);

        self.render_data = Some(Rendering {
            state: Default::default(),
            event_loop: Some(event_loop),
            window,
            display,
            render_cb: None,
        });
        self
    }

    pub fn with_custom_render_callback(
        mut self,
        render_function: RenderFun,
    ) -> Result<Self, AppBuilderErr> {
        if let Some(render_data) = self.render_data.as_mut() {
            (*render_data).render_cb = Some(render_function);
            Ok(self)
        } else {
            Err(AppBuilderErr(
                "'with_custom_render_callback' Requires rendering to be setup already!"
                    .to_string(),
            ))
        }
    }
}

// A default opengl window and context for mujoco
fn build_mujoco_gl_context(
    event_loop: &impl GliumEventLoop,
) -> (glium::winit::window::Window, glium::Display<WindowSurface>) {
    let window_attributes = Window::default_attributes();
    let config_template_builder = ConfigTemplateBuilder::new();
    let display_builder =
        DisplayBuilder::new().with_window_attributes(Some(window_attributes));

    let (window, gl_config) = event_loop
        .build(display_builder, config_template_builder, |mut configs| {
            // Use the first available configuration
            configs.next().unwrap()
        })
        .unwrap();
    let window = window.unwrap();

    // We need to specially request OpenGL 3.1 for MuJoCo to work
    let window_handle = window.window_handle().unwrap();
    let context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(Some(Version { major: 3, minor: 1 })))
        .build(Some(window_handle.into()));

    // Default to 800x600 if the window size is invisible
    let (width, height) = window.inner_size().into();
    let attrs = SurfaceAttributesBuilder::<WindowSurface>::new().build(
        window_handle.into(),
        NonZeroU32::new(width).unwrap(),
        NonZeroU32::new(height).unwrap(),
    );

    // Finally construct the display
    let surface = unsafe {
        gl_config
            .display()
            .create_window_surface(&gl_config, &attrs)
            .unwrap()
    };

    let current_context = Some(unsafe {
        gl_config
            .display()
            .create_context(&gl_config, &context_attributes)
            .expect("Failed to create gl context")
    })
    .unwrap()
    .make_current(&surface)
    .unwrap();
    let display = Display::from_context_surface(current_context, surface).unwrap();

    (window, display)
}
