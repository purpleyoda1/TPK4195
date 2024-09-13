// Uncomment these following global attributes to silence most warnings of "low" interest:
/*
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(unused_unsafe)]
#![allow(unused_variables)]
*/
extern crate gl;
extern crate nalgebra_glm as glm;
use std::collections::HashSet;
use std::ffi::CString;
use std::{ mem, ptr, os::raw::c_void };
use std::thread;
use std::sync::{Mutex, Arc, RwLock};

mod shader;
mod util;

use glm::{cos, pi, proj, Vec3, look_at, translate};
use glutin::event::ElementState;
use glutin::event::{Event, WindowEvent, DeviceEvent, KeyboardInput, ElementState::{Pressed, Released}, VirtualKeyCode::{self, *}};
use glutin::event_loop::ControlFlow;
use shader::Shader;

// initial window size
const INITIAL_SCREEN_W: u32 = 800;
const INITIAL_SCREEN_H: u32 = 600;

// == // Helper functions to make interacting with OpenGL a little bit prettier. You *WILL* need these! // == //

// Get the size of an arbitrary array of numbers measured in bytes
// Example usage:  byte_size_of_array(my_array)
fn byte_size_of_array<T>(val: &[T]) -> isize {
    std::mem::size_of_val(&val[..]) as isize
}

// Get the OpenGL-compatible pointer to an arbitrary array of numbers
// Example usage:  pointer_to_array(my_array)
fn pointer_to_array<T>(val: &[T]) -> *const c_void {
    &val[0] as *const T as *const c_void
}

// Get the size of the given type in bytes
// Example usage:  size_of::<u64>()
fn size_of<T>() -> i32 {
    mem::size_of::<T>() as i32
}

// Get an offset in bytes for n units of type T, represented as a relative pointer
// Example usage:  offset::<u64>(4)
fn offset<T>(n: u32) -> *const c_void {
    (n * mem::size_of::<T>() as u32) as *const T as *const c_void
}

// Get a null pointer (equivalent to an offset of 0)
// ptr::null()

// Set up camera functions
struct Camera {
    x: f32,
    y: f32,
    z: f32,
    yaw: f32,
    pitch: f32,
    transformation: glm::Mat4,
}

impl Camera {
    fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            transformation: glm::Mat4::identity(),
        }
    }
}

fn update_camera_matrix(camera: &mut Camera) {
    camera.transformation = glm::mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );

    let cos_pitch = camera.pitch.to_radians().cos();
    let sin_pitch = camera.pitch.to_radians().sin();
    let cos_yaw = camera.yaw.to_radians().cos();
    let sin_yaw = camera.yaw.to_radians().sin();

    let pitch_rotation = glm::mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, cos_pitch, -sin_pitch, 0.0,
        0.0, sin_pitch, cos_pitch, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    let yaw_rotation = glm::mat4(
        cos_yaw, -sin_yaw, 0.0, 0.0,
        sin_yaw, cos_yaw, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    let rotation_matrix = pitch_rotation * yaw_rotation;

    // Translation matrix based on camera position
    let translation_vector = Vec3::new(-camera.x, -camera.y, -camera.z);
    let translation_matrix = translate(&glm::Mat4::identity(), &translation_vector);

    // Combine rotation and translation to form the view matrix
    camera.transformation = translation_matrix * rotation_matrix;
}

// == // Generate your VAO here
unsafe fn create_vao(vertices: &Vec<f32>, indices: &Vec<u32>, color: &Vec<f32>) -> u32 {
    let mut vao = 0;
    let mut position_vbo = 0;
    let mut color_vbo = 0;
    let mut ibo = 0;

    // Generate and bind VAO
    gl::GenVertexArrays(1, &mut vao);
    gl::BindVertexArray(vao);

    // Generate, bind, and fill VBO with vertex position data
    gl::GenBuffers(1, &mut position_vbo);
    gl::BindBuffer(gl::ARRAY_BUFFER, position_vbo);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(vertices),
        pointer_to_array(vertices),
        gl::STATIC_DRAW
    );

    // Generate VAP
    gl::VertexAttribPointer(
        0,
        3,
        gl::FLOAT,
        gl::FALSE,
        0,
        ptr::null()
    );
    gl::EnableVertexAttribArray(0);

    // Generate, bind, and fill VBO with vertex color data
    gl::GenBuffers(1, &mut color_vbo);
    gl::BindBuffer(gl::ARRAY_BUFFER, color_vbo);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(color),
        pointer_to_array(color),
        gl::STATIC_DRAW
    );

    // Generate VAP
    gl::VertexAttribPointer(
        1,
        4,
        gl::FLOAT,
        gl::FALSE,
        0,
        ptr::null()
    );
    gl::EnableVertexAttribArray(1);

    // Generate, bind and fill IBO
    gl::GenBuffers(1, &mut ibo);
    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ibo);
    gl::BufferData(
        gl::ELEMENT_ARRAY_BUFFER,
        byte_size_of_array(indices),
        pointer_to_array(indices),
        gl::STATIC_DRAW
    );
    
    // Return VAO ID
    gl::BindVertexArray(0);
    vao
}


fn main() {
    // Set up the necessary objects to deal with windows and event handling
    let el = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("Gloom-rs")
        .with_resizable(true)
        .with_inner_size(glutin::dpi::LogicalSize::new(INITIAL_SCREEN_W, INITIAL_SCREEN_H));
    let cb = glutin::ContextBuilder::new()
        .with_vsync(true);
    let windowed_context = cb.build_windowed(wb, &el).unwrap();
    // Uncomment these if you want to use the mouse for controls, but want it to be confined to the screen and/or invisible.
    // windowed_context.window().set_cursor_grab(true).expect("failed to grab cursor");
    // windowed_context.window().set_cursor_visible(false);

    // Set up a shared vector for keeping track of currently pressed keys
    let arc_pressed_keys = Arc::new(Mutex::new(Vec::<VirtualKeyCode>::with_capacity(10)));
    // Make a reference of this vector to send to the render thread
    let pressed_keys = Arc::clone(&arc_pressed_keys);

    // Set up shared tuple for tracking mouse movement between frames
    let arc_mouse_delta = Arc::new(Mutex::new((0f32, 0f32)));
    // Make a reference of this tuple to send to the render thread
    let mouse_delta = Arc::clone(&arc_mouse_delta);

    // Set up shared tuple for tracking changes to the window size
    let arc_window_size = Arc::new(Mutex::new((INITIAL_SCREEN_W, INITIAL_SCREEN_H, false)));
    // Make a reference of this tuple to send to the render thread
    let window_size = Arc::clone(&arc_window_size);

    // Spawn a separate thread for rendering, so event handling doesn't block rendering
    let render_thread = thread::spawn(move || {
        // Acquire the OpenGL Context and load the function pointers.
        // This has to be done inside of the rendering thread, because
        // an active OpenGL context cannot safely traverse a thread boundary
        let context = unsafe {
            let c = windowed_context.make_current().unwrap();
            gl::load_with(|symbol| c.get_proc_address(symbol) as *const _);
            c
        };

        let mut window_aspect_ratio = INITIAL_SCREEN_W as f32 / INITIAL_SCREEN_H as f32;

        // Set up openGL
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::DepthFunc(gl::LESS);
            gl::Enable(gl::CULL_FACE);
            gl::Disable(gl::MULTISAMPLE);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl::DebugMessageCallback(Some(util::debug_callback), ptr::null());

            // Print some diagnostics
            println!("{}: {}", util::get_gl_string(gl::VENDOR), util::get_gl_string(gl::RENDERER));
            println!("OpenGL\t: {}", util::get_gl_string(gl::VERSION));
            println!("GLSL\t: {}", util::get_gl_string(gl::SHADING_LANGUAGE_VERSION));
        }

        // == // Set up your VAO around here
        let vertices: Vec<f32> = vec![
            -0.8, -0.8, -0.5,
            0.0, -0.8, -0.5,
            0.8, -0.8, -0.5,
            -0.4, 0.0, -0.5,
            0.4, 0.0, -0.5,
            0.0, 0.8, -0.5,
        ];

        let color: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.5,
            1.0, 0.0, 0.0, 0.5,
            1.0, 0.0, 0.0, 0.5,
            0.0, 0.0, 1.0, 0.5,
            0.0, 0.0, 1.0, 0.5,
            0.0, 1.0, 0.0, 0.5,
        ];

        let  indices: Vec<u32> = vec![
            0, 1, 3,
            1, 2, 4,
            3, 4, 5,
        ];

        let my_vao = unsafe { create_vao(&vertices, &indices, &color) };


        // == // Set up your shaders here
        let my_shader = unsafe {
            shader::ShaderBuilder::new()
                .attach_file("shaders/simple.vert")
                .attach_file("shaders/simple.frag")
                .link()
        };
        
        // Set up projection and depth transformation
        let projection: glm::Mat4 = glm::perspective(1.6, 1.6, 1.0, 100.0);
        let depth_transformation =  glm::mat4(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 49.5, -50.5,
            0.0, 0.0, 0.0, 1.0
        );

        // Initialize camera
        let mut camera = Camera::new();
        let speed = 2.0;
        let rotation_speed = 30.0;

        // The main rendering loop
        let first_frame_time = std::time::Instant::now();
        let mut previous_frame_time = first_frame_time;
        loop {
            // Compute time passed since the previous frame and since the start of the program
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(first_frame_time).as_secs_f32();
            let delta_time = now.duration_since(previous_frame_time).as_secs_f32();
            previous_frame_time = now;

            // Handle resize events
            if let Ok(mut new_size) = window_size.lock() {
                if new_size.2 {
                    context.resize(glutin::dpi::PhysicalSize::new(new_size.0, new_size.1));
                    window_aspect_ratio = new_size.0 as f32 / new_size.1 as f32;
                    (*new_size).2 = false;
                    println!("Window was resized to {}x{}", new_size.0, new_size.1);
                    unsafe { gl::Viewport(0, 0, new_size.0 as i32, new_size.1 as i32); }
                }
            }

            // Handle keyboard input
            if let Ok(keys) = pressed_keys.lock() {
                for key in keys.iter() {
                    match key {
                        VirtualKeyCode::W => camera.z -= speed * delta_time,
                        VirtualKeyCode::S => camera.z += speed * delta_time,
                        VirtualKeyCode::A => camera.x -= speed * delta_time,
                        VirtualKeyCode::D => camera.x += speed * delta_time,            
                        VirtualKeyCode::Space => camera.y -= speed * delta_time,
                        VirtualKeyCode::LShift => camera.y += speed * delta_time,
                        VirtualKeyCode::Left => camera.yaw -=  rotation_speed * delta_time,
                        VirtualKeyCode::Right => camera.yaw += rotation_speed * delta_time,            
                        VirtualKeyCode::Up => camera.pitch -= rotation_speed * delta_time,
                        VirtualKeyCode::Down => camera.pitch += rotation_speed * delta_time,
                        _ => {}  
                    }
                }
            }
            

            // Handle mouse movement. delta contains the x and y movement of the mouse since last frame in pixels
            if let Ok(mut delta) = mouse_delta.lock() {

                // == // Optionally access the accumulated mouse movement between
                // == // frames here with `delta.0` and `delta.1`

                *delta = (0.0, 0.0); // reset when done
            }

            // == // Please compute camera transforms here (exercise 2 & 3)
            update_camera_matrix(&mut camera);

            unsafe {
                // Clear the color and depth buffers
                gl::ClearColor(0.035, 0.046, 0.078, 1.0); // night sky
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                // Activate shaders
                my_shader.activate();

                // Prepare affine transformation
                let combined_transformation = projection * depth_transformation * camera.transformation;
                gl::UniformMatrix4fv(0, 1, gl::TRUE, combined_transformation.as_ptr());
                
                // Bind VAO
                gl::BindVertexArray(my_vao);

                // Draw elements
                gl::DrawElements(
                    gl::TRIANGLES,
                    9,
                    gl::UNSIGNED_INT,
                    std::ptr::null()
                );

                //Unbind
                gl::BindVertexArray(0);
            }

            // Display the new color buffer on the display
            context.swap_buffers().unwrap(); // we use "double buffering" to avoid artifacts
        }
    });


    // == //
    // == // From here on down there are only internals.
    // == //


    // Keep track of the health of the rendering thread
    let render_thread_healthy = Arc::new(RwLock::new(true));
    let render_thread_watchdog = Arc::clone(&render_thread_healthy);
    thread::spawn(move || {
        if !render_thread.join().is_ok() {
            if let Ok(mut health) = render_thread_watchdog.write() {
                println!("Render thread panicked!");
                *health = false;
            }
        }
    });

    // Start the event loop -- This is where window events are initially handled
    el.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        // Terminate program if render thread panics
        if let Ok(health) = render_thread_healthy.read() {
            if *health == false {
                *control_flow = ControlFlow::Exit;
            }
        }

        match event {
            Event::WindowEvent { event: WindowEvent::Resized(physical_size), .. } => {
                println!("New window size received: {}x{}", physical_size.width, physical_size.height);
                if let Ok(mut new_size) = arc_window_size.lock() {
                    *new_size = (physical_size.width, physical_size.height, true);
                }
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            }
            // Keep track of currently pressed keys to send to the rendering thread
            Event::WindowEvent { event: WindowEvent::KeyboardInput {
                    input: KeyboardInput { state: key_state, virtual_keycode: Some(keycode), .. }, .. }, .. } => {

                if let Ok(mut keys) = arc_pressed_keys.lock() {
                    match key_state {
                        Released => {
                            if keys.contains(&keycode) {
                                let i = keys.iter().position(|&k| k == keycode).unwrap();
                                keys.remove(i);
                            }
                        },
                        Pressed => {
                            if !keys.contains(&keycode) {
                                keys.push(keycode);
                            }
                        }
                    }
                }

                // Handle Escape and Q keys separately
                match keycode {
                    Escape => { *control_flow = ControlFlow::Exit; }
                    Q      => { *control_flow = ControlFlow::Exit; }
                    _      => { }
                }
            }
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                // Accumulate mouse movement
                if let Ok(mut position) = arc_mouse_delta.lock() {
                    *position = (position.0 + delta.0 as f32, position.1 + delta.1 as f32);
                }
            }
            _ => { }
        }
    });
}
