use spuc3d::renderer::core::{
    transform::Transform,
    math::{Matrix, Vector},
    rasterizer::Rasterizer,
    pipeline::Pipeline,
};
use spuc3d::renderer::geometry::{Mesh, Vertex};
use spuc3d::renderer::shader::DefaultShader;
use spuc3d::renderer::texture::Texture;
use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Write, Seek, SeekFrom},
    mem,
    os::unix::io::AsRawFd,
    ptr,
    thread, 
    time::{Duration, Instant},
};
// Importar nuestra implementación alternativa usando minifb
use std::slice;
use libc;
use framebuffer::Framebuffer;
use std::error::Error;

// Estructuras para el framebuffer de Linux
#[repr(C)]
#[derive(Clone)]
struct FbVarScreeninfo {
    xres: u32,            // visible resolution
    yres: u32,
    xres_virtual: u32,    // virtual resolution
    yres_virtual: u32,
    xoffset: u32,         // offset from virtual to visible
    yoffset: u32,
    bits_per_pixel: u32,  // bpp
    grayscale: u32,
    red: BitField,
    green: BitField,
    blue: BitField,
    transp: BitField,
    nonstd: u32,
    activate: u32,
    height: u32,          // height of picture in mm
    width: u32,           // width of picture in mm
    accel_flags: u32,
    // timing: 12 bytes
    pixclock: u32,
    left_margin: u32,
    right_margin: u32,
    upper_margin: u32,
    lower_margin: u32,
    hsync_len: u32,
    vsync_len: u32,
    sync: u32,
    vmode: u32,
    rotate: u32,
    colorspace: u32,
    reserved: [u32; 4],
}

#[repr(C)]
#[derive(Clone)]
struct BitField {
    offset: u32,
    length: u32,
    msb_right: u32,
}

#[repr(C)]
struct FbFixScreeninfo {
    id: [u8; 16],         // identification string
    smem_start: usize,    // Start of frame buffer mem (physical address)
    smem_len: u32,        // Length of frame buffer mem
    r#type: u32,          // see FB_TYPE_*
    type_aux: u32,        // Interleave for interleaved Planes
    visual: u32,          // see FB_VISUAL_*
    xpanstep: u16,        // zero if no hardware panning
    ypanstep: u16,        // zero if no hardware panning
    ywrapstep: u16,       // zero if no hardware ywrap
    line_length: u32,     // length of a line in bytes
    mmio_start: usize,    // Start of Memory Mapped I/O (physical address)
    mmio_len: u32,        // Length of Memory Mapped I/O
    accel: u32,           // Indicate to driver which specific chip/card we have
    capabilities: u16,
    reserved: [u16; 2],
}

// Estructuras para entrada táctil
#[repr(C)]
struct InputEvent {
    time: TimeVal,
    r#type: u16,
    code: u16,
    value: i32,
}

#[repr(C)]
struct TimeVal {
    tv_sec: isize,
    tv_usec: isize,
}

// Constantes para ioctl
const FBIOGET_VSCREENINFO: u64 = 0x4600;
const FBIOPUT_VSCREENINFO: u64 = 0x4601;
const FBIOGET_FSCREENINFO: u64 = 0x4602;
const FBIOPAN_DISPLAY: u64 = 0x4606;

// Constantes para eventos de entrada
const EV_ABS: u16 = 0x03;
const ABS_X: u16 = 0x00;
const ABS_Y: u16 = 0x01;
const ABS_MT_POSITION_X: u16 = 0x35;
const ABS_MT_POSITION_Y: u16 = 0x36;
const EV_SYN: u16 = 0x00;
const SYN_REPORT: u16 = 0x00;
const EV_KEY: u16 = 0x01;
const BTN_TOUCH: u16 = 0x14a;

// Enlaza con la función ioctl de libc
extern "C" {
    fn ioctl(fd: i32, request: u64, ...) -> i32;
}

// Estructuras y constantes para la entrada táctil
struct TouchInput {
    fd: File,
    x: i32,
    y: i32,
    touch_active: bool,
}

// Shader básico para nuestra demostración
struct DemoShader {}

impl DemoShader {
    fn new() -> Self {
        Self {}
    }
}

impl spuc3d::renderer::shader::Shader for DemoShader {
    fn vertex_shader(&self, input: &Vertex) -> Vec<f32> {
        // Convertimos la posición del vértice a un vector de f32
        vec![input.position[0], input.position[1], input.position[2], 1.0]
    }

    fn fragment_shader(&self, color: &[f32]) -> Vec<f32> {
        // Usamos el color que recibimos
        color.to_vec()
    }
}

// Implementación para entrada táctil
impl TouchInput {
    // Abre e inicializa el dispositivo táctil
    fn open(path: &str) -> io::Result<Self> {
        let fd = OpenOptions::new()
            .read(true)
            .open(path)?;
        
        Ok(Self {
            fd,
            x: 0,
            y: 0,
            touch_active: false,
        })
    }
    
    // Lee eventos táctiles de forma no bloqueante
    fn poll_events(&mut self) -> io::Result<bool> {
        // Configurar la lectura no bloqueante
        use std::os::unix::io::AsRawFd;
        let flags = unsafe { libc::fcntl(self.fd.as_raw_fd(), libc::F_GETFL, 0) };
        unsafe { libc::fcntl(self.fd.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) };
        
        let mut event = InputEvent {
            time: TimeVal { tv_sec: 0, tv_usec: 0 },
            r#type: 0,
            code: 0,
            value: 0,
        };
        
        let event_size = mem::size_of::<InputEvent>();
        let mut buf = unsafe { 
            slice::from_raw_parts_mut(
                &mut event as *mut _ as *mut u8, 
                event_size
            ) 
        };
        
        let mut has_updates = false;
        
        // Leer eventos hasta que no haya más
        loop {
            match self.fd.read(&mut buf) {
                Ok(size) if size == event_size => {
                    match event.r#type {
                        EV_ABS => {
                            match event.code {
                                ABS_X | ABS_MT_POSITION_X => {
                                    self.x = event.value;
                                    has_updates = true;
                                },
                                ABS_Y | ABS_MT_POSITION_Y => {
                                    self.y = event.value;
                                    has_updates = true;
                                },
                                _ => {}
                            }
                        },
                        EV_KEY => {
                            if event.code == BTN_TOUCH {
                                self.touch_active = event.value != 0;
                                has_updates = true;
                            }
                        },
                        _ => {}
                    }
                },
                Ok(_) => break, // Tamaño incorrecto
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => break, // No hay más eventos
                Err(e) => return Err(e), // Error de lectura
            }
        }
        
        Ok(has_updates)
    }
    
    // Obtiene la posición actual del toque
    fn get_position(&self) -> (i32, i32) {
        (self.x, self.y)
    }
    
    // Indica si hay un toque activo
    fn is_touching(&self) -> bool {
        self.touch_active
    }
}

// Función para crear un cubo simple
fn create_cube() -> Mesh {
    // Vértices del cubo (8 vértices)
    let vertices = vec![
        // Cara frontal (z positivo)
        Vertex {
            position: [-0.5, -0.5, 0.5],  // 0: Inferior izquierda
            normal: [0.0, 0.0, 1.0],
            tex_coords: [0.0, 0.0],
            color: [1.0, 0.0, 0.0, 1.0],  // Rojo
        },
        Vertex {
            position: [0.5, -0.5, 0.5],   // 1: Inferior derecha
            normal: [0.0, 0.0, 1.0],
            tex_coords: [1.0, 0.0],
            color: [0.0, 1.0, 0.0, 1.0],  // Verde
        },
        Vertex {
            position: [0.5, 0.5, 0.5],    // 2: Superior derecha
            normal: [0.0, 0.0, 1.0],
            tex_coords: [1.0, 1.0],
            color: [0.0, 0.0, 1.0, 1.0],  // Azul
        },
        Vertex {
            position: [-0.5, 0.5, 0.5],   // 3: Superior izquierda
            normal: [0.0, 0.0, 1.0],
            tex_coords: [0.0, 1.0],
            color: [1.0, 1.0, 0.0, 1.0],  // Amarillo
        },
        // Cara trasera (z negativo)
        Vertex {
            position: [-0.5, -0.5, -0.5], // 4: Inferior izquierda
            normal: [0.0, 0.0, -1.0],
            tex_coords: [1.0, 0.0],
            color: [1.0, 0.0, 1.0, 1.0],  // Magenta
        },
        Vertex {
            position: [0.5, -0.5, -0.5],  // 5: Inferior derecha
            normal: [0.0, 0.0, -1.0],
            tex_coords: [0.0, 0.0],
            color: [0.0, 1.0, 1.0, 1.0],  // Cian
        },
        Vertex {
            position: [0.5, 0.5, -0.5],   // 6: Superior derecha
            normal: [0.0, 0.0, -1.0],
            tex_coords: [0.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],  // Blanco
        },
        Vertex {
            position: [-0.5, 0.5, -0.5],  // 7: Superior izquierda
            normal: [0.0, 0.0, -1.0],
            tex_coords: [1.0, 1.0],
            color: [0.5, 0.5, 0.5, 1.0],  // Gris
        },
    ];

    // Índices para las caras del cubo (12 triángulos, 36 índices)
    let indices = vec![
        // Cara frontal
        0, 1, 2, 0, 2, 3,
        // Cara trasera
        4, 6, 5, 4, 7, 6,
        // Cara superior
        3, 2, 6, 3, 6, 7,
        // Cara inferior
        0, 5, 1, 0, 4, 5,
        // Cara izquierda
        0, 3, 7, 0, 7, 4,
        // Cara derecha
        1, 5, 6, 1, 6, 2,
    ];

    Mesh { vertices, indices }
}

// Función para crear una pirámide
fn create_pyramid() -> Mesh {
    // Vértices de la pirámide (5 vértices)
    let vertices = vec![
        // Base (cuadrado)
        Vertex {
            position: [-0.5, -0.5, 0.0],  // 0: Inferior izquierda
            normal: [0.0, -1.0, 0.0],
            tex_coords: [0.0, 0.0],
            color: [1.0, 0.0, 0.0, 1.0],  // Rojo
        },
        Vertex {
            position: [0.5, -0.5, 0.0],   // 1: Inferior derecha
            normal: [0.0, -1.0, 0.0],
            tex_coords: [1.0, 0.0],
            color: [0.0, 1.0, 0.0, 1.0],  // Verde
        },
        Vertex {
            position: [0.5, 0.5, 0.0],    // 2: Superior derecha
            normal: [0.0, -1.0, 0.0],
            tex_coords: [1.0, 1.0],
            color: [0.0, 0.0, 1.0, 1.0],  // Azul
        },
        Vertex {
            position: [-0.5, 0.5, 0.0],   // 3: Superior izquierda
            normal: [0.0, -1.0, 0.0],
            tex_coords: [0.0, 1.0],
            color: [1.0, 1.0, 0.0, 1.0],  // Amarillo
        },
        // Punta de la pirámide
        Vertex {
            position: [0.0, 0.0, 1.0],    // 4: Punta
            normal: [0.0, 0.0, 1.0],
            tex_coords: [0.5, 0.5],
            color: [1.0, 1.0, 1.0, 1.0],  // Blanco
        },
    ];

    // Índices para las caras de la pirámide
    let indices = vec![
        // Base (cuadrado)
        0, 2, 1, 0, 3, 2,
        // Caras laterales (triángulos)
        0, 1, 4, // Frente
        1, 2, 4, // Derecha
        2, 3, 4, // Atrás
        3, 0, 4, // Izquierda
    ];

    Mesh { vertices, indices }
}

// Función para crear una esfera (aproximada con un icosaedro)
fn create_sphere(radius: f32, subdivisions: u32) -> Mesh {
    // Vector para almacenar vértices
    let mut vertices = Vec::new();
    // Vector para almacenar índices
    let mut indices = Vec::new();

    // Factor para normalizar a un radio de 1.0
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let r = radius;

    // Vértices de un icosaedro regular
    let base_vertices = [
        [-r, 0.0, t*r], [r, 0.0, t*r], [-r, 0.0, -t*r], [r, 0.0, -t*r],
        [0.0, t*r, r], [0.0, t*r, -r], [0.0, -t*r, r], [0.0, -t*r, -r],
        [t*r, r, 0.0], [-t*r, r, 0.0], [t*r, -r, 0.0], [-t*r, -r, 0.0]
    ];

    // Caras del icosaedro
    let base_faces = [
        [0, 4, 1], [0, 9, 4], [9, 5, 4], [4, 5, 8], [4, 8, 1],
        [8, 10, 1], [8, 3, 10], [5, 3, 8], [5, 2, 3], [2, 7, 3],
        [7, 10, 3], [7, 6, 10], [7, 11, 6], [11, 0, 6], [0, 1, 6],
        [6, 1, 10], [9, 0, 11], [9, 11, 2], [9, 2, 5], [7, 2, 11]
    ];

    // Crear vértices
    for i in 0..12 {
        let pos = base_vertices[i];
        // Normalizar la posición para que quede en la superficie de la esfera
        let mut normal = [pos[0], pos[1], pos[2]];
        let length = (normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]).sqrt();
        normal[0] /= length;
        normal[1] /= length;
        normal[2] /= length;

        // Coordenadas de textura (aproximadas)
        let u = 0.5 + (normal[0].atan2(normal[2]) / (2.0 * std::f32::consts::PI));
        let v = 0.5 - (normal[1].asin() / std::f32::consts::PI);

        // Color basado en la normal
        let color = [
            (normal[0] + 1.0) / 2.0,
            (normal[1] + 1.0) / 2.0,
            (normal[2] + 1.0) / 2.0,
            1.0
        ];

        vertices.push(Vertex {
            position: pos,
            normal,
            tex_coords: [u, v],
            color,
        });
    }

    // Crear índices (caras triangulares)
    for face in &base_faces {
        indices.push(face[0]);
        indices.push(face[1]);
        indices.push(face[2]);
    }

    // Subdividir malla si es necesario
    for _ in 0..subdivisions {
        let mut new_indices = Vec::new();
        let mut midpoints = std::collections::HashMap::new();

        // Procesar triángulos
        for chunk in indices.chunks(3) {
            let i1 = chunk[0];
            let i2 = chunk[1];
            let i3 = chunk[2];

            // // Obtener vértices del triángulo
            // let v1 = &vertices[i1 as usize];
            // let v2 = &vertices[i2 as usize];
            // let v3 = &vertices[i3 as usize];

            // Crear puntos medios
            let m1 = add_midpoint(&mut vertices, &mut midpoints, i1, i2, radius);
            let m2 = add_midpoint(&mut vertices, &mut midpoints, i2, i3, radius);
            let m3 = add_midpoint(&mut vertices, &mut midpoints, i3, i1, radius);

            // Crear 4 triángulos
            new_indices.extend_from_slice(&[i1, m1, m3]);
            new_indices.extend_from_slice(&[i2, m2, m1]);
            new_indices.extend_from_slice(&[i3, m3, m2]);
            new_indices.extend_from_slice(&[m1, m2, m3]);
        }

        // Reemplazar índices
        indices = new_indices;
    }

    Mesh { vertices, indices }
}

// Función auxiliar para crear un punto medio entre dos vértices (para la esfera)
fn add_midpoint(
    vertices: &mut Vec<Vertex>,
    midpoints: &mut std::collections::HashMap<(u32, u32), u32>,
    i1: u32,
    i2: u32,
    radius: f32
) -> u32 {
    // Obtener vértices a partir de los índices
    let v1 = &vertices[i1 as usize];
    let v2 = &vertices[i2 as usize];

    // Ordenar índices para evitar duplicados
    let (i_min, i_max) = if i1 < i2 { (i1, i2) } else { (i2, i1) };
    let key = (i_min, i_max);

    // Verificar si ya existe este punto medio
    if let Some(&index) = midpoints.get(&key) {
        return index;
    }

    // Crear nuevo punto medio
    let position = [
        (v1.position[0] + v2.position[0]) * 0.5,
        (v1.position[1] + v2.position[1]) * 0.5,
        (v1.position[2] + v2.position[2]) * 0.5,
    ];

    // Normalizar a la superficie de la esfera
    let mut normal = [position[0], position[1], position[2]];
    let length = (normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]).sqrt();
    normal[0] /= length;
    normal[1] /= length;
    normal[2] /= length;

    // Escalar al radio deseado
    let position = [
        normal[0] * radius,
        normal[1] * radius,
        normal[2] * radius,
    ];

    // Coordenadas de textura (aproximadas)
    let u = 0.5 + (normal[0].atan2(normal[2]) / (2.0 * std::f32::consts::PI));
    let v = 0.5 - (normal[1].asin() / std::f32::consts::PI);

    // Color basado en la normal
    let color = [
        (normal[0] + 1.0) / 2.0,
        (normal[1] + 1.0) / 2.0,
        (normal[2] + 1.0) / 2.0,
        1.0
    ];

    // Crear nuevo vértice
    let vertex = Vertex {
        position,
        normal,
        tex_coords: [u, v],
        color,
    };

    // Añadir vértice y registrar su índice
    let new_index = vertices.len() as u32;
    vertices.push(vertex);
    midpoints.insert(key, new_index);

    new_index
}

// Función para crear una textura de tablero de ajedrez
fn create_checkerboard_texture(width: u32, height: u32) -> Texture {
    // Cada pixel tiene 4 bytes (RGBA)
    let mut data = vec![0u8; (width * height * 4) as usize];
    let cell_size = 16; // Tamaño de cada cuadro
    
    for y in 0..height {
        for x in 0..width {
            let cell_x = x / cell_size;
            let cell_y = y / cell_size;
            
            // Patrón de tablero alternando blanco y negro
            let is_white = (cell_x + cell_y) % 2 == 0;
            
            // Índice base para el pixel (x,y) en el vector de bytes
            let idx = ((y * width + x) * 4) as usize;
            
            // Color blanco o negro en formato RGBA (individual bytes)
            if is_white {
                // Blanco (255, 255, 255, 255)
                data[idx] = 255;     // R
                data[idx + 1] = 255; // G
                data[idx + 2] = 255; // B
                data[idx + 3] = 255; // A
            } else {
                // Negro (0, 0, 0, 255)
                data[idx] = 0;       // R
                data[idx + 1] = 0;   // G
                data[idx + 2] = 0;   // B
                data[idx + 3] = 255; // A
            }
        }
    }
    
    Texture::new(width, height, data)
}

// Función para mostrar una animación de objetos relacionados jerárquicamente
fn main() -> Result<(), Box<dyn Error>> {
    // Configuramos el logger
    spuc3d::init();
    
    // Abrimos e inicializamos el framebuffer
    let mut fb = Framebuffer::new("/dev/fb0")?;
    
    // Configurar framebuffer para doble buffering
    let mut var_info = fb.var_screen_info.clone();
    var_info.yres_virtual = var_info.yres * 2;
    var_info.activate = 0x4000; // FB_ACTIVATE_NOW | FB_ACTIVATE_FORCE

    // Aplicar la configuración
    Framebuffer::put_var_screeninfo(&fb.device, &var_info)?;

    // Recargar la información del framebuffer
    fb.var_screen_info = Framebuffer::get_var_screeninfo(&fb.device)?;

    println!("Framebuffer info:");
    println!("  Resolution: {}x{}", fb.var_screen_info.xres, fb.var_screen_info.yres);
    println!("  Virtual: {}x{}", fb.var_screen_info.xres_virtual, fb.var_screen_info.yres_virtual);
    println!("  BPP: {}", fb.var_screen_info.bits_per_pixel);
    println!("  Line length: {}", fb.fix_screen_info.line_length);

    // Obtenemos las dimensiones reales del framebuffer
    let width = fb.var_screen_info.xres as usize;
    let height = fb.var_screen_info.yres as usize;
    println!("Usando framebuffer con resolución: {}x{}", width, height);
    
    // Abrimos el dispositivo táctil
    let mut touch = match TouchInput::open("/dev/input/event1") {
        Ok(touch) => {
            println!("Panel táctil inicializado correctamente");
            Some(touch)
        },
        Err(e) => {
            eprintln!("Error al abrir el panel táctil, continuando sin entrada: {}", e);
            None
        }
    };
    
    // Crear rasterizador ajustado a la resolución del framebuffer
    let mut rasterizer = Rasterizer::new(width as u32, height as u32);
    let clear_color = 0x000022; // Azul muy oscuro
    
    // Crear shader
    let shader = DemoShader::new();
    
    // Crear pipeline con una referencia mutable al rasterizador (sin clonar)
    let mut pipeline = Pipeline::new(shader, &mut rasterizer);
    
    // Configurar la cámara (alejada para ver toda la escena pero no demasiado)
    let eye = [0.0, 1.0, 5.0];  // Acercamos la cámara para ver mejor
    let target = [0.0, 0.0, 0.0];
    let up = [0.0, 1.0, 0.0];
    pipeline.look_at(&eye, &target, &up);
    
    // Configurar la proyección con un FOV más amplio y más cercano
    let fov = std::f32::consts::PI / 6.0; // 30 grados (más estrecho para menos distorsión)
    let aspect_ratio = width as f32 / height as f32;
    let near = 1.0;  // Plano cercano más alejado para evitar problemas de precisión
    let far = 50.0;  // No necesitamos ver tan lejos
    pipeline.set_perspective(fov, aspect_ratio, near, far);
    
    // Crear las geometrías
    let cube = create_cube();
    let pyramid = create_pyramid();
    let sphere = create_sphere(0.5, 1); // Radio 0.5, 1 subdivisión
    
    // Crear textura de tablero de ajedrez
    let checkerboard_texture = create_checkerboard_texture(256, 256);
    
    // Crear transformaciones para objetos
    let mut root_transform = Transform::new();
    let mut sun_transform = Transform::new();
    let mut planet_transform = Transform::new();
    let mut moon_transform = Transform::new();
    let mut satellite_transform = Transform::new();
    
    // Configurar jerarquía de transformaciones
    planet_transform.set_parent(Some(&sun_transform));
    moon_transform.set_parent(Some(&planet_transform));
    satellite_transform.set_parent(Some(&moon_transform));
    
    // Variables para interacción táctil
    let mut last_touch_pos = (0, 0);
    let mut root_rotation_y = 0.0f32;
    let mut view_distance = 8.0f32;
    let mut auto_rotate = true;
    
    // Para medir el tiempo y controlar FPS
    let start_time = Instant::now();
    let mut last_frame_time = Instant::now();
    let frame_duration = Duration::from_millis(16); // ~60 FPS
    
    // Bucle de animación
    let mut running = true;
    while running {
        // Esperar hasta que sea tiempo del siguiente frame
        let now = Instant::now();
        let elapsed_since_last_frame = now - last_frame_time;
        if elapsed_since_last_frame < frame_duration {
            let sleep_time = frame_duration - elapsed_since_last_frame;
            thread::sleep(sleep_time);
            continue;
        }
        last_frame_time = now;
        frames_count += 1;

        if now.duration_since(last_fps_time).as_secs() >= 1 {
            println!("FPS: {}", frames_count);
            frames_count = 0;
            last_fps_time = now;
        }
        
        // Calcular tiempo transcurrido para animación
        let elapsed = start_time.elapsed().as_secs_f32();
        
        // Procesar eventos táctiles
        if let Some(ref mut touch) = touch {
            if let Ok(true) = touch.poll_events() {
                let (x, y) = touch.get_position();
                
                if touch.is_touching() {
                    // Si no es el primer toque, calcular el delta
                    if last_touch_pos.0 != 0 && last_touch_pos.1 != 0 {
                        let dx = x - last_touch_pos.0;
                        let dy = y - last_touch_pos.1;
                        
                        // Rotar vista según movimiento horizontal
                        root_rotation_y += dx as f32 * 0.01;
                        
                        // Zoom según movimiento vertical
                        view_distance -= dy as f32 * 0.05;
                        view_distance = view_distance.max(3.0).min(15.0);
                        
                        // Actualizar la cámara
                        let eye = [
                            view_distance * root_rotation_y.sin(),
                            2.0,
                            view_distance * root_rotation_y.cos()
                        ];
                        pipeline.look_at(&eye, &target, &up);
                        
                        // Desactivar rotación automática cuando hay interacción
                        auto_rotate = false;
                    }
                    
                    last_touch_pos = (x, y);
                } else {
                    // Reiniciar posición de toque cuando se levanta el dedo
                    last_touch_pos = (0, 0);
                }
            }
        }
        
        // Limpiar buffers
        pipeline.clear_rasterizer(clear_color, f32::INFINITY);
        
        // Sistema solar: transformaciones anidadas
        // ----------------------------------------
        
        // Transformación raíz (afecta a todo el sistema)
        root_transform = Transform::new();
        if auto_rotate {
            root_transform.rotate_y(elapsed * 0.1);
        } else {
            root_transform.rotate_y(root_rotation_y);
        }
        
        // Sol (esfera central)
        sun_transform = Transform::new();
        sun_transform.set_parent(Some(&root_transform));
        sun_transform.set_uniform_scale(1.5);
        sun_transform.rotate_y(elapsed * 0.2);            // Dibujar el sol con textura
        pipeline.set_model_transform(&mut sun_transform);
        pipeline.render(&sphere, Some(&checkerboard_texture));
        
        // Planeta (cubo orbitando)
        planet_transform = Transform::new();
        planet_transform.set_parent(Some(&sun_transform));
        planet_transform.set_position_xyz(3.0 * elapsed.cos(), 0.0, 3.0 * elapsed.sin());
        planet_transform.set_uniform_scale(0.8);
        planet_transform.rotate_y(elapsed * 2.0);
        
        // Dibujar el planeta con textura
        pipeline.set_model_transform(&mut planet_transform);
        pipeline.render(&cube, Some(&checkerboard_texture));
        
        // Luna (pirámide orbitando el planeta)
        moon_transform = Transform::new();
        moon_transform.set_parent(Some(&planet_transform));
        moon_transform.set_position_xyz(1.5 * (elapsed * 2.0).cos(), 0.0, 1.5 * (elapsed * 2.0).sin());
        moon_transform.set_uniform_scale(0.5);
        moon_transform.rotate_z(elapsed * 1.5);
        
        // Dibujar la luna sin textura (uso del color del vértice)
        pipeline.set_model_transform(&mut moon_transform);
        pipeline.render(&pyramid, None);
        
        // Satélite (esfera pequeña orbitando la luna)
        satellite_transform = Transform::new();
        satellite_transform.set_parent(Some(&moon_transform));
        satellite_transform.set_position_xyz(0.8 * (elapsed * 3.0).cos(), 0.8 * (elapsed * 3.0).sin(), 0.0);
        satellite_transform.set_uniform_scale(0.3);
        
        // Dibujar el satélite con textura
        pipeline.set_model_transform(&mut satellite_transform);
        pipeline.render(&sphere, Some(&checkerboard_texture));

        let bytes_per_pixel = (fb.var_screen_info.bits_per_pixel / 8) as usize;

        // Preparar un buffer con los datos del rasterizador
        let mut frame_buffer = vec![0u8; width * height * bytes_per_pixel];

        // Obtener el color buffer desde el pipeline para evitar problemas de préstamo
        let color_buffer = pipeline.get_color_buffer();
        
        // Convertir datos del rasterizador al formato de framebuffer
        for y in 0..height {
            for x in 0..width {
                let src_idx = y * width + x;
                let dst_idx = (y * width + x) * bytes_per_pixel;
                
                let color = color_buffer[src_idx];
                
                let r = ((color >> 16) & 0xFF) as u8;
                let g = ((color >> 8) & 0xFF) as u8;
                let b = (color & 0xFF) as u8;
                
                frame_buffer[dst_idx] = b;
                frame_buffer[dst_idx + 1] = g;
                frame_buffer[dst_idx + 2] = r;
                if bytes_per_pixel >= 4 {
                    frame_buffer[dst_idx + 3] = 0;
                }
            }
        }

        // En lugar de usar write_frame, vamos a escribir directamente al buffer activo
        // para manejar correctamente el doble buffering
        let buffer_offset = if fb.var_screen_info.yoffset == 0 {
            // Escribir en el segundo buffer
            fb.var_screen_info.yres as usize * fb.fix_screen_info.line_length as usize
        } else {
            // Escribir en el primer buffer
            0
        };

        // Obtener un slice mutable al framebuffer
        let fb_slice = &mut *fb.frame;

        // Copiar los datos en la posición correcta del buffer
        for y in 0..height {
            for x in 0..width {
                let src_idx = (y * width + x) * bytes_per_pixel;
                let dst_idx = buffer_offset + (y * fb.fix_screen_info.line_length as usize) + (x * bytes_per_pixel);

                // Comprobar que estamos dentro de los límites
                if dst_idx + bytes_per_pixel <= fb_slice.len() && src_idx + bytes_per_pixel <= frame_buffer.len() {
                    for i in 0..bytes_per_pixel {
                        fb_slice[dst_idx + i] = frame_buffer[src_idx + i];
                    }
                }
            }
        }

        // Configurar la información para hacer el pan (flip)
        let mut var_info = fb.var_screen_info.clone();

        // Alternar entre los dos buffers
        if var_info.yoffset == 0 {
            var_info.yoffset = var_info.yres;
        } else {
            var_info.yoffset = 0;
        }

        // Realizar el pan_display (equivalente a flip)
        match Framebuffer::pan_display(&fb.device, &var_info) {
            Ok(_) => {},
            Err(e) => eprintln!("Error al hacer flip del framebuffer: {}", e)
        }

        // Actualizar la información del framebuffer para el próximo frame
        fb.var_screen_info = var_info;

        // Verificar si debemos salir
        if elapsed > 120.0 {
            running = false;
        }
    }
    
    // // Limpiar recursos
    // fb.close();
    
    println!("Demostración finalizada");
    Ok(())
}

// Función para guardar un frame como imagen (no implementada)
#[allow(dead_code)]
fn save_frame(buffer: &[u32], width: u32, height: u32, filename: String) {
    // Esta función requeriría una biblioteca como 'image' para guardar el buffer como PNG
    println!("Guardando frame como {}", filename);
}
