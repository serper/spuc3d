use spuc3d::renderer::{core::{
    pipeline::Pipeline, rasterizer::Rasterizer, transform::Transform
}, geometry::load_obj};
use spuc3d::renderer::geometry::{Mesh, Vertex};
use spuc3d::renderer::shader::DefaultShader;
use spuc3d::renderer::texture::Texture;
use spuc3d::renderer::geometry::obj_loader;

use std::{
    fs::{File, OpenOptions},
    io::{self, Read},
    mem,
    thread, 
    time::{Duration, Instant},
};
// Importar nuestra implementación alternativa usando minifb
use std::slice;
use framebuffer::Framebuffer;
use std::error::Error;
use std::rc::Rc;
use std::cell::RefCell;

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

// // Constantes para ioctl
// const FBIOGET_VSCREENINFO: u64 = 0x4600;
// const FBIOPUT_VSCREENINFO: u64 = 0x4601;
// const FBIOGET_FSCREENINFO: u64 = 0x4602;
// const FBIOPAN_DISPLAY: u64 = 0x4606;

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

// Estructuras y constantes para la entrada táctil
struct TouchInput {
    fd: File,
    x: i32,
    y: i32,
    touch_active: bool,
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

fn create_triangle() -> Mesh {
    // Definir los 3 vértices del triángulo con profundidad
    let vertices = vec![
        Vertex::new(
            [-0.5, -0.5, -1.0],  // Z negativo (más cerca de la cámara)
            [0.0, 0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ),
        Vertex::new(
            [0.5, -0.5, -1.0],   // Z negativo (más cerca de la cámara)
            [0.0, 0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ),
        Vertex::new(
            [0.0, 0.5, -1.0],    // Z negativo (más cerca de la cámara)
            [0.0, 0.0, 1.0],
            [0.5, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ),
    ];

    // Índices para definir el triángulo
    let indices = vec![0, 1, 2];

    // Crear y devolver la malla
    Mesh::new(vertices, indices, None)
}

fn create_cube() -> Mesh {
    // Definir los 8 vértices del cubo (centrado en el origen)
    let vertices = vec![
        // Frontal (Z+)
        Vertex::new([-0.5, -0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
        Vertex::new([0.5, -0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 0.0], [0.0, 1.0, 0.0, 1.0]),
        Vertex::new([0.5, 0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 1.0], [0.0, 0.0, 1.0, 1.0]),
        Vertex::new([-0.5, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0], [1.0, 1.0, 0.0, 1.0]),
        
        // Trasera (Z-)
        Vertex::new([-0.5, -0.5, -0.5], [0.0, 0.0, -1.0], [1.0, 0.0], [1.0, 0.0, 1.0, 1.0]),
        Vertex::new([0.5, -0.5, -0.5], [0.0, 0.0, -1.0], [0.0, 0.0], [0.0, 1.0, 1.0, 1.0]),
        Vertex::new([0.5, 0.5, -0.5], [0.0, 0.0, -1.0], [0.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
        Vertex::new([-0.5, 0.5, -0.5], [0.0, 0.0, -1.0], [1.0, 1.0], [0.5, 0.5, 0.5, 1.0]),
    ];

    // Definir los 12 triángulos del cubo (6 caras, 2 triángulos por cara)
    let indices = vec![
        // Frontal
        0, 1, 2, 0, 2, 3,
        // Trasera
        5, 4, 7, 5, 7, 6,
        // Superior
        3, 2, 6, 3, 6, 7,
        // Inferior
        0, 4, 5, 0, 5, 1,
        // Izquierda
        0, 3, 7, 0, 7, 4,
        // Derecha
        1, 5, 6, 1, 6, 2,
    ];

    Mesh::new(vertices, indices, None)
}

// Función para crear una pirámide
fn create_pyramid() -> Mesh {
    // Vértices de la pirámide con normales corregidas
    let mut vertices = vec![
        // Base (cuadrado) - todos apuntando hacia abajo para la cara inferior
        Vertex {
            position: [-0.5, -0.5, 0.0],  // 0: Inferior izquierda
            normal: [0.0, 0.0, -1.0],     // Normal apuntando hacia abajo (-Z)
            tex_coords: [0.0, 0.0],
            color: [1.0, 0.0, 0.0, 1.0],  // Rojo
        },
        Vertex {
            position: [0.5, -0.5, 0.0],   // 1: Inferior derecha
            normal: [0.0, 0.0, -1.0],     // Normal apuntando hacia abajo (-Z)
            tex_coords: [1.0, 0.0],
            color: [0.0, 1.0, 0.0, 1.0],  // Verde
        },
        Vertex {
            position: [0.5, 0.5, 0.0],    // 2: Superior derecha
            normal: [0.0, 0.0, -1.0],     // Normal apuntando hacia abajo (-Z)
            tex_coords: [1.0, 1.0],
            color: [0.0, 0.0, 1.0, 1.0],  // Azul
        },
        Vertex {
            position: [-0.5, 0.5, 0.0],   // 3: Superior izquierda
            normal: [0.0, 0.0, -1.0],     // Normal apuntando hacia abajo (-Z)
            tex_coords: [0.0, 1.0],
            color: [1.0, 1.0, 0.0, 1.0],  // Amarillo
        },
    ];
    
    // Para las caras laterales, necesitamos vértices adicionales con normales específicas
    // Frente
    vertices.push(Vertex {
        position: [-0.5, -0.5, 0.0],
        normal: [-0.4472, -0.4472, 0.7746],  // Normal de la cara frontal
        tex_coords: [0.0, 0.0],
        color: [1.0, 0.0, 0.0, 1.0],
    });
    vertices.push(Vertex {
        position: [0.5, -0.5, 0.0],
        normal: [-0.4472, -0.4472, 0.7746],
        tex_coords: [1.0, 0.0],
        color: [0.0, 1.0, 0.0, 1.0],
    });
    vertices.push(Vertex {
        position: [0.0, 0.0, 1.0],
        normal: [-0.4472, -0.4472, 0.7746],
        tex_coords: [0.5, 0.5],
        color: [1.0, 1.0, 1.0, 1.0],
    });
    
    // Derecha
    vertices.push(Vertex {
        position: [0.5, -0.5, 0.0],
        normal: [0.4472, -0.4472, 0.7746],  // Normal de la cara derecha
        tex_coords: [0.0, 0.0],
        color: [0.0, 1.0, 0.0, 1.0],
    });
    vertices.push(Vertex {
        position: [0.5, 0.5, 0.0],
        normal: [0.4472, -0.4472, 0.7746],
        tex_coords: [1.0, 0.0],
        color: [0.0, 0.0, 1.0, 1.0],
    });
    vertices.push(Vertex {
        position: [0.0, 0.0, 1.0],
        normal: [0.4472, -0.4472, 0.7746],
        tex_coords: [0.5, 0.5],
        color: [1.0, 1.0, 1.0, 1.0],
    });
    
    // Atrás
    vertices.push(Vertex {
        position: [0.5, 0.5, 0.0],
        normal: [0.4472, 0.4472, 0.7746],  // Normal de la cara trasera
        tex_coords: [0.0, 0.0],
        color: [0.0, 0.0, 1.0, 1.0],
    });
    vertices.push(Vertex {
        position: [-0.5, 0.5, 0.0],
        normal: [0.4472, 0.4472, 0.7746],
        tex_coords: [1.0, 0.0],
        color: [1.0, 1.0, 0.0, 1.0],
    });
    vertices.push(Vertex {
        position: [0.0, 0.0, 1.0],
        normal: [0.4472, 0.4472, 0.7746],
        tex_coords: [0.5, 0.5],
        color: [1.0, 1.0, 1.0, 1.0],
    });
    
    // Izquierda
    vertices.push(Vertex {
        position: [-0.5, 0.5, 0.0],
        normal: [-0.4472, 0.4472, 0.7746],  // Normal de la cara izquierda
        tex_coords: [0.0, 0.0],
        color: [1.0, 1.0, 0.0, 1.0],
    });
    vertices.push(Vertex {
        position: [-0.5, -0.5, 0.0],
        normal: [-0.4472, 0.4472, 0.7746],
        tex_coords: [1.0, 0.0],
        color: [1.0, 0.0, 0.0, 1.0],
    });
    vertices.push(Vertex {
        position: [0.0, 0.0, 1.0],
        normal: [-0.4472, 0.4472, 0.7746],
        tex_coords: [0.5, 0.5],
        color: [1.0, 1.0, 1.0, 1.0],
    });

    // Índices para las caras de la pirámide
    let indices = vec![
        // Base (cuadrado) - en sentido antihorario para OpenGL
        0, 1, 2, 
        0, 2, 3,
        // Caras laterales (triángulos) - Usando los nuevos vértices con normales correctas
        4, 5, 6,  // Frente
        7, 8, 9,  // Derecha
        10, 11, 12, // Atrás
        13, 14, 15, // Izquierda
    ];

    Mesh::new(vertices, indices, None)
}

// Función para crear una esfera (aproximada con un icosaedro)
fn create_sphere(radius: f32, subdivisions: u32) -> Mesh {
    // Vector para almacenar vértices
    let mut vertices = Vec::new();
    // Mapa para evitar duplicados de vértices durante la subdivisión
    let mut vertex_indices = std::collections::HashMap::new();

    // Factor para normalizar a un radio de 1.0
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    
    // Vértices de un icosaedro regular
    let base_vertices = [
        [-1.0, 0.0, t], [1.0, 0.0, t], [-1.0, 0.0, -t], [1.0, 0.0, -t],
        [0.0, t, 1.0], [0.0, t, -1.0], [0.0, -t, 1.0], [0.0, -t, -1.0],
        [t, 1.0, 0.0], [-t, 1.0, 0.0], [t, -1.0, 0.0], [-t, -1.0, 0.0]
    ];

    // Caras del icosaedro
    let base_faces = [
        [0, 4, 1], [0, 9, 4], [9, 5, 4], [4, 5, 8], [4, 8, 1],
        [8, 10, 1], [8, 3, 10], [5, 3, 8], [5, 2, 3], [2, 7, 3],
        [7, 10, 3], [7, 6, 10], [7, 11, 6], [11, 0, 6], [0, 1, 6],
        [6, 1, 10], [9, 0, 11], [9, 11, 2], [9, 2, 5], [7, 2, 11]
    ];

    // Función para obtener el índice de un vértice o crear uno nuevo si no existe
    let mut get_vertex_index = |p: [f32; 3]| {
        let key = format!("{:.6},{:.6},{:.6}", p[0], p[1], p[2]);
        if let Some(&idx) = vertex_indices.get(&key) {
            return idx;
        }
        
        // Normalizar la posición
        let mut normal = [p[0], p[1], p[2]];
        let length = (normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]).sqrt();
        normal[0] /= length;
        normal[1] /= length;
        normal[2] /= length;
        
        // Posición en la superficie de la esfera
        let position = [
            normal[0] * radius,
            normal[1] * radius,
            normal[2] * radius
        ];
        
        // Coordenadas de textura esféricas
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
        let idx = vertices.len();
        vertices.push(Vertex {
            position,
            normal,
            tex_coords: [u, v],
            color,
        });
        
        vertex_indices.insert(key, idx);
        idx
    };

    // Subdividir cada triángulo del icosaedro
    let mut faces = Vec::new();
    for face in &base_faces {
        let v1 = base_vertices[face[0]];
        let v2 = base_vertices[face[1]];
        let v3 = base_vertices[face[2]];
        
        // Añadir el triángulo inicial a la lista de caras a procesar
        faces.push([v1, v2, v3]);
    }
    
    // Realizar las subdivisiones
    for _ in 0..subdivisions {
        let mut new_faces = Vec::new();
        for face in faces {
            let v1 = face[0];
            let v2 = face[1];
            let v3 = face[2];
            
            // Calcular puntos medios
            let v12 = [
                (v1[0] + v2[0]) / 2.0,
                (v1[1] + v2[1]) / 2.0,
                (v1[2] + v2[2]) / 2.0
            ];
            
            let v23 = [
                (v2[0] + v3[0]) / 2.0,
                (v2[1] + v3[1]) / 2.0,
                (v2[2] + v3[2]) / 2.0
            ];
            
            let v31 = [
                (v3[0] + v1[0]) / 2.0,
                (v3[1] + v1[1]) / 2.0,
                (v3[2] + v1[2]) / 2.0
            ];
            
            // Crear 4 nuevos triángulos
            new_faces.push([v1, v12, v31]);
            new_faces.push([v2, v23, v12]);
            new_faces.push([v3, v31, v23]);
            new_faces.push([v12, v23, v31]);
        }
        faces = new_faces;
    }
    
    // Generar los índices finales
    let mut indices = Vec::new();
    for face in faces {
        let i1 = get_vertex_index(face[0]);
        let i2 = get_vertex_index(face[1]);
        let i3 = get_vertex_index(face[2]);
        
        indices.push(i1 as u32);
        indices.push(i2 as u32);
        indices.push(i3 as u32);
    }

    Mesh::new(vertices, indices, None)
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

    // Crear vértice
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
    
    // Para poder detener el programa con Ctrl+C
    let mut running = true;
    ctrlc::set_handler(move || {
        println!("Deteniendo el programa...");
        running = false;
        std::process::exit(0);
    })?;
    println!("Iniciando el programa...");

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
    let enabled = false;
    let color: u32 = 0xFF0000; // Color rojo
    rasterizer.set_wireframe(enabled, Some(color)); // Color rojo para wireframe
    let clear_color = 0x000022; // Azul muy oscuro
    
    // Crear shader
    let shader = DefaultShader::new();
    
    // Crear pipeline con una referencia mutable al rasterizador (sin clonar)
    let mut pipeline = Pipeline::new(shader, &mut rasterizer);
    
    // Posicionar la cámara en el eje Z negativo para contrarrestar la inversión de la matriz de vista
    let eye = [0.0, 0.0, 4.0];
    let target = [0.0, 0.0, 0.0];
    let up = [0.0, 1.0, 0.0];
    pipeline.look_at(&eye, &target, &up);
    
    // Configurar la proyección con parámetros adecuados
    let fov = std::f32::consts::PI / 4.0; // 30 grados (más estrecho para mejor visualización)
    let aspect_ratio = width as f32 / height as f32;
    let near = 0.1; // Aumentar el plano cercano para evitar problemas de clipping
    let far = 10.0; // Reducir el plano lejano para mejorar la precisión del z-buffer
    pipeline.set_perspective(fov, aspect_ratio, near, far);
    
    // Crear las geometrías
    // let cube = create_cube();
    // let pyramid = create_pyramid();
    // let sphere = create_sphere(0.5, 2); // Radio 0.5, 1 subdivisión
    let plane = load_obj("airplane.obj", true, false)?;
    
    // // Crear textura de tablero de ajedrez
    // let texture_enabled = true; // Cambiar a true para usar la textura
    // let checkerboard_texture_data = create_checkerboard_texture(256, 256);
    // let checkerboard_texture = if texture_enabled {
    //         Some(&checkerboard_texture_data)
    //     } else {
    //         None
    //     };
    
    // Crear transformaciones para objetos como Rc<RefCell<Transform>>
    let root_transform = Transform::new();
    let sun_transform = Transform::new();
    // let planet_transform = Transform::new();
    // let moon_transform = Transform::new();
    // let satellite_transform = Transform::new();
    
    // Configurar jerarquía de transformaciones
    Transform::set_parent(&sun_transform, Some(&root_transform));
    // Transform::set_parent(&planet_transform, Some(&sun_transform));
    // Transform::set_parent(&moon_transform, Some(&planet_transform));
    // Transform::set_parent(&satellite_transform, Some(&moon_transform));
    
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
    let mut frames_count = 0;
    let mut last_fps_time = Instant::now();

    while running {
        // Esperar hasta que sea tiempo del siguiente frame
        let now = Instant::now();
        let elapsed_since_last_frame = now - last_frame_time;
        // if elapsed_since_last_frame < frame_duration {
        //     let sleep_time = frame_duration - elapsed_since_last_frame;
        //     thread::sleep(sleep_time);
        //     continue;
        // }
        // Calcular tiempo transcurrido para animación
        let elapsed = last_frame_time.elapsed().as_secs_f32();
        let elapse_since_start = start_time.elapsed().as_secs_f32();

        last_frame_time = now;
        frames_count += 1;

        if now.duration_since(last_fps_time).as_secs() >= 1 {
            println!("FPS: {}", frames_count);
            frames_count = 0;
            last_fps_time = now;
        }
        

        // Calcular los grados a rotar dependiendo del tiempo desde el último frame
        let delta_time = elapsed - last_frame_time.elapsed().as_secs_f32();
        let rotation_speed = 0.5; // Grados por segundo
        let rotation_angle = rotation_speed * delta_time;
        
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
        {
            let mut root = root_transform.borrow_mut();
            if auto_rotate {
                root.rotate_y(rotation_angle);
            } else {
                root.rotate_y(root_rotation_y);
            }
        }
        // Sol (esfera central)
        {
            let mut sun = sun_transform.borrow_mut();
            sun.set_uniform_scale(0.022);
            sun.rotate_y(rotation_angle * 0.2);
            sun.rotate_x(rotation_angle * 0.1);
            pipeline.set_model_transform(&mut sun);
        }
        pipeline.render_parallel(&plane, None);
        // // Planeta (cubo orbitando)
        // {
        //     let mut planet = planet_transform.borrow_mut();
        //     planet.set_position_xyz(2.6 * (elapse_since_start * 0.5).cos(), 0.0, 2.6 * (elapse_since_start * 0.5).sin());
        //     planet.set_uniform_scale(0.8);
        //     planet.rotate_y(rotation_angle * 0.6);
        //     pipeline.set_model_transform(&mut planet);
        // }
        // pipeline.render_parallel(&cube, checkerboard_texture);
        // // Luna (esfera orbitando el planeta)
        // {
        //     let mut moon = moon_transform.borrow_mut();
        //     moon.set_position_xyz(1.5 * elapse_since_start.cos(), 0.0, 1.5 * elapse_since_start.sin());
        //     moon.set_uniform_scale(0.6);
        //     moon.rotate_z(rotation_angle * 1.5);
        //     pipeline.set_model_transform(&mut moon);
        // }
        // pipeline.render_parallel(&sphere, checkerboard_texture);
        // // Satélite (esfera pequeña orbitando la luna)
        // {
        //     let mut satellite = satellite_transform.borrow_mut();
        //     satellite.set_position_xyz(1.0 * elapse_since_start.cos(), 0.8 * elapse_since_start.sin(), 0.0);
        //     satellite.set_uniform_scale(0.3);
        //     satellite.rotate_y(rotation_angle * 2.5);
        //     satellite.rotate_z(rotation_angle * 3.0);
        //     pipeline.set_model_transform(&mut satellite);
        // }
        // pipeline.render_parallel(&sphere, checkerboard_texture);

        let bytes_per_pixel = (fb.var_screen_info.bits_per_pixel / 8) as usize;

        // // Preparar un buffer con los datos del rasterizador
        // let mut frame_buffer = vec![0u8; width * height * bytes_per_pixel];

        // // Obtener el color buffer desde el pipeline para evitar problemas de préstamo
        // let color_buffer = pipeline.get_color_buffer();
        
        // // Convertir datos del rasterizador al formato de framebuffer
        // for y in 0..height {
        //     for x in 0..width {
        //         let src_idx = y * width + x;
        //         let dst_idx = (y * width + x) * bytes_per_pixel;
                
        //         let color = color_buffer[src_idx];
                
        //         let r = ((color >> 16) & 0xFF) as u8;
        //         let g = ((color >> 8) & 0xFF) as u8;
        //         let b = (color & 0xFF) as u8;
                
        //         frame_buffer[dst_idx] = b;
        //         frame_buffer[dst_idx + 1] = g;
        //         frame_buffer[dst_idx + 2] = r;
        //         if bytes_per_pixel >= 4 {
        //             frame_buffer[dst_idx + 3] = 0;
        //         }
        //     }
        // }

        // En lugar de usar write_frame, vamos a escribir directamente al buffer activo
        // para manejar correctamente el doble buffering
        // let buffer_offset = if fb.var_screen_info.yoffset == 0 {
        //     // Escribir en el segundo buffer
        //     fb.var_screen_info.yres as usize * fb.fix_screen_info.line_length as usize
        // } else {
        //     // Escribir en el primer buffer
        //     0
        // };

        // // Obtener un slice mutable al framebuffer
        // let fb_slice = &mut *fb.frame;

        // // Copiar los datos en la posición correcta del buffer
        // for y in 0..height {
        //     for x in 0..width {
        //         let src_idx = (y * width + x) * bytes_per_pixel;
        //         let dst_idx = buffer_offset + (y * fb.fix_screen_info.line_length as usize) + (x * bytes_per_pixel);

        //         // Comprobar que estamos dentro de los límites
        //         if dst_idx + bytes_per_pixel <= fb_slice.len() && src_idx + bytes_per_pixel <= frame_buffer.len() {
        //             for i in 0..bytes_per_pixel {
        //                 fb_slice[dst_idx + i] = frame_buffer[src_idx + i];
        //             }
        //         }
        //     }
        // }

        // Obtener slice mutable al framebuffer, ajustando el offset y el tamaño para el buffer activo
        let buffer_offset = if fb.var_screen_info.yoffset == 0 {
            // Escribir en el segundo buffer
            fb.var_screen_info.yres as usize * fb.fix_screen_info.line_length as usize
        } else {
            // Escribir en el primer buffer
            0
        };
        let fb_slice = &mut *fb.frame;
        let fb_slice = &mut fb_slice[buffer_offset..buffer_offset + (height * fb.fix_screen_info.line_length as usize)];
        
        pipeline.write_color_buffer_to_framebuffer(fb_slice, fb.var_screen_info.xres, fb.var_screen_info.yres, bytes_per_pixel);

        // Configurar la información para hacer el pan (flip)
        if fb.var_screen_info.yoffset == 0 {
            fb.var_screen_info.yoffset = fb.var_screen_info.yres;
        } else {
            fb.var_screen_info.yoffset = 0;
        }

        // Realizar el pan_display (equivalente a flip)
        match Framebuffer::pan_display(&fb.device, &fb.var_screen_info) {
            Ok(_) => {},
            Err(e) => eprintln!("Error al hacer flip del framebuffer: {}", e)
        }
    }
    
    println!("Demostración finalizada");
    Ok(())
}

// Función para guardar un frame como imagen (no implementada)
#[allow(dead_code)]
fn save_frame(buffer: &[u32], width: u32, height: u32, filename: String) {
    // Esta función requeriría una biblioteca como 'image' para guardar el buffer como PNG
    println!("Guardando frame como {}", filename);
}
