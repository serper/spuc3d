use std::sync::OnceLock;
use std::clone::Clone;

pub trait MathBackend: Send + Sync {
    // Métodos para operaciones con vectores
    fn vector_add(&self, v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3];
    fn vector_subtract(&self,v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3];
    fn vector_dot(&self,v1: &[f32; 3], v2: &[f32; 3]) -> f32;
    fn vector_cross(&self,v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3];
    fn vector_scale(&self,v: &[f32; 3], scalar: f32) -> [f32; 3];
    fn vector_length(&self,v: &[f32; 3]) -> f32;
    fn vector_normalize(&self,v: &[f32; 3]) -> [f32; 3];
    fn vector_distance(&self,v1: &[f32; 3], v2: &[f32; 3]) -> f32;
    fn vector_distance_squared(&self,v1: &[f32; 3], v2: &[f32; 3]) -> f32;
    fn vector_length_squared(&self,v: &[f32; 3]) -> f32;
    fn vector_angle(&self,v1: &[f32; 3], v2: &[f32; 3]) -> f32;
    fn vector_project(&self,v: &[f32; 3], onto: &[f32; 3]) -> [f32; 3];
    fn vector_reflect(&self,v: &[f32; 3], normal: &[f32; 3]) -> [f32; 3];
    fn vector_refract(&self,v: &[f32; 3], normal: &[f32; 3], eta: f32) -> [f32; 3];
    fn vector_lerp(&self,v1: &[f32; 3], v2: &[f32; 3], t: f32) -> [f32; 3];
    fn vector_slerp(&self,v1: &[f32; 3], v2: &[f32; 3], t: f32) -> [f32; 3];
    fn vector_squad(&self,v1: &[f32; 3], v2: &[f32; 3], v3: &[f32; 3], v4: &[f32; 3], t: f32) -> [f32; 3];
    fn vector_squad_slerp(&self,v1: &[f32; 3], v2: &[f32; 3], v3: &[f32; 3], v4: &[f32; 3], t: f32) -> [f32; 3];
    fn vector_to_array(&self,v: &[f32; 3]) -> [f32; 3];
    fn vector_from_array(&self, arr: &[f32; 3]) -> [f32; 3];
    fn vector_to_string(&self,v: &[f32; 3]) -> String;
    fn vector_from_string(&self, s: &str) -> [f32; 3];
    fn vector_to_tuple(&self,v: &[f32; 3]) -> (f32, f32, f32);
    fn vector_from_tuple(&self, t: (f32, f32, f32)) -> [f32; 3];
    fn vector_to_vec(&self,v: &[f32; 3]) -> Vec<f32>;
    fn vector_from_vec(&self, v: &Vec<f32>) -> [f32; 3];
    
    // // Métodos SIMD para procesamiento por lotes
    // fn batch_transform_points(&self, points: &[[f32; 3]], matrix: &[[f32; 4]; 4]) -> Vec<[f32; 4]>;
    // fn batch_transform_points_quaternion(&self, 
    //                                    points: &[[f32; 3]], 
    //                                    position: &[f32; 3],
    //                                    rotation: &[f32; 4],  // quaternion como [w, x, y, z]
    //                                    scale: &[f32; 3]) -> Vec<[f32; 4]>;
    // fn batch_normalize_vectors(&self, vectors: &[[f32; 3]]) -> Vec<[f32; 3]>;
    // fn batch_cross_product(&self, vectors_a: &[[f32; 3]], vectors_b: &[[f32; 3]]) -> Vec<[f32; 3]>;

    // Métodos para operaciones con matrices
    fn matrix_multiply(&self, m1: &[[f32; 4]; 4], m2: &[[f32; 4]; 4]) -> [[f32; 4]; 4];
    fn matrix_translate(&self, m: &mut [[f32; 4]; 4], tx: f32, ty: f32, tz: f32);
    fn matrix_scale(&self, m: &mut [[f32; 4]; 4], sx: f32, sy: f32, sz: f32);
    fn matrix_rotate_x(&self, m: &mut [[f32; 4]; 4], angle: f32);
    fn matrix_rotate_y(&self, m: &mut [[f32; 4]; 4], angle: f32);
    fn matrix_rotate_z(&self, m: &mut [[f32; 4]; 4], angle: f32);
    fn matrix_transpose(&self, m: &[[f32; 4]; 4]) -> [[f32; 4]; 4];
    fn matrix_inverse(&self, m: &[[f32; 4]; 4]) -> [[f32; 4]; 4];
    fn matrix_determinant(&self, m: &[[f32; 4]; 4]) -> f32;
    fn matrix_perspective(&self, fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4];
    fn matrix_normalize(&self, v: &[f32; 3]) -> [f32; 3];
    fn matrix_cross(&self, v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3];
    fn matrix_dot(&self, v1: &[f32; 3], v2: &[f32; 3]) -> f32;
    fn matrix_look_at(&self, eye: &[f32; 3], center: &[f32; 3], up: &[f32; 3]) -> [[f32; 4]; 4];
    fn matrix_orthographic(&self, left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [[f32; 4]; 4];
    fn matrix_frustum(&self, left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [[f32; 4]; 4];
    fn matrix_to_array(&self, m: &[[f32; 4]; 4]) -> [[f32; 4]; 4];
    fn matrix_from_array(&self, arr: &[[f32; 4]; 4]) -> [[f32; 4]; 4];
    fn matrix_to_string(&self, m: &[[f32; 4]; 4]) -> String;
    fn matrix_from_string(&self, s: &str) -> [[f32; 4]; 4];
    fn matrix_to_tuple(&self, m: &[[f32; 4]; 4]) -> [[f32; 4]; 4];
    fn matrix_from_tuple(&self, t: [[f32; 4]; 4]) -> [[f32; 4]; 4];
    fn matrix_to_vec(&self, m: &[[f32; 4]; 4]) -> Vec<f32>;
    fn matrix_from_vec(&self, v: &Vec<f32>) -> [[f32; 4]; 4];
    
    // Métodos para operaciones con cuaterniones
    fn quaternion_dot(&self, q1: &[f32; 4], q2: &[f32; 4]) -> f32;
    fn quaternion_multiply(&self, q1: &[f32; 4], q2: &[f32; 4]) -> [f32; 4];
    fn quaternion_conjugate(&self, q: &[f32; 4]) -> [f32; 4];
    fn quaternion_normalize(&self, q: &[f32; 4]) -> [f32; 4];
    fn quaternion_to_matrix(&self, q: &[f32; 4]) -> [[f32; 4]; 4];
    fn quaternion_from_axis_angle(&self, axis: &[f32; 3], angle: f32) -> [f32; 4];
    fn quaternion_to_axis_angle(&self, q: &[f32; 4]) -> ([f32; 3], f32);
    fn quaternion_from_euler(&self, roll: f32, pitch: f32, yaw: f32) -> [f32; 4];
    fn quaternion_to_euler(&self, q: &[f32; 4]) -> (f32, f32, f32);
    fn quaternion_from_rotation_matrix(&self, m: &[[f32; 4]; 4]) -> [f32; 4];
    fn quaternion_to_rotation_matrix(&self, q: &[f32; 4]) -> [[f32; 4]; 4];
    fn quaternion_lerp(&self, q1: &[f32; 4], q2: &[f32; 4], t: f32) -> [f32; 4];
    fn quaternion_slerp(&self, q1: &[f32; 4], q2: &[f32; 4], t: f32) -> [f32; 4];
    fn quaternion_squad(&self, q1: &[f32; 4], q2: &[f32; 4], q3: &[f32; 4], q4: &[f32; 4], t: f32) -> [f32; 4];
    fn quaternion_squad_slerp(&self, q1: &[f32; 4], q2: &[f32; 4], q3: &[f32; 4], q4: &[f32; 4], t: f32) -> [f32; 4];
    fn quaternion_to_array(&self, q: &[f32; 4]) -> [f32; 4];
    fn quaternion_from_array(&self, arr: &[f32; 4]) -> [f32; 4];
    fn quaternion_to_string(&self, q: &[f32; 4]) -> String;
    fn quaternion_from_string(&self, s: &str) -> [f32; 4];
    fn quaternion_to_tuple(&self, q: &[f32; 4]) -> (f32, f32, f32, f32);
    fn quaternion_from_tuple(&self, t: (f32, f32, f32, f32)) -> [f32; 4];
    fn quaternion_to_vec(&self, q: &[f32; 4]) -> Vec<f32>;
    fn quaternion_from_vec(&self, v: &Vec<f32>) -> [f32; 4];

    // Métodos varios
    fn compute_barycentric_coordinates(&self,point: [f32; 2], v0: [f32; 2], v1: [f32; 2], v2: [f32; 2]) -> (f32, f32, f32);
}

//Backends soportados
pub struct CpuMathBackend;
pub struct SimdMathBackend;

// Función para seleccionar el mejor backend disponible
pub fn get_optimal_math_backend() -> Box<dyn MathBackend> {
    #[cfg(feature = "simd")]
    return Box::new(SimdMathBackend);
    
    #[cfg(not(feature = "simd"))]
    Box::new(CpuMathBackend)
}

// Implementamos tanto Clone como Copy para Vector
#[derive(Copy)]
pub struct Vector {
    x: f32,
    y: f32,
    z: f32,
    backend: &'static dyn MathBackend,
}

// Implementación manual de Clone para Vector
impl Clone for Vector {
    fn clone(&self) -> Self {
        *self // Copy permite usar simplemente *self para clonar
    }
}

impl std::fmt::Debug for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vector({}, {}, {})", self.x, self.y, self.z)
    }
}
impl Vector {
    pub fn new_with_values(x: f32, y: f32, z: f32) -> Self {
        // Usar OnceLock para inicializar el backend una sola vez
        // y compartirlo entre todas las instancias de Vector
        static BACKEND: OnceLock<Box<dyn MathBackend>> = OnceLock::new();
        let backend = BACKEND.get_or_init(|| get_optimal_math_backend());

        Self { x, y, z, backend: &**backend }
    }
    
    pub fn new() -> Self {
        // Valor por defecto: vector cero
        Self::new_with_values(0.0, 0.0, 0.0)
    }
    
    pub fn x(&self) -> f32 {
        self.x
    }

    pub fn y(&self) -> f32 {
        self.y
    }

    pub fn z(&self) -> f32 {
        self.z
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            backend: self.backend,
        }
    }

    pub fn subtract(&self, other: &Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            backend: self.backend,
        }
    }

    pub fn dot(&self, other: &Self) -> f32 {
        self.backend.vector_dot(&[self.x, self.y, self.z], &[other.x, other.y, other.z])
    }

    pub fn cross(&self, other: &Self) -> Self {
        let result = self.backend.vector_cross(&[self.x, self.y, self.z], &[other.x, other.y, other.z]);
        Self {
            x: result[0],
            y: result[1],
            z: result[2],
            backend: self.backend,
        }
    }

    pub fn add_scalar(&self, scalar: f32) -> Self {
        Self {
            x: self.x + scalar,
            y: self.y + scalar,
            z: self.z + scalar,
            backend: self.backend,
        }
    }
    
    pub fn subtract_scalar(&self, scalar: f32) -> Self {
        Self {
            x: self.x - scalar,
            y: self.y - scalar,
            z: self.z - scalar,
            backend: self.backend,
        }
    }

    pub fn multiply_scalar(&self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
            backend: self.backend,
        }
    }

    pub fn divide_scalar(&self, scalar: f32) -> Self {
        if scalar == 0.0 {
            panic!("Division by zero");
        }
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
            backend: self.backend,
        }
    }

    pub fn scale(&self, scalar: f32) -> Self {
        let result = self.backend.vector_scale(&[self.x, self.y, self.z], scalar);
        Self {
            x: result[0],
            y: result[1],
            z: result[2],
            backend: self.backend,
        }
    }

    pub fn length(&self) -> f32 {
        self.backend.vector_length(&[self.x, self.y, self.z])
    }

    pub fn length_squared(&self) -> f32 {
        self.backend.vector_length_squared(&[self.x, self.y, self.z])
    }

    pub fn normalize(&self) -> Self {
        let result = self.backend.vector_normalize(&[self.x, self.y, self.z]);
        Self {
            x: result[0],
            y: result[1],
            z: result[2],
            backend: self.backend,
        }
    }

    pub fn distance(&self, other: &Self) -> f32 {
        self.backend.vector_distance(&[self.x, self.y, self.z], &[other.x, other.y, other.z])
    }

    pub fn distance_squared(&self, other: &Self) -> f32 {
        self.backend.vector_distance_squared(&[self.x, self.y, self.z], &[other.x, other.y, other.z])
    }

    pub fn angle(&self, other: &Self) -> f32 {
        self.backend.vector_angle(&[self.x, self.y, self.z], &[other.x, other.y, other.z])
    }

    pub fn project(&self, onto: &Self) -> Self {
        let result = self.backend.vector_project(&[self.x, self.y, self.z], &[onto.x, onto.y, onto.z]);
        Self {
            x: result[0],
            y: result[1],
            z: result[2],
            backend: self.backend,
        }
    }

    pub fn reflect(&self, normal: &Self) -> Self {
        let result = self.backend.vector_reflect(&[self.x, self.y, self.z], &[normal.x, normal.y, normal.z]);
        Self {
            x: result[0],
            y: result[1],
            z: result[2],
            backend: self.backend,
        }
    }

    pub fn refract(&self, normal: &Self, eta: f32) -> Self {
        let result = self.backend.vector_refract(&[self.x, self.y, self.z], &[normal.x, normal.y, normal.z], eta);
        Self {
            x: result[0],
            y: result[1],
            z: result[2],
            backend: self.backend,
        }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let result = self.backend.vector_lerp(&[self.x, self.y, self.z], &[other.x, other.y, other.z], t);
        Self {
            x: result[0],
            y: result[1],
            z: result[2],
            backend: self.backend,
        }
    }

    pub fn slerp(&self, other: &Self, t: f32) -> Self {
        let result = self.backend.vector_slerp(&[self.x, self.y, self.z], &[other.x, other.y, other.z], t);
        Self {
            x: result[0],
            y: result[1],
            z: result[2],
            backend: self.backend,
        }
    }
    
    pub fn squad(&self, v2: &Self, v3: &Self, v4: &Self, t: f32) -> Self {
        let result = self.backend.vector_squad(
            &[self.x, self.y, self.z],
            &[v2.x, v2.y, v2.z],
            &[v3.x, v3.y, v3.z],
            &[v4.x, v4.y, v4.z],
            t,
        );
        Self {
            x: result[0],
            y: result[1],
            z: result[2],
            backend: self.backend,
        }
    }

    pub fn squad_slerp(&self, v2: &Self, v3: &Self, v4: &Self, t: f32) -> Self {
        let result = self.backend.vector_squad_slerp(
            &[self.x, self.y, self.z],
            &[v2.x, v2.y, v2.z],
            &[v3.x, v3.y, v3.z],
            &[v4.x, v4.y, v4.z],
            t,
        );
        Self {
            x: result[0],
            y: result[1],
            z: result[2],
            backend: self.backend,
        }
    }

    pub fn to_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub fn from_array(self, arr: &[f32; 3]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
            backend: self.backend,
        }
    }

    pub fn to_string(&self) -> String {
        format!("Vector({}, {}, {})", self.x, self.y, self.z)
    }

    pub fn from_string(self, s: &str) -> Self {
        let parts: Vec<&str> = s.trim_matches(|c| c == '(' || c == ')').split(',').collect();
        if parts.len() != 3 {
            panic!("Invalid vector string format");
        }
        Self {
            x: parts[0].trim().parse().unwrap(),
            y: parts[1].trim().parse().unwrap(),
            z: parts[2].trim().parse().unwrap(),
            backend: self.backend,
        }
    }

    pub fn to_tuple(&self) -> (f32, f32, f32) {
        (self.x, self.y, self.z)
    }

    pub fn from_tuple(self, t: (f32, f32, f32)) -> Self {
        Self {
            x: t.0,
            y: t.1,
            z: t.2,
            backend: self.backend,
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.x, self.y, self.z]
    }
    
    pub fn from_vec(self, v: &Vec<f32>) -> Self {
        if v.len() != 3 {
            panic!("Invalid vector length");
        }
        Self {
            x: v[0],
            y: v[1],
            z: v[2],
            backend: self.backend,
        }
    }
}

// Implementamos tanto Clone como Copy para Matrix
#[derive(Copy)]
pub struct Matrix {
    elements: [[f32; 4]; 4],  // Arrays fijos como [[f32; 4]; 4] ya implementan Copy
    backend: &'static dyn MathBackend,
}

// Implementación simplificada de Clone para Matrix
impl Clone for Matrix {
    fn clone(&self) -> Self {
        *self // Ahora podemos simplemente usar *self ya que Matrix implementa Copy
    }
}

impl std::fmt::Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Matrix [")?;
        for row in &self.elements {
            writeln!(f, "  [{:7.4}, {:7.4}, {:7.4}, {:7.4}]", row[0], row[1], row[2], row[3])?;
        }
        write!(f, "]")
    }
}

impl Matrix {
    pub fn new_with_values(elements: [[f32; 4]; 4]) -> Self {
        // Usar OnceLock para inicializar el backend una sola vez
        // y compartirlo entre todas las instancias de Matrix
        static BACKEND: OnceLock<Box<dyn MathBackend>> = OnceLock::new();
        let backend = BACKEND.get_or_init(|| get_optimal_math_backend());

        Self { elements, backend: &**backend }
    }
    
    pub fn new() -> Self {
        // Valor por defecto: matriz identidad
        Self::new_with_values([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn elements(&self) -> &[[f32; 4]; 4] {
        &self.elements
    }
    
    pub fn elements_mut(&mut self) -> &mut [[f32; 4]; 4] {
        &mut self.elements
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let result = self.backend.matrix_multiply(&self.elements, &other.elements);
        Self {
            elements: result,
            backend: self.backend,
        }
    }

    pub fn translate(&mut self, tx: f32, ty: f32, tz: f32) {
        self.backend.matrix_translate(&mut self.elements, tx, ty, tz);
    }

    pub fn scale(&mut self, sx: f32, sy: f32, sz: f32) {
        self.backend.matrix_scale(&mut self.elements, sx, sy, sz);
    }

    pub fn rotate_x(&mut self, angle: f32) {
        self.backend.matrix_rotate_x(&mut self.elements, angle);
    }

    pub fn rotate_y(&mut self, angle: f32) {
        self.backend.matrix_rotate_y(&mut self.elements, angle);
    }

    pub fn rotate_z(&mut self, angle: f32) {
        self.backend.matrix_rotate_z(&mut self.elements, angle);
    }

    pub fn transpose(&self) -> Self {

        let result = self.backend.matrix_transpose(&self.elements);
        Self {
            elements: result,
            backend: self.backend,
        }
    }

    pub fn inverse(&self) -> Self {
        let result = self.backend.matrix_inverse(&self.elements);
        Self {
            elements: result,
            backend: self.backend,
        }
    }

    pub fn determinant(&self) -> f32 {
        self.backend.matrix_determinant(&self.elements)
    }

    pub fn perspective(self, fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        let result = self.backend.matrix_perspective(fov, aspect, near, far);
        Self {
            elements: result,
            backend: self.backend,
        }
    }

    pub fn look_at(self, eye: &[f32; 3], center: &[f32; 3], up: &[f32; 3]) -> Self {
        let result = self.backend.matrix_look_at(eye, center, up);
        Self {
            elements: result,
            backend: self.backend,
        }
    }
    
    pub fn orthographic(self, left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let result = self.backend.matrix_orthographic(left, right, bottom, top, near, far);
        Self {
            elements: result,
            backend: self.backend,
        }
    }
    
    pub fn frustum(self, left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let result = self.backend.matrix_frustum(left, right, bottom, top, near, far);
        Self {
            elements: result,
            backend: self.backend,
        }
    }
    
    pub fn to_array(&self) -> [[f32; 4]; 4] {
        self.backend.matrix_to_array(&self.elements)
    }
    
    pub fn from_array(self, arr: &[[f32; 4]; 4]) -> Self {
        let result = self.backend.matrix_from_array(arr);
        Self {
            elements: result,
            backend: self.backend,
        }
    }
    
    pub fn to_string(&self) -> String {
        self.backend.matrix_to_string(&self.elements)
    }
    
    pub fn from_string(self, s: &str) -> Self {
        let result = self.backend.matrix_from_string(s);
        Self {
            elements: result,
            backend: self.backend,
        }
    }

    pub fn to_tuple(&self) -> [[f32; 4]; 4] {
        self.backend.matrix_to_tuple(&self.elements)
    }

    pub fn from_tuple(self, t: [[f32; 4]; 4]) -> Self {
        let result = self.backend.matrix_from_tuple(t);
        Self {
            elements: result,
            backend: self.backend,
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.backend.matrix_to_vec(&self.elements)
    }

    pub fn from_vec(self, v: &Vec<f32>) -> Self {
        let result = self.backend.matrix_from_vec(v);
        Self {
            elements: result,
            backend: self.backend,
        }
    }
}

// Implementamos tanto Clone como Copy para Quaternion
#[derive(Copy)]
pub struct Quaternion {
    w: f32,
    x: f32,
    y: f32,
    z: f32,
    backend: &'static dyn MathBackend,
}

// Implementación simplificada de Clone para Quaternion
impl Clone for Quaternion {
    fn clone(&self) -> Self {
        *self // Copy permite usar simplemente *self para clonar
    }
}

impl std::fmt::Debug for Quaternion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Quaternion({}, {}, {}, {})", self.w, self.x, self.y, self.z)
    }
}

impl Quaternion {
    pub fn new_with_values(w: f32, x: f32, y: f32, z: f32) -> Self {
        // Usar OnceLock para inicializar el backend una sola vez
        // y compartirlo entre todas las instancias de Matrix
        static BACKEND: OnceLock<Box<dyn MathBackend>> = OnceLock::new();
        let backend = BACKEND.get_or_init(|| get_optimal_math_backend());

        Self {
            w,
            x,
            y,
            z,
            backend: &**backend,
        }
    }
    
    pub fn new() -> Self {
        // Valor por defecto: cuaternión identidad (sin rotación)
        Self::new_with_values(1.0, 0.0, 0.0, 0.0)
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let result = self.backend.quaternion_multiply(
            &[self.w, self.x, self.y, self.z],
            &[other.w, other.x, other.y, other.z],
        );
        Self {
            w: result[0],
            x: result[1],
            y: result[2],
            z: result[3],
            backend: self.backend,
        }
    }
    
    pub fn conjugate(&self) -> Self {
        let result = self.backend.quaternion_conjugate(&[self.w, self.x, self.y, self.z]);
        Self {
            w: result[0],
            x: result[1],
            y: result[2],
            z: result[3],
            backend: self.backend,
        }
    }
    
    pub fn normalize(&self) -> Self {
        let result = self.backend.quaternion_normalize(&[self.w, self.x, self.y, self.z]);
        Self {
            w: result[0],
            x: result[1],
            y: result[2],
            z: result[3],
            backend: self.backend,
        }
    }

    pub fn to_matrix(&self) -> [[f32; 4]; 4] {
        let result = self.backend.quaternion_to_matrix(&[self.w, self.x, self.y, self.z]);
        result
    }

    pub fn from_axis_angle(axis: &[f32; 3], angle: f32) -> Self {
        static BACKEND: OnceLock<Box<dyn MathBackend>> = OnceLock::new();
        let backend = BACKEND.get_or_init(|| get_optimal_math_backend());
        let result = backend.quaternion_from_axis_angle(axis, angle);
        Self {
            w: result[0],
            x: result[1],
            y: result[2],
            z: result[3],
            backend: &**backend,
        }
    }

    pub fn to_axis_angle(&self) -> ([f32; 3], f32) {
        let (axis, angle) = self.backend.quaternion_to_axis_angle(&[self.w, self.x, self.y, self.z]);
        (axis, angle)
    }

    pub fn from_euler(roll: f32, pitch: f32, yaw: f32) -> Self {
        static BACKEND: OnceLock<Box<dyn MathBackend>> = OnceLock::new();
        let backend = BACKEND.get_or_init(|| get_optimal_math_backend());
        let result = backend.quaternion_from_euler(roll, pitch, yaw);
        Self {
            w: result[0],
            x: result[1],
            y: result[2],
            z: result[3],
            backend: &**backend,
        }
    }

    pub fn to_euler(&self) -> (f32, f32, f32) {
        let (roll, pitch, yaw) = self.backend.quaternion_to_euler(&[self.w, self.x, self.y, self.z]);
        (roll, pitch, yaw)
    }

    pub fn from_rotation_matrix(m: &[[f32; 4]; 4]) -> Self {
        static BACKEND: OnceLock<Box<dyn MathBackend>> = OnceLock::new();
        let backend = BACKEND.get_or_init(|| get_optimal_math_backend());
        let result = backend.quaternion_from_rotation_matrix(m);
        Self {
            w: result[0],
            x: result[1],
            y: result[2],
            z: result[3],
            backend: &**backend,
        }
    }

    pub fn to_rotation_matrix(&self) -> [[f32; 4]; 4] {
        let result = self.backend.quaternion_to_rotation_matrix(&[self.w, self.x, self.y, self.z]);
        result
    }

    pub fn slerp(&self, other: &Self, t: f32) -> Self {
        let result = self.backend.quaternion_slerp(
            &[self.w, self.x, self.y, self.z],
            &[other.w, other.x, other.y, other.z],
            t,
        );
        Self {
            w: result[0],
            x: result[1],
            y: result[2],
            z: result[3],
            backend: self.backend,
        }
    }

    pub fn squad(&self, q2: &Self, q3: &Self, q4: &Self, t: f32) -> Self {
        let result = self.backend.quaternion_squad(
            &[self.w, self.x, self.y, self.z],
            &[q2.w, q2.x, q2.y, q2.z],
            &[q3.w, q3.x, q3.y, q3.z],
            &[q4.w, q4.x, q4.y, q4.z],
            t,
        );
        Self {
            w: result[0],
            x: result[1],
            y: result[2],
            z: result[3],
            backend: self.backend,
        }
    }

    pub fn squad_slerp(&self, q2: &Self, q3: &Self, q4: &Self, t: f32) -> Self {
        let result = self.backend.quaternion_squad_slerp(
            &[self.w, self.x, self.y, self.z],
            &[q2.w, q2.x, q2.y, q2.z],
            &[q3.w, q3.x, q3.y, q3.z],
            &[q4.w, q4.x, q4.y, q4.z],
            t,
        );
        Self {
            w: result[0],
            x: result[1],
            y: result[2],
            z: result[3],
            backend: self.backend,
        }
    }

    pub fn to_array(&self) -> [f32; 4] {
        [self.w, self.x, self.y, self.z]
    }

    pub fn from_array(self, arr: &[f32; 4]) -> Self {
        Self {
            w: arr[0],
            x: arr[1],
            y: arr[2],
            z: arr[3],
            backend: self.backend,
        }
    }

    pub fn to_string(&self) -> String {
        format!("Quaternion({}, {}, {}, {})", self.w, self.x, self.y, self.z)
    }

    pub fn from_string(self, s: &str) -> Self {
        let parts: Vec<&str> = s.trim_matches(|c| c == '(' || c == ')').split(',').collect();
        if parts.len() != 4 {
            panic!("Invalid quaternion string format");
        }
        Self {
            w: parts[0].trim().parse().unwrap(),
            x: parts[1].trim().parse().unwrap(),
            y: parts[2].trim().parse().unwrap(),
            z: parts[3].trim().parse().unwrap(),
            backend: self.backend,
        }
    }

    pub fn to_tuple(&self) -> (f32, f32, f32, f32) {
        (self.w, self.x, self.y, self.z)
    }

    pub fn from_tuple(self, t: (f32, f32, f32, f32)) -> Self {
        Self {
            w: t.0,
            x: t.1,
            y: t.2,
            z: t.3,
            backend: self.backend,
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.w, self.x, self.y, self.z]
    }

    pub fn from_vec(self, v: &Vec<f32>) -> Self {
        if v.len() != 4 {
            panic!("Invalid quaternion length");
        }
        Self {
            w: v[0],
            x: v[1],
            y: v[2],
            z: v[3],
            backend: self.backend,
        }
    }

}

#[derive(Copy)]
pub struct Maths {
    backend: &'static dyn MathBackend,
}

// Implementación manual de Clone para Maths
impl Clone for Maths {
    fn clone(&self) -> Self {
        *self // Copy permite usar simplemente *self para clonar
    }
}

impl Maths {
    pub fn new() -> Self {
        // Usar OnceLock para inicializar el backend una sola vez
        // y compartirlo entre todas las instancias de Vector
        static BACKEND: OnceLock<Box<dyn MathBackend>> = OnceLock::new();
        let backend = BACKEND.get_or_init(|| get_optimal_math_backend());

        Self { backend: &**backend }
    }
    pub fn compute_barycentric_coordinates(&self, point: [f32; 2], v0: [f32; 2], v1: [f32; 2], v2: [f32; 2]) -> (f32, f32, f32) {
        self.backend.compute_barycentric_coordinates(point, v0, v1, v2)
    }
}