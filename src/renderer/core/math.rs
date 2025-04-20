// --- SIMD ---
use core_simd::simd::f32x4;
use core_simd::simd::f32x8;
use core_simd::simd::cmp::SimdPartialOrd;
use core_simd::simd::num::SimdFloat;

#[derive(Copy, Clone, Debug)]
pub struct Vector {
    simd: f32x4,
}

impl Vector {
    pub fn new_with_values(x: f32, y: f32, z: f32) -> Self {
        Self { simd: f32x4::from_array([x, y, z, 0.0]) }
    }
    pub fn new() -> Self {
        Self::new_with_values(0.0, 0.0, 0.0)
    }
    pub fn x(&self) -> f32 { self.simd[0] }
    pub fn y(&self) -> f32 { self.simd[1] }
    pub fn z(&self) -> f32 { self.simd[2] }
    pub fn add(&self, other: &Self) -> Self {
        Self { simd: self.simd + other.simd }
    }
    pub fn subtract(&self, other: &Self) -> Self {
        Self { simd: self.simd - other.simd }
    }
    pub fn dot(&self, other: &Self) -> f32 {
        (self.simd * other.simd).reduce_sum()
    }
    pub fn cross(&self, other: &Self) -> Self {
        let a = self.simd;
        let b = other.simd;
        let cross = f32x4::from_array([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
            0.0,
        ]);
        Self { simd: cross }
    }
    pub fn add_scalar(&self, scalar: f32) -> Self {
        Self { simd: self.simd + f32x4::splat(scalar) }
    }
    pub fn subtract_scalar(&self, scalar: f32) -> Self {
        Self { simd: self.simd - f32x4::splat(scalar) }
    }
    pub fn multiply_scalar(&self, scalar: f32) -> Self {
        Self { simd: self.simd * f32x4::splat(scalar) }
    }
    pub fn divide_scalar(&self, scalar: f32) -> Self {
        if scalar == 0.0 { panic!("Division by zero"); }
        Self { simd: self.simd / f32x4::splat(scalar) }
    }
    pub fn scale(&self, scalar: f32) -> Self {
        Self { simd: self.simd * f32x4::splat(scalar) }
    }
    pub fn length(&self) -> f32 {
        (self.simd * self.simd).reduce_sum().sqrt()
    }
    pub fn length_squared(&self) -> f32 {
        (self.simd * self.simd).reduce_sum()
    }
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len == 0.0 { return Self::new(); }
        Self { simd: self.simd / f32x4::splat(len) }
    }
    pub fn distance(&self) -> f32 {
        ((self.simd) * (self.simd)).reduce_sum().sqrt()
    }
    pub fn distance_squared(&self, other: &Self) -> f32 {
        ((self.simd - other.simd) * (self.simd - other.simd)).reduce_sum()
    }
    pub fn angle(&self, other: &Self) -> f32 {
        let dot = self.dot(other);
        let len1 = self.length();
        let len2 = other.length();
        (dot / (len1 * len2)).acos()
    }
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let vt = f32x4::splat(t);
        Self { simd: self.simd + (other.simd - self.simd) * vt }
    }
    pub fn slerp(&self, other: &Self, t: f32) -> Self {
        let dot = self.dot(other);
        let theta = dot.acos();
        let sin_theta = theta.sin();
        if sin_theta == 0.0 {
            return self.lerp(other, t);
        }
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;
        Self { simd: self.simd * f32x4::splat(a) + other.simd * f32x4::splat(b) }
    }
    pub fn squad(&self, v2: &Self, v3: &Self, v4: &Self, t: f32) -> Self {
        let a = self.slerp(v2, t);
        let b = v3.slerp(v4, t);
        a.lerp(&b, t)
    }
    pub fn squad_slerp(&self, v2: &Self, v3: &Self, v4: &Self, t: f32) -> Self {
        let a = self.slerp(v2, t);
        let b = v3.slerp(v4, t);
        a.slerp(&b, t)
    }
    pub fn project(&self, onto: &Self) -> Self {
        let dot = self.dot(onto);
        let onto_len2 = onto.length_squared();
        if onto_len2 == 0.0 { return Self::new(); }
        let scale = dot / onto_len2;
        Self { simd: onto.simd * f32x4::splat(scale) }
    }
    pub fn reflect(&self, normal: &Self) -> Self {
        let dot = self.dot(normal);
        Self { simd: self.simd - normal.simd * f32x4::splat(2.0 * dot) }
    }
    pub fn refract(&self, normal: &Self, eta: f32) -> Self {
        let dot = self.dot(normal);
        let k = 1.0 - eta * eta * (1.0 - dot * dot);
        if k < 0.0 {
            Self::new()
        } else {
            Self { simd: self.simd * f32x4::splat(eta) - normal.simd * f32x4::splat(eta * dot + k.sqrt()) }
        }
    }
    pub fn to_array(&self) -> [f32; 3] {
        let arr = self.simd.to_array();
        [arr[0], arr[1], arr[2]]
    }
    pub fn from_array(arr: [f32; 3]) -> Self {
        Self { simd: f32x4::from_array([arr[0], arr[1], arr[2], 0.0]) }
    }
    pub fn to_tuple(&self) -> (f32, f32, f32) {
        let arr = self.simd.to_array();
        (arr[0], arr[1], arr[2])
    }
    pub fn from_tuple(t: (f32, f32, f32)) -> Self {
        Self { simd: f32x4::from_array([t.0, t.1, t.2, 0.0]) }
    }
    pub fn to_vec(&self) -> Vec<f32> {
        let arr = self.simd.to_array();
        vec![arr[0], arr[1], arr[2]]
    }
    pub fn from_vec(v: &Vec<f32>) -> Self {
        if v.len() != 3 { panic!("Invalid vector length"); }
        Self { simd: f32x4::from_array([v[0], v[1], v[2], 0.0]) }
    }
    pub fn to_string(&self) -> String {
        let arr = self.simd.to_array();
        format!("({}, {}, {})", arr[0], arr[1], arr[2])
    }
    pub fn from_string(s: &str) -> Self {
        let parts: Vec<&str> = s.trim_matches(|c| c == '(' || c == ')').split(',').collect();
        if parts.len() != 3 { panic!("Invalid vector string format"); }
        Self {
            simd: f32x4::from_array([
                parts[0].trim().parse().unwrap(),
                parts[1].trim().parse().unwrap(),
                parts[2].trim().parse().unwrap(),
                0.0,
            ])
        }
    }
}

// Matrix SIMD
#[derive(Copy, Clone, Debug)]
pub struct Matrix {
    elements: [f32x4; 4], // 4 filas de 4 elementos
}

impl Matrix {
    pub fn new_with_values(elements: [[f32; 4]; 4]) -> Self {
        Self {
            elements: [
                f32x4::from_array(elements[0]),
                f32x4::from_array(elements[1]),
                f32x4::from_array(elements[2]),
                f32x4::from_array(elements[3]),
            ],
        }
    }
    pub fn new() -> Self {
        Self::new_with_values([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }
    pub fn elements(&self) -> [[f32; 4]; 4] {
        [
            self.elements[0].to_array(),
            self.elements[1].to_array(),
            self.elements[2].to_array(),
            self.elements[3].to_array(),
        ]
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let mut result = [f32x4::splat(0.0); 4];
        
        // Transposición de la segunda matriz para mejorar el acceso a memoria
        let other_transposed = [
            f32x4::from_array([other.elements[0][0], other.elements[1][0], other.elements[2][0], other.elements[3][0]]),
            f32x4::from_array([other.elements[0][1], other.elements[1][1], other.elements[2][1], other.elements[3][1]]),
            f32x4::from_array([other.elements[0][2], other.elements[1][2], other.elements[2][2], other.elements[3][2]]),
            f32x4::from_array([other.elements[0][3], other.elements[1][3], other.elements[2][3], other.elements[3][3]]),
        ];
        
        // Ahora podemos hacer multiplicación de vectores más directamente
        for i in 0..4 {
            result[i] = f32x4::from_array([
                (self.elements[i] * other_transposed[0]).reduce_sum(),
                (self.elements[i] * other_transposed[1]).reduce_sum(),
                (self.elements[i] * other_transposed[2]).reduce_sum(),
                (self.elements[i] * other_transposed[3]).reduce_sum(),
            ]);
        }
        Self { elements: result }
    }
    pub fn transpose(&self) -> Self {
        let m = self.elements();
        let mut t = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                t[j][i] = m[i][j];
            }
        }
        Self::new_with_values(t)
    }
    pub fn to_array(&self) -> [[f32; 4]; 4] {
        self.elements()
    }
    pub fn from_array(arr: [[f32; 4]; 4]) -> Self {
        Self::new_with_values(arr)
    }
    pub fn to_vec(&self) -> Vec<f32> {
        let m = self.elements();
        m.iter().flat_map(|row| row.iter()).cloned().collect()
    }
    pub fn from_vec(v: &Vec<f32>) -> Self {
        if v.len() != 16 { panic!("Invalid matrix vector format"); }
        let mut arr = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                arr[i][j] = v[i*4 + j];
            }
        }
        Self::new_with_values(arr)
    }
    pub fn to_string(&self) -> String {
        let m = self.elements();
        format!("[[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]]",
            m[0][0], m[0][1], m[0][2], m[0][3],
            m[1][0], m[1][1], m[1][2], m[1][3],
            m[2][0], m[2][1], m[2][2], m[2][3],
            m[3][0], m[3][1], m[3][2], m[3][3])
    }
    pub fn from_string(s: &str) -> Self {
        let parts: Vec<&str> = s.trim_matches(|c| c == '[' || c == ']').split("],").collect();
        if parts.len() != 4 { panic!("Invalid matrix string format"); }
        let mut arr = [[0.0; 4]; 4];
        for i in 0..4 {
            let row_parts: Vec<&str> = parts[i].trim_matches(|c| c == '[' || c == ']').split(',').collect();
            if row_parts.len() != 4 { panic!("Invalid matrix row format"); }
            for j in 0..4 {
                arr[i][j] = row_parts[j].trim().parse().unwrap();
            }
        }
        Self::new_with_values(arr)
    }
    // Métodos matemáticos básicos para Matrix
    pub fn translate(&self, tx: f32, ty: f32, tz: f32) -> Self {
        let mut m = self.elements();
        m[0][3] += tx;
        m[1][3] += ty;
        m[2][3] += tz;
        m[3][3] = 1.0;
        Self::new_with_values(m)
    }
    pub fn scale(&self, sx: f32, sy: f32, sz: f32) -> Self {
        let mut m = self.elements();
        m[0][0] *= sx;
        m[1][1] *= sy;
        m[2][2] *= sz;
        m[3][3] = 1.0;
        Self::new_with_values(m)
    }
    pub fn rotate_x(&self, angle: f32) -> Self {
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        let rotation = Self::new_with_values([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_angle, -sin_angle, 0.0],
            [0.0, sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        self.multiply(&rotation)
    }
    pub fn rotate_y(&self, angle: f32) -> Self {
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        let rotation = Self::new_with_values([
            [cos_angle, 0.0, sin_angle, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_angle, 0.0, cos_angle, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        self.multiply(&rotation)
    }
    pub fn rotate_z(&self, angle: f32) -> Self {
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        let rotation = Self::new_with_values([
            [cos_angle, -sin_angle, 0.0, 0.0],
            [sin_angle, cos_angle, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        self.multiply(&rotation)
    }
    pub fn determinant(&self) -> f32 {
        let m = self.elements();
        // Expansión de Laplace para 4x4 (no optimizado, pero correcto)
        let a = m[0][0]; let b = m[0][1]; let c = m[0][2]; let d = m[0][3];
        let e = m[1][0]; let f = m[1][1]; let g = m[1][2]; let h = m[1][3];
        let i = m[2][0]; let j = m[2][1]; let k = m[2][2]; let l = m[2][3];
        let m0 = m[3][0]; let n = m[3][1]; let o = m[3][2]; let p = m[3][3];
        a*f*k*p - a*f*l*o - a*g*j*p + a*g*l*n + a*h*j*o - a*h*k*n
        - b*e*k*p + b*e*l*o + b*g*i*p - b*g*l*m0 - b*h*i*o + b*h*k*m0
        + c*e*j*p - c*e*l*n - c*f*i*p + c*f*l*m0 + c*h*i*n - c*h*j*m0
        - d*e*j*o + d*e*k*n + d*f*i*o - d*f*k*m0 - d*g*i*n + d*g*j*m0
    }
    pub fn inverse(&self) -> Self {
        let m = self.elements();
        let mut inv = [[0.0; 4]; 4];
        // Algoritmo de inversión de matriz 4x4 (Gauss-Jordan, no optimizado)
        let mut aug = [[0.0; 8]; 4];
        for i in 0..4 {
            for j in 0..4 {
                aug[i][j] = m[i][j];
                aug[i][j+4] = if i == j { 1.0 } else { 0.0 };
            }
        }
        for i in 0..4 {
            let mut max_row = i;
            for k in (i+1)..4 {
                if aug[k][i].abs() > aug[max_row][i].abs() {
                    max_row = k;
                }
            }
            aug.swap(i, max_row);
            let pivot = aug[i][i];
            if pivot == 0.0 { panic!("Matrix is not invertible"); }
            for j in 0..8 {
                aug[i][j] /= pivot;
            }
            for k in 0..4 {
                if k != i {
                    let factor = aug[k][i];
                    for j in 0..8 {
                        aug[k][j] -= factor * aug[i][j];
                    }
                }
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                inv[i][j] = aug[i][j+4];
            }
        }
        Self::new_with_values(inv)
    }
    // Métodos de proyección y vista
    pub fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        let tan_half_fov = (fov / 2.0).tan();
        let f = 1.0 / tan_half_fov;
        Self::new_with_values([
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ])
    }
    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        Self::new_with_values([
            [2.0 / (right - left), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (top - bottom), 0.0, 0.0],
            [0.0, 0.0, -2.0 / (far - near), 0.0],
            [-(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1.0],
        ])
    }
    pub fn frustum(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let x = 2.0 * near / (right - left);
        let y = 2.0 * near / (top - bottom);
        let a = (right + left) / (right - left);
        let b = (top + bottom) / (top - bottom);
        let c = -(far + near) / (far - near);
        let d = -(2.0 * far * near) / (far - near);
        Self::new_with_values([
            [x, 0.0, a, 0.0],
            [0.0, y, b, 0.0],
            [0.0, 0.0, c, d],
            [0.0, 0.0, -1.0, 0.0],
        ])
    }
    pub fn look_at(eye: &[f32; 3], center: &[f32; 3], up: &[f32; 3]) -> Self {
        // Crear vectores SIMD directamente
        let eye_vec = Vector::new_with_values(eye[0], eye[1], eye[2]);
        let f = Vector::new_with_values(
            center[0] - eye[0],
            center[1] - eye[1],
            center[2] - eye[2],
        ).normalize();
        let s = f.cross(&Vector::new_with_values(up[0], up[1], up[2])).normalize();
        let u = s.cross(&f);
        
        // Calcular dot products de una vez
        let s_dot_eye = s.dot(&eye_vec);
        let u_dot_eye = u.dot(&eye_vec);
        let f_dot_eye = f.dot(&eye_vec);
        
        // Construir matriz directamente
        Self::new_with_values([
            [s.x(), s.y(), s.z(), -s_dot_eye],
            [u.x(), u.y(), u.z(), -u_dot_eye],
            [-f.x(), -f.y(), -f.z(), f_dot_eye],
            [0.0, 0.0, 0.0, 1.0]
        ])
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Quaternion {
    simd: f32x4, // [w, x, y, z]
}

impl Quaternion {
    pub fn new_with_values(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { simd: f32x4::from_array([w, x, y, z]) }
    }
    pub fn new() -> Self {
        Self::new_with_values(1.0, 0.0, 0.0, 0.0)
    }
    pub fn multiply(&self, other: &Self) -> Self {
        // Multiplicación de cuaterniones optimizada con SIMD
        // (w1, x1, y1, z1) * (w2, x2, y2, z2)
        let a = self.simd;
        let b = other.simd;
        // Descomponer componentes
        let w1 = a[0]; let x1 = a[1]; let y1 = a[2]; let z1 = a[3];
        let w2 = b[0]; let x2 = b[1]; let y2 = b[2]; let z2 = b[3];
        // Usar SIMD para calcular los productos cruzados y sumas
        let w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
        let x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
        let y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
        let z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
        Self { simd: f32x4::from_array([w, x, y, z]) }
    }
    pub fn conjugate(&self) -> Self {
        let arr = self.simd.to_array();
        Self { simd: f32x4::from_array([arr[0], -arr[1], -arr[2], -arr[3]]) }
    }
    pub fn normalize(&self) -> Self {
        let mag2 = (self.simd * self.simd).reduce_sum();
        let inv_mag = 1.0 / mag2.sqrt();
        Self { simd: self.simd * f32x4::splat(inv_mag) }
    }
    pub fn to_array(&self) -> [f32; 4] {
        self.simd.to_array()
    }
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let vt = f32x4::splat(t);
        Self { simd: self.simd + (other.simd - self.simd) * vt }
    }
    pub fn slerp(&self, other: &Self, t: f32) -> Self {
        let dot = (self.simd * other.simd).reduce_sum();
        let mut other_simd = other.simd;
        let mut dot_val = dot;
        if dot < 0.0 {
            other_simd = -other_simd;
            dot_val = -dot;
        }
        if dot_val > 0.9995 {
            return self.lerp(&Self { simd: other_simd }, t).normalize();
        }
        let theta_0 = dot_val.acos();
        let theta = theta_0 * t;
        let sin_theta_0 = theta_0.sin();
        let sin_theta = theta.sin();
        let s0 = ((theta_0 - theta).sin()) / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;
        Self { simd: self.simd * f32x4::splat(s0) + other_simd * f32x4::splat(s1) }
    }
    pub fn from_array(arr: [f32; 3]) -> Self {
        Self { simd: f32x4::from_array([arr[0], arr[1], arr[2], 0.0]) }
    }
    pub fn from_tuple(t: (f32, f32, f32)) -> Self {
        Self { simd: f32x4::from_array([t.0, t.1, t.2, 0.0]) }
    }
    pub fn to_tuple(&self) -> (f32, f32, f32) {
        let arr = self.simd.to_array();
        (arr[0], arr[1], arr[2])
    }
    pub fn to_vec(&self) -> Vec<f32> {
        let arr = self.simd.to_array();
        vec![arr[0], arr[1], arr[2]]
    }
    pub fn from_vec(v: &Vec<f32>) -> Self {
        if v.len() != 3 { panic!("Invalid vector length"); }
        Self { simd: f32x4::from_array([v[0], v[1], v[2], 0.0]) }
    }
    pub fn to_string(&self) -> String {
        let arr = self.simd.to_array();
        format!("({}, {}, {})", arr[0], arr[1], arr[2])
    }
    pub fn from_string(s: &str) -> Self {
        let parts: Vec<&str> = s.trim_matches(|c| c == '(' || c == ')').split(',').collect();
        if parts.len() != 3 { panic!("Invalid vector string format"); }
        Self {
            simd: f32x4::from_array([
                parts[0].trim().parse().unwrap(),
                parts[1].trim().parse().unwrap(),
                parts[2].trim().parse().unwrap(),
                0.0,
            ])
        }
    }
    pub fn from_rotation_matrix(m: &[[f32; 4]; 4]) -> Self {
        let trace = m[0][0] + m[1][1] + m[2][2];
        let (w, x, y, z);
        if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            w = 0.25 / s;
            x = (m[2][1] - m[1][2]) * s;
            y = (m[0][2] - m[2][0]) * s;
            z = (m[1][0] - m[0][1]) * s;
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
            w = (m[2][1] - m[1][2]) / s;
            x = 0.25 * s;
            y = (m[0][1] + m[1][0]) / s;
            z = (m[0][2] + m[2][0]) / s;
        } else if m[1][1] > m[2][2] {
            let s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt();
            w = (m[0][2] - m[2][0]) / s;
            x = (m[0][1] + m[1][0]) / s;
            y = 0.25 * s;
            z = (m[1][2] + m[2][1]) / s;
        } else {
            let s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt();
            w = (m[1][0] - m[0][1]) / s;
            x = (m[0][2] + m[2][0]) / s;
            y = (m[1][2] + m[2][1]) / s;
            z = 0.25 * s;
        }
        Self { simd: f32x4::from_array([w, x, y, z]) }
    }
    pub fn from_axis_angle(axis: [f32; 3], angle: f32) -> Self {
        let half_angle = angle * 0.5;
        let (sin_half, cos_half) = half_angle.sin_cos();
        let norm = (axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]).sqrt();
        let (x, y, z) = if norm > 0.0 {
            (axis[0]/norm, axis[1]/norm, axis[2]/norm)
        } else {
            (0.0, 0.0, 0.0)
        };
        Self { simd: f32x4::from_array([cos_half, x * sin_half, y * sin_half, z * sin_half]) }
    }
    pub fn from_euler(roll: f32, pitch: f32, yaw: f32) -> Self {
        // Orden: yaw (Z), pitch (Y), roll (X) - Tait-Bryan angles
        let (sr, cr) = (roll * 0.5).sin_cos();
        let (sp, cp) = (pitch * 0.5).sin_cos();
        let (sy, cy) = (yaw * 0.5).sin_cos();
        let w = cr * cp * cy + sr * sp * sy;
        let x = sr * cp * cy - cr * sp * sy;
        let y = cr * sp * cy + sr * cp * sy;
        let z = cr * cp * sy - sr * sp * cy;
        Self { simd: f32x4::from_array([w, x, y, z]) }
    }
    pub fn to_euler(&self) -> (f32, f32, f32) {
        // Devuelve (roll, pitch, yaw) en radianes, evitando bloqueo de gimbal
        let arr = self.simd.to_array();
        let (w, x, y, z) = (arr[0], arr[1], arr[2], arr[3]);
        // Pitch (Y)
        let sinp = 2.0 * (w * y - z * x);
        let pitch = if sinp.abs() >= 1.0 {
            sinp.signum() * (std::f32::consts::PI / 2.0)
        } else {
            sinp.asin()
        };
        // Roll (X)
        let sinr_cosp = 2.0 * (w * x + y * z);
        let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        let roll = sinr_cosp.atan2(cosr_cosp);
        // Yaw (Z)
        let siny_cosp = 2.0 * (w * z + x * y);
        let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        let yaw = siny_cosp.atan2(cosy_cosp);
        (roll, pitch, yaw)
    }
    pub fn to_rotation_matrix(&self) -> Matrix {
        let arr = self.simd.to_array();
        let (w, x, y, z) = (arr[0], arr[1], arr[2], arr[3]);
        Matrix::new_with_values([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),     2.0 * (x * z + w * y),     0.0],
            [2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),     0.0],
            [2.0 * (x * z - w * y),       2.0 * (y * z + w * x),     1.0 - 2.0 * (x * x + y * y), 0.0],
            [0.0,                        0.0,                        0.0,                        1.0],
        ])
    }
}

/// Calcula coordenadas baricéntricas para 8 puntos en paralelo (SIMD)
pub fn compute_barycentric_coordinates_simd(
    pxs: f32x8,
    pys: f32x8,
    v0: [f32; 2],
    v1: [f32; 2],
    v2: [f32; 2],
) -> (f32x8, f32x8, f32x8) {
    let v0x = f32x8::splat(v0[0]);
    let v0y = f32x8::splat(v0[1]);
    let v1x = f32x8::splat(v1[0]);
    let v1y = f32x8::splat(v1[1]);
    let v2x = f32x8::splat(v2[0]);
    let v2y = f32x8::splat(v2[1]);
    // Área del triángulo
    let mut area = (v1x - v0x) * (v2y - v0y) - (v2x - v0x) * (v1y - v0y);
    let mask = area.abs().simd_lt(f32x8::splat(1e-6)); // Use simd_lt for explicit mask generation
    area = mask.select(f32x8::splat(1e-6), area); // Use mask.select
    // Alpha
    let alpha = ((v1x - pxs) * (v2y - pys) - (v2x - pxs) * (v1y - pys)) / area;
    // Beta
    let beta = ((v2x - pxs) * (v0y - pys) - (v0x - pxs) * (v2y - pys)) / area;
    // Gamma
    let gamma = f32x8::splat(1.0) - alpha - beta;
    (alpha, beta, gamma)
}

pub fn compute_barycentric_coordinates(point: [f32; 2], v0: [f32; 2], v1: [f32; 2], v2: [f32; 2]) -> (f32, f32, f32) {
    // Usar SIMD para calcular vectores y productos cruzados 2D
    let p = f32x4::from_array([point[0], point[1], 0.0, 0.0]);
    let a = f32x4::from_array([v0[0], v0[1], 0.0, 0.0]);
    let b = f32x4::from_array([v1[0], v1[1], 0.0, 0.0]);
    let c = f32x4::from_array([v2[0], v2[1], 0.0, 0.0]);
    
    // Área del triángulo (v1-v0) x (v2-v0)
    let v0_to_v1 = b - a;
    let v0_to_v2 = c - a;
    
    // Área usando producto cruz 2D
    let area = v0_to_v1[0] * v0_to_v2[1] - v0_to_v1[1] * v0_to_v2[0];
    
    if area.abs() < 1e-6 {
        return (0.0, 0.0, 0.0);
    }
    
    // Vectores desde vértices al punto
    let v1_to_v2 = c - b;
    let p_to_a = a - p;
    let p_to_b = b - p;
    let p_to_c = c - p;
    
    // Cálculo de coordenadas baricéntricas mediante áreas
    let alpha = (p_to_b[0] * p_to_c[1] - p_to_c[0] * p_to_b[1]) / area;
    let beta = (p_to_c[0] * p_to_a[1] - p_to_a[0] * p_to_c[1]) / area;
    let gamma = 1.0 - alpha - beta;
    
    (alpha, beta, gamma)
}