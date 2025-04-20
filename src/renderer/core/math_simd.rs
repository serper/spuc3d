use core_simd::simd::prelude::*;
use super::math::{MathBackend, SimdMathBackend};

#[cfg(target_arch = "arm")]
impl MathBackend for SimdMathBackend {

    // Vector operations
    fn vector_add(&self, v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3] {
        let va = f32x4::from_array([v1[0], v1[1], v1[2], 0.0]);
        let vb = f32x4::from_array([v2[0], v2[1], v2[2], 0.0]);
        
        let result = va + vb;
        let arr = result.to_array();
        
        [arr[0], arr[1], arr[2]]
    }

    fn vector_subtract(&self, v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3] {
        let va = f32x4::from_array([v1[0], v1[1], v1[2], 0.0]);
        let vb = f32x4::from_array([v2[0], v2[1], v2[2], 0.0]);
        
        let result = va - vb;
        let arr = result.to_array();
        
        [arr[0], arr[1], arr[2]]
    }

    fn vector_dot(&self, v1: &[f32; 3], v2: &[f32; 3]) -> f32 {
        let va = f32x4::from_array([v1[0], v1[1], v1[2], 0.0]);
        let vb = f32x4::from_array([v2[0], v2[1], v2[2], 0.0]);
        (va * vb).reduce_sum()
    }

    fn vector_cross(&self, v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3] {
        let a = f32x4::from_array([v1[0], v1[1], v1[2], 0.0]);
        let b = f32x4::from_array([v2[0], v2[1], v2[2], 0.0]);
        
        let cross = f32x4::from_array([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
            0.0,
        ]);
        
        cross.to_array()[..3].try_into().unwrap()
    }

    fn vector_scale(&self,v: &[f32; 3], scalar: f32) -> [f32; 3] {
        let va = f32x4::from_array([v[0], v[1], v[2], 0.0]);
        let vs = f32x4::splat(scalar);
        
        let result = va * vs;
        let arr = result.to_array();
        
        [arr[0], arr[1], arr[2]]
    }
    
    fn vector_length(&self,v: &[f32; 3]) -> f32 {
        let va = f32x4::from_array([v[0], v[1], v[2], 0.0]);
        let vmul = va * va;
        let vsum = vmul.reduce_sum();
        vsum.sqrt()
    }

    fn vector_normalize(&self, v: &[f32; 3]) -> [f32; 3] {
        let va = f32x4::from_array([v[0], v[1], v[2], 0.0]);
        let magnitude_squared = (va * va).reduce_sum();
        let inv_magnitude = 1.0 / magnitude_squared.sqrt();
        let normalized = va * f32x4::splat(inv_magnitude);
        let result = normalized.to_array();
        [result[0], result[1], result[2]]
    }

    fn vector_distance(&self, v1: &[f32; 3], v2: &[f32; 3]) -> f32 {
        let diff = self.vector_subtract(v1, v2);
        self.vector_length(&diff)
    }

    fn vector_distance_squared(&self, v1: &[f32; 3], v2: &[f32; 3]) -> f32 {
        let diff = self.vector_subtract(v1, v2);
        self.vector_length_squared(&diff)
    }

    fn vector_length_squared(&self, v: &[f32; 3]) -> f32 {
        let va = f32x4::from_array([v[0], v[1], v[2], 0.0]);
        let vmul = va * va;
        vmul.reduce_sum()
    }

    fn vector_angle(&self, v1: &[f32; 3], v2: &[f32; 3]) -> f32 {
        let va = f32x4::from_array([v1[0], v1[1], v1[2], 0.0]);
        let vb = f32x4::from_array([v2[0], v2[1], v2[2], 0.0]);
        let dot_product = (va * vb).reduce_sum();
        let length_v1_squared = self.vector_length_squared(v1);
        let length_v2_squared = self.vector_length_squared(v2);
        let length_v1 = length_v1_squared.sqrt();
        let length_v2 = length_v2_squared.sqrt();
        (dot_product / (length_v1 * length_v2)).acos()
    }

    fn vector_project(&self, v: &[f32; 3], onto: &[f32; 3]) -> [f32; 3] {
        let dot_product = self.vector_dot(v, onto);
        let length_squared = self.vector_length_squared(onto);
        let scale = dot_product / length_squared;
        self.vector_scale(onto, scale)
    }

    fn vector_reflect(&self, v: &[f32; 3], normal: &[f32; 3]) -> [f32; 3] {
        let dot_product = self.vector_dot(v, normal);
        let scaled_normal = self.vector_scale(normal, 2.0 * dot_product);
        self.vector_subtract(v, &scaled_normal)
    }

    fn vector_refract(&self, v: &[f32; 3], normal: &[f32; 3], eta: f32) -> [f32; 3] {
        let dot_product = self.vector_dot(v, normal);
        let k = 1.0 - eta * eta * (1.0 - dot_product * dot_product);
        if k < 0.0 {
            return [0.0, 0.0, 0.0]; // Total internal reflection
        } else {
            let scaled_normal = self.vector_scale(normal, eta * dot_product + k.sqrt());
            return self.vector_subtract(v, &scaled_normal);
        }
    }

    fn vector_lerp(&self, v1: &[f32; 3], v2: &[f32; 3], t: f32) -> [f32; 3] {
        let va = f32x4::from_array([v1[0], v1[1], v1[2], 0.0]);
        let vb = f32x4::from_array([v2[0], v2[1], v2[2], 0.0]);
        let vt = f32x4::splat(t);

        let result = va + (vb - va) * vt;
        let arr = result.to_array();

        [arr[0], arr[1], arr[2]]
    }

    fn vector_slerp(&self, v1: &[f32; 3], v2: &[f32; 3], t: f32) -> [f32; 3] {
        let va = f32x4::from_array([v1[0], v1[1], v1[2], 0.0]);
        let vb = f32x4::from_array([v2[0], v2[1], v2[2], 0.0]);
        
        let dot_product = self.vector_dot(v1, v2);
        let theta = dot_product.acos();
        let sin_theta = theta.sin();
        
        if sin_theta == 0.0 {
            return self.vector_lerp(v1, v2, t);
        }
        
        let a = (1.0 - t) * theta.sin() / sin_theta;
        let b = t * theta.sin() / sin_theta;
        
        let result = va * f32x4::splat(a) + vb * f32x4::splat(b);
        let arr = result.to_array();
        
        [arr[0], arr[1], arr[2]]
    }

    fn vector_squad(&self, v1: &[f32; 3], v2: &[f32; 3], v3: &[f32; 3], v4: &[f32; 3], t: f32) -> [f32; 3] {
        let a = self.vector_slerp(v1, v2, t);
        let b = self.vector_slerp(v3, v4, t);
        self.vector_lerp(&a, &b, t)
    }

    fn vector_squad_slerp(&self, v1: &[f32; 3], v2: &[f32; 3], v3: &[f32; 3], v4: &[f32; 3], t: f32) -> [f32; 3] {
        let a = self.vector_slerp(v1, v2, t);
        let b = self.vector_slerp(v3, v4, t);
        self.vector_slerp(&a, &b, t)
    }

    fn vector_to_array(&self, v: &[f32; 3]) -> [f32; 3] {
        let va = f32x4::from_array([v[0], v[1], v[2], 0.0]);
        let arr = va.to_array();
        [arr[0], arr[1], arr[2]]
    }

    fn vector_from_array(&self, arr: &[f32; 3]) -> [f32; 3] {
        let va = f32x4::from_array([arr[0], arr[1], arr[2], 0.0]);
        let result = va.to_array();
        [result[0], result[1], result[2]]
    }

    fn vector_to_string(&self, v: &[f32; 3]) -> String {
        format!("({}, {}, {})", v[0], v[1], v[2])
    }

    fn vector_from_string(&self, s: &str) -> [f32; 3] {
        let parts: Vec<&str> = s.trim_matches(|c| c == '(' || c == ')').split(',').collect();
        if parts.len() != 3 {
            panic!("Invalid vector string format");
        }
        [
            parts[0].trim().parse().unwrap(),
            parts[1].trim().parse().unwrap(),
            parts[2].trim().parse().unwrap(),
        ]
    }

    fn vector_to_tuple(&self, v: &[f32; 3]) -> (f32, f32, f32) {
        (v[0], v[1], v[2])
    }

    fn vector_from_tuple(&self, t: (f32, f32, f32)) -> [f32; 3] {
        [t.0, t.1, t.2]
    }

    fn vector_to_vec(&self, v: &[f32; 3]) -> Vec<f32> {
        vec![v[0], v[1], v[2]]
    }

    fn vector_from_vec(&self, v: &Vec<f32>) -> [f32; 3] {
        if v.len() != 3 {
            panic!("Invalid vector vector format");
        }
        [v[0], v[1], v[2]]
    }


    // Matrix operations
    fn matrix_multiply(&self, m1: &[[f32; 4]; 4], m2: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut result = [[0.0; 4]; 4];
        
        // Cargar las filas de la primera matriz
        let m1_row0 = f32x4::from_array(m1[0]);
        let m1_row1 = f32x4::from_array(m1[1]);
        let m1_row2 = f32x4::from_array(m1[2]);
        let m1_row3 = f32x4::from_array(m1[3]);
        
        // Cargar las columnas de la segunda matriz
        let m2_col0 = f32x4::from_array([m2[0][0], m2[1][0], m2[2][0], m2[3][0]]);
        let m2_col1 = f32x4::from_array([m2[0][1], m2[1][1], m2[2][1], m2[3][1]]);
        let m2_col2 = f32x4::from_array([m2[0][2], m2[1][2], m2[2][2], m2[3][2]]);
        let m2_col3 = f32x4::from_array([m2[0][3], m2[1][3], m2[2][3], m2[3][3]]);
        
        // Calcular el producto punto de cada fila por cada columna
        result[0][0] = (m1_row0 * m2_col0).reduce_sum();
        result[0][1] = (m1_row0 * m2_col1).reduce_sum();
        result[0][2] = (m1_row0 * m2_col2).reduce_sum();
        result[0][3] = (m1_row0 * m2_col3).reduce_sum();
        
        result[1][0] = (m1_row1 * m2_col0).reduce_sum();
        result[1][1] = (m1_row1 * m2_col1).reduce_sum();
        result[1][2] = (m1_row1 * m2_col2).reduce_sum();
        result[1][3] = (m1_row1 * m2_col3).reduce_sum();
        
        result[2][0] = (m1_row2 * m2_col0).reduce_sum();
        result[2][1] = (m1_row2 * m2_col1).reduce_sum();
        result[2][2] = (m1_row2 * m2_col2).reduce_sum();
        result[2][3] = (m1_row2 * m2_col3).reduce_sum();
        
        result[3][0] = (m1_row3 * m2_col0).reduce_sum();
        result[3][1] = (m1_row3 * m2_col1).reduce_sum();
        result[3][2] = (m1_row3 * m2_col2).reduce_sum();
        result[3][3] = (m1_row3 * m2_col3).reduce_sum();
        
        result
    }

    fn matrix_translate(&self, m: &mut [[f32; 4]; 4], tx: f32, ty: f32, tz: f32) {
        m[0][3] += tx;
        m[1][3] += ty;
        m[2][3] += tz;
        m[3][3] = 1.0;
    }

    fn matrix_scale(&self, m: &mut [[f32; 4]; 4], sx: f32, sy: f32, sz: f32) {
        m[0][0] *= sx;
        m[1][1] *= sy;
        m[2][2] *= sz;
        m[3][3] = 1.0;
    }

    fn matrix_rotate_x(&self, m: &mut [[f32; 4]; 4], angle: f32) {
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        
        let rotation_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_angle, -sin_angle, 0.0],
            [0.0, sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        // *m = self.matrix_multiply(m, &rotation_matrix);
        *m = self.matrix_multiply(&rotation_matrix, m);
    }

    fn matrix_rotate_y(&self, m: &mut [[f32; 4]; 4], angle: f32) {
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        
        let rotation_matrix = [
            [cos_angle, 0.0, sin_angle, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_angle, 0.0, cos_angle, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        // *m = self.matrix_multiply(m, &rotation_matrix);
        *m = self.matrix_multiply(&rotation_matrix, m);
    }

    fn matrix_rotate_z(&self, m: &mut [[f32; 4]; 4], angle: f32) {
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        
        let rotation_matrix = [
            [cos_angle, -sin_angle, 0.0, 0.0],
            [sin_angle, cos_angle, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        // *m = self.matrix_multiply(m, &rotation_matrix);
        *m = self.matrix_multiply(&rotation_matrix, m);
    }

    fn matrix_transpose(&self, m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[j][i] = m[i][j];
            }
        }
        result
    }

    fn matrix_inverse(&self, m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        // Crear matriz aumentada [A|I]
        let mut augmented = [[0.0f32; 8]; 4];
        for i in 0..4 {
            for j in 0..4 {
                augmented[i][j] = m[i][j];            // Parte A
                augmented[i][j+4] = if i == j { 1.0 } else { 0.0 };  // Parte I
            }
        }
        
        // Algoritmo de Gauss-Jordan con optimizaciones SIMD
        for i in 0..4 {
            // Normalizar la fila pivote
            let pivot = augmented[i][i];
            if pivot == 0.0 {
                panic!("Matrix is not invertible");
            }
            
            // Vectorizar la división de la fila pivote
            let pivot_recip = f32x4::splat(1.0 / pivot);
            
            // Procesar la fila pivote en bloques de 4 elementos
            for j in (0..8).step_by(4) {
                let row_chunk = f32x4::from_array([
                    augmented[i][j], augmented[i][j+1], 
                    augmented[i][j+2], augmented[i][j+3]
                ]);
                let normalized_chunk = row_chunk * pivot_recip;
                let result_array = normalized_chunk.to_array();
                
                augmented[i][j] = result_array[0];
                augmented[i][j+1] = result_array[1];
                augmented[i][j+2] = result_array[2];
                augmented[i][j+3] = result_array[3];
            }
            
            // Eliminar el elemento pivote de las otras filas
            for k in 0..4 {
                if k != i {
                    let factor = augmented[k][i];
                    let factor_vec = f32x4::splat(factor);
                    
                    // Actualizar todas las filas excepto la pivote
                    for j in (0..8).step_by(4) {
                        let pivot_row = f32x4::from_array([
                            augmented[i][j], augmented[i][j+1], 
                            augmented[i][j+2], augmented[i][j+3]
                        ]);
                        let cur_row = f32x4::from_array([
                            augmented[k][j], augmented[k][j+1], 
                            augmented[k][j+2], augmented[k][j+3]
                        ]);
                        
                        // row[k] = row[k] - factor * row[i]
                        let product = factor_vec * pivot_row;
                        let result = cur_row - product;
                        let result_array = result.to_array();
                        
                        augmented[k][j] = result_array[0];
                        augmented[k][j+1] = result_array[1];
                        augmented[k][j+2] = result_array[2];
                        augmented[k][j+3] = result_array[3];
                    }
                }
            }
        }
        
        // Extraer la parte inversa
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = augmented[i][j+4];
            }
        }
        
        result
    }

    fn matrix_determinant(&self, m: &[[f32; 4]; 4]) -> f32 {
        // Función de determinante para matrices 4x4
        // Implementación simplificada, no optimizada para Simd
        let a = m[0][0] * m[1][1] * m[2][2] * m[3][3]
              + m[0][1] * m[1][2] * m[2][3] * m[3][0]
              + m[0][2] * m[1][3] * m[2][0] * m[3][1]
              + m[0][3] * m[1][0] * m[2][1] * m[3][2]
              - (m[0][3] * m[1][2] * m[2][1] * m[3][0]
              + m[0][2] * m[1][1] * m[2][0] * m[3][3]
              + m[0][1] * m[1][0] * m[2][3] * m[3][2]
              + m[0][0] * m[1][3] * m[2][2] * m[3][1]);
        a
    }

    // fn matrix_perspective(&self, fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    //     let f = 1.0 / (fov / 2.0).tan();
    //     let nf = 1.0 / (near - far);
        
    //     [
    //         [f / aspect, 0.0, 0.0, 0.0],
    //         [0.0, f, 0.0, 0.0],
    //         [0.0, 0.0, (far + near) * nf, -1.0],
    //         [0.0, 0.0, 2.0 * far * near * nf, 0.0],
    //     ]
    // }

    fn matrix_perspective(&self, fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        let tan_half_fov = (fov / 2.0).tan();
        let f = 1.0 / tan_half_fov;
        
        // Para matrices en formato fila-mayor
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
            [0.0, 0.0, -1.0, 0.0]
        ]
    }

    fn matrix_normalize(&self, v: &[f32; 3]) -> [f32; 3] {
        let length = self.vector_length(v);
        if length == 0.0 {
            return [0.0, 0.0, 0.0];
        }
        self.vector_scale(v, 1.0 / length)
    }

    fn matrix_cross(&self, v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3] {
        let a = f32x4::from_array([v1[0], v1[1], v1[2], 0.0]);
        let b = f32x4::from_array([v2[0], v2[1], v2[2], 0.0]);
        
        let cross = f32x4::from_array([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
            0.0,
        ]);
        
        cross.to_array()[..3].try_into().unwrap()
    }    

    fn matrix_dot(&self, v1: &[f32; 3], v2: &[f32; 3]) -> f32 {
        let va = f32x4::from_array([v1[0], v1[1], v1[2], 0.0]);
        let vb = f32x4::from_array([v2[0], v2[1], v2[2], 0.0]);
        (va * vb).reduce_sum()
    }
    
    fn matrix_look_at(&self, eye: &[f32; 3], center: &[f32; 3], up: &[f32; 3]) -> [[f32; 4]; 4] {
        let f = self.matrix_normalize(&[
            center[0] - eye[0],
            center[1] - eye[1],
            center[2] - eye[2],
        ]);

        let s = self.matrix_normalize(&self.matrix_cross(&f, up));
        let u = self.matrix_cross(&s, &f);

        [
            [ s[0], s[1], s[2], -self.vector_dot(&s, eye) ],
            [ u[0], u[1], u[2], -self.vector_dot(&u, eye) ],
            [-f[0],-f[1],-f[2],  self.vector_dot(&f, eye) ],
            [ 0.0,  0.0,  0.0, 1.0 ],
        ]
    }

    fn matrix_orthographic(&self, left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        [
            [2.0 / (right - left), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (top - bottom), 0.0, 0.0],
            [0.0, 0.0, -2.0 / (far - near), 0.0],
            [-(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1.0],
        ]
    }

    fn matrix_frustum(&self, left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        let x = 2.0 * near / (right - left);
        let y = 2.0 * near / (top - bottom);
        let a = (right + left) / (right - left);
        let b = (top + bottom) / (top - bottom);
        let c = -(far + near) / (far - near);
        let d = -(2.0 * far * near) / (far - near);

        [
            [x, 0.0, a, 0.0],
            [0.0, y, b, 0.0],
            [0.0, 0.0, c, d],
            [0.0, 0.0, -1.0, 0.0],
        ]
    }

    fn matrix_to_array(&self, m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = m[i][j];
            }
        }
        result
    }

    fn matrix_from_array(&self, arr: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = arr[i][j];
            }
        }
        result
    }

    fn matrix_to_string(&self, m: &[[f32; 4]; 4]) -> String {
        format!("[[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]]",
            m[0][0], m[0][1], m[0][2], m[0][3],
            m[1][0], m[1][1], m[1][2], m[1][3],
            m[2][0], m[2][1], m[2][2], m[2][3],
            m[3][0], m[3][1], m[3][2], m[3][3])
    }

    fn matrix_from_string(&self, s: &str) -> [[f32; 4]; 4] {
        let parts: Vec<&str> = s.trim_matches(|c| c == '[' || c == ']').split("],").collect();
        if parts.len() != 4 {
            panic!("Invalid matrix string format");
        }
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            let row_parts: Vec<&str> = parts[i].trim_matches(|c| c == '[' || c == ']').split(',').collect();
            if row_parts.len() != 4 {
                panic!("Invalid matrix row format");
            }
            for j in 0..4 {
                result[i][j] = row_parts[j].trim().parse().unwrap();
            }
        }
        result
    }

    fn matrix_to_tuple(&self, m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        [
            [m[0][0], m[0][1], m[0][2], m[0][3]],
            [m[1][0], m[1][1], m[1][2], m[1][3]],
            [m[2][0], m[2][1], m[2][2], m[2][3]],
            [m[3][0], m[3][1], m[3][2], m[3][3]],
        ]
    }

    fn matrix_from_tuple(&self, t: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
        [
            [t[0][0], t[0][1], t[0][2], t[0][3]],
            [t[1][0], t[1][1], t[1][2], t[1][3]],
            [t[2][0], t[2][1], t[2][2], t[2][3]],
            [t[3][0], t[3][1], t[3][2], t[3][3]],
        ]
    }

    fn matrix_to_vec(&self, m: &[[f32; 4]; 4]) -> Vec<f32> {
        let mut result = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                result.push(m[i][j]);
            }
        }
        result
    }

    fn matrix_from_vec(&self, v: &Vec<f32>) -> [[f32; 4]; 4] {
        if v.len() != 16 {
            panic!("Invalid matrix vector format");
        }
        [
            [v[0], v[1], v[2], v[3]],
            [v[4], v[5], v[6], v[7]],
            [v[8], v[9], v[10], v[11]],
            [v[12], v[13], v[14], v[15]],
        ]
    }



    // Quaternion operations
    fn quaternion_dot(&self, q1: &[f32; 4], q2: &[f32; 4]) -> f32 {
        let qa = f32x4::from_array([q1[0], q1[1], q1[2], q1[3]]);
        let qb = f32x4::from_array([q2[0], q2[1], q2[2], q2[3]]);
        (qa * qb).reduce_sum()
    }

    // fn quaternion_multiply(&self, q1: &[f32; 4], q2: &[f32; 4]) -> [f32; 4] {
    //     // q1 = [w, x, y, z], q2 = [w, x, y, z]
    //     let a = f32x4::from_array(*q1); // [w1, x1, y1, z1]
    //     let b = f32x4::from_array(*q2); // [w2, x2, y2, z2]
        
    //     // Crear patrones para el primer cuaternión
    //     let a_wzyx = f32x4::from_array([q1[0], q1[3], q1[2], q1[1]]); // [w1, z1, y1, x1]
    //     let a_ywxz = f32x4::from_array([q1[2], q1[0], q1[1], q1[3]]); // [y1, w1, x1, z1]
    //     let a_zxwy = f32x4::from_array([q1[3], q1[1], q1[0], q1[2]]); // [z1, x1, w1, y1]
        
    //     // Crear patrones para el segundo cuaternión
    //     let b_zwxy = f32x4::from_array([q2[3], q2[0], q2[1], q2[2]]); // [z2, w2, x2, y2]
    //     let b_yxwz = f32x4::from_array([q2[2], q2[1], q2[0], q2[3]]); // [y2, x2, w2, z2]
    //     let b_xzyw = f32x4::from_array([q2[1], q2[3], q2[2], q2[0]]); // [x2, z2, y2, w2]
        
    //     // Calcular productos
    //     let p1 = a * b_xzyw;        // [w1*x2, x1*z2, y1*y2, z1*w2]
    //     let p2 = a_wzyx * b_zwxy;   // [w1*z2, z1*w2, y1*x2, x1*y2]
    //     let p3 = a_ywxz * b_yxwz;   // [y1*y2, w1*x2, x1*w2, z1*z2]
    //     let p4 = a_zxwy * b;        // [z1*w2, x1*x2, w1*y2, y1*z2]
        
    //     // Combinar con signos correctos
    //     let result_w = p1[0] - p2[1] - p3[0] - p4[3];
    //     let result_x = p1[3] + p2[0] - p3[2] - p4[1];
    //     let result_y = p1[1] - p2[3] + p3[1] + p4[0];
    //     let result_z = p1[2] + p2[2] - p3[3] + p4[2];
        
    //     [result_w, result_x, result_y, result_z]
    // }

    // fn quaternion_multiply(&self,q1: &[f32; 4], q2: &[f32; 4]) -> [f32; 4] {
    //     let a = f32x4::from_array(*q1); // q1 = [w1, x1, y1, z1]
    //     let b = f32x4::from_array(*q2); // q2 = [w2, x2, y2, z2]

    //     // Extraer productos intermedios con permutaciones (swizzles)
    //     let w1 = a[0];
    //     let x1 = a[1];
    //     let y1 = a[2];
    //     let z1 = a[3];

    //     let qx = f32x4::from_array([w1,  x1,  y1, -z1]);
    //     let qy = f32x4::from_array([-x1, w1,  z1,  y1]);
    //     let qz = f32x4::from_array([-y1, -z1, w1,  x1]);
    //     let qw = f32x4::from_array([z1, -y1,  x1, w1]);

    //     let result = qx * f32x4::splat(b[1]) +
    //                 qy * f32x4::splat(b[2]) +
    //                 qz * f32x4::splat(b[3]) +
    //                 qw * f32x4::splat(b[0]);

    //     result.to_array()
    // }

    fn quaternion_multiply(&self, q1: &[f32; 4], q2: &[f32; 4]) -> [f32; 4] {
        let w1 = q1[0]; let x1 = q1[1]; let y1 = q1[2]; let z1 = q1[3];
        let w2 = q2[0]; let x2 = q2[1]; let y2 = q2[2]; let z2 = q2[3];

        let r = f32x4::from_array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  // w
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  // x
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  // y
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  // z
        ]);

        r.to_array()
    }

    // fn quaternion_multiply(&self, q1: &[f32; 4], q2: &[f32; 4]) -> [f32; 4] { // Temporalmente sin SIMD
    //     let (w1, x1, y1, z1) = (q1[0], q1[1], q1[2], q1[3]);
    //     let (w2, x2, y2, z2) = (q2[0], q2[1], q2[2], q2[3]);

    //     let w = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    //     let x = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    //     let y = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    //     let z = w1*z2 + x1*y2 - y1*x2 + z1*w2;

    //     [w, x, y, z]
    // }

    fn quaternion_conjugate(&self, q: &[f32; 4]) -> [f32; 4] {
        [q[0], -q[1], -q[2], -q[3]]
    }
    
    fn quaternion_normalize(&self, q: &[f32; 4]) -> [f32; 4] {
        // Cargar el cuaternión como vector SIMD
        let q_simd = f32x4::from_array(*q);
        
        // Calcular el cuadrado de la magnitud: w² + x² + y² + z²
        let magnitude_squared = (q_simd * q_simd).reduce_sum();
        
        // Calcular la inversa de la magnitud
        let inv_magnitude = 1.0 / magnitude_squared.sqrt();
        
        // Normalizar multiplicando cada componente por la inversa de la magnitud
        let normalized = q_simd * f32x4::splat(inv_magnitude);
        
        // Convertir de vuelta a array
        normalized.to_array()
    }
    
    fn quaternion_to_matrix(&self, q: &[f32; 4]) -> [[f32; 4]; 4] {
        let qx = q[1];
        let qy = q[2];
        let qz = q[3];
        let qw = q[0];
        
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy), 0.0],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx), 0.0],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
    
    fn quaternion_from_axis_angle(&self, axis: &[f32; 3], angle: f32) -> [f32; 4] {
        // Calculamos el medio ángulo y su seno y coseno
        let half_angle = angle * 0.5;
        let sin_half_angle = half_angle.sin();
        let cos_half_angle = half_angle.cos();
        
        // Convertimos el eje a un vector SIMD
        // Agregamos un cero al final para hacerlo de 4 elementos
        let axis_simd = f32x4::from_array([axis[0], axis[1], axis[2], 0.0]);
        
        // Multiplicamos el eje por el seno del medio ángulo
        let scaled_axis = axis_simd * f32x4::splat(sin_half_angle);
        
        // Extraemos los componentes del eje escalado
        let scaled_axis_array = scaled_axis.to_array();
        
        // Construimos el cuaternión: [w, x, y, z] = [cos(θ/2), axis.x*sin(θ/2), axis.y*sin(θ/2), axis.z*sin(θ/2)]
        [
            cos_half_angle,         // w
            scaled_axis_array[0],   // x
            scaled_axis_array[1],   // y
            scaled_axis_array[2]    // z
        ]
    }
    
    fn quaternion_to_axis_angle(&self, q: &[f32; 4]) -> ([f32; 3], f32) {
        // q = [w, x, y, z] en nuestro formato
        let q_simd = f32x4::from_array(*q);
        
        // Calculamos el ángulo (2 * arccos(w))
        let angle = 2.0 * q[0].acos();
        
        // Calculamos el seno del medio ángulo
        let s = (1.0 - q[0] * q[0]).sqrt();
        
        // Si el seno es muy pequeño, devolvemos un eje arbitrario
        if s < 0.001 {
            return ([1.0, 0.0, 0.0], angle); // Eje arbitrario
        }
        
        // Escalamos el vector parte del cuaternión por 1/s
        let inv_s = 1.0 / s;
        let scaled_axis = q_simd * f32x4::splat(inv_s);
        
        // Extraemos los componentes xyz (ignorando w que está en la posición 0)
        let result_array = scaled_axis.to_array();
        
        // Devolvemos (axis, angle)
        ([result_array[1], result_array[2], result_array[3]], angle)
    }
    
    fn quaternion_from_euler(&self, roll: f32, pitch: f32, yaw: f32) -> [f32; 4] {
        let cy = yaw * 0.5;
        let sy = cy.sin();
        let cy = cy.cos();
        
        let cp = pitch * 0.5;
        let sp = cp.sin();
        let cp = cp.cos();
        
        let cr = roll * 0.5;
        let sr = cr.sin();
        let cr = cr.cos();
        
        [
            cr * cp * cy + sr * sp * sy, // w
            sr * cp * cy - cr * sp * sy, // x
            cr * sp * cy + sr * cp * sy, // y
            cr * cp * sy - sr * sp * cy  // z
        ]
    }
    
    fn quaternion_to_euler(&self, q: &[f32; 4]) -> (f32, f32, f32) {
        let qx = q[1];
        let qy = q[2];
        let qz = q[3];
        let qw = q[0];
        
        // Roll (x-axis rotation)
        let roll_x = (2.0 * (qw * qx + qy * qz)).atan2(1.0 - 2.0 * (qx * qx + qy * qy));
        
        // Pitch (y-axis rotation)
        let sinp = 2.0 * (qw * qy - qz * qx);
        let pitch_y = if sinp.abs() >= 1.0 {
            std::f32::consts::PI / 2.0 * sinp.signum()
        } else {
            sinp.asin()
        };
        
        // Yaw (z-axis rotation)
        let yaw_z = (2.0 * (qw * qz + qx * qy)).atan2(1.0 - 2.0 * (qy * qy + qz * qz));
        
        (roll_x, pitch_y, yaw_z)
    }
    
    fn quaternion_from_rotation_matrix(&self, m: &[[f32; 4]; 4]) -> [f32; 4] {
        let trace = m[0][0] + m[1][1] + m[2][2];
        
        if trace > 0.0 {
            // Caso 1: La traza es positiva
            let s = 0.5 / (trace + 1.0).sqrt();
            [
                0.25 / s,
                (m[2][1] - m[1][2]) * s,
                (m[0][2] - m[2][0]) * s,
                (m[1][0] - m[0][1]) * s
            ]
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            // Caso 2: m[0][0] es el mayor elemento diagonal
            let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
            [
                (m[2][1] - m[1][2]) / s,
                0.25 * s,
                (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s
            ]
        } else if m[1][1] > m[2][2] {
            // Caso 3: m[1][1] es el mayor elemento diagonal
            let s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt();
            [
                (m[0][2] - m[2][0]) / s,
                (m[0][1] + m[1][0]) / s,
                0.25 * s,
                (m[1][2] + m[2][1]) / s
            ]
        } else {
            // Caso 4: m[2][2] es el mayor elemento diagonal
            let s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt();
            [
                (m[1][0] - m[0][1]) / s,
                (m[0][2] + m[2][0]) / s,
                (m[1][2] + m[2][1]) / s,
                0.25 * s
            ]
        }
    }
    
    fn quaternion_to_rotation_matrix(&self, q: &[f32; 4]) -> [[f32; 4]; 4] {
        let qx = q[1];
        let qy = q[2];
        let qz = q[3];
        let qw = q[0];
        
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy), 0.0],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx), 0.0],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
    
    fn quaternion_lerp(&self, q1: &[f32; 4], q2: &[f32; 4], t: f32) -> [f32; 4] {
        let qa = f32x4::from_array(*q1);
        let qb = f32x4::from_array(*q2);

        let t_vec = f32x4::splat(t);
        let q_lerp = qa * (f32x4::splat(1.0) - t_vec) + qb * t_vec;

        q_lerp.to_array()
    }

    fn quaternion_slerp(&self, q1: &[f32; 4], q2: &[f32; 4], t: f32) -> [f32; 4] {
        let qa = f32x4::from_array(*q1);
        let qb = f32x4::from_array(*q2);

        // Calcular el coseno del ángulo entre los cuaterniones
        let dot = self.quaternion_dot(q1, q2);
        
        // Si el coseno es negativo, invertimos el segundo cuaternión
        let qb = if dot < 0.0 {
            -qb
        } else {
            qb
        };
        
        // Calcular el ángulo entre los cuaterniones
        let theta_0 = dot.acos();
        
        // Si el ángulo es cero, devolvemos el primer cuaternión
        if theta_0 == 0.0 {
            return *q1;
        }
        
        // Calcular el seno del ángulo
        let sin_theta_0 = theta_0.sin();
        
        // Calcular los coeficientes de interpolación
        let a = (theta_0 * t).sin() / sin_theta_0;
        let b = (theta_0 * (1.0 - t)).sin() / sin_theta_0;
        
        // Realizar la interpolación esférica
        let q_slerp = qa * f32x4::splat(a) + qb * f32x4::splat(b);
        
        q_slerp.to_array()
    }
    
    fn quaternion_squad(&self, q1: &[f32; 4], q2: &[f32; 4], q3: &[f32; 4], q4: &[f32; 4], t: f32) -> [f32; 4] {
        let a = self.quaternion_slerp(q1, q2, t);
        let b = self.quaternion_slerp(q3, q4, t);
        
        self.quaternion_slerp(&a, &b, t)
    }
    
    fn quaternion_squad_slerp(&self, q1: &[f32; 4], q2: &[f32; 4], q3: &[f32; 4], q4: &[f32; 4], t: f32) -> [f32; 4] {
        let a = self.quaternion_slerp(q1, q2, t);
        let b = self.quaternion_slerp(q3, q4, t);
        
        self.quaternion_slerp(&a, &b, t)
    }
    
    fn quaternion_to_array(&self, q: &[f32; 4]) -> [f32; 4] {
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = q[i];
        }
        result
    }
    
    fn quaternion_from_array(&self, arr: &[f32; 4]) -> [f32; 4] {
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = arr[i];
        }
        result
    }
    
    fn quaternion_to_string(&self, q: &[f32; 4]) -> String {
        format!("({}, {}, {}, {})", q[0], q[1], q[2], q[3])
    }
    
    fn quaternion_from_string(&self, s: &str) -> [f32; 4] {
        let parts: Vec<&str> = s.trim_matches(|c| c == '(' || c == ')').split(',').collect();
        if parts.len() != 4 {
            panic!("Invalid quaternion string format");
        }
        [
            parts[0].trim().parse().unwrap(),
            parts[1].trim().parse().unwrap(),
            parts[2].trim().parse().unwrap(),
            parts[3].trim().parse().unwrap(),
        ]
    }
    
    fn quaternion_to_tuple(&self, q: &[f32; 4]) -> (f32, f32, f32, f32) {
        (q[0], q[1], q[2], q[3])
    }
    
    fn quaternion_from_tuple(&self, t: (f32, f32, f32, f32)) -> [f32; 4] {
        [t.0, t.1, t.2, t.3]
    }
    
    fn quaternion_to_vec(&self, q: &[f32; 4]) -> Vec<f32> {
        vec![q[0], q[1], q[2], q[3]]
    }
    
    fn quaternion_from_vec(&self, v: &Vec<f32>) -> [f32; 4] {
        if v.len() != 4 {
            panic!("Invalid quaternion vector format");
        }
        [v[0], v[1], v[2], v[3]]
    }

    // Métodos de utilidad
    // fn compute_barycentric_coordinates(&self, point: [f32; 2], v0: [f32; 2], v1: [f32; 2], v2: [f32; 2]) -> (f32, f32, f32) {
    //     // Calcular las áreas usando aritmética de diferencias fijas
    //     let area = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]);
        
    //     // Evitar división por cero
    //     if area.abs() < 1e-6 {
    //         return (0.0, 0.0, 0.0);
    //     }
        
    //     // Calcular las áreas relativas para las coordenadas baricéntricas
    //     let alpha = ((v1[0] - point[0]) * (v2[1] - point[1]) - (v2[0] - point[0]) * (v1[1] - point[1])) / area;
    //     let beta = ((v2[0] - point[0]) * (v0[1] - point[1]) - (v0[0] - point[0]) * (v2[1] - point[1])) / area;
    //     let gamma = 1.0 - alpha - beta;
        
    //     (alpha, beta, gamma)
    // }

    fn compute_barycentric_coordinates(&self, point: [f32; 2], v0: [f32; 2], v1: [f32; 2], v2: [f32; 2]) -> (f32, f32, f32) {
        // Calcular el área del triángulo
        let v10_x = v1[0] - v0[0];
        let v10_y = v1[1] - v0[1];
        let v20_x = v2[0] - v0[0];
        let v20_y = v2[1] - v0[1];
        
        let area = v10_x * v20_y - v20_x * v10_y;
        
        // Comprobación de área casi cero
        if area.abs() < 1e-6 {
            return (0.0, 0.0, 0.0);
        }
        
        // Usando SIMD para calcular las coordenadas baricéntricas en paralelo
        // Reorganizamos los datos para aprovechar mejor las operaciones SIMD
        // [v1.x-point.x, v2.x-point.x, v0.x-point.x, unused]
        let dx = f32x4::from_array([
            v1[0] - point[0],
            v2[0] - point[0],
            v0[0] - point[0],
            0.0
        ]);
        
        // [v1.y-point.y, v2.y-point.y, v0.y-point.y, unused]
        let dy = f32x4::from_array([
            v1[1] - point[1],
            v2[1] - point[1],
            v0[1] - point[1],
            0.0
        ]);
        
        // [v2.y-point.y, v0.y-point.y, v1.y-point.y, unused]
        let dy_shifted = f32x4::from_array([
            v2[1] - point[1],
            v0[1] - point[1],
            v1[1] - point[1],
            0.0
        ]);
        
        // [v2.x-point.x, v0.x-point.x, v1.x-point.x, unused]
        let dx_shifted = f32x4::from_array([
            v2[0] - point[0],
            v0[0] - point[0],
            v1[0] - point[0],
            0.0
        ]);
        
        // Cálculo de los determinantes para las coordenadas baricéntricas
        // alpha: (v1.x-p.x)*(v2.y-p.y) - (v2.x-p.x)*(v1.y-p.y)
        // beta:  (v2.x-p.x)*(v0.y-p.y) - (v0.x-p.x)*(v2.y-p.y)
        // gamma: (v0.x-p.x)*(v1.y-p.y) - (v1.x-p.x)*(v0.y-p.y)
        let det = dx * dy_shifted - dx_shifted * dy;
        
        let coords = det / f32x4::splat(area);
        
        // Extraer los resultados
        let alpha = coords.as_array()[0];
        let beta = coords.as_array()[1];
        let gamma = coords.as_array()[2];
        
        (alpha, beta, gamma)
    }
}
