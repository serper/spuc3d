use super::math::{MathBackend, CpuMathBackend};

impl MathBackend for CpuMathBackend {
    
    // Vector operations
    fn vector_add(&self, v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3] {
        [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]
    }

    fn vector_subtract(&self, v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3] {
        [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]
    }

    fn vector_dot(&self, v1: &[f32; 3], v2: &[f32; 3]) -> f32 {
        v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    }

    fn vector_cross(&self, v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3] {
        [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ]
    }

    fn vector_scale(&self,v: &[f32; 3], scalar: f32) -> [f32; 3] {
        [v[0] * scalar, v[1] * scalar, v[2] * scalar]
    }
    
    fn vector_length(&self,v: &[f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    fn vector_normalize(&self, v: &[f32; 3]) -> [f32; 3] {
        let mag = self.vector_length(v);
        [
            v[0] / mag,
            v[1] / mag,
            v[2] / mag,
        ]
    }

    fn vector_distance(&self,v1: &[f32; 3], v2: &[f32; 3]) -> f32 {
        ((v1[0] - v2[0]).powi(2) + (v1[1] - v2[1]).powi(2) + (v1[2] - v2[2]).powi(2)).sqrt()
    }

    fn vector_distance_squared(&self, v1: &[f32; 3], v2: &[f32; 3]) -> f32 {
        (v1[0] - v2[0]).powi(2) + (v1[1] - v2[1]).powi(2) + (v1[2] - v2[2]).powi(2)
    }

    fn vector_length_squared(&self, v: &[f32; 3]) -> f32 {
        v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    }

    fn vector_angle(&self, v1: &[f32; 3], v2: &[f32; 3]) -> f32 {
        let dot_product = self.vector_dot(v1, v2);
        let lengths = self.vector_length(v1) * self.vector_length(v2);
        (dot_product / lengths).acos()
    }
    fn vector_project(&self, v: &[f32; 3], onto: &[f32; 3]) -> [f32; 3] {
        let dot_product = self.vector_dot(v, onto);
        let length_squared = self.vector_length_squared(onto);
        self.vector_scale(onto, dot_product / length_squared)
    }


    fn vector_reflect(&self,v: &[f32; 3], normal: &[f32; 3]) -> [f32; 3] {
        let dot_product = self.vector_dot(v, normal);
        [
            v[0] - 2.0 * dot_product * normal[0],
            v[1] - 2.0 * dot_product * normal[1],
            v[2] - 2.0 * dot_product * normal[2],
        ]
    }

    fn vector_refract(&self,v: &[f32; 3], normal: &[f32; 3], eta: f32) -> [f32; 3] {
        let dot_product = self.vector_dot(v, normal);
        let k = 1.0 - eta * eta * (1.0 - dot_product * dot_product);
        if k < 0.0 {
            return [0.0, 0.0, 0.0]; // Total internal reflection
        } else {
            return [
                eta * v[0] - (eta * dot_product + k.sqrt()) * normal[0],
                eta * v[1] - (eta * dot_product + k.sqrt()) * normal[1],
                eta * v[2] - (eta * dot_product + k.sqrt()) * normal[2],
            ];
        }
    }

    fn vector_lerp(&self,v1: &[f32; 3], v2: &[f32; 3], t: f32) -> [f32; 3] {
        [
            v1[0] + t * (v2[0] - v1[0]),
            v1[1] + t * (v2[1] - v1[1]),
            v1[2] + t * (v2[2] - v1[2]),
        ]
    }

    fn vector_slerp(&self,v1: &[f32; 3], v2: &[f32; 3], t: f32) -> [f32; 3] {
        let dot_product = self.vector_dot(v1, v2);
        let theta = dot_product.acos();
        let sin_theta = (1.0 - dot_product * dot_product).sqrt();
        if sin_theta < 1e-6 {
            return self.vector_lerp(v1, v2, t);
        } else {
            let a = (1.0 - t) * theta.sin() / sin_theta;
            let b = t * theta.sin() / sin_theta;
            return [
                a * v1[0] + b * v2[0],
                a * v1[1] + b * v2[1],
                a * v1[2] + b * v2[2],
            ];
        }
    }

    fn vector_squad(&self,v1: &[f32; 3], v2: &[f32; 3], v3: &[f32; 3], v4: &[f32; 3], t: f32) -> [f32; 3] {
        let p1 = self.vector_slerp(v1, v2, t);
        let p2 = self.vector_slerp(v3, v4, t);
        self.vector_slerp(&p1, &p2, t)
    }

    fn vector_squad_slerp(&self,v1: &[f32; 3], v2: &[f32; 3], v3: &[f32; 3], v4: &[f32; 3], t: f32) -> [f32; 3] {
        let p1 = self.vector_slerp(v1, v2, t);
        let p2 = self.vector_slerp(v3, v4, t);
        self.vector_slerp(&p1, &p2, t)
    }

    fn vector_to_array(&self,v: &[f32; 3]) -> [f32; 3] {
        [v[0], v[1], v[2]]
    }

    fn vector_from_array(&self,arr: &[f32; 3]) -> [f32; 3] {
        [arr[0], arr[1], arr[2]]
    }

    fn vector_to_string(&self,v: &[f32; 3]) -> String {
        format!("({}, {}, {})", v[0], v[1], v[2])
    }

    fn vector_from_string(&self,s: &str) -> [f32; 3] {
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

    fn vector_to_tuple(&self,v: &[f32; 3]) -> (f32, f32, f32) {
        (v[0], v[1], v[2])
    }

    fn vector_from_tuple(&self,t: (f32, f32, f32)) -> [f32; 3] {
        [t.0, t.1, t.2]
    }

    fn vector_to_vec(&self,v: &[f32; 3]) -> Vec<f32> {
        vec![v[0], v[1], v[2]]
    }

    fn vector_from_vec(&self,v: &Vec<f32>) -> [f32; 3] {
        if v.len() != 3 {
            panic!("Invalid vector length");
        }
        [v[0], v[1], v[2]]
    }

    // Matrix operations
    fn matrix_multiply(&self, m1: &[[f32; 4]; 4], m2: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = m1[i][0] * m2[0][j] + m1[i][1] * m2[1][j] + m1[i][2] * m2[2][j] + m1[i][3] * m2[3][j];
            }
        }
        result
    }

    fn matrix_translate(&self, m: &mut [[f32; 4]; 4], tx: f32, ty: f32, tz: f32) {
        m[0][3] += tx;
        m[1][3] += ty;
        m[2][3] += tz;
        m[3][3] = 1.0; // Homogeneous coordinate
    }

    fn matrix_scale(&self, m: &mut [[f32; 4]; 4], sx: f32, sy: f32, sz: f32) {
        m[0][0] *= sx;
        m[1][1] *= sy;
        m[2][2] *= sz;
        m[3][3] = 1.0; // Homogeneous coordinate
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
        let mut result = [[0.0; 4]; 4];
        let det = self.matrix_determinant(m);
        if det == 0.0 {
            panic!("Matrix is not invertible");
        }
        for i in 0..4 {
            for j in 0..4 {
                result[j][i] = m[(i + 1) % 4][(j + 1) % 4] * m[(i + 2) % 4][(j + 2) % 4] * m[(i + 3) % 4][(j + 3) % 4] -
                               m[(i + 1) % 4][(j + 2) % 4] * m[(i + 2) % 4][(j + 1) % 4] * m[(i + 3) % 4][(j + 3) % 4];
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] /= det;
            }
        }
        result
    }

    fn matrix_determinant(&self, m: &[[f32; 4]; 4]) -> f32 {
        m[0][0] * (m[1][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) - m[1][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1]) + m[1][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1])) -
        m[0][1] * (m[1][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) - m[1][2] * (m[2][0] * m[3][3] - m[2][3] * m[3][0]) + m[1][3] * (m[2][0] * m[3][2] - m[2][2] * m[3][0])) +
        m[0][2] * (m[1][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1]) - m[1][1] * (m[2][0] * m[3][3] - m[2][3] * m[3][0]) + m[1][3] * (m[2][0] * m[3][1] - m[2][1] * m[3][0])) -
        m[0][3] * (m[1][0] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]) - m[1][1] * (m[2][0] * m[3][2] - m[2][2] * m[3][0]) + m[1][2] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]))
    }    
    
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
        [v[0] / length, v[1] / length, v[2] / length]
    }

    fn matrix_cross(&self, v1: &[f32; 3], v2: &[f32; 3]) -> [f32; 3] {
        [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ]
    }

    fn matrix_dot(&self, v1: &[f32; 3], v2: &[f32; 3]) -> f32 {
        v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
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
        [
            [(2.0 * near) / (right - left), 0.0, 0.0, 0.0],
            [0.0, (2.0 * near) / (top - bottom), 0.0, 0.0],
            [(right + left) / (right - left), (top + bottom) / (top - bottom), -(far + near) / (far - near), -1.0],
            [0.0, 0.0, -(2.0 * far * near) / (far - near), 0.0],
        ]
    }

    fn matrix_to_array(&self, m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        [
            [m[0][0], m[0][1], m[0][2], m[0][3]],
            [m[1][0], m[1][1], m[1][2], m[1][3]],
            [m[2][0], m[2][1], m[2][2], m[2][3]],
            [m[3][0], m[3][1], m[3][2], m[3][3]],
        ]
    }

    fn matrix_from_array(&self, arr: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        [
            [arr[0][0], arr[0][1], arr[0][2], arr[0][3]],
            [arr[1][0], arr[1][1], arr[1][2], arr[1][3]],
            [arr[2][0], arr[2][1], arr[2][2], arr[2][3]],
            [arr[3][0], arr[3][1], arr[3][2], arr[3][3]],
        ]
    }

    fn matrix_to_string(&self, m: &[[f32; 4]; 4]) -> String {
        format!("[[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]]",
            m[0][0], m[0][1], m[0][2], m[0][3],
            m[1][0], m[1][1], m[1][2], m[1][3],
            m[2][0], m[2][1], m[2][2], m[2][3],
            m[3][0], m[3][1], m[3][2], m[3][3])
    }

    fn matrix_from_string(&self, s: &str) -> [[f32; 4]; 4] {
        let parts: Vec<&str> = s.trim_matches(|c| c == '[' || c == ']').split("], [").collect();
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
        vec![
            m[0][0], m[0][1], m[0][2], m[0][3],
            m[1][0], m[1][1], m[1][2], m[1][3],
            m[2][0], m[2][1], m[2][2], m[2][3],
            m[3][0], m[3][1], m[3][2], m[3][3],
        ]
    }

    fn matrix_from_vec(&self, v: &Vec<f32>) -> [[f32; 4]; 4] {
        if v.len() != 16 {
            panic!("Invalid matrix vector length");
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
        q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]
    }

    fn quaternion_multiply(&self, q1: &[f32; 4], q2: &[f32; 4]) -> [f32; 4] {
        [
            q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
            q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
            q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
            q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
        ]
    }

    fn quaternion_conjugate(&self, q: &[f32; 4]) -> [f32; 4] {
        [
            q[0],
            -q[1],
            -q[2],
            -q[3],
        ]
    }

    fn quaternion_normalize(&self, q: &[f32; 4]) -> [f32; 4] {
        let mag = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        [
            q[0] / mag,
            q[1] / mag,
            q[2] / mag,
            q[3] / mag,
        ]
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
        let half_angle = angle / 2.0;
        let sin_half_angle = half_angle.sin();
        [
            (half_angle.cos()),
            axis[0] * sin_half_angle,
            axis[1] * sin_half_angle,
            axis[2] * sin_half_angle,
        ]
    }

    fn quaternion_to_axis_angle(&self, q: &[f32; 4]) -> ([f32; 3], f32) {
        let angle = 2.0 * q[0].acos();
        let s = (1.0 - q[0] * q[0]).sqrt();
        if s < 1e-6 {
            return ([1.0, 0.0, 0.0], angle); // Arbitrary axis
        } else {
            return ([q[1] / s, q[2] / s, q[3] / s], angle);
        }
    }

    fn quaternion_from_euler(&self, roll: f32, pitch: f32, yaw: f32) -> [f32; 4] {
        let cy = yaw * 0.5;
        let sy = yaw.sin() * 0.5;
        let cp = pitch * 0.5;
        let sp = pitch.sin() * 0.5;
        let cr = roll * 0.5;
        let sr = roll.sin() * 0.5;

        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    }

    fn quaternion_to_euler(&self, q: &[f32; 4]) -> (f32, f32, f32) {
        let sinr_cosp = 2.0 * (q[0] * q[1] + q[2] * q[3]);
        let cosr_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
        let roll = sinr_cosp.atan2(cosr_cosp);

        let sinp = 2.0 * (q[0] * q[2] - q[3] * q[1]);
        let pitch = if sinp.abs() >= 1.0 {
            std::f32::consts::PI / 2.0 * sinp.signum()
        } else {
            sinp.asin()
        };

        let siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2]);
        let cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }

    fn quaternion_from_rotation_matrix(&self, m: &[[f32; 4]; 4]) -> [f32; 4] {
        let trace = m[0][0] + m[1][1] + m[2][2];
        let mut q = [0.0; 4];
        if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            q[0] = 0.25 * s;
            q[1] = (m[2][1] - m[1][2]) / s;
            q[2] = (m[0][2] - m[2][0]) / s;
            q[3] = (m[1][0] - m[0][1]) / s;
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
            q[0] = (m[2][1] - m[1][2]) / s;
            q[1] = 0.25 * s;
            q[2] = (m[0][1] + m[1][0]) / s;
            q[3] = (m[0][2] + m[2][0]) / s;
        } else if m[1][1] > m[2][2] {
            let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
            q[0] = (m[0][2] - m[2][0]) / s;
            q[1] = (m[0][1] + m[1][0]) / s;
            q[2] = 0.25 * s;
            q[3] = (m[1][2] + m[2][1]) / s;
        } else {
            let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
            q[0] = (m[1][0] - m[0][1]) / s;
            q[1] = (m[0][2] + m[2][0]) / s;
            q[2] = (m[1][2] + m[2][1]) / s;
            q[3] = 0.25 * s;
        }

        q
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
        [
            q1[0] + t * (q2[0] - q1[0]),
            q1[1] + t * (q2[1] - q1[1]),
            q1[2] + t * (q2[2] - q1[2]),
            q1[3] + t * (q2[3] - q1[3]),
        ]
    }

    fn quaternion_slerp(&self, q1: &[f32; 4], q2: &[f32; 4], t: f32) -> [f32; 4] {
        let dot_product = self.quaternion_dot(q1, q2);
        let theta = dot_product.acos();
        let sin_theta = (1.0 - dot_product * dot_product).sqrt();
        if sin_theta < 1e-6 {
            return self.quaternion_lerp(q1, q2, t);
        } else {
            let a = (1.0 - t) * theta.sin() / sin_theta;
            let b = t * theta.sin() / sin_theta;
            return [
                a * q1[0] + b * q2[0],
                a * q1[1] + b * q2[1],
                a * q1[2] + b * q2[2],
                a * q1[3] + b * q2[3],
            ];
        }
    }

    fn quaternion_squad(&self, q1: &[f32; 4], q2: &[f32; 4], q3: &[f32; 4], q4: &[f32; 4], t: f32) -> [f32; 4] {
        let p1 = self.quaternion_slerp(q1, q2, t);
        let p2 = self.quaternion_slerp(q3, q4, t);
        self.quaternion_slerp(&p1, &p2, t)
    }

    fn quaternion_squad_slerp(&self, q1: &[f32; 4], q2: &[f32; 4], q3: &[f32; 4], q4: &[f32; 4], t: f32) -> [f32; 4] {
        let p1 = self.quaternion_slerp(q1, q2, t);
        let p2 = self.quaternion_slerp(q3, q4, t);
        self.quaternion_slerp(&p1, &p2, t)
    }

    fn quaternion_to_array(&self, q: &[f32; 4]) -> [f32; 4] {
        [q[0], q[1], q[2], q[3]]
    }

    fn quaternion_from_array(&self, arr: &[f32; 4]) -> [f32; 4] {
        [arr[0], arr[1], arr[2], arr[3]]    
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
            panic!("Invalid quaternion vector length");
        }
        [v[0], v[1], v[2], v[3]]
    }

    // Métodos de utilidad
    fn compute_barycentric_coordinates(&self, point: [f32; 2], v0: [f32; 2], v1: [f32; 2], v2: [f32; 2]) -> (f32, f32, f32) {
        // Calcular las áreas usando aritmética de diferencias fijas
        let area = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]);
        
        // Evitar división por cero
        if area.abs() < 1e-6 {
            return (0.0, 0.0, 0.0);
        }
        
        // Calcular las áreas relativas para las coordenadas baricéntricas
        let alpha = ((v1[0] - point[0]) * (v2[1] - point[1]) - (v2[0] - point[0]) * (v1[1] - point[1])) / area;
        let beta = ((v2[0] - point[0]) * (v0[1] - point[1]) - (v0[0] - point[0]) * (v2[1] - point[1])) / area;
        let gamma = 1.0 - alpha - beta;
        
        (alpha, beta, gamma)
    }

}
