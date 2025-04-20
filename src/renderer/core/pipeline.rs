use super::rasterizer::Rasterizer;
use super::transform::Transform;
use super::math::{Vector, Matrix, Quaternion};
use core_simd::simd::cmp::SimdPartialOrd;
use crate::renderer::geometry::{Mesh, Vertex};
use crate::renderer::shader::Shader;
use crate::renderer::texture::Texture;
use crate::renderer::core::math;

pub struct VertexOutput {
    pub position: [f32; 4],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub tex_coords: [f32; 2],
    pub world_position: [f32; 3], // Posición en el espacio mundial para cálculos de iluminación
}

pub struct FragmentOutput {
    pub color: [f32; 4],
}

pub struct Pipeline<'a, S: Shader> {
    shader: S,
    rasterizer: &'a mut Rasterizer,
    // Matrices de transformación (enfoque basado en matrices)
    model_matrix: Matrix,
    view_matrix: Matrix,
    projection_matrix: Matrix,
    coordinate_system: CoordinateSystem,
    
    // Transformaciones basadas en quaternions
    model_position: Vector,
    model_rotation: Quaternion,
    model_scale: Vector,
    view_position: Vector,
    view_rotation: Quaternion,
    
    // OPTIMIZACIÓN: Caché de matrices combinadas
    cached_model_view_matrix: Matrix, // Caché de model * view
    cached_mvp_matrix: Matrix,        // Caché de projection * view * model (MVP)
    cache_valid: bool,                // Indica si la caché es válida
    
    // OPTIMIZACIÓN: Precálculo para transformaciones quaternion
    model_rotation_normalized: Quaternion, // Quaternion de rotación normalizado
    model_rotation_conjugate: Quaternion,  // Conjugado del quaternion normalizado
    rotation_precalc_valid: bool,          // Indica si los precálculos son válidos
}

impl<'a, S: Shader> Pipeline<'a, S> {    
    pub fn new(shader: S, rasterizer: &'a mut Rasterizer) -> Self {
        // Crear una matriz identidad correcta para el modelo
        // Manualmente definir la matriz identidad 4x4
        let model_matrix = Matrix::new_with_values([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        
        // Matriz de vista: mirar desde [0,0,5] hacia el origen [0,0,0]
        let view_matrix = Matrix::look_at(
            &[0.0, 0.0, 5.0],  // posición de la cámara
            &[0.0, 0.0, 0.0],  // punto al que mira la cámara
            &[0.0, 1.0, 0.0]   // vector "arriba" de la cámara
        );
        
        // Matriz de proyección perspectiva con parámetros razonables
        let aspect_ratio = rasterizer.width as f32 / rasterizer.height as f32;
        
        // Ajustar near y far según la dirección de la cámara
        let near = 0.1;
        let far = 100.0;
        
        if cfg!(debug_assertions) {
            println!("Configurando proyección inicial: FOV=45°, Aspect={:.2}, Near={:.2}, Far={:.2}", 
                     aspect_ratio, near, far);
        }
        
        // Crear matriz de proyección ajustada a la dirección de la cámara
        let projection_matrix = Matrix::perspective(
            std::f32::consts::PI / 4.0,  // 45 grados FOV
            aspect_ratio,
            near,  // near plane
            far    // far plane
        );
        
        // Inicializar la versión de quaternions de las transformaciones
        let model_position = Vector::new();
        let model_rotation = Quaternion::new();
        let model_scale = Vector::new_with_values(1.0, 1.0, 1.0);
        
        // Extraer posición y rotación de la matriz de vista
        let view_position = Vector::new_with_values(0.0, 0.0, 5.0);
        let view_rotation = Quaternion::new(); // Por defecto, sin rotación
        
        // OPTIMIZACIÓN: Inicializar la caché de matrices combinadas
        let cached_model_view_matrix = view_matrix.multiply(&model_matrix);
        let cached_mvp_matrix = projection_matrix.multiply(&cached_model_view_matrix);
        
        // OPTIMIZACIÓN: Precalcular valores para transformaciones quaternion
        let model_rotation_normalized = model_rotation.normalize();
        let model_rotation_conjugate = model_rotation_normalized.conjugate();
        
        Self { 
            shader, 
            rasterizer,
            model_matrix,
            view_matrix,
            projection_matrix,
            coordinate_system: CoordinateSystem::OpenGL,
            model_position,
            model_rotation,
            model_scale,
            view_position,
            view_rotation,
            cached_model_view_matrix,
            cached_mvp_matrix,
            cache_valid: true,
            model_rotation_normalized,
            model_rotation_conjugate,
            rotation_precalc_valid: true,
        }
    }

    pub fn is_rotation_precalc_valid(&self) -> bool {
        self.rotation_precalc_valid
    }
    pub fn is_cache_valid(&self) -> bool {
        self.cache_valid
    }
    
    pub fn is_model_matrix_valid(&self) -> bool {
        self.cache_valid
    }

    pub fn is_view_matrix_valid(&self) -> bool {
        self.cache_valid
    }

    pub fn is_projection_matrix_valid(&self) -> bool {
        self.cache_valid
    }

    pub fn get_model_scale(&self) -> &Vector {
        &self.model_scale
    }

    pub fn get_model_position(&self) -> &Vector {
        &self.model_position
    }

    pub fn get_model_rotation(&self) -> &Quaternion {
        &self.model_rotation
    }

    pub fn get_model_rotation_normalized(&self) -> &Quaternion {
        &self.model_rotation_normalized
    }

    pub fn get_model_rotation_conjugate(&self) -> &Quaternion {
        &self.model_rotation_conjugate
    }

    // OPTIMIZACIÓN: Método para recalcular la caché de matrices
    pub fn update_matrix_cache(&mut self) {
        if !self.cache_valid {
            // Recalcular la caché de matrices
            self.cached_model_view_matrix = self.view_matrix.multiply(&self.model_matrix);
            self.cached_mvp_matrix = self.projection_matrix.multiply(&self.cached_model_view_matrix);
            self.cache_valid = true;
            
            if cfg!(debug_assertions) {
                println!("Caché de matrices MVP actualizada");
            }
        }
    }
    
    // OPTIMIZACIÓN: Método para actualizar precálculos de quaternion
    pub fn update_quaternion_precalc(&mut self) {
        if !self.rotation_precalc_valid {
            self.model_rotation_normalized = self.model_rotation.normalize();
            self.model_rotation_conjugate = self.model_rotation_normalized.conjugate();
            self.rotation_precalc_valid = true;
            
            if cfg!(debug_assertions) {
                println!("Precálculos de quaternion actualizados");
            }
        }
    }
    
    // OPTIMIZACIÓN: Métodos para invalidar las cachés
    fn invalidate_matrix_cache(&mut self) {
        self.cache_valid = false;
    }
    
    fn invalidate_quaternion_precalc(&mut self) {
        self.rotation_precalc_valid = false;
    }

    pub fn rasterizer(&self) -> &Rasterizer {
        &self.rasterizer
    }

    pub fn rasterizer_mut(&mut self) -> &mut Rasterizer {
        self.rasterizer
    }
    
    pub fn get_color_buffer(&self) -> Vec<u32> {
        self.rasterizer.get_color_buffer()
    }

    pub fn clear_rasterizer(&mut self, color: u32, depth: f32) {
        self.rasterizer.clear(color, depth);
    }
    
    /// Escribe el color_buffer directamente en el framebuffer destino (BGR/BGRX) usando SIMD
    pub fn write_color_buffer_to_framebuffer(&self, framebuffer: &mut [u8], width: u32, height: u32, bytes_per_pixel: usize) {
        use core_simd::simd::u32x8;
        let color_buffer = &self.rasterizer.color_buffer;
        let total_pixels = (width * height) as usize;
        let simd_width = 8;
        let mut i = 0;
        while i + simd_width <= total_pixels {
            let colors = u32x8::from_array([
                color_buffer[i + 0],
                color_buffer[i + 1],
                color_buffer[i + 2],
                color_buffer[i + 3],
                color_buffer[i + 4],
                color_buffer[i + 5],
                color_buffer[i + 6],
                color_buffer[i + 7],
            ]);
            let r = ((colors >> 16) & u32x8::splat(0xFF)).to_array();
            let g = ((colors >> 8) & u32x8::splat(0xFF)).to_array();
            let b = (colors & u32x8::splat(0xFF)).to_array();
            for lane in 0..simd_width {
                let dst_idx = (i + lane) * bytes_per_pixel;
                if dst_idx + 2 < framebuffer.len() {
                    framebuffer[dst_idx + 0] = b[lane] as u8;
                    framebuffer[dst_idx + 1] = g[lane] as u8;
                    framebuffer[dst_idx + 2] = r[lane] as u8;
                    if bytes_per_pixel >= 4 && dst_idx + 3 < framebuffer.len() {
                        framebuffer[dst_idx + 3] = 0;
                    }
                }
            }
            i += simd_width;
        }
        // Resto escalar
        while i < total_pixels {
            let color = color_buffer[i];
            let r = ((color >> 16) & 0xFF) as u8;
            let g = ((color >> 8) & 0xFF) as u8;
            let b = (color & 0xFF) as u8;
            let dst_idx = i * bytes_per_pixel;
            if dst_idx + 2 < framebuffer.len() {
                framebuffer[dst_idx + 0] = b;
                framebuffer[dst_idx + 1] = g;
                framebuffer[dst_idx + 2] = r;
                if bytes_per_pixel >= 4 && dst_idx + 3 < framebuffer.len() {
                    framebuffer[dst_idx + 3] = 0;
                }
            }
            i += 1;
        }
    }
    
    /// Establece la matriz de modelo (transforma vértices del espacio local al mundial)
    pub fn set_model_matrix(&mut self, matrix: Matrix) {
        self.model_matrix = matrix;
        
        // OPTIMIZACIÓN: Invalidar caché de matrices
        self.invalidate_matrix_cache();
        
        // Si estamos en modo quaternion, extraemos los componentes de la matriz
        // Extraer la traslación de la última columna de la matriz
        let elements = matrix.elements();
        self.model_position = Vector::new_with_values(
            elements[0][3], 
            elements[1][3], 
            elements[2][3]
        );
        
        // Extraer la rotación como quaternion
        self.model_rotation = Quaternion::from_rotation_matrix(&elements);
        
        // OPTIMIZACIÓN: Invalidar precálculos de quaternion
        self.invalidate_quaternion_precalc();
        
        // Extraer la escala (aproximadamente)
        // Tomamos la magnitud de los vectores columna de la matriz
        let scale_x = Vector::new_with_values(
            elements[0][0],
            elements[1][0],
            elements[2][0]
        ).length();
        
        let scale_y = Vector::new_with_values(
            elements[0][1],
            elements[1][1],
            elements[2][1]
        ).length();
        
        let scale_z = Vector::new_with_values(
            elements[0][2],
            elements[1][2],
            elements[2][2]
        ).length();
        
        self.model_scale = Vector::new_with_values(scale_x, scale_y, scale_z);
        
        if cfg!(debug_assertions) {
            println!("Matriz de modelo convertida a quaternion: Pos={:?}, Rot={:?}, Scale={:?}",
                     self.model_position, self.model_rotation, self.model_scale);
        }
    }
    
    /// Establece la matriz de modelo a partir de una transformación
    pub fn set_model_transform(&mut self, transform: &mut Transform) {
        // En lugar de usar clone(), copiar los elementos de la matriz manualmente
        let world_matrix = transform.world_matrix();
        let world_elements = world_matrix.elements();
        
        // Crear una nueva matriz con los mismos elementos
        self.model_matrix = Matrix::new_with_values(world_elements);
        
        // OPTIMIZACIÓN: Invalidar caché de matrices
        self.invalidate_matrix_cache();
        
        // Si estamos en modo quaternion, también actualizamos esos componentes
        self.model_position = transform.position().clone();
        self.model_rotation = transform.rotation().clone();
        
        // OPTIMIZACIÓN: Invalidar precálculos de quaternion
        self.invalidate_quaternion_precalc();
        
        self.model_scale = transform.scale().clone();
        
        // Depuración: imprimir la matriz para verificar
        if cfg!(debug_assertions) {
            self.debug_matrix("Matriz modelo aplicada", &self.model_matrix);
        }
    }

    /// Establece la matriz de vista (transforma del espacio mundial al de la cámara)
    pub fn set_view_matrix(&mut self, matrix: Matrix) {
        self.view_matrix = matrix;
        
        // OPTIMIZACIÓN: Invalidar caché de matrices
        self.invalidate_matrix_cache();
        
        // Si estamos en modo quaternion, extraemos los componentes de la matriz de vista
        // La matriz de vista es el inverso de la matriz de la cámara
        // Por lo tanto, necesitamos invertirla para obtener la posición y rotación real
        
        // Para extraer la posición de la cámara, debemos considerar que la matriz
        // de vista es una concatenación de una rotación R y una traslación T:
        // V = R * T, donde T = translate(-eye)
        
        // Extraer la submatriz de rotación 3x3
        let elements = matrix.elements();
        let rotation_matrix = [
            [elements[0][0], elements[0][1], elements[0][2], 0.0],
            [elements[1][0], elements[1][1], elements[1][2], 0.0],
            [elements[2][0], elements[2][1], elements[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        
        // Extraer la rotación como quaternion
        self.view_rotation = Quaternion::from_rotation_matrix(&rotation_matrix);
        
        // Para obtener la posición de la cámara (eye), calculamos -R^-1 * t
        // donde t es el vector de traslación en la matriz de vista
        let translation = Vector::new_with_values(elements[0][3], elements[1][3], elements[2][3]);
        
        // Crear una matriz de rotación a partir de la rotación inversa (transpuesta)
        let inverse_rotation = [
            [elements[0][0], elements[1][0], elements[2][0], 0.0],
            [elements[0][1], elements[1][1], elements[2][1], 0.0],
            [elements[0][2], elements[1][2], elements[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        
        // Aplicar la rotación inversa a la traslación
        let mut eye = Vector::new();
        for i in 0..3 {
            let mut sum = 0.0;
            for j in 0..3 {
                sum += -inverse_rotation[i][j] * translation.to_array()[j];
            }
            match i {
                0 => eye = Vector::new_with_values(sum, 0.0, 0.0),
                1 => eye = Vector::new_with_values(eye.x(), sum, 0.0),
                2 => eye = Vector::new_with_values(eye.x(), eye.y(), sum),
                _ => {}
            }
        }
        
        self.view_position = eye;
        
        if cfg!(debug_assertions) {
            println!("Matriz de vista convertida a quaternion: Eye={:?}, Rot={:?}",
                     self.view_position, self.view_rotation);
        }
    }
    
    /// Configura la matriz de vista a partir de posición, objetivo y vector arriba
    pub fn look_at(&mut self, eye: &[f32; 3], target: &[f32; 3], up: &[f32; 3]) {
        // Imprimir parámetros para debug
        if cfg!(debug_assertions) {
            println!("look_at: eye={:?}, target={:?}, up={:?}", eye, target, up);
        }
        
        let view_matrix = Matrix::look_at(eye, target, up);
        self.view_matrix = view_matrix;
        
        // OPTIMIZACIÓN: Invalidar caché de matrices
        self.invalidate_matrix_cache();
        
        // Actualizar también la posición y rotación de la vista para el modo quaternion
        // La posición de la cámara es simplemente el punto eye
        self.view_position = Vector::new_with_values(eye[0], eye[1], eye[2]);
        
        // Calcular la rotación mirando desde eye hacia target
        // Para calcular la rotación, construimos un sistema de coordenadas ortogonales
        
        // 1. Vector dirección normalizado (eye hacia target): -Z en el sistema de la cámara
        let direction = Vector::new_with_values(
            target[0] - eye[0],
            target[1] - eye[1],
            target[2] - eye[2]
        ).normalize();
        
        // 2. Vector derecha (perpendicular a up y direction): X en el sistema de la cámara
        let up_vector = Vector::new_with_values(up[0], up[1], up[2]);
        let right = direction.cross(&up_vector).normalize();
        
        // 3. Vector up corregido (perpendicular a direction y right): Y en el sistema de la cámara
        let corrected_up = right.cross(&direction).normalize();
        
        // 4. Construir una matriz de rotación con estos vectores
        let rotation_matrix = [
            [right.x(), corrected_up.x(), -direction.x(), 0.0],
            [right.y(), corrected_up.y(), -direction.y(), 0.0],
            [right.z(), corrected_up.z(), -direction.z(), 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        
        // 5. Convertir la matriz de rotación a quaternion
        self.view_rotation = Quaternion::from_rotation_matrix(&rotation_matrix);
        
        if cfg!(debug_assertions) {
            println!("Look-at convertido a quaternion: Eye={:?}, Rot={:?}", 
                self.view_position, self.view_rotation);
        }
    }
    
    /// Método de depuración para imprimir una matriz
    pub fn debug_matrix(&self, name: &str, matrix: &Matrix) {
        println!("{}: {:?}", name, matrix);
    }

    /// Establece la matriz de proyección (transforma del espacio de cámara al de recorte)
    pub fn set_projection_matrix(&mut self, matrix: Matrix) {
        self.projection_matrix = matrix;
        
        // OPTIMIZACIÓN: Invalidar caché de matrices
        self.invalidate_matrix_cache();
    }    
    
    /// Configura una matriz de proyección en perspectiva    
    pub fn set_perspective(&mut self, fov_radians: f32, aspect_ratio: f32, near: f32, far: f32) {
        // Verificar si la cámara está mirando en dirección -Z (según la matriz de vista)
        // Esto se puede detectar mirando el elemento [2][2] de la matriz de vista
        let inverse_z = self.view_matrix.elements()[2][2] < 0.0;
        
        if cfg!(debug_assertions) {
            println!("Configurando matriz de proyección: FOV={:.2}°, Aspect={:.2}, Near={:.2}, Far={:.2}, InverseZ={}", 
                 fov_radians * 180.0 / std::f32::consts::PI, aspect_ratio, near, far, inverse_z);
        }
        
        let projection = Matrix::perspective(fov_radians, aspect_ratio, near, far);
        
        self.projection_matrix = projection;
        
        // OPTIMIZACIÓN: Invalidar caché de matrices
        self.invalidate_matrix_cache();

        if cfg!(debug_assertions) {
            println!("Matriz de proyección configurada: {:?}", self.projection_matrix);
        }
    }

    /// Configura una matriz de proyección ortográfica
    pub fn set_orthographic(&mut self, left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) {
        let projection = Matrix::orthographic(left, right, bottom, top, near, far);
        self.projection_matrix = projection;
        
        // OPTIMIZACIÓN: Invalidar caché de matrices
        self.invalidate_matrix_cache();

        if cfg!(debug_assertions) {
            println!("Matriz de proyección ortográfica configurada: {:?}", self.projection_matrix);
        }
    }

    /// Establece el sistema de coordenadas a utilizar
    pub fn set_coordinate_system(&mut self, system: CoordinateSystem) {
        self.coordinate_system = system;
    }
    
    /// Establece la posición del modelo para el modo quaternion
    pub fn set_model_position(&mut self, position: Vector) {
        self.model_position = position;
        
        // También actualizamos la matriz de modelo para mantener consistencia
        self.update_model_matrix_from_quaternion_components();
    }
    
    /// Establece la rotación del modelo para el modo quaternion
    pub fn set_model_rotation(&mut self, rotation: Quaternion) {
        self.model_rotation = rotation;
        
        // OPTIMIZACIÓN: Invalidar precálculos de quaternion
        self.invalidate_quaternion_precalc();
        
        // También actualizamos la matriz de modelo para mantener consistencia
        self.update_model_matrix_from_quaternion_components();
    }
    
    /// Establece la escala del modelo para el modo quaternion
    pub fn set_model_scale(&mut self, scale: Vector) {
        self.model_scale = scale;
        
        // También actualizamos la matriz de modelo para mantener consistencia
        self.update_model_matrix_from_quaternion_components();
    }
    
    /// OPTIMIZACIÓN: Actualiza la matriz de modelo a partir de los componentes quaternion
    fn update_model_matrix_from_quaternion_components(&mut self) {
        // Crear una matriz de rotación a partir del quaternion
        let model_matrix = self.model_rotation.to_rotation_matrix();
        // Aplicar escala a los vectores columna
        let mut elements = model_matrix.elements();
        for i in 0..3 {
            elements[i][0] *= self.model_scale.x();
            elements[i][1] *= self.model_scale.y();
            elements[i][2] *= self.model_scale.z();
        }
        // Aplicar traslación
        elements[0][3] = self.model_position.x();
        elements[1][3] = self.model_position.y();
        elements[2][3] = self.model_position.z();
        // Actualizar la matriz de modelo
        self.model_matrix = Matrix::new_with_values(elements);
        // Invalidar la caché de matrices
        self.invalidate_matrix_cache();
    }
    
    /// Establece la posición de la vista para el modo quaternion
    pub fn set_view_position(&mut self, position: Vector) {
        self.view_position = position;
        
        // También actualizamos la matriz de vista para mantener consistencia
        self.update_view_matrix_from_quaternion_components();
    }
    
    /// Establece la rotación de la vista para el modo quaternion
    pub fn set_view_rotation(&mut self, rotation: Quaternion) {
        self.view_rotation = rotation;
        
        // También actualizamos la matriz de vista para mantener consistencia
        self.update_view_matrix_from_quaternion_components();
    }
    
    /// OPTIMIZACIÓN: Actualiza la matriz de vista a partir de los componentes quaternion
    fn update_view_matrix_from_quaternion_components(&mut self) {
        // Para crear una matriz de vista, necesitamos:
        // 1. Crear la matriz inversa de la cámara (que sería la matriz de mundo)
        // 2. La matriz de mundo es una combinación de rotación y traslación
        
        // Crear una matriz de rotación a partir del quaternion de vista
        let rotation_matrix = self.view_rotation.to_rotation_matrix().elements();
        
        // La matriz de cámara en el mundo sería rotación * traslación
        // Pero necesitamos la inversa para la vista
        // Primero, la parte de rotación invertida es la transpuesta
        let mut rotation_inverse = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                rotation_inverse[i][j] = rotation_matrix[j][i];
            }
        }
        // Luego, la traslación invertida es -R^-1 * t
        let mut translation_inverse = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                translation_inverse[i] -= rotation_inverse[i][j] * self.view_position.to_array()[j];
            }
        }
        // Combinamos para formar la matriz de vista
        let view_elements = [
            [rotation_inverse[0][0], rotation_inverse[0][1], rotation_inverse[0][2], translation_inverse[0]],
            [rotation_inverse[1][0], rotation_inverse[1][1], rotation_inverse[1][2], translation_inverse[1]],
            [rotation_inverse[2][0], rotation_inverse[2][1], rotation_inverse[2][2], translation_inverse[2]],
            [0.0, 0.0, 0.0, 1.0]
        ];
        
        // Actualizar la matriz de vista
        self.view_matrix = Matrix::new_with_values(view_elements);
        
        // Invalidar la caché de matrices
        self.invalidate_matrix_cache();
    }
    
    /// OPTIMIZACIÓN: Métodos para obtener las matrices (útiles para el renderizado de sombras)
    pub fn get_model_matrix(&self) -> Matrix {
        self.model_matrix
    }

    pub fn get_view_matrix(&self) -> Matrix {
        self.view_matrix
    }

    pub fn get_projection_matrix(&self) -> Matrix {
        self.projection_matrix
    }
    
    // OPTIMIZACIÓN: Obtener matrices combinadas (si es necesario actualizar la caché)
    pub fn get_model_view_matrix(&mut self) -> Matrix {
        self.update_matrix_cache();
        self.cached_model_view_matrix
    }
    
    pub fn get_mvp_matrix(&mut self) -> Matrix {
        self.update_matrix_cache();
        self.cached_mvp_matrix
    }

    /// Procesa un vértice a través del pipeline de renderizado    
    pub fn process_vertex(&mut self, vertex: &Vertex) -> VertexOutput {
        // OPTIMIZACIÓN: Asegurar que los precálculos están actualizados
        self.update_quaternion_precalc();
        // Transformación al espacio de mundo con quaternions optimizados
        let world_pos = Pipeline::<S>::transform_point_quaternion_parallel(
            &vertex.position,
            &self.model_scale,
            &self.model_position,
            &self.model_rotation_normalized,
            &self.model_rotation_conjugate,
        );
        // Ahora usar la matriz de vista-proyección para transformar al espacio de clip
        let view_proj_matrix = self.projection_matrix.multiply(&self.view_matrix);
        let clip_pos = Pipeline::<S>::transform_point_parallel(&world_pos, &view_proj_matrix);
        // División de perspectiva para obtener NDC
        let w = clip_pos[3];
        // Manejar w cercanos a cero o negativos
        if w.abs() < 0.001 {
            return VertexOutput {
                position: [-999.0, -999.0, 0.0, 1.0], // Fuera de pantalla
                normal: vertex.normal,
                color: vertex.color,
                tex_coords: vertex.tex_coords,
                world_position: [world_pos[0], world_pos[1], world_pos[2]],
            };
        }
        let ndc_x = clip_pos[0] / w;
        let ndc_y = clip_pos[1] / w;
        let ndc_z = clip_pos[2] / w;
        // Transformación de NDC a coordenadas de pantalla
        let screen_x = (ndc_x + 1.0) * 0.5 * self.rasterizer.width as f32;
        let screen_y = (1.0 - ndc_y) * 0.5 * self.rasterizer.height as f32; // Y invertida para OpenGL
        let screen_z = (ndc_z + 1.0) * 0.5; // Z de [0, 1] para buffer de profundidad

        // Calcular la normal transformada (solo rotación y escala, sin traslación)
        let normal_quat = Quaternion::new_with_values(0.0, vertex.normal[0], vertex.normal[1], vertex.normal[2]);
        let rotated_normal = self.model_rotation_normalized
            .multiply(&normal_quat)
            .multiply(&self.model_rotation_conjugate);
        let n = rotated_normal.to_array();
        let transformed_normal = [n[1], n[2], n[3]];
        let norm = (transformed_normal[0].powi(2) + transformed_normal[1].powi(2) + transformed_normal[2].powi(2)).sqrt();
        let transformed_normal = if norm > 0.0 {
            [
                transformed_normal[0] / norm,
                transformed_normal[1] / norm,
                transformed_normal[2] / norm,
            ]
        } else {
            transformed_normal
        };

        VertexOutput {
            position: [screen_x, screen_y, screen_z, w],
            normal: transformed_normal,
            color: vertex.color,
            tex_coords: vertex.tex_coords,
            world_position: [world_pos[0], world_pos[1], world_pos[2]],
        }
    }

    // --- NUEVA FUNCIÓN AUXILIAR PARA PARALELISMO ---
    fn transform_point_quaternion_parallel(position: &[f32; 3], model_scale: &Vector, model_position: &Vector, model_rotation_normalized: &Quaternion, model_rotation_conjugate: &Quaternion) -> [f32; 4] {
        let scaled_x = position[0] * model_scale.x();
        let scaled_y = position[1] * model_scale.y();
        let scaled_z = position[2] * model_scale.z();
        let point_quaternion = Quaternion::new_with_values(0.0, scaled_x, scaled_y, scaled_z);
        let rotated_quaternion = model_rotation_normalized
            .multiply(&point_quaternion)
            .multiply(model_rotation_conjugate);
        let quaternion_array = rotated_quaternion.to_array();
        [
            quaternion_array[1] + model_position.x(),
            quaternion_array[2] + model_position.y(),
            quaternion_array[3] + model_position.z(),
            1.0
        ]
    }

    fn transform_point_parallel(point: &[f32; 4], matrix: &Matrix) -> [f32; 4] {
        let mut result = [0.0; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i] += matrix.elements()[i][j] * point[j];
            }
        }
        result
    }

    fn process_vertex_parallel(
        vertex: &Vertex,
        model_scale: &Vector,
        model_position: &Vector,
        model_rotation_normalized: &Quaternion,
        model_rotation_conjugate: &Quaternion,
        projection_matrix: &Matrix,
        view_matrix: &Matrix,
        width: u32,
        height: u32,
    ) -> VertexOutput {
        let world_pos = Self::transform_point_quaternion_parallel(
            &vertex.position,
            model_scale,
            model_position,
            model_rotation_normalized,
            model_rotation_conjugate,
        );
        // Calcular la normal transformada (solo rotación y escala, sin traslación)
        let normal_quat = Quaternion::new_with_values(0.0, vertex.normal[0], vertex.normal[1], vertex.normal[2]);
        let rotated_normal = model_rotation_normalized
            .multiply(&normal_quat)
            .multiply(model_rotation_conjugate);
        let n = rotated_normal.to_array();
        let transformed_normal = [n[1], n[2], n[3]];
        let norm = (transformed_normal[0].powi(2) + transformed_normal[1].powi(2) + transformed_normal[2].powi(2)).sqrt();
        let transformed_normal = if norm > 0.0 {
            [
                transformed_normal[0] / norm,
                transformed_normal[1] / norm,
                transformed_normal[2] / norm,
            ]
        } else {
            transformed_normal
        };
        let view_proj_matrix = projection_matrix.multiply(view_matrix);
        let clip_pos = Self::transform_point_parallel(&world_pos, &view_proj_matrix);
        let w = clip_pos[3];
        if w.abs() < 0.001 {
            return VertexOutput {
                position: [-999.0, -999.0, 0.0, 1.0],
                normal: transformed_normal,
                color: vertex.color,
                tex_coords: vertex.tex_coords,
                world_position: [world_pos[0], world_pos[1], world_pos[2]],
            };
        }
        let ndc_x = clip_pos[0] / w;
        let ndc_y = clip_pos[1] / w;
        let ndc_z = clip_pos[2] / w;
        let screen_x = (ndc_x + 1.0) * 0.5 * width as f32;
        let screen_y = (1.0 - ndc_y) * 0.5 * height as f32;
        let screen_z = (ndc_z + 1.0) * 0.5;
        VertexOutput {
            position: [screen_x, screen_y, screen_z, w],
            normal: transformed_normal,
            color: vertex.color,
            tex_coords: vertex.tex_coords,
            world_position: [world_pos[0], world_pos[1], world_pos[2]],
        }
    }

    pub fn render_parallel(&mut self, mesh: &Mesh, texture: Option<&Texture>) {
        self.update_matrix_cache();
        self.update_quaternion_precalc();
        let indices = &mesh.indices;
        let vertices = &mesh.vertices;
        let tex_ref = texture.or_else(|| mesh.diffuse_texture.as_ref());
        // Obtener slices mutables de los buffers globales
        let color_buffer = &mut self.rasterizer.color_buffer;
        let depth_buffer = &mut self.rasterizer.depth_buffer;
        let width = self.rasterizer.width;
        let height = self.rasterizer.height;
        let num_threads = rayon::current_num_threads() as u32;
        let band_height = (height + num_threads - 1) / num_threads;

        // --- Extraer datos inmutables para el paralelismo ---
        let model_scale = self.model_scale;
        let model_position = self.model_position;
        let model_rotation_normalized = self.model_rotation_normalized;
        let model_rotation_conjugate = self.model_rotation_conjugate;
        let projection_matrix = self.projection_matrix;
        let view_matrix = self.view_matrix;

        // Preprocesar todos los triángulos (transformación + culling) en paralelo
        use rayon::prelude::*;
        let triangles: Vec<[Vertex; 3]> = indices
            .par_chunks(3)
            .filter_map(|chunk| {
                if chunk.len() < 3 {
                    return None;
                }
                let i0 = chunk[0] as usize;
                let i1 = chunk[1] as usize;
                let i2 = chunk[2] as usize;
                if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
                    return None;
                }
                let v0 = &vertices[i0];
                let v1 = &vertices[i1];
                let v2 = &vertices[i2];
                let v0_out = Pipeline::<S>::process_vertex_parallel(
                    v0, &model_scale, &model_position, &model_rotation_normalized, &model_rotation_conjugate, &projection_matrix, &view_matrix, width, height
                );
                let v1_out = Pipeline::<S>::process_vertex_parallel(
                    v1, &model_scale, &model_position, &model_rotation_normalized, &model_rotation_conjugate, &projection_matrix, &view_matrix, width, height
                );
                let v2_out = Pipeline::<S>::process_vertex_parallel(
                    v2, &model_scale, &model_position, &model_rotation_normalized, &model_rotation_conjugate, &projection_matrix, &view_matrix, width, height
                );
                // Culling 2D (igual que antes, pero sin self)
                let edge1 = [v1_out.position[0] - v0_out.position[0], v1_out.position[1] - v0_out.position[1]];
                let edge2 = [v2_out.position[0] - v0_out.position[0], v2_out.position[1] - v0_out.position[1]];
                let cross_product = edge1[0] * edge2[1] - edge1[1] * edge2[0];
                if cross_product > 0.0 {
                    return None;
                }
                Some([
                    Vertex {
                        position: [v0_out.position[0], v0_out.position[1], v0_out.position[2]],
                        normal: [v0_out.normal[0], v0_out.normal[1], v0_out.normal[2]],
                        tex_coords: v0_out.tex_coords,
                        color: v0_out.color,
                    },
                    Vertex {
                        position: [v1_out.position[0], v1_out.position[1], v1_out.position[2]],
                        normal: [v1_out.normal[0], v1_out.normal[1], v1_out.normal[2]],
                        tex_coords: v1_out.tex_coords,
                        color: v1_out.color,
                    },
                    Vertex {
                        position: [v2_out.position[0], v2_out.position[1], v2_out.position[2]],
                        normal: [v2_out.normal[0], v2_out.normal[1], v2_out.normal[2]],
                        tex_coords: v2_out.tex_coords,
                        color: v2_out.color,
                    },
                ])
            })
            .collect();

        // Rasterizar por bandas horizontales en paralelo
        color_buffer
            .chunks_mut((band_height * width) as usize)
            .zip(depth_buffer.chunks_mut((band_height * width) as usize))
            .enumerate()
            .par_bridge()
            .for_each(|(band_idx, (color_slice, depth_slice))| {
                let y_start = band_idx as u32 * band_height;
                let y_end = ((band_idx as u32 + 1) * band_height).min(height);
                for tri in triangles.iter() {
                    let min_x = tri.iter().map(|v| v.position[0]).fold(f32::INFINITY, f32::min).floor() as i32;
                    let max_x = tri.iter().map(|v| v.position[0]).fold(f32::NEG_INFINITY, f32::max).ceil() as i32;
                    let min_y = tri.iter().map(|v| v.position[1]).fold(f32::INFINITY, f32::min).floor() as i32;
                    let max_y = tri.iter().map(|v| v.position[1]).fold(f32::NEG_INFINITY, f32::max).ceil() as i32;
                    let start_x = min_x.max(0);
                    let end_x = max_x.min((width - 1) as i32);
                    let start_y = min_y.max(y_start as i32);
                    let end_y = max_y.min((y_end - 1) as i32);
                    // --- Comprobación de vértices finitos ---
                    if tri.iter().any(|v| !v.position[0].is_finite() || !v.position[1].is_finite() || !v.position[2].is_finite()) {
                        continue;
                    }
                    // --- Comprobación de rangos válidos ---
                    if start_x > end_x || start_y > end_y {
                        continue;
                    }
                    for y in start_y..=end_y {
                        let px_start = start_x as u32;
                        let px_end = end_x as u32;
                        let mut x = px_start;
                        while x + 7 <= px_end {
                            use core_simd::simd::{f32x8, u32x8};
                            let pxs = f32x8::from_array([
                                (x + 0) as f32 + 0.5,
                                (x + 1) as f32 + 0.5,
                                (x + 2) as f32 + 0.5,
                                (x + 3) as f32 + 0.5,
                                (x + 4) as f32 + 0.5,
                                (x + 5) as f32 + 0.5,
                                (x + 6) as f32 + 0.5,
                                (x + 7) as f32 + 0.5,
                            ]);
                            let pys = f32x8::splat(y as f32 + 0.5);
                            let (alpha, beta, gamma) = math::compute_barycentric_coordinates_simd(
                                pxs, pys,
                                [tri[0].position[0], tri[0].position[1]],
                                [tri[1].position[0], tri[1].position[1]],
                                [tri[2].position[0], tri[2].position[1]],
                            );
                            let mask = alpha.simd_ge(f32x8::splat(0.0)) & beta.simd_ge(f32x8::splat(0.0)) & gamma.simd_ge(f32x8::splat(0.0));
                            if mask.any() {
                                // Interpolación paralela
                                let depth = alpha * f32x8::splat(tri[0].position[2]) + beta * f32x8::splat(tri[1].position[2]) + gamma * f32x8::splat(tri[2].position[2]);
                                let r = alpha * f32x8::splat(tri[0].color[0]) + beta * f32x8::splat(tri[1].color[0]) + gamma * f32x8::splat(tri[2].color[0]);
                                let g = alpha * f32x8::splat(tri[0].color[1]) + beta * f32x8::splat(tri[1].color[1]) + gamma * f32x8::splat(tri[2].color[1]);
                                let b = alpha * f32x8::splat(tri[0].color[2]) + beta * f32x8::splat(tri[1].color[2]) + gamma * f32x8::splat(tri[2].color[2]);
                                let u = alpha * f32x8::splat(tri[0].tex_coords[0]) + beta * f32x8::splat(tri[1].tex_coords[0]) + gamma * f32x8::splat(tri[2].tex_coords[0]);
                                let v = alpha * f32x8::splat(tri[0].tex_coords[1]) + beta * f32x8::splat(tri[1].tex_coords[1]) + gamma * f32x8::splat(tri[2].tex_coords[1]);
                                let ux = u32x8::from_array([
                                    x + 0, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7
                                ]);
                                let uy = u32x8::splat(y as u32);
                                let local_idx = (uy - u32x8::splat(y_start)) * u32x8::splat(width) + ux;
                                let local_idx_arr = local_idx.to_array();
                                let depth_arr = depth.to_array();
                                let r_arr = r.to_array();
                                let g_arr = g.to_array();
                                let b_arr = b.to_array();
                                let u_arr = u.to_array();
                                let v_arr = v.to_array();
                                for lane in 0..8 {
                                    if mask.test(lane) {
                                        let idx = local_idx_arr[lane] as usize;
                                        if idx < depth_slice.len() && idx < color_slice.len() {
                                            if depth_arr[lane] < depth_slice[idx] {
                                                let (r, g, b) = (r_arr[lane], g_arr[lane], b_arr[lane]);
                                                let (u, v) = (u_arr[lane], v_arr[lane]);
                                                let final_color = if let Some(tex) = tex_ref {
                                                    let tex_color = tex.sample(u, v);
                                                    let final_r = (r * tex_color[0] * 255.0) as u32;
                                                    let final_g = (g * tex_color[1] * 255.0) as u32;
                                                    let final_b = (b * tex_color[2] * 255.0) as u32;
                                                    (final_r << 16) | (final_g << 8) | final_b
                                                } else {
                                                    let final_r = (r * 255.0) as u32;
                                                    let final_g = (g * 255.0) as u32;
                                                    let final_b = (b * 255.0) as u32;
                                                    (final_r << 16) | (final_g << 8) | final_b
                                                };
                                                depth_slice[idx] = depth_arr[lane];
                                                color_slice[idx] = final_color;
                                            }
                                        }
                                    }
                                }
                            }
                            x += 8;
                        }
                        // Resto escalar
                        while x <= px_end {
                            let px = x as f32 + 0.5;
                            let py = y as f32 + 0.5;
                            let (alpha, beta, gamma) = math::compute_barycentric_coordinates(
                                [px, py],
                                [tri[0].position[0], tri[0].position[1]],
                                [tri[1].position[0], tri[1].position[1]],
                                [tri[2].position[0], tri[2].position[1]],
                            );
                            if alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 {
                                let depth = alpha * tri[0].position[2] + beta * tri[1].position[2] + gamma * tri[2].position[2];
                                let ux = x as u32;
                                let uy = y as u32;
                                let local_idx = ((uy - y_start) * width + ux) as usize;
                                if local_idx < depth_slice.len() && local_idx < color_slice.len() {
                                    if let Some(tex) = tex_ref {
                                        let u = alpha * tri[0].tex_coords[0] + beta * tri[1].tex_coords[0] + gamma * tri[2].tex_coords[0];
                                        let v = alpha * tri[0].tex_coords[1] + beta * tri[1].tex_coords[1] + gamma * tri[2].tex_coords[1];
                                        let tex_color = tex.sample(u, v);
                                        let r = alpha * tri[0].color[0] + beta * tri[1].color[0] + gamma * tri[2].color[0];
                                        let g = alpha * tri[0].color[1] + beta * tri[1].color[1] + gamma * tri[2].color[1];
                                        let b = alpha * tri[0].color[2] + beta * tri[1].color[2] + gamma * tri[2].color[2];
                                        let final_r = (r * tex_color[0] * 255.0) as u32;
                                        let final_g = (g * tex_color[1] * 255.0) as u32;
                                        let final_b = (b * tex_color[2] * 255.0) as u32;
                                        let final_color = (final_r << 16) | (final_g << 8) | final_b;
                                        if depth < depth_slice[local_idx] {
                                            depth_slice[local_idx] = depth;
                                            color_slice[local_idx] = final_color;
                                        }
                                    } else {
                                        let r = alpha * tri[0].color[0] + beta * tri[1].color[0] + gamma * tri[2].color[0];
                                        let g = alpha * tri[0].color[1] + beta * tri[1].color[1] + gamma * tri[2].color[1];
                                        let b = alpha * tri[0].color[2] + beta * tri[1].color[2] + gamma * tri[2].color[2];
                                        let final_r = (r * 255.0) as u32;
                                        let final_g = (g * 255.0) as u32;
                                        let final_b = (b * 255.0) as u32;
                                        let final_color = (final_r << 16) | (final_g << 8) | final_b;
                                        if depth < depth_slice[local_idx] {
                                            depth_slice[local_idx] = depth;
                                            color_slice[local_idx] = final_color;
                                        }
                                    }
                                }
                            }
                            x += 1;
                        }
                    }
                }
            });
    }
}

pub struct TileBuffer {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub color_buffer: Vec<u32>,
    pub depth_buffer: Vec<f32>,
}

impl TileBuffer {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            color_buffer: vec![0; (width * height) as usize],
            depth_buffer: vec![f32::INFINITY; (width * height) as usize],
        }
    }

    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, color: u32, depth: f32) {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) as usize;
            if depth < self.depth_buffer[idx] {
                self.depth_buffer[idx] = depth;
                self.color_buffer[idx] = color;
            }
        }
    }
}

pub enum CoordinateSystem {
    OpenGL,
    Vulkan,
    Metal,
    DirectX,
}

impl CoordinateSystem {
    pub fn transform_position(&self, position: &mut [f32; 4]) {
        match self {
            CoordinateSystem::OpenGL => {
                // OpenGL uses a right-handed coordinate system with Y up and Z forward.
                // No transformation needed for OpenGL.
            }
            CoordinateSystem::Vulkan => {
                // Vulkan uses a right-handed coordinate system with Y down and Z forward.
                position[1] = -position[1];
            }
            CoordinateSystem::Metal => {
                // Metal uses a right-handed coordinate system with Y up and Z forward.
                // No transformation needed for Metal.
            }
            CoordinateSystem::DirectX => {
                // DirectX uses a left-handed coordinate system with Y up and Z forward.
                position[2] = -position[2];
            }
        }
    }
}
