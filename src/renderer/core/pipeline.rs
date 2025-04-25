use super::rasterizer::Rasterizer;
use super::transform::Transform;
use super::math::{Vector, Matrix, Quaternion};
use crate::renderer::geometry::{Mesh, Vertex};
use crate::renderer::geometry::obj_loader::Scene;
use crate::renderer::shader::Shader;
use crate::renderer::texture::Texture;

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

#[derive(PartialEq)]
pub enum RenderMode {
    Wireframe,
    Color,
    Texture,
}

pub struct Pipeline<'a> {
    pub shader: Box<dyn Shader + Sync>,
    pub rasterizer: &'a mut Rasterizer,
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

    pub render_mode: RenderMode,
}

impl<'a> Pipeline<'a> {    
    pub fn new(shader: Box<dyn Shader + Sync>, rasterizer: &'a mut Rasterizer) -> Self {
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
            render_mode: RenderMode::Texture, // Por defecto
        }
    }

    pub fn set_shader(&mut self, shader: Box<dyn Shader + Sync>) {
        self.shader = shader;
    }

    pub fn set_render_mode(&mut self, mode: RenderMode) {
        self.render_mode = mode;
    }

    pub fn get_render_mode(&self) -> &RenderMode {
        &self.render_mode
    }

    pub fn get_render_mode_string(&self) -> String {
        match self.render_mode {
            RenderMode::Wireframe => "Wireframe".to_string(),
            RenderMode::Color => "Color".to_string(),
            RenderMode::Texture => "Texture".to_string(),
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
        use core_simd::simd::u32x4;
        let color_buffer = &self.rasterizer.color_buffer;
        let total_pixels = (width * height) as usize;
        let simd_width = 4;
        let mut i = 0;
        while i + simd_width <= total_pixels {
            let colors = u32x4::from_array([
                color_buffer[i + 0],
                color_buffer[i + 1],
                color_buffer[i + 2],
                color_buffer[i + 3],
            ]);
            let r = ((colors >> 16) & u32x4::splat(0xFF)).to_array();
            let g = ((colors >> 8) & u32x4::splat(0xFF)).to_array();
            let b = (colors & u32x4::splat(0xFF)).to_array();
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
        self.update_quaternion_precalc();
        let world_pos = Pipeline::transform_point_quaternion_parallel(
            &vertex.position,
            &self.model_scale,
            &self.model_position,
            &self.model_rotation_normalized,
            &self.model_rotation_conjugate,
        );
        let view_proj_matrix = self.projection_matrix.multiply(&self.view_matrix);
        let clip_pos = Pipeline::transform_point_parallel(&world_pos, &view_proj_matrix);
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
    fn transform_point_quaternion_parallel(position: &[f32; 4], model_scale: &Vector, model_position: &Vector, model_rotation_normalized: &Quaternion, model_rotation_conjugate: &Quaternion) -> [f32; 4] {
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

    pub fn render(&mut self, scene: &Scene) {
        for mesh in &scene.meshes {
            let texture = mesh.diffuse_texture.as_ref();
            self.render_parallel(mesh, texture);
        }
    }

    /// Configura la matriz de vista usando eye, target y up
    pub fn look_at(&mut self, eye: &[f32; 3], target: &[f32; 3], up: &[f32; 3]) {
        self.view_matrix = Matrix::look_at(eye, target, up);
        self.view_position = Vector::new_with_values(eye[0], eye[1], eye[2]);
        self.cache_valid = false;
    }

    /// Configura la matriz de proyección perspectiva
    pub fn set_perspective(&mut self, fov_y: f32, aspect: f32, near: f32, far: f32) {
        self.projection_matrix = Matrix::perspective(fov_y, aspect, near, far);
        self.cache_valid = false;
    }

    /// Configura la transformación del modelo a partir de un Transform
    pub fn set_model_transform(&mut self, transform: &mut Transform) {
        self.model_position = *transform.position();
        self.model_scale = *transform.scale();
        self.model_rotation = *transform.rotation();
        self.cache_valid = false;
        self.rotation_precalc_valid = false;
    }
    
    pub fn render_parallel(&mut self, mesh: &Mesh, texture: Option<&Texture>) {
        self.update_matrix_cache();
        self.update_quaternion_precalc();
        let indices = &mesh.indices;
        let vertices = &mesh.vertices;
        let tex_ref = texture.or_else(|| mesh.diffuse_texture.as_ref());
        let shader = &self.shader;
        let width = self.rasterizer.width;
        let height = self.rasterizer.height;
        // let num_threads = rayon::current_num_threads() as u32;
        // let band_height = (height + num_threads - 1) / num_threads;
        let model_scale = self.model_scale;
        let model_position = self.model_position;
        let model_rotation_normalized = self.model_rotation_normalized;
        let model_rotation_conjugate = self.model_rotation_conjugate;
        let projection_matrix = self.projection_matrix;
        let view_matrix = self.view_matrix;

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
                let v0_out = Pipeline::process_vertex_parallel(
                    v0, &model_scale, &model_position, &model_rotation_normalized, &model_rotation_conjugate, &projection_matrix, &view_matrix, width, height
                );
                let v1_out = Pipeline::process_vertex_parallel(
                    v1, &model_scale, &model_position, &model_rotation_normalized, &model_rotation_conjugate, &projection_matrix, &view_matrix, width, height
                );
                let v2_out = Pipeline::process_vertex_parallel(
                    v2, &model_scale, &model_position, &model_rotation_normalized, &model_rotation_conjugate, &projection_matrix, &view_matrix, width, height
                );
                let edge1 = [v1_out.position[0] - v0_out.position[0], v1_out.position[1] - v0_out.position[1]];
                let edge2 = [v2_out.position[0] - v0_out.position[0], v2_out.position[1] - v0_out.position[1]];
                let cross_product = edge1[0] * edge2[1] - edge1[1] * edge2[0];
                if cross_product > 0.0 {
                    return None;
                }
                // Al construir vértices tras el vertex shader, asegúrate de que position es [f32; 4]
                Some([
                    Vertex::new4([
                        v0_out.position[0], v0_out.position[1], v0_out.position[2], v0_out.position[3]
                    ], v0_out.normal, v0_out.tex_coords, v0_out.color),
                    Vertex::new4([
                        v1_out.position[0], v1_out.position[1], v1_out.position[2], v1_out.position[3]
                    ], v1_out.normal, v1_out.tex_coords, v1_out.color),
                    Vertex::new4([
                        v2_out.position[0], v2_out.position[1], v2_out.position[2], v2_out.position[3]
                    ], v2_out.normal, v2_out.tex_coords, v2_out.color),
                ])
            })
            .collect();

        match self.render_mode {
            RenderMode::Wireframe => {
                self.rasterizer.raster(&triangles);
            }
            RenderMode::Color => {
                self.rasterizer.raster_color(shader, &triangles, width, height);
            }
            RenderMode::Texture => {
                self.rasterizer.raster_texture(shader, &triangles, width, height, tex_ref);
            }
        }
    }

    /// Dibuja texto 2D en coordenadas de pantalla con escala y rotación
    pub fn draw_text_transformed(
        &mut self, 
        x: i32, 
        y: i32, 
        text: &str, 
        font: &crate::renderer::font::Font, 
        color: u32,
        scale: f32,
        rotation: f32
    ) {
        self.rasterizer.draw_text_matrix_transform(x, y, text, font, color, scale, rotation);
    }
    
    /// Dibuja texto en una posición 3D con escala y rotación (el texto siempre estará orientado hacia la cámara)
    pub fn draw_text_3d_transformed(
        &mut self, 
        position: &[f32; 3], 
        text: &str, 
        font: &crate::renderer::font::Font, 
        color: u32,
        scale: f32,
        rotation: f32
    ) -> bool {
        // Crear la matriz de vista-proyección combinada
        self.update_matrix_cache();
        let view_proj = self.cached_mvp_matrix.clone();
        
        // Convertir la matriz a formato [[f32; 4]; 4]
        let mut view_proj_data = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                view_proj_data[i][j] = view_proj.get(i, j);
            }
        }
        
        // Delegamos al método del rasterizador
        self.rasterizer.draw_text_3d_transformed(position, &view_proj_data, text, font, color, scale, rotation)
    }
    
    /// Dibuja texto con posición, rotación y escala completas (usando un Transform)
    pub fn draw_text_with_transform_extended(
        &mut self, 
        text: &str, 
        font: &crate::renderer::font::Font, 
        transform: &Transform, 
        color: u32,
        text_scale: f32,
        text_rotation: f32
    ) -> bool {
        // Obtener la posición mundial desde la transformación
        let position = transform.position();
        let world_pos = [position.x(), position.y(), position.z()];
        
        // Usar el método extendido para dibujar el texto
        self.draw_text_3d_transformed(&world_pos, text, font, color, text_scale, text_rotation)
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
