use super::transform::Transform;
use super::math::{Vector, Matrix, Quaternion};
use super::rasterizer::Rasterizer;
use crate::renderer::geometry::{Mesh, Vertex};
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
    
    // Modo de transformación seleccionado
    transformation_mode: TransformationMode,
    
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
        let view_matrix = Matrix::new().look_at(
            &[0.0, 0.0, 5.0],  // posición de la cámara
            &[0.0, 0.0, 0.0],  // punto al que mira la cámara
            &[0.0, 1.0, 0.0]   // vector "arriba" de la cámara
        );
        
        // Matriz de proyección perspectiva con parámetros razonables
        let aspect_ratio = rasterizer.width as f32 / rasterizer.height as f32;
        
        // Verificar si la cámara está mirando en dirección -Z
        // En este caso sabemos que sí porque eye.z = 5.0 y target.z = 0.0
        let looking_along_negative_z = true;
        
        // Ajustar near y far según la dirección de la cámara
        let near = 0.1;
        let far = 100.0;
        
        if cfg!(debug_assertions) {
            println!("Configurando proyección inicial: FOV=45°, Aspect={:.2}, Near={:.2}, Far={:.2}", 
                     aspect_ratio, near, far);
        }
        
        // Crear matriz de proyección ajustada a la dirección de la cámara
        let projection_matrix = Matrix::new().perspective(
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
            
            // Campos para transformaciones basadas en quaternions
            model_position,
            model_rotation,
            model_scale,
            view_position,
            view_rotation,
            
            // Por defecto, usar el enfoque basado en matrices
            transformation_mode: TransformationMode::Matrix,
            
            // OPTIMIZACIÓN: Inicializar caché
            cached_model_view_matrix,
            cached_mvp_matrix,
            cache_valid: true,
            
            // OPTIMIZACIÓN: Valores precalculados para quaternions
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
    
    pub fn get_color_buffer(&self) -> &[u32] {
        &self.rasterizer.color_buffer
    }

    pub fn clear_rasterizer(&mut self, color: u32, depth: f32) {
        self.rasterizer.clear(color, depth);
    }
    
    /// Establece la matriz de modelo (transforma vértices del espacio local al mundial)
    pub fn set_model_matrix(&mut self, matrix: Matrix) {
        self.model_matrix = matrix;
        
        // OPTIMIZACIÓN: Invalidar caché de matrices
        self.invalidate_matrix_cache();
        
        // Si estamos en modo quaternion, extraemos los componentes de la matriz
        if matches!(self.transformation_mode, TransformationMode::Quaternion) {
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
    }
    
    /// Establece la matriz de modelo a partir de una transformación
    pub fn set_model_transform(&mut self, transform: &mut Transform) {
        // En lugar de usar clone(), copiar los elementos de la matriz manualmente
        let world_matrix = transform.world_matrix();
        let world_elements = world_matrix.elements();
        
        // Crear una nueva matriz con los mismos elementos
        self.model_matrix = Matrix::new_with_values(*world_elements);
        
        // OPTIMIZACIÓN: Invalidar caché de matrices
        self.invalidate_matrix_cache();
        
        // Si estamos usando quaternions, también actualizamos esos componentes
        if matches!(self.transformation_mode, TransformationMode::Quaternion) {
            self.model_position = transform.position().clone();
            self.model_rotation = transform.rotation().clone();
            
            // OPTIMIZACIÓN: Invalidar precálculos de quaternion
            self.invalidate_quaternion_precalc();
            
            self.model_scale = transform.scale().clone();
        }
        
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
        if matches!(self.transformation_mode, TransformationMode::Quaternion) {
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
    }
    
    /// Configura la matriz de vista a partir de posición, objetivo y vector arriba
    pub fn look_at(&mut self, eye: &[f32; 3], target: &[f32; 3], up: &[f32; 3]) {
        // Imprimir parámetros para debug
        if cfg!(debug_assertions) {
            println!("look_at: eye={:?}, target={:?}, up={:?}", eye, target, up);
        }
        
        let view_matrix = Matrix::new().look_at(eye, target, up);
        self.view_matrix = view_matrix;
        
        // OPTIMIZACIÓN: Invalidar caché de matrices
        self.invalidate_matrix_cache();
        
        // Actualizar también la posición y rotación de la vista para el modo quaternion
        if matches!(self.transformation_mode, TransformationMode::Quaternion) {
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
        
        let projection = Matrix::new().perspective(fov_radians, aspect_ratio, near, far);
        
        self.projection_matrix = projection;
        
        // OPTIMIZACIÓN: Invalidar caché de matrices
        self.invalidate_matrix_cache();

        if cfg!(debug_assertions) {
            println!("Matriz de proyección configurada: {:?}", self.projection_matrix);
        }
    }

    /// Configura una matriz de proyección ortográfica
    pub fn set_orthographic(&mut self, left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) {
        let projection = Matrix::new().orthographic(left, right, bottom, top, near, far);
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
    
    /// Establece el modo de transformación a utilizar
    pub fn set_transformation_mode(&mut self, mode: TransformationMode) {
        self.transformation_mode = mode;
        println!("Modo de transformación cambiado a: {:?}", self.transformation_mode);
    }
    
    /// Obtiene el modo de transformación actual
    pub fn transformation_mode(&self) -> &TransformationMode {
        &self.transformation_mode
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
        let rotation_matrix = self.model_rotation.to_rotation_matrix();
        
        // Aplicar escala a los vectores columna
        let mut model_elements = rotation_matrix;
        for i in 0..3 {
            model_elements[0][i] *= self.model_scale.x();
            model_elements[1][i] *= self.model_scale.y();
            model_elements[2][i] *= self.model_scale.z();
        }
        
        // Aplicar traslación
        model_elements[0][3] = self.model_position.x();
        model_elements[1][3] = self.model_position.y();
        model_elements[2][3] = self.model_position.z();
        
        // Actualizar la matriz de modelo
        self.model_matrix = Matrix::new_with_values(model_elements);
        
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
        let rotation_matrix = self.view_rotation.to_rotation_matrix();
        
        // Crear una matriz de traslación para la posición de la cámara
        let translation_matrix = [
            [1.0, 0.0, 0.0, self.view_position.x()],
            [0.0, 1.0, 0.0, self.view_position.y()],
            [0.0, 0.0, 1.0, self.view_position.z()],
            [0.0, 0.0, 0.0, 1.0]
        ];
        
        // La matriz de cámara en el mundo sería rotación * traslación
        // Pero necesitamos la inversa para la vista
        
        // Primero, la parte de rotación invertida es la transpuesta
        let rotation_inverse = [
            [rotation_matrix[0][0], rotation_matrix[1][0], rotation_matrix[2][0], 0.0],
            [rotation_matrix[0][1], rotation_matrix[1][1], rotation_matrix[2][1], 0.0],
            [rotation_matrix[0][2], rotation_matrix[1][2], rotation_matrix[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        
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
    
    /// Configura el modelo a partir de una transformación, usando el modo de transformación actual
    pub fn set_model_transform_with_mode(&mut self, transform: &mut Transform) {
        match self.transformation_mode {
            TransformationMode::Matrix => {
                // Usar la versión original basada en matrices
                self.set_model_transform(transform);
            },
            TransformationMode::Quaternion => {
                // Usar la versión basada en quaternions
                self.model_position = transform.position().clone();
                self.model_rotation = transform.rotation().clone();
                
                // OPTIMIZACIÓN: Invalidar precálculos de quaternion
                self.invalidate_quaternion_precalc();
                
                self.model_scale = transform.scale().clone();
                
                // También actualizamos la matriz por compatibilidad
                self.set_model_transform(transform);
                
                if cfg!(debug_assertions) {
                    println!("Transformación con quaternions aplicada: Pos={:?}, Rot={:?}, Scale={:?}", 
                             self.model_position, self.model_rotation, self.model_scale);
                }
            }
        }
    }
    
    // OPTIMIZACIÓN: Métodos para obtener las matrices (útiles para el renderizado de sombras)
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
        let world_pos: [f32; 4];
        
        // OPTIMIZACIÓN: Asegurar que la caché de matrices está actualizada
        self.update_matrix_cache();
        
        // Aplicar transformación de modelo según el modo seleccionado
        match self.transformation_mode {
            TransformationMode::Matrix => {
                // Método tradicional basado en matrices
                let vertex_position = [vertex.position[0], vertex.position[1], vertex.position[2], 1.0];
                
                // Transformación al espacio de mundo
                world_pos = self.transform_point(&vertex_position, &self.model_matrix);
                
                // OPTIMIZACIÓN: Transformar directamente al espacio de clip usando la matriz MVP
                let clip_pos = self.transform_point(&vertex_position, &self.cached_mvp_matrix);
                
                // División de perspectiva para obtener NDC
                let w = clip_pos[3];
                
                // Manejar w cercanos a cero o negativos
                if w.abs() < 0.001 {
                    // println!("W muy pequeño: {}", w);
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
                
                return VertexOutput {
                    position: [screen_x, screen_y, screen_z, w],
                    normal: vertex.normal,
                    color: vertex.color,
                    tex_coords: vertex.tex_coords,
                    world_position: [world_pos[0], world_pos[1], world_pos[2]],
                };
            },
            TransformationMode::Quaternion => {
                // Método basado en quaternions para mayor eficiencia en rotaciones
                // OPTIMIZACIÓN: Asegurar que los precálculos están actualizados
                self.update_quaternion_precalc();
                
                // Transformación al espacio de mundo con quaternions optimizados
                world_pos = self.transform_point_quaternion_based_optimized(&vertex.position);
                
                // Ahora usar la matriz de vista-proyección para transformar al espacio de clip
                let view_proj_matrix = self.projection_matrix.multiply(&self.view_matrix);
                let clip_pos = self.transform_point(&world_pos, &view_proj_matrix);
                
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
                
                return VertexOutput {
                    position: [screen_x, screen_y, screen_z, w],
                    normal: vertex.normal,
                    color: vertex.color,
                    tex_coords: vertex.tex_coords,
                    world_position: [world_pos[0], world_pos[1], world_pos[2]],
                };
            }
        }
    }

    // OPTIMIZACIÓN: Versión optimizada de transform_point_quaternion_based que usa los valores precalculados
    fn transform_point_quaternion_based_optimized(&self, position: &[f32; 3]) -> [f32; 4] {
        // 1. Escalar el punto primero
        let scaled_x = position[0] * self.model_scale.x();
        let scaled_y = position[1] * self.model_scale.y();
        let scaled_z = position[2] * self.model_scale.z();

        // 2. Crear un quaternión puro para el punto escalado
        let point_quaternion = Quaternion::new_with_values(0.0, scaled_x, scaled_y, scaled_z);

        // 3. Usar los quaternions normalizados y conjugados precalculados
        let rotated_quaternion = self.model_rotation_normalized
            .multiply(&point_quaternion)
            .multiply(&self.model_rotation_conjugate);

        // 4. Extraer el punto rotado
        let quaternion_array = rotated_quaternion.to_array();

        // 5. Aplicar traslación
        [
            quaternion_array[1] + self.model_position.x(),
            quaternion_array[2] + self.model_position.y(),
            quaternion_array[3] + self.model_position.z(),
            1.0
        ]
    }

    // Método original para mantener compatibilidad
    fn transform_point_quaternion_based(&self, position: &[f32; 3]) -> [f32; 4] {
        // 1. Escalar el punto primero
        let scaled_x = position[0] * self.model_scale.x();
        let scaled_y = position[1] * self.model_scale.y();
        let scaled_z = position[2] * self.model_scale.z();

        // 2. Crear un quaternión puro para el punto escalado
        let point_quaternion = Quaternion::new_with_values(0.0, scaled_x, scaled_y, scaled_z);

        // 3. Asegurar que el quaternión de rotación esté normalizado
        let rotation_normalized = self.model_rotation.normalize();

        // 4. Rotar el punto escalado
        let rotated_quaternion = rotation_normalized
            .multiply(&point_quaternion)
            .multiply(&rotation_normalized.conjugate());

        // 5. Extraer el punto rotado
        let quaternion_array = rotated_quaternion.to_array();

        // 6. Aplicar traslación
        [
            quaternion_array[1] + self.model_position.x(),
            quaternion_array[2] + self.model_position.y(),
            quaternion_array[3] + self.model_position.z(),
            1.0
        ]
    }
    
    // Método auxiliar para transformar un punto mediante una matriz
    fn transform_point(&self, point: &[f32; 4], matrix: &Matrix) -> [f32; 4] {
        let mut result = [0.0; 4];
        
        // Multiplicar el punto por la matriz
        for i in 0..4 {
            for j in 0..4 {
                result[i] += matrix.elements()[i][j] * point[j];
            }
        }
        
        result
    }

    /// Determina si un triángulo debe ser descartado según su orientación (backface culling)
    pub fn should_cull_triangle(&self, v0: &VertexOutput, v1: &VertexOutput, v2: &VertexOutput) -> bool {
        // Calcular los vectores del triángulo en espacio de pantalla
        let edge1 = [v1.position[0] - v0.position[0], v1.position[1] - v0.position[1]];
        let edge2 = [v2.position[0] - v0.position[0], v2.position[1] - v0.position[1]];
        
        // Calcular el producto cruz 2D (componente z) para determinar la orientación
        let cross_product = edge1[0] * edge2[1] - edge1[1] * edge2[0];
        
        // En un sistema OpenGL, los triángulos en sentido antihorario son frontales
        // Si el producto cruz es negativo, el triángulo está orientado en sentido horario (trasero)
        cross_product > 0.0
    }

    pub fn render(&mut self, mesh: &Mesh, texture: Option<&Texture>) {
        // OPTIMIZACIÓN: Asegurar que las cachés están actualizadas antes de renderizar
        self.update_matrix_cache();
        if matches!(self.transformation_mode, TransformationMode::Quaternion) {
            self.update_quaternion_precalc();
        }
        
        // Procesar cada triángulo
        for chunk in mesh.indices.chunks(3) {
            if chunk.len() < 3 {
                if cfg!(debug_assertions) {
                    println!("Índice fuera de rango: {:?}", chunk);
                }
                continue;
            }
            
            // Obtener índices del triángulo
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            
            // Verificar que los índices son válidos
            if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
                if cfg!(debug_assertions) {
                    println!("Índice fuera de rango: {:?}", chunk);
                }
                continue;
            }
            
            // Obtener los vértices
            let v0 = &mesh.vertices[i0];
            let v1 = &mesh.vertices[i1];
            let v2 = &mesh.vertices[i2];
            
            // OPTIMIZACIÓN: Transformar los vértices utilizando el proceso optimizado
            let v0_output = self.process_vertex(v0);
            let v1_output = self.process_vertex(v1);
            let v2_output = self.process_vertex(v2);
            
            if self.should_cull_triangle(&v0_output, &v1_output, &v2_output) {
                continue;
            }
            
            // Crear vértices para el rasterizador
            let v0_vertex = Vertex {
                position: [v0_output.position[0], v0_output.position[1], v0_output.position[2]],
                normal: [v0_output.normal[0], v0_output.normal[1], v0_output.normal[2]],
                tex_coords: v0_output.tex_coords,
                color: v0_output.color,
            };
            
            let v1_vertex = Vertex {
                position: [v1_output.position[0], v1_output.position[1], v1_output.position[2]],
                normal: [v1_output.normal[0], v1_output.normal[1], v1_output.normal[2]],
                tex_coords: v1_output.tex_coords,
                color: v1_output.color,
            };
            
            let v2_vertex = Vertex {
                position: [v2_output.position[0], v2_output.position[1], v2_output.position[2]],
                normal: [v2_output.normal[0], v2_output.normal[1], v2_output.normal[2]],
                tex_coords: v2_output.tex_coords,
                color: v2_output.color,
            };
            
            // Si hay una textura, pasar esta información al rasterizador
            if let Some(tex) = texture.or_else(|| mesh.diffuse_texture.as_ref()) {
                self.rasterizer.draw_textured_triangle(&v0_vertex, &v1_vertex, &v2_vertex, tex);
            } else {
                self.rasterizer.draw_triangle(&v0_vertex, &v1_vertex, &v2_vertex);
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

/// Define el modo de transformación a utilizar en el pipeline
#[derive(Debug)]
pub enum TransformationMode {
    /// Utiliza matrices 4x4 para todas las transformaciones
    Matrix,
    /// Utiliza quaternions para rotaciones y vectores para traslación y escala
    Quaternion,
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
