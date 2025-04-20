// Implementación de transformaciones 3D para el motor de renderizado
use super::math::{Matrix, Vector, Quaternion};
use std::rc::{Rc, Weak};
use std::cell::RefCell;

/// Estructura que representa una transformación en el espacio 3D
/// Incluye posición, rotación y escala
#[derive(Debug)]
pub struct Transform {
    // Componentes básicos de la transformación
    position: Vector,       // Posición en el espacio
    rotation: Quaternion,   // Rotación como cuaternión
    scale: Vector,          // Escala en cada eje

    // Matrices precalculadas para optimizar el rendimiento
    local_matrix: Matrix,   // Matriz de transformación local
    world_matrix: Matrix,   // Matriz de transformación mundial
    
    // Estado de las matrices
    is_local_matrix_dirty: bool,
    is_world_matrix_dirty: bool,
    
    // Referencia al padre (opcional)
    parent: Option<Weak<RefCell<Transform>>>,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vector::new(),
            rotation: Quaternion::new(),
            scale: Vector::new_with_values(1.0, 1.0, 1.0),
            local_matrix: Matrix::new(),
            world_matrix: Matrix::new(),
            is_local_matrix_dirty: true,
            is_world_matrix_dirty: true,
            parent: None,
        }
    }
}

impl Transform {
    /// Crea una nueva transformación con valores por defecto
    pub fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            position: Vector::new(),
            rotation: Quaternion::new(),
            scale: Vector::new_with_values(1.0, 1.0, 1.0),
            local_matrix: Matrix::new(),
            world_matrix: Matrix::new(),
            is_local_matrix_dirty: true,
            is_world_matrix_dirty: true,
            parent: None,
        }))
    }
    
    /// Obtiene la posición de la transformación
    #[inline]
    pub fn position(&self) -> &Vector {
        &self.position
    }
    
    /// Establece la posición de la transformación
    pub fn set_position(&mut self, position: Vector) -> &mut Self {
        self.position = position;
        self.mark_dirty();
        self
    }
    
    /// Establece la posición de la transformación a partir de coordenadas
    pub fn set_position_xyz(&mut self, x: f32, y: f32, z: f32) -> &mut Self {
        self.position = Vector::new_with_values(x, y, z);
        self.mark_dirty();
        self
    }
    
    /// Mueve la transformación sumando un vector a la posición actual
    pub fn translate(&mut self, translation: &Vector) -> &mut Self {
        let new_pos = self.position.add(translation);
        self.position = new_pos;
        self.mark_dirty();
        self
    }
    
    /// Mueve la transformación sumando coordenadas a la posición actual
    pub fn translate_xyz(&mut self, x: f32, y: f32, z: f32) -> &mut Self {
        let translation = Vector::new_with_values(x, y, z);
        self.translate(&translation)
    }
    
    /// Obtiene la rotación como cuaternión
    #[inline]
    pub fn rotation(&self) -> &Quaternion {
        &self.rotation
    }
    
    /// Establece la rotación utilizando un cuaternión
    pub fn set_rotation(&mut self, rotation: Quaternion) -> &mut Self {
        self.rotation = rotation;
        self.mark_dirty();
        self
    }
    
    /// Establece la rotación a partir de ángulos de Euler (en radianes)
    pub fn set_rotation_euler(&mut self, x: f32, y: f32, z: f32) -> &mut Self {
        self.rotation = Quaternion::from_euler(x, y, z);
        self.mark_dirty();
        self
    }
    
    /// Rota alrededor del eje X
    pub fn rotate_x(&mut self, angle: f32) -> &mut Self {
        let rotation = Quaternion::from_axis_angle(&[1.0, 0.0, 0.0], angle);
        self.rotation = rotation.multiply(&self.rotation);
        self.mark_dirty();
        self
    }
    
    /// Rota alrededor del eje Y
    pub fn rotate_y(&mut self, angle: f32) -> &mut Self {
        let rotation = Quaternion::from_axis_angle(&[0.0, 1.0, 0.0], angle);
        self.rotation = rotation.multiply(&self.rotation);
        self.mark_dirty();
        self
    }
    
    /// Rota alrededor del eje Z
    pub fn rotate_z(&mut self, angle: f32) -> &mut Self {
        let rotation = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], angle);
        self.rotation = rotation.multiply(&self.rotation);
        self.mark_dirty();
        self
    }
    
    /// Obtiene la escala de la transformación
    #[inline]
    pub fn scale(&self) -> &Vector {
        &self.scale
    }
    
    /// Establece la escala de la transformación
    pub fn set_scale(&mut self, scale: Vector) -> &mut Self {
        self.scale = scale;
        self.mark_dirty();
        self
    }
    
    /// Establece la escala utilizando valores individuales para cada eje
    pub fn set_scale_xyz(&mut self, x: f32, y: f32, z: f32) -> &mut Self {
        self.scale = Vector::new_with_values(x, y, z);
        self.mark_dirty();
        self
    }
    
    /// Establece una escala uniforme para todos los ejes
    pub fn set_uniform_scale(&mut self, scale: f32) -> &mut Self {
        self.scale = Vector::new_with_values(scale, scale, scale);
        self.mark_dirty();
        self
    }
    
    /// Obtiene la matriz de transformación local
    pub fn local_matrix(&mut self) -> &Matrix {
        if self.is_local_matrix_dirty {
            self.update_local_matrix();
        }
        &self.local_matrix
    }
    
    /// Obtiene la matriz de transformación mundial
    pub fn world_matrix(&mut self) -> &Matrix {
        if self.is_world_matrix_dirty {
            self.update_world_matrix();
        }
        &self.world_matrix
    }
    
    /// Establece el padre de esta transformación
    pub fn set_parent(this: &Rc<RefCell<Self>>, parent: Option<&Rc<RefCell<Transform>>>) {
        let mut this_mut = this.borrow_mut();
        this_mut.parent = parent.map(|p| Rc::downgrade(p));
        this_mut.mark_dirty();
    }
    
    /// Marcar las matrices como sucias para ser recalculadas
    fn mark_dirty(&mut self) {
        self.is_local_matrix_dirty = true;
        self.is_world_matrix_dirty = true;
    }
    
    /// Actualiza la matriz local
    fn update_local_matrix(&mut self) {
        // Crear matriz de escala
        let mut scale_matrix = Matrix::new();
        scale_matrix.scale(self.scale.x(), self.scale.y(), self.scale.z());
        
        // Crear matriz de rotación a partir del cuaternión
        let rotation_matrix = self.rotation.to_rotation_matrix();
        
        // Crear matriz de traslación
        let mut translation_matrix = Matrix::new();
        translation_matrix.translate(self.position.x(), self.position.y(), self.position.z());
        
        // Combinar las matrices: Translation * Rotation * Scale
        let temp = Matrix::new_with_values(rotation_matrix).multiply(&Matrix::new_with_values(*scale_matrix.elements()));
        self.local_matrix = translation_matrix.multiply(&temp);
        
        self.is_local_matrix_dirty = false;
    }
    
    /// Actualiza la matriz mundial
    fn update_world_matrix(&mut self) {
        if self.is_local_matrix_dirty {
            self.update_local_matrix();
        }
        if let Some(ref weak_parent) = self.parent {
            if let Some(parent_rc) = weak_parent.upgrade() {
                let mut parent = parent_rc.borrow_mut();
                if parent.is_world_matrix_dirty {
                    parent.update_world_matrix();
                }
                if self.is_local_matrix_dirty {
                    self.update_local_matrix();
                }
                let result_matrix = parent.world_matrix.multiply(&self.local_matrix);
                let result_elements = result_matrix.elements();
                let world_elements = self.world_matrix.elements_mut();
                for i in 0..4 {
                    for j in 0..4 {
                        world_elements[i][j] = result_elements[i][j];
                    }
                }
            } else {
                // Si el padre ya no existe, usar la local
                let local_elements = self.local_matrix.elements();
                let world_elements = self.world_matrix.elements_mut();
                for i in 0..4 {
                    for j in 0..4 {
                        world_elements[i][j] = local_elements[i][j];
                    }
                }
            }
        } else {
            let local_elements = self.local_matrix.elements();
            let world_elements = self.world_matrix.elements_mut();
            for i in 0..4 {
                for j in 0..4 {
                    world_elements[i][j] = local_elements[i][j];
                }
            }
        }
        self.is_world_matrix_dirty = false;
    }
    
    /// Aplica esta transformación a un vector
    pub fn transform_point(&mut self, point: &Vector) -> Vector {
        let matrix = self.world_matrix();
        
        // Convertir el punto a coordenadas homogéneas
        let homogeneous = [point.x(), point.y(), point.z(), 1.0];
        
        // Multiplicar el punto por la matriz
        let mut result = [0.0, 0.0, 0.0, 0.0];
        for i in 0..4 {
            for j in 0..4 {
                result[i] += matrix.elements()[i][j] * homogeneous[j];
            }
        }
        
        // Convertir de nuevo a coordenadas cartesianas
        if result[3] != 0.0 && result[3] != 1.0 {
            Vector::new_with_values(
                result[0] / result[3],
                result[1] / result[3],
                result[2] / result[3],
            )
        } else {
            Vector::new_with_values(result[0], result[1], result[2])
        }
    }
    
    /// Aplica solo la rotación de esta transformación a un vector
    pub fn transform_direction(&mut self, direction: &Vector) -> Vector {
        let matrix = self.world_matrix();
        
        // Para vectores de dirección, ignoramos la traslación
        let homogeneous = [direction.x(), direction.y(), direction.z(), 0.0];
        
        // Multiplicar la dirección por la matriz
        let mut result = [0.0, 0.0, 0.0, 0.0];
        for i in 0..4 {
            for j in 0..4 {
                result[i] += matrix.elements()[i][j] * homogeneous[j];
            }
        }
        
        Vector::new_with_values(result[0], result[1], result[2])
    }
    
    /// Convierte la transformación a una cadena legible
    pub fn to_string(&self) -> String {
        format!(
            "Transform {{ position: {:?}, rotation: {:?}, scale: {:?} }}",
            self.position.to_array(),
            self.rotation.to_tuple(),
            self.scale.to_array()
        )
    }
}
