use super::Shader;
use std::any::Any;
use crate::renderer::shader::{Vertex, VertexShader};

// Representa una fuente de luz en la escena
pub struct Light {
    direction: [f32; 3],
    position: [f32; 3],
    color: [f32; 3],
    intensity: f32,
    attenuation: [f32; 3],
    cast_shadows: bool,
    light_type: LightType,
    shadow_map: Option<ShadowMap>, // Shadow map propio de la luz
}

// Implementación por defecto para una luz
impl Light {
    // Crea una luz direccional (como el sol)
    pub fn directional(direction: [f32; 3], color: [f32; 3], intensity: f32, cast_shadows: bool) -> Self {
        Self {
            direction: Self::normalize_vec3(&direction),
            position: [0.0, 0.0, 0.0], // No relevante para luz direccional
            color,
            intensity,
            attenuation: [1.0, 0.0, 0.0], // Sin atenuación para luz direccional
            cast_shadows,
            light_type: LightType::Directional,
            shadow_map: None,
        }
    }

    // Crea una luz puntual (como una bombilla)
    pub fn point(position: [f32; 3], color: [f32; 3], intensity: f32, attenuation: [f32; 3], cast_shadows: bool) -> Self {
        Self {
            direction: [0.0, 0.0, 0.0], // No relevante para luz puntual
            position,
            color,
            intensity,
            attenuation,
            cast_shadows,
            light_type: LightType::Point,
            shadow_map: None,
        }
    }

    // Normaliza un vector 3D
    fn normalize_vec3(v: &[f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(0.0001);
        [v[0] / len, v[1] / len, v[2] / len]
    }

    pub fn set_shadow_map(&mut self, shadow_map: ShadowMap) {
        self.shadow_map = Some(shadow_map);
    }
    pub fn clear_shadow_map(&mut self) {
        self.shadow_map = None;
    }
}

// Tipos de luz soportados
pub enum LightType {
    Directional, // Luz direccional (sol)
    Point,       // Luz puntual (bombilla)
    Spot,        // Luz focal (linterna) - implementación futura
}

// Estructura para manejar el mapa de sombras
pub struct ShadowMap {
    pub data: Vec<f32>,           // Datos del mapa de profundidad
    pub width: usize,             // Ancho del mapa
    pub height: usize,            // Alto del mapa
    pub light_view_proj: [[f32; 4]; 4], // Matriz vista-proyección desde la luz
}

impl ShadowMap {
    pub fn new(width: usize, height: usize, light_view_proj: [[f32; 4]; 4]) -> Self {
        // Inicializar con valores de profundidad máxima (1.0)
        let data = vec![1.0; width * height];
        Self {
            data,
            width,
            height, 
            light_view_proj,
        }
    }

    // Obtiene el valor de profundidad en las coordenadas normalizadas (0-1)
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let u = (x.clamp(0.0, 1.0) * (self.width as f32 - 1.0)) as usize;
        let v = (y.clamp(0.0, 1.0) * (self.height as f32 - 1.0)) as usize;
        let idx = v * self.width + u;
        if idx < self.data.len() {
            self.data[idx]
        } else {
            1.0 // Fuera de rango, asumimos sin sombra
        }
    }
}

// Shader por defecto con soporte para múltiples luces y sombras
pub struct MultiShader {
    // Propiedades de iluminación
    lights: Vec<Light>,
    ambient_intensity: f32,
    shadow_bias: f32,
    specular_intensity: f32,
    shininess: f32,
}

impl MultiShader {
    pub fn new() -> Self {
        // Crear una luz direccional por defecto
        let default_light = Light::directional(
            [-0.5, -1.0, -0.5], // Dirección
            [1.0, 1.0, 1.0],    // Color blanco
            0.8,                // Intensidad
            true                // Genera sombras
        );

        Self {
            lights: vec![default_light],
            ambient_intensity: 0.2,
            shadow_bias: 0.005,
            specular_intensity: 0.3,
            shininess: 32.0,
        }
    }

    // Añade una nueva luz a la escena
    pub fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }

    // Calcula la iluminación según el modelo de Phong para múltiples luces
    fn calculate_lighting(&self, position: &[f32; 3], normal: &[f32; 3], view_dir: &[f32; 3]) -> [f32; 3] {
        // Componente ambiental global
        let mut result = [
            self.ambient_intensity,
            self.ambient_intensity,
            self.ambient_intensity,
        ];

        // Procesamos cada luz
        for light in &self.lights {
            let (light_dir, light_intensity) = match light.light_type {
                LightType::Directional => {
                    // Para luz direccional, usamos la dirección invertida y la intensidad sin atenuación
                    ([-light.direction[0], -light.direction[1], -light.direction[2]], light.intensity)
                },
                LightType::Point => {
                    // Para luz puntual, calculamos dirección desde el punto hacia la luz
                    let dx = light.position[0] - position[0];
                    let dy = light.position[1] - position[1];
                    let dz = light.position[2] - position[2];
                    
                    // Calculamos la distancia
                    let distance = (dx*dx + dy*dy + dz*dz).sqrt().max(0.001);
                    
                    // Normalizamos la dirección
                    let dir = [dx/distance, dy/distance, dz/distance];
                    
                    // Calculamos atenuación basada en la distancia
                    let attenuation = 1.0 / (
                        light.attenuation[0] +
                        light.attenuation[1] * distance +
                        light.attenuation[2] * distance * distance
                    );
                    
                    (dir, light.intensity * attenuation)
                },
                LightType::Spot => {
                    // No implementado completamente, tratamos como direccional
                    ([-light.direction[0], -light.direction[1], -light.direction[2]], light.intensity)
                }
            };

            // Calculamos el factor de sombra
            let shadow_factor = if light.cast_shadows && light.shadow_map.is_some() {
                self.calculate_shadow_factor(position, &light)
            } else {
                1.0 // Sin sombra
            };

            // Componente difusa: N·L
            let n_dot_l = normal[0] * light_dir[0] +
                         normal[1] * light_dir[1] +
                         normal[2] * light_dir[2];
            
            let diffuse_factor = n_dot_l.max(0.0) * light_intensity;
            
            // Calcular el vector de reflexión: R = 2(N·L)N - L
            let reflect_x = 2.0 * n_dot_l * normal[0] - light_dir[0];
            let reflect_y = 2.0 * n_dot_l * normal[1] - light_dir[1];
            let reflect_z = 2.0 * n_dot_l * normal[2] - light_dir[2];
            
            // Calcular el producto punto entre reflexión y vista
            let r_dot_v = (reflect_x * view_dir[0] +
                          reflect_y * view_dir[1] +
                          reflect_z * view_dir[2]).max(0.0);
            
            // Componente especular
            let specular_factor = r_dot_v.powf(self.shininess) * self.specular_intensity * light_intensity;
            
            // Factores afectados por sombras (difuso y especular)
            let shadowed_factor = (diffuse_factor + specular_factor) * shadow_factor;
            
            // Acumular la contribución de esta luz
            result[0] += light.color[0] * shadowed_factor;
            result[1] += light.color[1] * shadowed_factor;
            result[2] += light.color[2] * shadowed_factor;
        }
        
        // Limitar los resultados a 1.0
        [
            result[0].min(1.0),
            result[1].min(1.0),
            result[2].min(1.0),
        ]
    }
    
    // Calcula el factor de sombra para un punto y una luz específica
    fn calculate_shadow_factor(&self, position: &[f32; 3], light: &Light) -> f32 {
        // Si la luz no tiene shadow map, no hay sombra
        let shadow_map = match &light.shadow_map {
            Some(map) => map,
            None => return 1.0,
        };
        
        // Transformamos la posición al espacio de la luz
        let light_space_pos = self.transform_to_light_space(position, &shadow_map.light_view_proj);
        
        // Si el punto está fuera del frustum de la luz, no hay sombra
        if light_space_pos[0] < -1.0 || light_space_pos[0] > 1.0 ||
           light_space_pos[1] < -1.0 || light_space_pos[1] > 1.0 ||
           light_space_pos[2] < 0.0 || light_space_pos[2] > 1.0 {
            return 1.0;
        }
        
        // Convertimos a coordenadas UV para el shadow map
        let shadow_u = light_space_pos[0] * 0.5 + 0.5;
        let shadow_v = light_space_pos[1] * 0.5 + 0.5;
        
        // Obtenemos la profundidad almacenada en el shadow map
        let shadow_depth = shadow_map.sample(shadow_u, shadow_v);
        
        // La profundidad actual en el espacio de luz
        let current_depth = light_space_pos[2];
        
        // Comparamos con bias para evitar acné de sombra
        if current_depth - self.shadow_bias > shadow_depth {
            0.3 // En sombra (permitimos algo de luz ambiental)
        } else {
            1.0 // Sin sombra
        }
    }
    
    // Transforma una posición del mundo al espacio de coordenadas de la luz
    fn transform_to_light_space(&self, position: &[f32; 3], light_vp: &[[f32; 4]; 4]) -> [f32; 3] {
        // Posición en espacio homogéneo
        let position_h = [position[0], position[1], position[2], 1.0];
        
        // Transformamos con la matriz vista-proyección de la luz
        let mut result = [0.0, 0.0, 0.0, 0.0];
        for i in 0..4 {
            for j in 0..4 {
                result[i] += light_vp[i][j] * position_h[j];
            }
        }
        
        // Perspectiva dividida para obtener coordenadas normalizadas
        let w = result[3].max(0.0001);
        [result[0] / w, result[1] / w, result[2] / w]
    }
}

impl Shader for MultiShader {
    fn as_any(&self) -> &dyn Any { self }

    fn get_shader_name(&self) -> String {
        "MultiShader".to_string()
    }
    
    fn vertex_shader(&self, input: &Vertex) -> VertexShader {
        // Pasamos los datos necesarios al fragment shader

        // Normal - asegurémonos que está normalizada
        let normal_len = (
            input.normal[0] * input.normal[0] +
            input.normal[1] * input.normal[1] +
            input.normal[2] * input.normal[2]
        ).sqrt().max(0.001); // Evitar división por cero

        // Creamos el VertexShader con los datos necesarios
        VertexShader {
            position: [input.position[0], input.position[1], input.position[2], 1.0],
            normal: [input.normal[0] / normal_len, input.normal[1] / normal_len, input.normal[2] / normal_len],
            color: [input.color[0], input.color[1], input.color[2], input.color[3]],
            tex_coords: [input.tex_coords[0], input.tex_coords[1]],
            world_position: [input.position[0], input.position[1], input.position[2], 1.0], // Asignar la posición del vértice al espacio mundial
        }
    }
    
    fn fragment_shader(&self, input: &VertexShader) -> Vec<f32> {
        // Extraer datos del vertex shader
        let normal = [input.normal[0], input.normal[1], input.normal[2]];
        let vertex_color = [input.color[0], input.color[1], input.color[2], input.color[3]];
        let world_position = [input.world_position[0], input.world_position[1], input.world_position[2]];
        
        // Dirección de vista (desde punto hacia cámara)
        let view_dir = [0.0, 0.0, 1.0];
        
        // Calcular iluminación completa con todas las luces y sombras
        let lighting = self.calculate_lighting(&world_position, &normal, &view_dir);
        
        // Aplicar iluminación al color del vértice
        let final_color = [
            (vertex_color[0] * lighting[0]).min(1.0),
            (vertex_color[1] * lighting[1]).min(1.0),
            (vertex_color[2] * lighting[2]).min(1.0),
            vertex_color[3], // Mantener el alfa original
        ];
        
        // Devolver el color final
        vec![final_color[0], final_color[1], final_color[2], final_color[3]]
    }
    fn fragment_shader_tex(&self, input: &VertexShader, texture: Option<&crate::renderer::texture::Texture>, u: f32, v: f32) -> Vec<f32> {
        // Llamar al fragment shader normal
        let color = self.fragment_shader(input);
        
        // Si hay textura, mezclar con el color de la textura
        if let Some(tex) = texture {
            let tex_color = tex.sample(u, v);
            vec![
                color[0] * tex_color[0],
                color[1] * tex_color[1],
                color[2] * tex_color[2],
                color[3],
            ]
        } else {
            color
        }
    }
}