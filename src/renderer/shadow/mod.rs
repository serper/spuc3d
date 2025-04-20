use crate::renderer::core::{
    math::{Matrix, Vector},
    pipeline::Pipeline,
    rasterizer::Rasterizer,
};
use crate::renderer::geometry::Mesh;
use crate::renderer::shader::{Light, LightType, ShadowMap};

// Shader simplificado para renderizar mapas de profundidad
pub struct DepthShader {}

// Implementación del shader de profundidad
impl crate::renderer::shader::Shader for DepthShader {
    fn vertex_shader(&self, input: &crate::renderer::geometry::Vertex) -> Vec<f32> {
        // Solo pasamos la posición, ya que solo nos interesa la profundidad
        vec![
            input.position[0],
            input.position[1],
            input.position[2],
        ]
    }
    
    fn fragment_shader(&self, _input: &[f32]) -> Vec<f32> {
        // Para el shadow map, solo importa el valor de profundidad que se guarda automáticamente
        // No necesitamos calcular un color de salida
        vec![1.0, 1.0, 1.0, 1.0]
    }
}

pub struct ShadowRenderer {
    shadow_width: usize,
    shadow_height: usize,
    shadow_rasterizer: Rasterizer,
    depth_shader: DepthShader,
}

impl ShadowRenderer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            shadow_width: width,
            shadow_height: height,
            shadow_rasterizer: Rasterizer::new(width, height),
            depth_shader: DepthShader{},
        }
    }
    
    // Renderiza la escena desde la perspectiva de la luz para generar un mapa de sombras
    pub fn render_shadow_map(&mut self, meshes: &[&Mesh], light: &Light) -> ShadowMap {
        // Limpiar el buffer de profundidad con valores máximos
        self.shadow_rasterizer.clear(0, 1.0);
        
        // Crear un pipeline específico para el renderizado de sombras
        let mut shadow_pipeline = Pipeline::new(
            self.depth_shader, 
            &mut self.shadow_rasterizer
        );
        
        // Configurar la matriz de vista desde la perspectiva de la luz
        let (eye, target, up) = match light.light_type {
            LightType::Directional => {
                // Para luz direccional, la posición de la cámara está en dirección opuesta a la luz
                // a una distancia suficiente para cubrir la escena
                let light_dir = light.direction();
                let light_pos = [
                    -light_dir[0] * 20.0, 
                    -light_dir[1] * 20.0, 
                    -light_dir[2] * 20.0
                ];
                let target = [0.0, 0.0, 0.0]; // Mirando al centro de la escena
                let up = if light_dir[1].abs() > 0.99 {
                    // Si la luz viene directamente desde arriba o abajo, usar Z como up
                    [0.0, 0.0, 1.0]
                } else {
                    // En otro caso, usar Y como up
                    [0.0, 1.0, 0.0]
                };
                
                (light_pos, target, up)
            },
            LightType::Point => {
                // Para luz puntual, la posición de la cámara es la posición de la luz
                let light_pos = light.position();
                let target = [0.0, 0.0, 0.0]; // Mirando al centro de la escena
                let up = [0.0, 1.0, 0.0];
                
                (light_pos, target, up)
            },
            LightType::Spot => {
                // Para luz spot, la posición es la posición de la luz y la dirección es hacia donde apunta
                let light_pos = light.position();
                let light_dir = light.direction();
                let target = [
                    light_pos[0] + light_dir[0],
                    light_pos[1] + light_dir[1],
                    light_pos[2] + light_dir[2],
                ];
                let up = [0.0, 1.0, 0.0];
                
                (light_pos, target, up)
            },
        };
        
        // Configurar la vista desde la perspectiva de la luz
        shadow_pipeline.look_at(&eye, &target, &up);
        
        // Para luces direccionales, usar proyección ortográfica
        // Para luces puntuales y focales, usar proyección perspectiva
        match light.light_type {
            LightType::Directional => {
                // Proyección ortográfica (sin perspectiva)
                shadow_pipeline.set_orthographic(-10.0, 10.0, -10.0, 10.0, 1.0, 50.0);
            },
            _ => {
                // Proyección perspectiva para luces puntuales y spot
                shadow_pipeline.set_perspective(
                    std::f32::consts::FRAC_PI_2, // 90 grados FOV
                    1.0, // Aspect ratio cuadrado para shadow map
                    0.1, // Near plane
                    50.0, // Far plane
                );
            }
        }
        
        // Renderizar cada malla en la escena
        for mesh in meshes {
            shadow_pipeline.render(mesh, None); // No necesitamos textura para shadow map
        }
        
        // Obtener la matriz vista-proyección combinada
        let light_view = shadow_pipeline.get_view_matrix();
        let light_proj = shadow_pipeline.get_projection_matrix();
        let light_view_proj = light_proj.multiply(&light_view);
        
        // Crear y devolver el mapa de sombras usando el buffer de profundidad
        let depth_buffer = self.shadow_rasterizer.get_depth_buffer();
        let mut shadow_map = ShadowMap::new(
            self.shadow_width, 
            self.shadow_height, 
            *light_view_proj.elements()
        );
        
        // Copiar el buffer de profundidad
        shadow_map.data.copy_from_slice(depth_buffer);
        
        shadow_map
    }
}
