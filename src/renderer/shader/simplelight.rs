use super::Shader;
use std::any::Any;
use crate::renderer::shader::{Vertex, VertexShader};

pub struct SimpleShader {
    pub light_dir: [f32; 3], // Dirección de la luz (debe estar normalizada)
    pub light_color: [f32; 3],
    pub ambient: [f32; 3],
    pub specular_strength: f32,
    pub shininess: f32,
    pub view_pos: [f32; 3],
}

impl SimpleShader {
    pub fn new(light_dir: [f32; 3], light_color: [f32; 3], ambient: [f32; 3], specular_strength: f32, shininess: f32, view_pos: [f32; 3]) -> Self {
        SimpleShader {
            light_dir,
            light_color,
            ambient,
            specular_strength,
            shininess,
            view_pos,
        }
    }
}
impl SimpleShader {
    pub fn new_default() -> Self {
        SimpleShader {
            light_dir: [0.0, 0.0, -1.0],
            light_color: [1.0, 1.0, 1.0],
            ambient: [0.2, 0.2, 0.2],
            specular_strength: 0.5,
            shininess: 32.0,
            view_pos: [0.0, 0.0, 3.0],
        }
    }
}

impl Shader for SimpleShader {
    fn as_any(&self) -> &dyn Any { self }

    fn get_shader_name(&self) -> String {
        "SimpleShader".to_string()
    }
    
    fn vertex_shader(&self, input: &Vertex) -> VertexShader {
        let normal_len = (
            input.normal[0] * input.normal[0] +
            input.normal[1] * input.normal[1] +
            input.normal[2] * input.normal[2]
        ).sqrt().max(0.001);

        VertexShader {
            position: [input.position[0], input.position[1], input.position[2], 1.0],
            normal: [input.normal[0] / normal_len, input.normal[1] / normal_len, input.normal[2] / normal_len],
            color: [input.color[0], input.color[1], input.color[2], input.color[3]],
            tex_coords: [input.tex_coords[0], input.tex_coords[1]],
            world_position: [input.position[0], input.position[1], input.position[2], 1.0],
        }
    }
    fn fragment_shader(&self, input: &VertexShader) -> Vec<f32> {
        let normal = [input.normal[0], input.normal[1], input.normal[2]];
        let vertex_color = [input.color[0], input.color[1], input.color[2], input.color[3]];
        let world_position = [input.world_position[0], input.world_position[1], input.world_position[2]];
        let mut norm = normal;
        let nlen = (norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]).sqrt().max(0.001);
        norm[0] /= nlen; norm[1] /= nlen; norm[2] /= nlen;
        let light_dir = self.light_dir;
        let diff = norm[0]*light_dir[0] + norm[1]*light_dir[1] + norm[2]*light_dir[2];
        let diff = diff.max(0.0);
        let view_dir = [
            self.view_pos[0] - world_position[0],
            self.view_pos[1] - world_position[1],
            self.view_pos[2] - world_position[2],
        ];
        let vlen = (view_dir[0]*view_dir[0] + view_dir[1]*view_dir[1] + view_dir[2]*view_dir[2]).sqrt().max(0.001);
        let view_dir = [view_dir[0]/vlen, view_dir[1]/vlen, view_dir[2]/vlen];
        let reflect_dir = [
            2.0*diff*norm[0] - light_dir[0],
            2.0*diff*norm[1] - light_dir[1],
            2.0*diff*norm[2] - light_dir[2],
        ];
        let spec = (view_dir[0]*reflect_dir[0] + view_dir[1]*reflect_dir[1] + view_dir[2]*reflect_dir[2]).max(0.0).powf(self.shininess) * self.specular_strength;
        let mut color = [0.0; 3];
        for i in 0..3 {
            color[i] = self.ambient[i] + diff * self.light_color[i] + spec * self.light_color[i];
            color[i] = color[i].min(1.0f32);
        }
        vec![color[0], color[1], color[2], vertex_color[3]]
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