// --- SHADER SOLO AMBIENTAL ---
use super::Shader;
use std::any::Any;
use crate::renderer::shader::{Vertex, VertexShader};

pub struct DefaultShader {
}

impl DefaultShader {
    pub fn new() -> Self {
        DefaultShader {
        }
    }
}

impl Shader for DefaultShader {
    fn as_any(&self) -> &dyn Any { self }

    fn get_shader_name(&self) -> String {
        "DefaultShader".to_string()
    }
    
    fn vertex_shader(&self, input: &Vertex) -> VertexShader {
        VertexShader {
            position: [input.position[0], input.position[1], input.position[2], 1.0],
            normal: [input.normal[0], input.normal[1], input.normal[2]],
            color: [input.color[0], input.color[1], input.color[2], input.color[3]],
            tex_coords: [input.tex_coords[0], input.tex_coords[1]],
            world_position: [input.position[0], input.position[1], input.position[2], 1.0],
        }
    }
    fn fragment_shader(&self, input: &VertexShader) -> Vec<f32> {
        vec![
            input.color[0],
            input.color[1],
            input.color[2],
            input.color[3]
        ]
    }
    fn fragment_shader_tex(&self, input: &VertexShader, texture: Option<&crate::renderer::texture::Texture>, u: f32, v: f32) -> Vec<f32> {
        let mut color = [input.color[0], input.color[1], input.color[2]];
        if let Some(tex) = texture {
            let tex_color = tex.sample(u, v);
            color[0] *= tex_color[0];
            color[1] *= tex_color[1];
            color[2] *= tex_color[2];
        }
        vec![color[0], color[1], color[2], input.color[3]]
    }
}
