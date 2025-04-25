use crate::renderer::geometry::Vertex;
use std::any::Any;

pub struct VertexShader {
    pub position: [f32; 4],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub tex_coords: [f32; 2],
    pub world_position: [f32; 4], // Posición en el espacio mundial para cálculos de iluminación
}

impl VertexShader {
    pub fn new(position: [f32; 4], normal: [f32; 3], color: [f32; 4], tex_coords: [f32; 2], world_position: [f32; 4]) -> Self {
        Self {
            position,
            normal,
            color,
            tex_coords,
            world_position,
        }
    }

    pub fn from_vertex(vertex: &Vertex) -> Self {
        Self {
            position: vertex.position,
            normal: vertex.normal,
            color: vertex.color,
            tex_coords: vertex.tex_coords,
            world_position: vertex.position, // Asignar la posición del vértice al espacio mundial
        }
    }

    pub fn from_vertex_with_world_position(vertex: &Vertex, world_position: [f32; 4]) -> Self {
        Self {
            position: vertex.position,
            normal: vertex.normal,
            color: vertex.color,
            tex_coords: vertex.tex_coords,
            world_position,
        }
    }

    pub fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0, 1.0],
            normal: [0.0, 0.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            tex_coords: [0.0, 0.0],
            world_position: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

pub trait Shader {
    fn as_any(&self) -> &dyn Any;
    fn vertex_shader(&self, input: &Vertex) -> VertexShader;
    fn fragment_shader(&self, input: &VertexShader) -> Vec<f32>;
    fn fragment_shader_tex(&self, input: &VertexShader, _texture: Option<&crate::renderer::texture::Texture>, _u: f32, _v: f32) -> Vec<f32>;
    fn get_shader_name(&self) -> String;
}

pub mod default;
pub use default::DefaultShader;

pub mod simplelight;
pub use simplelight::SimpleShader;

pub mod multilight;
pub use multilight::MultiShader;
