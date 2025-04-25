use crate::renderer::texture::Texture;

pub struct Vertex {
    pub position: [f32; 4],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex {
    pub fn new(position: [f32; 3], normal: [f32; 3], tex_coords: [f32; 2], color: [f32; 4]) -> Self {
        Self {
            position: [position[0], position[1], position[2], 1.0],
            normal,
            tex_coords,
            color,
        }
    }
    // Nuevo constructor para [f32; 4]
    pub fn new4(position: [f32; 4], normal: [f32; 3], tex_coords: [f32; 2], color: [f32; 4]) -> Self {
        Self {
            position,
            normal,
            tex_coords,
            color,
        }
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub diffuse_texture: Option<Texture>,
    pub name: Option<String>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>, diffuse_texture: Option<Texture>, name: Option<String>) -> Self {
        Self { vertices, indices, diffuse_texture, name }
    }

    pub fn get_vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn get_index_count(&self) -> usize {
        self.indices.len()
    }

    pub fn get_diffuse_texture(&self) -> Option<&Texture> {
        self.diffuse_texture.as_ref()
    }

    pub fn set_diffuse_texture(&mut self, texture: Texture) {
        self.diffuse_texture = Some(texture);
    }

    pub fn clear_diffuse_texture(&mut self) {
        self.diffuse_texture = None;
    }

    pub fn has_diffuse_texture(&self) -> bool {
        self.diffuse_texture.is_some()
    }
}

pub mod obj_loader;
pub use obj_loader::load_obj;