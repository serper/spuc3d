use crate::renderer::geometry::Mesh;
use crate::renderer::texture::Texture;

#[no_mangle]
pub extern "C" fn create_mesh(vertices: *const f32, vertex_count: usize, indices: *const u32, index_count: usize) -> *mut Mesh {
    let vertices = unsafe { std::slice::from_raw_parts(vertices, vertex_count * 8) };
    let indices = unsafe { std::slice::from_raw_parts(indices, index_count) };

    let mut mesh_vertices = Vec::new();
    for i in 0..vertex_count {
        let position = [vertices[i * 8], vertices[i * 8 + 1], vertices[i * 8 + 2]];
        let normal = [vertices[i * 8 + 3], vertices[i * 8 + 4], vertices[i * 8 + 5]];
        let tex_coords = [vertices[i * 8 + 6], vertices[i * 8 + 7]];
        mesh_vertices.push(crate::renderer::geometry::Vertex::new(position, normal, tex_coords, [0.0, 0.0, 0.0, 1.0]));
    }

    let mesh = Mesh::new(mesh_vertices, indices.to_vec(), None);
    Box::into_raw(Box::new(mesh))
}

#[no_mangle]
pub extern "C" fn destroy_mesh(mesh: *mut Mesh) {
    if mesh.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(mesh));
    }
}

#[no_mangle]
pub extern "C" fn create_texture(width: u32, height: u32, data: *const u8) -> *mut Texture {
    let data = unsafe { std::slice::from_raw_parts(data, (width * height * 4) as usize) };
    let texture = Texture::new(width, height, data.to_vec());
    Box::into_raw(Box::new(texture))
}

#[no_mangle]
pub extern "C" fn destroy_texture(texture: *mut Texture) {
    if texture.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(texture));
    }
}