use crate::renderer::geometry::Mesh;
use crate::renderer::texture::Texture;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct Material {
    pub name: String,
    pub diffuse: [f32; 4],
    pub diffuse_map: Option<String>, // Ruta a la textura difusa
}

fn parse_mtl(path: &str) -> HashMap<String, Material> {
    let mut materials = HashMap::new();
    let Ok(file) = fs::File::open(path) else { return materials; };
    let reader = BufReader::new(file);
    let mut current: Option<Material> = None;
    for line in reader.lines().flatten() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() { continue; }
        match parts[0] {
            "newmtl" => {
                if let Some(mat) = current.take() {
                    materials.insert(mat.name.clone(), mat);
                }
                current = Some(Material { name: parts[1].to_string(), ..Default::default() });
            },
            "Kd" => {
                if let Some(mat) = current.as_mut() {
                    if parts.len() >= 4 {
                        mat.diffuse = [
                            parts[1].parse().unwrap_or(1.0),
                            parts[2].parse().unwrap_or(1.0),
                            parts[3].parse().unwrap_or(1.0),
                            1.0
                        ];
                    }
                }
            },
            "map_Kd" | "map_Ka" => {
                if let Some(mat) = current.as_mut() {
                    if parts.len() >= 2 {
                        // Normaliza la ruta a barras normales
                        let tex_path = parts[1].replace('\\', "/");
                        mat.diffuse_map = Some(tex_path);
                    }
                }
            },
            _ => {}
        }
    }
    if let Some(mat) = current { materials.insert(mat.name.clone(), mat); }
    materials
}

/// Estructura para soportar múltiples meshes y texturas
pub struct Scene {
    pub meshes: Vec<Mesh>,
}

/// Carga un modelo OBJ con soporte para múltiples objetos y texturas
pub fn load_obj(path: &str, flip_v: bool, flip_h: bool) -> Result<Scene, String> {
    use crate::renderer::geometry::Vertex;
    let file = File::open(path).map_err(|e| format!("No se pudo abrir el archivo: {}", e))?;
    let reader = BufReader::new(file);
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut texcoords: Vec<[f32; 2]> = Vec::new();
    let mut materials: HashMap<String, Material> = HashMap::new();
    let obj_dir = Path::new(path).parent().map(|p| p.to_path_buf());

    let mut meshes: Vec<Mesh> = Vec::new();
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut vertex_map = std::collections::HashMap::new();
    let mut current_material: Option<String> = None;
    let mut current_texture: Option<Texture> = None;
    let mut current_name: Option<String> = None;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Error de lectura: {}", e))?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() { continue; }
        match parts[0] {
            "mtllib" => {
                if let Some(dir) = &obj_dir {
                    let mtl_path = dir.join(parts[1]);
                    let mtl_path_str = mtl_path.to_string_lossy();
                    materials = parse_mtl(&mtl_path_str);
                }
            },
            "usemtl" => {
                // Si ya hay datos, guardar el mesh anterior
                if !vertices.is_empty() && !indices.is_empty() {
                    println!("[OBJ] Nuevo mesh: {} vértices, {} índices, textura: {:?}", vertices.len(), indices.len(), current_texture.as_ref().map(|t| format!("{}x{}", t.width, t.height)));
                    meshes.push(Mesh {
                        vertices: std::mem::take(&mut vertices),
                        indices: std::mem::take(&mut indices),
                        diffuse_texture: current_texture.clone(),
                        name: current_name.clone(),
                    });
                    vertex_map.clear();
                }
                current_material = Some(parts[1].to_string());
                // Buscar la textura asociada
                current_texture = None;
                if let Some(mat) = materials.get(&parts[1].to_string()) {
                    if let Some(tex_name) = &mat.diffuse_map {
                        if let Some(dir) = &obj_dir {
                            let tex_path = dir.join(tex_name);
                            if tex_path.extension().map(|e| e == "dds").unwrap_or(false) {
                                current_texture = load_dds_texture(&tex_path, flip_v, flip_h);
                            } else if tex_path.extension().map(|e| e == "png").unwrap_or(false) {
                                if let Ok(img) = image::open(&tex_path) {
                                    let rgba = img.to_rgba8();
                                    let (w, h) = rgba.dimensions();
                                    let data = rgba.into_raw();
                                    current_texture = Some(Texture::new_with_flip(w, h, data, flip_v, flip_h));
                                }
                            } else if let Ok(img) = image::open(&tex_path) {
                                let rgba = img.to_rgba8();
                                let (w, h) = rgba.dimensions();
                                let data = rgba.into_raw();
                                current_texture = Some(Texture::new(w, h, data));
                            }
                        }
                    }
                }
                println!("[OBJ] usemtl: {} => textura: {:?}", parts[1], current_texture.as_ref().map(|t| format!("{}x{}", t.width, t.height)));
            },
            "o" | "g" => {
                // Nuevo objeto/grupo: guardar el mesh anterior si hay datos
                if !vertices.is_empty() && !indices.is_empty() {
                    meshes.push(Mesh {
                        vertices: std::mem::take(&mut vertices),
                        indices: std::mem::take(&mut indices),
                        diffuse_texture: current_texture.clone(),
                        name: current_name.clone(),
                    });
                    vertex_map.clear();
                }
                current_name = Some(parts[1].to_string());
            },
            "v" => {
                if parts.len() >= 4 {
                    let x = parts[1].parse().unwrap_or(0.0);
                    let y = parts[2].parse().unwrap_or(0.0);
                    let z = parts[3].parse().unwrap_or(0.0);
                    positions.push([x, y, z]);
                }
            },
            "vn" => {
                if parts.len() >= 4 {
                    let x = parts[1].parse().unwrap_or(0.0);
                    let y = parts[2].parse().unwrap_or(0.0);
                    let z = parts[3].parse().unwrap_or(0.0);
                    normals.push([x, y, z]);
                }
            },
            "vt" => {
                if parts.len() >= 3 {
                    let u = parts[1].parse().unwrap_or(0.0);
                    let v = parts[2].parse().unwrap_or(0.0);
                    texcoords.push([u, v]);
                }
            },
            "f" => {
                if parts.len() >= 4 {
                    let mut v_idx = [0; 3];
                    let mut vt_idx = [0; 3];
                    let mut vn_idx = [0; 3];
                    for i in 0..3 {
                        let idx = parts[i + 1];
                        let mut split = idx.split('/');
                        v_idx[i] = split.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                        vt_idx[i] = split.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                        vn_idx[i] = split.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                    }
                    for i in 0..3 {
                        let key = (v_idx[i], vt_idx[i], vn_idx[i], current_material.clone());
                        let vert_index = if let Some(&idx) = vertex_map.get(&key) {
                            idx
                        } else {
                            let pos = positions.get(v_idx[i].wrapping_sub(1)).copied().unwrap_or([0.0,0.0,0.0]);
                            let norm = normals.get(vn_idx[i].wrapping_sub(1)).copied().unwrap_or([0.0,0.0,1.0]);
                            let tex = texcoords.get(vt_idx[i].wrapping_sub(1)).copied().unwrap_or([0.0,0.0]);
                            let color = current_material.as_ref()
                                .and_then(|mat| materials.get(mat))
                                .map(|mat| mat.diffuse)
                                .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                            // Al cargar vértices desde OBJ, asegúrate de que position es [f32; 4] (w=1.0)
                            let vertex = Vertex::new4([
                                pos[0],
                                pos[1],
                                pos[2],
                                1.0
                            ], norm, tex, color);
                            vertices.push(vertex);
                            let idx = (vertices.len() - 1) as u32;
                            vertex_map.insert(key, idx);
                            idx
                        };
                        indices.push(vert_index);
                    }
                }
            },
            _ => {}
        }
    }
    // Guardar el último mesh
    if !vertices.is_empty() && !indices.is_empty() {
        println!("[OBJ] Último mesh: {} vértices, {} índices, textura: {:?}", vertices.len(), indices.len(), current_texture.as_ref().map(|t| format!("{}x{}", t.width, t.height)));
        meshes.push(Mesh {
            vertices,
            indices,
            diffuse_texture: current_texture,
            name: current_name.clone(),
        });
    }
    // Si no hay normales, generarlas por triángulo y asignarlas a los vértices
    if normals.is_empty() {
        normals = vec![[0.0, 0.0, 0.0]; positions.len()];
        let mut counts = vec![0u32; positions.len()];
        for mesh in &meshes {
            for tri in mesh.indices.chunks(3) {
                if tri.len() < 3 { continue; }
                let i0 = tri[0] as usize;
                let i1 = tri[1] as usize;
                let i2 = tri[2] as usize;
                let v0 = mesh.vertices[i0].position;
                let v1 = mesh.vertices[i1].position;
                let v2 = mesh.vertices[i2].position;
                let u = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                let v = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
                let n = [
                    u[1]*v[2] - u[2]*v[1],
                    u[2]*v[0] - u[0]*v[2],
                    u[0]*v[1] - u[1]*v[0],
                ];
                for &idx in &[i0, i1, i2] {
                    normals[idx][0] += n[0];
                    normals[idx][1] += n[1];
                    normals[idx][2] += n[2];
                    counts[idx] += 1;
                }
            }
        }
        // Normalizar
        for (n, c) in normals.iter_mut().zip(counts.iter()) {
            if *c > 0 {
                let len = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
                if len > 0.0 {
                    n[0] /= len; n[1] /= len; n[2] /= len;
                }
            }
        }
        // Asignar normales a los vértices
        for mesh in &mut meshes {
            for v in &mut mesh.vertices {
                let idx = positions.iter().position(|&p| p[..3] == v.position[..3]).unwrap_or(0);
                v.normal = normals[idx];
            }
        }
    }
    Ok(Scene { meshes })
}

fn load_dds_texture(path: &Path, flip_v: bool, flip_h: bool) -> Option<Texture> {
    use ddsfile::{Dds, DxgiFormat};
    use std::fs::File;
    use std::io::Read;
    let mut file = File::open(path).ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;
    let dds = Dds::read(&mut &buf[..]).ok()?;
    let width = dds.get_width();
    let height = dds.get_height();
    if let Some(format) = dds.get_dxgi_format() {
        let (encoding, format_ok) = match format {
            DxgiFormat::BC1_UNorm => (bcndecode::BcnEncoding::Bc1, true),
            DxgiFormat::BC3_UNorm => (bcndecode::BcnEncoding::Bc3, true),
            _ => (bcndecode::BcnEncoding::Bc1, false), // Valor por defecto, no se usará
        };
        if format_ok {
            if let Ok(decoded) = bcndecode::decode(
                &dds.data,
                width as usize,
                height as usize,
                encoding,
                bcndecode::BcnDecoderFormat::RGBA,
            ) {
                return Some(Texture::new_with_flip(width, height, decoded, flip_v, flip_h));
            }
        }
    }
    // Soporte para RGBA8 sin comprimir
    let pf = &dds.header.spf;
    if pf.fourcc.is_none()
        && pf.rgb_bit_count == Some(32)
        && pf.r_bit_mask == Some(0x00ff_0000)
        && pf.g_bit_mask == Some(0x0000_ff00)
        && pf.b_bit_mask == Some(0x0000_00ff)
        && pf.a_bit_mask == Some(0xff00_0000)
    {
        let mut data = dds.data.clone();
        for px in data.chunks_exact_mut(4) {
            let a = px[3];
            let r = px[2];
            let g = px[1];
            let b = px[0];
            px[0] = r;
            px[1] = g;
            px[2] = b;
            px[3] = a;
        }
        return Some(Texture::new_with_flip(width, height, data, flip_v, flip_h));
    }
    None
}
