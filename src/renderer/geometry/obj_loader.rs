use crate::renderer::geometry::{Mesh, Vertex};
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
            "map_Kd" => {
                if let Some(mat) = current.as_mut() {
                    if parts.len() >= 2 {
                        mat.diffuse_map = Some(parts[1].to_string());
                    }
                }
            },
            _ => {}
        }
    }
    if let Some(mat) = current { materials.insert(mat.name.clone(), mat); }
    materials
}

/// Carga un modelo OBJ simple (solo soporta v, vn, vt, f)
pub fn load_obj(path: &str, flip_v: bool, flip_h: bool) -> Result<Mesh, String> {
    let file = File::open(path).map_err(|e| format!("No se pudo abrir el archivo: {}", e))?;
    let reader = BufReader::new(file);
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut texcoords: Vec<[f32; 2]> = Vec::new();
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut vertex_map = std::collections::HashMap::new();
    let mut materials: HashMap<String, Material> = HashMap::new();
    let mut current_material: Option<String> = None;
    let obj_dir = Path::new(path).parent().map(|p| p.to_path_buf());
    let mut diffuse_texture: Option<Texture> = None;

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
                    if diffuse_texture.is_none() {
                        for mat in materials.values() {
                            if let Some(tex_name) = &mat.diffuse_map {
                                let tex_name = tex_name.replace('\\', "/").replace('\\', "/");
                                let tex_path = dir.join(&tex_name);
                                if tex_path.extension().map(|e| e == "dds").unwrap_or(false) {
                                    if let Some(tex) = load_dds_texture(&tex_path) {
                                        diffuse_texture = Some(tex);
                                        break;
                                    }
                                } else if tex_path.extension().map(|e| e == "png").unwrap_or(false) {
                                    match image::open(&tex_path) {
                                        Ok(img) => {
                                            let rgba = img.to_rgba8();
                                            let (w, h) = rgba.dimensions();
                                            let data = rgba.into_raw();
                                            diffuse_texture = Some(Texture::new(w, h, data));
                                            break;
                                        }
                                        Err(e) => {
                                            println!("Error al abrir PNG {:?}: {}", tex_path, e);
                                        }
                                    }
                                } else if let Ok(img) = image::open(&tex_path) {
                                    let rgba = img.to_rgba8();
                                    let (w, h) = rgba.dimensions();
                                    let data = rgba.into_raw();
                                    diffuse_texture = Some(Texture::new(w, h, data));
                                    break;
                                }
                            }
                        }
                    }
                }
            },
            "usemtl" => {
                current_material = Some(parts[1].to_string());
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
                    let mut u = parts[1].parse().unwrap_or(0.0);
                    let mut v = parts[2].parse().unwrap_or(0.0);
                    if flip_v { v = 1.0 - v; }
                    if flip_h {
                        let temp = u;
                        u = v;
                        v = temp;
                    }
                    texcoords.push([u, v]);
                }
            },
            "f" => {
                if parts.len() >= 4 {
                    for i in 1..4 {
                        let idx = parts[i];
                        let mut split = idx.split('/');
                        let v_idx = split.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                        let vt_idx = split.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                        let vn_idx = split.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                        let key = (v_idx, vt_idx, vn_idx, current_material.clone());
                        let vert_index = if let Some(&idx) = vertex_map.get(&key) {
                            idx
                        } else {
                            let pos = positions.get(v_idx.wrapping_sub(1)).copied().unwrap_or([0.0,0.0,0.0]);
                            let norm = normals.get(vn_idx.wrapping_sub(1)).copied().unwrap_or([0.0,0.0,1.0]);
                            let tex = texcoords.get(vt_idx.wrapping_sub(1)).copied().unwrap_or([0.0,0.0]);
                            let color = current_material.as_ref()
                                .and_then(|mat| materials.get(mat))
                                .map(|mat| mat.diffuse)
                                .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                            let v = Vertex { position: pos, normal: norm, tex_coords: tex, color };
                            vertices.push(v);
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
    Ok(Mesh { vertices, indices, diffuse_texture })
}

fn load_dds_texture(path: &Path) -> Option<Texture> {
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
                return Some(Texture::new(width, height, decoded));
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
        return Some(Texture::new(width, height, data));
    }
    None
}
