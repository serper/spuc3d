use crate::renderer::geometry::Vertex;
use crate::renderer::texture::Texture;
use crate::renderer::core::math;
use crate::renderer::core::pipeline::TileBuffer;
use core_simd::simd::{f32x8, u32x8};
use core_simd::simd::cmp::SimdPartialOrd;

pub struct Rasterizer {
    pub width: u32,
    pub height: u32,
    pub color_buffer: Vec<u32>,
    pub depth_buffer: Vec<f32>,
    pub wireframe_enabled: bool,
    pub wireframe_color: Option<u32>,
}

impl Rasterizer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            color_buffer: vec![0; (width * height) as usize],
            depth_buffer: vec![f32::INFINITY; (width * height) as usize],
            wireframe_enabled: false,
            wireframe_color: None,
        }
    }

    pub fn clear(&mut self, color: u32, depth: f32) {
        self.color_buffer.fill(color);
        self.depth_buffer.fill(depth);
    }

    pub fn set_wireframe(&mut self, enabled: bool, color: Option<u32>) {
        self.wireframe_enabled = enabled;
        self.wireframe_color = color;
    }
    
    pub fn set_pixel(&mut self, x: u32, y: u32, color: u32, depth: f32) {
        let index = (y * self.width + x) as usize;
        if index < self.color_buffer.len() {
            if depth < self.depth_buffer[index] {
                self.depth_buffer[index] = depth;
                self.color_buffer[index] = color;
            }
        }
    }

    pub fn get_color_buffer(&self) -> Vec<u32> {
        self.color_buffer.clone()
    }

    /// Dibuja una línea utilizando el algoritmo de Bresenham con interpolación de profundidad
    /// 
    /// # Parámetros
    /// 
    /// * `x0` - Coordenada X inicial
    /// * `y0` - Coordenada Y inicial
    /// * `z0` - Profundidad inicial
    /// * `x1` - Coordenada X final
    /// * `y1` - Coordenada Y final
    /// * `z1` - Profundidad final
    /// * `color` - Color de la línea en formato ARGB
    pub fn draw_line(&mut self, x0: i32, y0: i32, z0: f32, x1: i32, y1: i32, z1: f32, color0: Option<u32>, color1: Option<u32>) {
        // Obtener las dimensiones del rasterizador previamente para evitar préstamos simultáneos
        let width = self.width as i32;
        let height = self.height as i32;
        
        // Restringir las coordenadas a los límites del rasterizador
        let clip_line = |x: i32, y: i32| -> (bool, u32, u32) {
            if x < 0 || y < 0 || x >= width || y >= height {
                (false, 0, 0)
            } else {
                (true, x as u32, y as u32)
            }
        };

        // Implementación del algoritmo de Bresenham
        let mut x = x0;
        let mut y = y0;
        
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs(); // Negativo para simplificar el algoritmo
        
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        
        let mut err = dx + dy; // Error
        
        // Distancia total para la interpolación
        let total_steps = dx.max(dy.abs());
        let mut current_step = 0;
        
        loop {
            // Calcular la profundidad interpolada
            let t = if total_steps > 0 {
                current_step as f32 / total_steps as f32
            } else {
                0.0
            };
            let interpolated_z = z0 * (1.0 - t) + z1 * t;
            
            // Dibujar el pixel si está dentro de los límites
            let (valid, px, py) = clip_line(x, y);
            if valid {
                // Si se proporciona un color, usarlo; de lo contrario, usar el color interpolado
                let final_color = match (color0, color1) {
                    (Some(c0), Some(c1)) => {
                        let r = ((c0 >> 16) & 0xFF) as f32 * (1.0 - t) + ((c1 >> 16) & 0xFF) as f32 * t;
                        let g = ((c0 >> 8) & 0xFF) as f32 * (1.0 - t) + ((c1 >> 8) & 0xFF) as f32 * t;
                        let b = (c0 & 0xFF) as f32 * (1.0 - t) + (c1 & 0xFF) as f32 * t;
                        ((r as u32) << 16 | (g as u32) << 8 | (b as u32)) as u32
                    },
                    _ => color0.unwrap_or(0xFFFFFFFF),
                };
                self.set_pixel(px, py, final_color, interpolated_z);
            }
            
            // Verificar si hemos llegado al final
            if x == x1 && y == y1 {
                break;
            }
            
            // Calcular el siguiente punto
            let e2 = 2 * err;
            
            if e2 >= dy {
                if x == x1 {
                    break;
                }
                err += dy;
                x += sx;
                current_step += 1;
            }
            
            if e2 <= dx {
                if y == y1 {
                    break;
                }
                err += dx;
                y += sy;
                current_step += 1;
            }
        }
    }

    /// Dibuja un triángulo con textura
    pub fn draw_textured_triangle(&mut self, v0: &Vertex, v1: &Vertex, v2: &Vertex, texture: &Texture) {
        if self.wireframe_enabled {
            // Dibujar solo los bordes
            let color0 = self.wireframe_color.unwrap_or(v0.color[0] as u32);
            let color1 = self.wireframe_color.unwrap_or(v1.color[0] as u32);
            let color2 = self.wireframe_color.unwrap_or(v2.color[0] as u32);

            self.draw_line(
                v0.position[0] as i32, v0.position[1] as i32, v0.position[2],
                v1.position[0] as i32, v1.position[1] as i32, v1.position[2],
                Some(color0), Some(color1),
            );
            self.draw_line(
                v1.position[0] as i32, v1.position[1] as i32, v1.position[2],
                v2.position[0] as i32, v2.position[1] as i32, v2.position[2],
                Some(color1), Some(color2),
            );
            self.draw_line(
                v2.position[0] as i32, v2.position[1] as i32, v2.position[2],
                v0.position[0] as i32, v0.position[1] as i32, v0.position[2],
                Some(color2), Some(color0),
            );
        } else {
            // Encontrar límites del triángulo
            let min_x = (self.width as f32 - 1.0).min(
                v0.position[0].max(0.0).min(v1.position[0].max(0.0).min(v2.position[0].max(0.0)))
            ) as u32;
            let min_y = (self.height as f32 - 1.0).min(
                v0.position[1].max(0.0).min(v1.position[1].max(0.0).min(v2.position[1].max(0.0)))
            ) as u32;
            let max_x = (self.width as f32 - 1.0).min(
                v0.position[0].max(v1.position[0].max(v2.position[0])).max(0.0)
            ) as u32;
            let max_y = (self.height as f32 - 1.0).min(
                v0.position[1].max(v1.position[1].max(v2.position[1])).max(0.0)
            ) as u32;
            
            // Para cada píxel dentro de los límites
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let point = [x as f32 + 0.5, y as f32 + 0.5];
                    let (alpha, beta, gamma) = math::compute_barycentric_coordinates(
                        point,
                        [v0.position[0], v0.position[1]],
                        [v1.position[0], v1.position[1]],
                        [v2.position[0], v2.position[1]],
                    );
                    if alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 {
                        let depth = alpha * v0.position[2] + beta * v1.position[2] + gamma * v2.position[2];
                        // Interpolar coordenadas de textura
                        let u = alpha * v0.tex_coords[0] + beta * v1.tex_coords[0] + gamma * v2.tex_coords[0];
                        let v = alpha * v0.tex_coords[1] + beta * v1.tex_coords[1] + gamma * v2.tex_coords[1];
                        // Obtener color de la textura
                        let tex_color = texture.sample(u, v);
                        // Interpolar colores de vértices
                        let r = alpha * v0.color[0] + beta * v1.color[0] + gamma * v2.color[0];
                        let g = alpha * v0.color[1] + beta * v1.color[1] + gamma * v2.color[1];
                        let b = alpha * v0.color[2] + beta * v1.color[2] + gamma * v2.color[2];
                        // Combinar color de vértice con color de textura
                        let final_r = (r * tex_color[0] * 255.0) as u32;
                        let final_g = (g * tex_color[1] * 255.0) as u32;
                        let final_b = (b * tex_color[2] * 255.0) as u32;
                        let final_color = (final_r << 16) | (final_g << 8) | final_b;
                        self.set_pixel(x, y, final_color, depth);
                    }
                }
            }
        }
    }

    /// Dibuja un triángulo interpolando los colores de los vértices (sin textura)
    pub fn draw_triangle(&mut self, v0: &Vertex, v1: &Vertex, v2: &Vertex) {
        if self.wireframe_enabled {
            let color0 = self.wireframe_color.unwrap_or(v0.color[0] as u32);
            let color1 = self.wireframe_color.unwrap_or(v1.color[0] as u32);
            let color2 = self.wireframe_color.unwrap_or(v2.color[0] as u32);

            self.draw_line(
                v0.position[0] as i32, v0.position[1] as i32, v0.position[2],
                v1.position[0] as i32, v1.position[1] as i32, v1.position[2],
                Some(color0), Some(color1),
            );
            self.draw_line(
                v1.position[0] as i32, v1.position[1] as i32, v1.position[2],
                v2.position[0] as i32, v2.position[1] as i32, v2.position[2],
                Some(color1), Some(color2),
            );
            self.draw_line(
                v2.position[0] as i32, v2.position[1] as i32, v2.position[2],
                v0.position[0] as i32, v0.position[1] as i32, v0.position[2],
                Some(color2), Some(color0),
            );
        } else {
            let min_x = (self.width as f32 - 1.0).min(
                v0.position[0].max(0.0).min(v1.position[0].max(0.0).min(v2.position[0].max(0.0)))
            ) as u32;
            let min_y = (self.height as f32 - 1.0).min(
                v0.position[1].max(0.0).min(v1.position[1].max(0.0).min(v2.position[1].max(0.0)))
            ) as u32;
            let max_x = (self.width as f32 - 1.0).min(
                v0.position[0].max(v1.position[0].max(v2.position[0])).max(0.0)
            ) as u32;
            let max_y = (self.height as f32 - 1.0).min(
                v0.position[1].max(v1.position[1].max(v2.position[1])).max(0.0)
            ) as u32;
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let point = [x as f32 + 0.5, y as f32 + 0.5];
                    let (alpha, beta, gamma) = math::compute_barycentric_coordinates(
                        point,
                        [v0.position[0], v0.position[1]],
                        [v1.position[0], v1.position[1]],
                        [v2.position[0], v2.position[1]],
                    );
                    if alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 {
                        let depth = alpha * v0.position[2] + beta * v1.position[2] + gamma * v2.position[2];
                        // Interpolar colores de vértices
                        let r = alpha * v0.color[0] + beta * v1.color[0] + gamma * v2.color[0];
                        let g = alpha * v0.color[1] + beta * v1.color[1] + gamma * v2.color[1];
                        let b = alpha * v0.color[2] + beta * v1.color[2] + gamma * v2.color[2];
                        let final_r = (r * 255.0) as u32;
                        let final_g = (g * 255.0) as u32;
                        let final_b = (b * 255.0) as u32;
                        let final_color = (final_r << 16) | (final_g << 8) | final_b;
                        self.set_pixel(x, y, final_color, depth);
                    }
                }
            }
        }
    }

    /// Fusiona los tiles locales al buffer global del rasterizador
    pub fn merge_tile(&mut self, tile: &TileBuffer) {
        let simd_width = 8;
        let tile_len = (tile.width * tile.height) as usize;
        let mut i = 0;
        while i + simd_width <= tile_len {
            let t_depth = f32x8::from_array([
                tile.depth_buffer[i],
                tile.depth_buffer[i + 1],
                tile.depth_buffer[i + 2],
                tile.depth_buffer[i + 3],
                tile.depth_buffer[i + 4],
                tile.depth_buffer[i + 5],
                tile.depth_buffer[i + 6],
                tile.depth_buffer[i + 7],
            ]);
            let t_color = u32x8::from_array([
                tile.color_buffer[i],
                tile.color_buffer[i + 1],
                tile.color_buffer[i + 2],
                tile.color_buffer[i + 3],
                tile.color_buffer[i + 4],
                tile.color_buffer[i + 5],
                tile.color_buffer[i + 6],
                tile.color_buffer[i + 7],
            ]);
            let mut g_depth = f32x8::from_array([
                self.depth_buffer[i],
                self.depth_buffer[i + 1],
                self.depth_buffer[i + 2],
                self.depth_buffer[i + 3],
                self.depth_buffer[i + 4],
                self.depth_buffer[i + 5],
                self.depth_buffer[i + 6],
                self.depth_buffer[i + 7],
            ]);
            let mut g_color = u32x8::from_array([
                self.color_buffer[i],
                self.color_buffer[i + 1],
                self.color_buffer[i + 2],
                self.color_buffer[i + 3],
                self.color_buffer[i + 4],
                self.color_buffer[i + 5],
                self.color_buffer[i + 6],
                self.color_buffer[i + 7],
            ]);
            let mask = t_depth.simd_lt(g_depth);
            g_depth = mask.select(t_depth, g_depth);
            g_color = mask.select(t_color, g_color);
            let g_depth_arr = g_depth.to_array();
            let g_color_arr = g_color.to_array();
            for j in 0..simd_width {
                self.depth_buffer[i + j] = g_depth_arr[j];
                self.color_buffer[i + j] = g_color_arr[j];
            }
            i += simd_width;
        }
        // Resto escalar
        while i < tile_len {
            let gx = tile.x + (i as u32 % tile.width);
            let gy = tile.y + (i as u32 / tile.width);
            if gx < self.width && gy < self.height {
                let global_idx = (gy * self.width + gx) as usize;
                let tile_depth = tile.depth_buffer[i];
                if tile_depth < self.depth_buffer[global_idx] {
                    self.depth_buffer[global_idx] = tile_depth;
                    self.color_buffer[global_idx] = tile.color_buffer[i];
                }
            }
            i += 1;
        }
    }
}

pub fn edge_function(v0: [f32; 3], v1: [f32; 3], p: [f32; 3]) -> f32 {
    (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])
}