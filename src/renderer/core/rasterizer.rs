use crate::renderer::geometry::Vertex;
use crate::renderer::shader::VertexShader;
use crate::renderer::texture::Texture;
use crate::renderer::core::math;
use rayon::prelude::*;

pub struct Rasterizer {
    pub width: u32,
    pub height: u32,
    pub color_buffer: Vec<u32>,
    pub depth_buffer: Vec<f32>,
    pub wireframe_enabled: bool,
    pub wireframe_color: Option<u32>,
}

pub struct TextCache {
    pub text: String,
    pub color: u32,
    pub scale: f32,
    pub rotation: f32,
    pub buffer: Vec<u32>,
    pub alpha_buffer: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub diagonal: u32,
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

    pub fn set_wireframe_color(&mut self, color: Option<u32>) {
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
    /// * `color0` - Color del punto inicial (opcional)
    /// * `color1` - Color del punto final (opcional)
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
                    // Proteger contra NaN/infinito en baricéntricas
                    if !alpha.is_finite() || !beta.is_finite() || !gamma.is_finite() {
                        continue;
                    }
                    if alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 {
                        let depth = alpha * v0.position[2] + beta * v1.position[2] + gamma * v2.position[2];
                        // Proteger contra NaN/infinito en depth
                        if !depth.is_finite() {
                            continue;
                        }
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
                    // Proteger contra NaN/infinito en baricéntricas
                    if !alpha.is_finite() || !beta.is_finite() || !gamma.is_finite() {
                        continue;
                    }
                    if alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 {
                        let depth = alpha * v0.position[2] + beta * v1.position[2] + gamma * v2.position[2];
                        // Proteger contra NaN/infinito en depth
                        if !depth.is_finite() {
                            continue;
                        }
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

    pub fn raster(
        &mut self,
        triangles: &[[Vertex; 3]],
    ) {
        use std::sync::{Arc, Mutex};
        let color_buffer = Arc::new(Mutex::new(&mut self.color_buffer));
        let depth_buffer = Arc::new(Mutex::new(&mut self.depth_buffer));
        let wireframe_color = self.wireframe_color;
        let width = self.width;
        let height = self.height;
        triangles.par_iter().for_each(|tri| {
            let v0 = &tri[0];
            let v1 = &tri[1];
            let v2 = &tri[2];
            let color0 = wireframe_color.unwrap_or(v0.color[0] as u32);
            let color1 = wireframe_color.unwrap_or(v1.color[0] as u32);
            let color2 = wireframe_color.unwrap_or(v2.color[0] as u32);
            // Rasterizar líneas (acceso concurrente a buffers)
            let mut color_buf = color_buffer.lock().unwrap();
            let mut depth_buf = depth_buffer.lock().unwrap();
            // draw_line adaptada para acceso a buffers externos
            Rasterizer::draw_line_static(
                &mut color_buf,
                &mut depth_buf,
                width,
                height,
                v0.position[0] as i32, v0.position[1] as i32, v0.position[2],
                v1.position[0] as i32, v1.position[1] as i32, v1.position[2],
                Some(color0), Some(color1),
            );
            Rasterizer::draw_line_static(
                &mut color_buf,
                &mut depth_buf,
                width,
                height,
                v1.position[0] as i32, v1.position[1] as i32, v1.position[2],
                v2.position[0] as i32, v2.position[1] as i32, v2.position[2],
                Some(color1), Some(color2),
            );
            Rasterizer::draw_line_static(
                &mut color_buf,
                &mut depth_buf,
                width,
                height,
                v2.position[0] as i32, v2.position[1] as i32, v2.position[2],
                v0.position[0] as i32, v0.position[1] as i32, v0.position[2],
                Some(color2), Some(color0),
            );
        });
    }

    pub fn raster_color(
        &mut self,
        shader: &Box<dyn crate::renderer::shader::Shader + Sync>,
        triangles: &[[Vertex; 3]],
        width: u32,
        height: u32,
    ) {
        use std::sync::{Arc, Mutex};
        use rayon::prelude::*;
        
        let color_buffer = Arc::new(Mutex::new(&mut self.color_buffer));
        let depth_buffer = Arc::new(Mutex::new(&mut self.depth_buffer));
        
        // Dividir en bandas horizontales
        let num_cores = num_cpus::get_physical() as u32;
        let num_bands = num_cores * 2; // Aumentar el número de bandas para mejorar la paralelización
        let band_height = (height + num_bands - 1) / num_bands;
        
        (0..num_bands).into_par_iter().for_each(|band_idx| {
            let start_y = band_idx as u32 * band_height;
            let end_y = ((band_idx as u32 + 1) * band_height).min(height);
            
            // Array estático para inputs del shader (evita creación repetida)
            let mut shader_input = VertexShader::default();
            
            for tri in triangles {
                // Bounding box más rápida
                let min_x = tri.iter().map(|v| v.position[0]).fold(f32::INFINITY, f32::min).floor().max(0.0) as u32;
                let max_x = tri.iter().map(|v| v.position[0]).fold(f32::NEG_INFINITY, f32::max).ceil().min(width as f32 - 1.0) as u32;
                let min_y = tri.iter().map(|v| v.position[1]).fold(f32::INFINITY, f32::min).floor().max(0.0) as u32;
                let max_y = tri.iter().map(|v| v.position[1]).fold(f32::NEG_INFINITY, f32::max).ceil().min(height as f32 - 1.0) as u32;
                
                // Saltar triángulos fuera de esta banda
                if max_y < start_y || min_y >= end_y {
                    continue;
                }
                
                // Limitar el procesamiento a la banda
                let process_min_y = min_y.max(start_y);
                let process_max_y = max_y.min(end_y - 1);
                
                // Precargar vértices
                let v0 = [tri[0].position[0], tri[0].position[1]];
                let v1 = [tri[1].position[0], tri[1].position[1]];
                let v2 = [tri[2].position[0], tri[2].position[1]];
                
                for y in process_min_y..=process_max_y {
                    for x in min_x..=max_x {
                        let px = x as f32 + 0.5;
                        let py = y as f32 + 0.5;

                        let (alpha, beta, gamma) = crate::renderer::core::math::compute_barycentric_coordinates(
                            [px, py], v0, v1, v2,
                        );
                        
                        // Early rejection
                        if alpha < 0.0 || beta < 0.0 || gamma < 0.0 {
                            continue;
                        }
                        
                        // Z-interpolación directa (evita calcular gamma = 1-alpha-beta)
                        let depth = alpha * tri[0].position[2] + beta * tri[1].position[2] + gamma * tri[2].position[2];
                        
                        let idx = (y * width + x) as usize;
                        
                        // Early Z test
                        {
                            let depth_buf = depth_buffer.lock().unwrap();
                            if depth >= depth_buf[idx] {
                                continue;
                            }
                        }
                        
                        // Reutilizar el array para atributos interpolados
                        // Posición
                        for i in 0..3 {
                            shader_input.position[i] = alpha * tri[0].position[i] + beta * tri[1].position[i] + gamma * tri[2].position[i];
                        }
                        // Normal
                        for i in 0..3 {
                            shader_input.normal[i] = alpha * tri[0].normal[i] + beta * tri[1].normal[i] + gamma * tri[2].normal[i];
                        }
                        // Tex coords
                        shader_input.tex_coords = [ alpha * tri[0].tex_coords[0] + beta * tri[1].tex_coords[0] + gamma * tri[2].tex_coords[0],
                                                    alpha * tri[0].tex_coords[1] + beta * tri[1].tex_coords[1] + gamma * tri[2].tex_coords[1]];

                        // Color
                        for i in 0..4 {
                            shader_input.color[i] = alpha * tri[0].color[i] + beta * tri[1].color[i] + gamma * tri[2].color[i];
                        }
                        // Posición duplicada (según el código original)
                        for i in 0..3 {
                            shader_input.world_position[i] = shader_input.position[i]; // Reutilizar valores ya calculados
                        }
                        shader_input.world_position[3] = 1.0;
                        
                        let frag_color = shader.fragment_shader(&shader_input);
                        
                        // Conversión más directa a u32
                        let final_color = ((frag_color[0] * 255.0) as u32) << 16 | 
                                        ((frag_color[1] * 255.0) as u32) << 8 | 
                                        (frag_color[2] * 255.0) as u32;
                        
                        {
                            let mut color_buf = color_buffer.lock().unwrap();
                            color_buf[idx] = final_color;
                        }
                        {
                            let mut depth_buf = depth_buffer.lock().unwrap();
                            depth_buf[idx] = depth;
                        }
                    }
                }
            }
        });
    }

    pub fn raster_texture(
        &mut self,
        shader: &Box<dyn crate::renderer::shader::Shader + Sync>,
        triangles: &[[Vertex; 3]],
        width: u32,
        height: u32,
        tex_ref: Option<&Texture>,
    ) {
        use std::sync::{Arc, Mutex};
        use rayon::prelude::*;
        
        let color_buffer = Arc::new(Mutex::new(&mut self.color_buffer));
        let depth_buffer = Arc::new(Mutex::new(&mut self.depth_buffer));
        
        let num_cores = num_cpus::get_physical() as u32;
        let num_bands = num_cores * 2; // Aumentar el número de bandas para mejorar la paralelización
        let band_height = (height + num_bands - 1) / num_bands;
        
        (0..num_bands).into_par_iter().for_each(|band_idx| {
            let start_y = band_idx as u32 * band_height;
            let end_y = ((band_idx as u32 + 1) * band_height).min(height);
            
            // Array fijo para el shader
            let mut shader_input = VertexShader::default();
            
            for tri in triangles {
                let min_x = tri.iter().map(|v| v.position[0]).fold(f32::INFINITY, f32::min).floor().max(0.0) as u32;
                let max_x = tri.iter().map(|v| v.position[0]).fold(f32::NEG_INFINITY, f32::max).ceil().min(width as f32 - 1.0) as u32;
                let min_y = tri.iter().map(|v| v.position[1]).fold(f32::INFINITY, f32::min).floor().max(0.0) as u32;
                let max_y = tri.iter().map(|v| v.position[1]).fold(f32::NEG_INFINITY, f32::max).ceil().min(height as f32 - 1.0) as u32;
                
                if max_y < start_y || min_y >= end_y {
                    continue;
                }
                
                // Precomputar datos de perspectiva
                let inv_w0 = 1.0 / tri[0].position[3];
                let inv_w1 = 1.0 / tri[1].position[3];
                let inv_w2 = 1.0 / tri[2].position[3];
                let u0 = tri[0].tex_coords[0] * inv_w0;
                let u1 = tri[1].tex_coords[0] * inv_w1;
                let u2 = tri[2].tex_coords[0] * inv_w2;
                let v0 = tri[0].tex_coords[1] * inv_w0;
                let v1 = tri[1].tex_coords[1] * inv_w1;
                let v2 = tri[2].tex_coords[1] * inv_w2;
                
                // Precargar vértices para cálculo de coordenadas baricéntricas
                let v0_pos = [tri[0].position[0], tri[0].position[1]];
                let v1_pos = [tri[1].position[0], tri[1].position[1]];
                let v2_pos = [tri[2].position[0], tri[2].position[1]];
                
                let process_min_y = min_y.max(start_y);
                let process_max_y = max_y.min(end_y - 1);
                
                for y in process_min_y..=process_max_y {
                    for x in min_x..=max_x {
                        let px = x as f32 + 0.5;
                        let py = y as f32 + 0.5;
                        
                        let (alpha, beta, gamma) = crate::renderer::core::math::compute_barycentric_coordinates(
                            [px, py], v0_pos, v1_pos, v2_pos,
                        );
                        
                        // Early rejection
                        if alpha < 0.0 || beta < 0.0 || gamma < 0.0 {
                            continue;
                        }
                        
                        let depth = alpha * tri[0].position[2] + beta * tri[1].position[2] + gamma * tri[2].position[2];
                        
                        let idx = (y * width + x) as usize;
                        
                        // Early Z test
                        {
                            let depth_buf = depth_buffer.lock().unwrap();
                            if depth >= depth_buf[idx] {
                                continue;
                            }
                        }
                        
                        // Calcular atributos una sola vez
                        // Posición
                        for i in 0..3 {
                            shader_input.position[i] = alpha * tri[0].position[i] + beta * tri[1].position[i] + gamma * tri[2].position[i];
                            // También copiar a la parte duplicada
                            shader_input.world_position[i] = shader_input.position[i];
                        }
                        // Normal
                        for i in 0..3 {
                            shader_input.normal[i] = alpha * tri[0].normal[i] + beta * tri[1].normal[i] + gamma * tri[2].normal[i];
                        }
                        // Tex coords (sin perspectiva aquí, se corrigen después)
                        shader_input.tex_coords[0] = alpha * tri[0].tex_coords[0] + beta * tri[1].tex_coords[0] + gamma * tri[2].tex_coords[0];
                        shader_input.tex_coords[1] = alpha * tri[0].tex_coords[1] + beta * tri[1].tex_coords[1] + gamma * tri[2].tex_coords[1];
                        // Color
                        for i in 0..4 {
                            shader_input.color[i] = alpha * tri[0].color[i] + beta * tri[1].color[i] + gamma * tri[2].color[i];
                        }
                        
                        // Corrección de perspectiva más directa
                        let one_over_w = alpha * inv_w0 + beta * inv_w1 + gamma * inv_w2;
                        let u_over_w = alpha * u0 + beta * u1 + gamma * u2;
                        let v_over_w = alpha * v0 + beta * v1 + gamma * v2;
                        let tex_u = u_over_w / one_over_w;
                        let tex_v = v_over_w / one_over_w;
                        
                        let frag_color = shader.fragment_shader_tex(&shader_input, tex_ref, tex_u, tex_v);
                        
                        // Empaquetado más eficiente a u32
                        let final_color = ((frag_color[0] * 255.0) as u32) << 16 | 
                                        ((frag_color[1] * 255.0) as u32) << 8 | 
                                        (frag_color[2] * 255.0) as u32;
                        
                        {
                            let mut color_buf = color_buffer.lock().unwrap();
                            color_buf[idx] = final_color;
                        }
                        {
                            let mut depth_buf = depth_buffer.lock().unwrap();
                            depth_buf[idx] = depth;
                        }
                    }
                }
            }
        });
    }

    pub fn draw_text_matrix_transform(
        &mut self, 
        x: i32, 
        y: i32, 
        text: &str, 
        font: &crate::renderer::font::Font, 
        color: u32,
        scale: f32,
        rotation: f32
    ) -> TextCache {
        if false {
            let mut cursor_x = x;
            let mut cursor_y = y;
            
            for c in text.chars() {
                if c == '\n' {
                    cursor_x = x;
                    cursor_y += (font.line_height as f32 * scale) as i32;
                    continue;
                }
                
                if let Some(glyph) = font.get_glyph(c) {
                    let width = glyph.size[0];
                    let height = glyph.size[1];
                    let uv = glyph.uv;
                    
                    // Aplicar escala a las dimensiones del glifo
                    let scaled_width = (width as f32 * scale) as u32;
                    let scaled_height = (height as f32 * scale) as u32;
                    
                    for py in 0..scaled_height {
                        for px in 0..scaled_width {
                            // Mapear al espacio original del glifo
                            let glyph_x = px as f32 / scale;
                            let glyph_y = py as f32 / scale;
                            
                            // Verificar si está dentro de los límites del glifo original
                            if glyph_x < width as f32 && glyph_y < height as f32 {
                                // Calcular coordenadas de pantalla
                                let screen_x = cursor_x + (px as i32) + (glyph.offset[0] as f32 * scale) as i32;
                                let screen_y = cursor_y + (py as i32) + (glyph.offset[1] as f32 * scale) as i32;
                                
                                // Saltar si está fuera de la pantalla
                                if screen_x < 0 || screen_x >= self.width as i32 || 
                                   screen_y < 0 || screen_y >= self.height as i32 {
                                    continue;
                                }
                                
                                // Calcular coordenadas UV para este pixel
                                let u = uv[0] + (uv[2] - uv[0]) * (glyph_x / width as f32);
                                let v = uv[1] + (uv[3] - uv[1]) * (glyph_y / height as f32);
                                
                                // Obtener el color del atlas de fuente
                                let tex_color = font.atlas.sample(u, v);
                                
                                // Solo dibujar si el alpha es significativo
                                if tex_color[3] > 0.1 {
                                    let r = (((color >> 16) & 0xFF) as f32 * tex_color[0]) as u32;
                                    let g = (((color >> 8) & 0xFF) as f32 * tex_color[1]) as u32;
                                    let b = ((color & 0xFF) as f32 * tex_color[2]) as u32;
                                    
                                    let final_color = (r << 16) | (g << 8) | b;
                                    
                                    self.set_pixel(screen_x as u32, screen_y as u32, final_color, 0.0);
                                }
                            }
                        }
                    }
                      // Avanzar cursor con el espaciado adecuado
                    cursor_x += (glyph.advance as f32 * scale) as i32;
                }
            }
            
            // Crear y devolver una TextCache para el texto renderizado sin transformación
            // Estimar ancho basado en el texto y altura de línea (ya que no tenemos max_width)
            let avg_char_width = font.line_height as u32 / 2; // Estimación simple de ancho
            let scaled_width = text.len() as u32 * avg_char_width;
            let scaled_height = font.line_height;
            let diagonal = ((scaled_width as f32).powi(2) + (scaled_height as f32).powi(2)).sqrt();
            let rotated_size = diagonal.ceil() as u32;
            
            return TextCache {
                text: text.to_string(),
                color,
                scale,
                rotation,
                buffer: Vec::new(), // No almacenamos buffer real, ya que se ha dibujado directamente
                alpha_buffer: Vec::new(),
                width: scaled_width,
                height: scaled_height,
                diagonal: rotated_size,
            };
        }
        
        // Determinar las dimensiones del texto
        let mut text_width = 0;
        let mut text_height = font.line_height;
        let mut max_line_width = 0;
        let mut line_count = 1;
        
        // Primera pasada: calcular dimensiones
        let mut cursor_x = 0;
        for c in text.chars() {
            if c == '\n' {
                line_count += 1;
                max_line_width = max_line_width.max(cursor_x);
                cursor_x = 0;
                continue;
            }
            
            if let Some(glyph) = font.get_glyph(c) {
                cursor_x += glyph.advance;
            }
        }
        
        max_line_width = max_line_width.max(cursor_x);
        text_width = max_line_width;
        text_height = font.line_height * line_count;
        
        // Crear un buffer temporal para el texto sin rotar
        let scaled_width = (text_width as f32 * scale) as u32;
        let scaled_height = (text_height as f32 * scale) as u32;
        
        // Crear buffers temporales
        let buffer_size = (scaled_width * scaled_height) as usize;
        let mut temp_color_buffer = vec![0u32; buffer_size];
        let mut temp_alpha_buffer = vec![0.0f32; buffer_size];
        
        // Renderizar el texto al buffer temporal (sin rotación)
        let mut cursor_x = 0;
        let mut cursor_y = 0;
        
        for c in text.chars() {
            if c == '\n' {
                cursor_x = 0;
                cursor_y += font.line_height as i32;
                continue;
            }
            
            if let Some(glyph) = font.get_glyph(c) {
                let width = glyph.size[0];
                let height = glyph.size[1];
                let uv = glyph.uv;
                
                for py in 0..height {
                    let temp_y = cursor_y + py as i32 + glyph.offset[1];
                    if temp_y < 0 || temp_y >= text_height as i32 {
                        continue;
                    }
                    
                    let buffer_y = (temp_y as f32 * scale) as u32;
                    if buffer_y >= scaled_height {
                        continue;
                    }
                    
                    for px in 0..width {
                        let temp_x = cursor_x + px as i32 + glyph.offset[0];
                        if temp_x < 0 || temp_x >= text_width as i32 {
                            continue;
                        }
                        
                        let buffer_x = (temp_x as f32 * scale) as u32;
                        if buffer_x >= scaled_width {
                            continue;
                        }
                        
                        // Calcular coordenadas UV
                        let u = uv[0] + (uv[2] - uv[0]) * (px as f32 / width as f32);
                        let v = uv[1] + (uv[3] - uv[1]) * (py as f32 / height as f32);
                        
                        // Obtener color y alpha
                        let tex_color = font.atlas.sample(u, v);
                        
                        if tex_color[3] > 0.1 {
                            let idx = (buffer_y * scaled_width + buffer_x) as usize;
                            if idx < buffer_size {
                                let r = ((color >> 16) & 0xFF) as f32 * tex_color[0];
                                let g = ((color >> 8) & 0xFF) as f32 * tex_color[1];
                                let b = (color & 0xFF) as f32 * tex_color[2];
                                
                                let final_color = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
                                
                                // Almacenar en buffer temporal
                                temp_color_buffer[idx] = final_color;
                                temp_alpha_buffer[idx] = tex_color[3];
                            }
                        }
                    }
                }
                
                // Avanzar cursor con espaciado adecuado
                cursor_x += glyph.advance;
            }
        }
        
        // Determinar tamaño del área rotada (usando la diagonal)
        let diagonal = ((scaled_width as f32).powi(2) + (scaled_height as f32).powi(2)).sqrt();
        let rotated_size = diagonal.ceil() as u32;
        
        // Crear la cache con los datos preprocesados
        TextCache {
            text: text.to_string(),
            color,
            scale,
            rotation,
            buffer: temp_color_buffer,
            alpha_buffer: temp_alpha_buffer,
            width: scaled_width,
            height: scaled_height,
            diagonal: rotated_size,
        }
    }
    
    /// Dibuja texto precalculado desde una caché
    pub fn draw_cached_text(
        &mut self,
        x: i32,
        y: i32,
        cache: &TextCache
    ) {
        let sin_r = cache.rotation.sin();
        let cos_r = cache.rotation.cos();
        
        // Punto central del texto para la rotación
        let center_x = cache.width as f32 / 2.0;
        let center_y = cache.height as f32 / 2.0;
        
        // Usar el tamaño rotado de la cache
        let rotated_size = cache.diagonal;
        
        // Dibujar el buffer rotándolo
        for py in 0..rotated_size {
            for px in 0..rotated_size {
                // Coordenadas relativas al centro
                let rel_x = px as f32 - rotated_size as f32 / 2.0;
                let rel_y = py as f32 - rotated_size as f32 / 2.0;
                
                // Aplicar rotación inversa
                let orig_rel_x = rel_x * cos_r + rel_y * sin_r;
                let orig_rel_y = -rel_x * sin_r + rel_y * cos_r;
                
                // Coordenadas en el buffer original
                let buffer_x = orig_rel_x + center_x;
                let buffer_y = orig_rel_y + center_y;
                
                // Verificar si está dentro del buffer original
                if buffer_x >= 0.0 && buffer_x < cache.width as f32 && 
                   buffer_y >= 0.0 && buffer_y < cache.height as f32 {
                    
                    // Usar interpolación bilineal
                    let x0 = buffer_x.floor() as u32;
                    let y0 = buffer_y.floor() as u32;
                    let x1 = (x0 + 1).min(cache.width - 1);
                    let y1 = (y0 + 1).min(cache.height - 1);
                    
                    let x_frac = buffer_x - x0 as f32;
                    let y_frac = buffer_y - y0 as f32;
                    
                    // Índices en los buffers temporales
                    let idx00 = (y0 * cache.width + x0) as usize;
                    let idx01 = (y0 * cache.width + x1) as usize;
                    let idx10 = (y1 * cache.width + x0) as usize;
                    let idx11 = (y1 * cache.width + x1) as usize;
                    
                    let buffer_size = cache.buffer.len();
                    
                    // Obtener colores y alphas
                    let alpha00 = if idx00 < buffer_size { cache.alpha_buffer[idx00] } else { 0.0 };
                    let alpha01 = if idx01 < buffer_size { cache.alpha_buffer[idx01] } else { 0.0 };
                    let alpha10 = if idx10 < buffer_size { cache.alpha_buffer[idx10] } else { 0.0 };
                    let alpha11 = if idx11 < buffer_size { cache.alpha_buffer[idx11] } else { 0.0 };
                    
                    // Interpolar alpha
                    let alpha_top = alpha00 * (1.0 - x_frac) + alpha01 * x_frac;
                    let alpha_bottom = alpha10 * (1.0 - x_frac) + alpha11 * x_frac;
                    let final_alpha = alpha_top * (1.0 - y_frac) + alpha_bottom * y_frac;
                    
                    // Si hay algo que dibujar
                    if final_alpha > 0.1 {
                        let col00 = if idx00 < buffer_size { cache.buffer[idx00] } else { 0 };
                        let col01 = if idx01 < buffer_size { cache.buffer[idx01] } else { 0 };
                        let col10 = if idx10 < buffer_size { cache.buffer[idx10] } else { 0 };
                        let col11 = if idx11 < buffer_size { cache.buffer[idx11] } else { 0 };
                        
                        // Extraer componentes R,G,B
                        let r00 = ((col00 >> 16) & 0xFF) as f32;
                        let g00 = ((col00 >> 8) & 0xFF) as f32;
                        let b00 = (col00 & 0xFF) as f32;
                        
                        let r01 = ((col01 >> 16) & 0xFF) as f32;
                        let g01 = ((col01 >> 8) & 0xFF) as f32;
                        let b01 = (col01 & 0xFF) as f32;
                        
                        let r10 = ((col10 >> 16) & 0xFF) as f32;
                        let g10 = ((col10 >> 8) & 0xFF) as f32;
                        let b10 = (col10 & 0xFF) as f32;
                        
                        let r11 = ((col11 >> 16) & 0xFF) as f32;
                        let g11 = ((col11 >> 8) & 0xFF) as f32;
                        let b11 = (col11 & 0xFF) as f32;
                        
                        // Interpolar colores
                        let r_top = r00 * (1.0 - x_frac) + r01 * x_frac;
                        let g_top = g00 * (1.0 - x_frac) + g01 * x_frac;
                        let b_top = b00 * (1.0 - x_frac) + b01 * x_frac;
                        
                        let r_bottom = r10 * (1.0 - x_frac) + r11 * x_frac;
                        let g_bottom = g10 * (1.0 - x_frac) + g11 * x_frac;
                        let b_bottom = b10 * (1.0 - x_frac) + b11 * x_frac;
                        
                        let final_r = (r_top * (1.0 - y_frac) + r_bottom * y_frac) as u32;
                        let final_g = (g_top * (1.0 - y_frac) + g_bottom * y_frac) as u32;
                        let final_b = (b_top * (1.0 - y_frac) + b_bottom * y_frac) as u32;
                        
                        let final_color = (final_r << 16) | (final_g << 8) | final_b;
                        
                        // Calcular posición final en pantalla
                        let screen_x = x + px as i32 - rotated_size as i32 / 2;
                        let screen_y = y + py as i32 - rotated_size as i32 / 2;
                        
                        // Dibujar en la pantalla
                        if screen_x >= 0 && screen_x < self.width as i32 && 
                           screen_y >= 0 && screen_y < self.height as i32 {
                            self.set_pixel(screen_x as u32, screen_y as u32, final_color, 0.0);
                        }
                    }
                }
            }
        }
    }
    
    /// Dibuja texto con transformación matricial usando caché si está disponible
    pub fn draw_text_matrix_transform_cached(
        &mut self, 
        x: i32, 
        y: i32, 
        text: &str, 
        font: &crate::renderer::font::Font, 
        color: u32,
        scale: f32,
        rotation: f32,
        cache: &mut Option<TextCache>
    ) {
        // Verificar si podemos usar la caché
        if let Some(ref existing_cache) = cache {
            if existing_cache.text == text && 
               existing_cache.color == color && 
               existing_cache.scale == scale && 
               existing_cache.rotation == rotation {
                // Usar la caché existente
                self.draw_cached_text(x, y, existing_cache);
                return;
            }
        }
        
        // Si llegamos aquí, necesitamos crear una nueva caché
        let new_cache = self.cache_text(text, font, color, scale, rotation);
        self.draw_cached_text(x, y, &new_cache);
        
        // Actualizar la caché
        *cache = Some(new_cache);
    }
}

impl Rasterizer {
    // Versión estática de draw_line para acceso concurrente
    fn draw_line_static(
        color_buffer: &mut [u32],
        depth_buffer: &mut [f32],
        width: u32,
        height: u32,
        x0: i32, y0: i32, z0: f32,
        x1: i32, y1: i32, z1: f32,
        color0: Option<u32>, color1: Option<u32>,
    ) {
        // Obtener las dimensiones del rasterizador previamente para evitar préstamos simultáneos
        let width = width as i32;
        let height = height as i32;
        
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
                let index = (py * width as u32 + px) as usize;
                if index < color_buffer.len() {
                    if interpolated_z < depth_buffer[index] {
                        depth_buffer[index] = interpolated_z;
                        color_buffer[index] = final_color;
                    }
                }
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

    /// Cachea un texto para su reutilización posterior
    pub fn cache_text(
        &mut self,
        text: &str,
        font: &crate::renderer::font::Font,
        color: u32,
        scale: f32,
        rotation: f32
    ) -> TextCache {
        // Determinar las dimensiones del texto
        let mut text_width = 0;
        let mut text_height = font.line_height;
        let mut max_line_width = 0;
        let mut line_count = 1;
        
        // Primera pasada: calcular dimensiones
        let mut cursor_x = 0;
        for c in text.chars() {
            if c == '\n' {
                line_count += 1;
                max_line_width = max_line_width.max(cursor_x);
                cursor_x = 0;
                continue;
            }
            
            if let Some(glyph) = font.get_glyph(c) {
                cursor_x += glyph.advance;
            }
        }
        
        max_line_width = max_line_width.max(cursor_x);
        text_width = max_line_width;
        text_height = font.line_height * line_count;
        
        // Crear un buffer temporal para el texto sin rotar
        let scaled_width = (text_width as f32 * scale) as u32;
        let scaled_height = (text_height as f32 * scale) as u32;
        
        // Crear buffers temporales
        let buffer_size = (scaled_width * scaled_height) as usize;
        let mut temp_color_buffer = vec![0u32; buffer_size];
        let mut temp_alpha_buffer = vec![0.0f32; buffer_size];
        
        // Renderizar el texto al buffer temporal (sin rotación)
        let mut cursor_x = 0;
        let mut cursor_y = 0;
        
        for c in text.chars() {
            if c == '\n' {
                cursor_x = 0;
                cursor_y += font.line_height as i32;
                continue;
            }
            
            if let Some(glyph) = font.get_glyph(c) {
                let width = glyph.size[0];
                let height = glyph.size[1];
                let uv = glyph.uv;
                
                for py in 0..height {
                    let temp_y = cursor_y + py as i32 + glyph.offset[1];
                    if temp_y < 0 || temp_y >= text_height as i32 {
                        continue;
                    }
                    
                    let buffer_y = (temp_y as f32 * scale) as u32;
                    if buffer_y >= scaled_height {
                        continue;
                    }
                    
                    for px in 0..width {
                        let temp_x = cursor_x + px as i32 + glyph.offset[0];
                        if temp_x < 0 || temp_x >= text_width as i32 {
                            continue;
                        }
                        
                        let buffer_x = (temp_x as f32 * scale) as u32;
                        if buffer_x >= scaled_width {
                            continue;
                        }
                        
                        // Calcular coordenadas UV
                        let u = uv[0] + (uv[2] - uv[0]) * (px as f32 / width as f32);
                        let v = uv[1] + (uv[3] - uv[1]) * (py as f32 / height as f32);
                        
                        // Obtener color y alpha
                        let tex_color = font.atlas.sample(u, v);
                        
                        if tex_color[3] > 0.1 {
                            let idx = (buffer_y * scaled_width + buffer_x) as usize;
                            if idx < buffer_size {
                                let r = ((color >> 16) & 0xFF) as f32 * tex_color[0];
                                let g = ((color >> 8) & 0xFF) as f32 * tex_color[1];
                                let b = (color & 0xFF) as f32 * tex_color[2];
                                
                                let final_color = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
                                
                                // Almacenar en buffer temporal
                                temp_color_buffer[idx] = final_color;
                                temp_alpha_buffer[idx] = tex_color[3];
                            }
                        }
                    }
                }
                
                // Avanzar cursor con espaciado adecuado
                cursor_x += glyph.advance;
            }
        }
        
        // Determinar tamaño del área rotada (usando la diagonal)
        let diagonal = ((scaled_width as f32).powi(2) + (scaled_height as f32).powi(2)).sqrt();
        let rotated_size = diagonal.ceil() as u32;
        
        // Crear la cache con los datos preprocesados
        TextCache {
            text: text.to_string(),
            color,
            scale,
            rotation,
            buffer: temp_color_buffer,
            alpha_buffer: temp_alpha_buffer,
            width: scaled_width,
            height: scaled_height,
            diagonal: rotated_size,
        }
    }

    /// Dibuja texto en una posición 3D con escala y rotación
    pub fn draw_text_3d_transformed(
        &mut self, 
        position: &[f32; 3], 
        view_proj: &[[f32; 4]; 4],
        text: &str, 
        font: &crate::renderer::font::Font, 
        color: u32,
        scale: f32,
        rotation: f32
    ) -> bool {
        // Transformar la posición 3D a coordenadas de pantalla
        let pos = [position[0], position[1], position[2], 1.0];
        
        // Multiplicar por la matriz view-projection
        let mut clip_pos = [0.0; 4];
        for i in 0..4 {
            for j in 0..4 {
                clip_pos[i] += view_proj[i][j] * pos[j];
            }
        }
        
        // Verificar si está detrás de la cámara
        if clip_pos[3] < 0.1 {
            return false; // No dibujar si está detrás de la cámara
        }
        
        // División de perspectiva
        let ndc_x = clip_pos[0] / clip_pos[3];
        let ndc_y = clip_pos[1] / clip_pos[3];
        let ndc_z = clip_pos[2] / clip_pos[3];
        
        // Si está fuera de los límites NDC, no dibujar
        if ndc_x < -1.0 || ndc_x > 1.0 || ndc_y < -1.0 || ndc_y > 1.0 || ndc_z < 0.0 || ndc_z > 1.0 {
            return false;
        }
        
        // Convertir de coordenadas NDC a coordenadas de pantalla
        let screen_x = ((ndc_x + 1.0) * 0.5 * self.width as f32) as i32;
        let screen_y = ((1.0 - ndc_y) * 0.5 * self.height as f32) as i32;
        
        // Calcular la escala basada en la profundidad
        let depth_scale = scale / clip_pos[3].max(0.1);
        
        // Ahora dibujar el texto en la posición de pantalla calculada        // Dibujar el texto y descartar el valor de retorno
        let _ = self.draw_text_matrix_transform(screen_x, screen_y, text, font, color, depth_scale, rotation);
        
        true
    }
}