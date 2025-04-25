use crate::renderer::texture::Texture;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// Representa un glifo (carácter) en el atlas de fuente
#[derive(Debug, Clone, Copy)]
pub struct Glyph {
    /// Coordenadas UV en el atlas (x1, y1, x2, y2)
    pub uv: [f32; 4],
    /// Dimensiones del glifo en pixeles (ancho, alto)
    pub size: [u32; 2],
    /// Offset de posicionamiento (x, y)
    pub offset: [i32; 2],
    /// Avance horizontal después de dibujar este glifo
    pub advance: i32,
}

/// Estructura que representa una fuente de mapa de bits
pub struct Font {
    /// Textura que contiene el atlas de caracteres
    pub atlas: Texture,
    /// Altura de línea base de la fuente
    pub line_height: u32,
    /// Mapa de glifos (caracteres) disponibles
    glyphs: std::collections::HashMap<char, Glyph>,
}

impl Font {
    /// Crea una nueva fuente a partir de un atlas y definiciones de glifos
    pub fn new(atlas: Texture, line_height: u32) -> Self {
        Self {
            atlas,
            line_height,
            glyphs: std::collections::HashMap::new(),
        }
    }
    
    /// Añade un glifo al mapa de caracteres
    pub fn add_glyph(&mut self, character: char, glyph: Glyph) {
        self.glyphs.insert(character, glyph);
    }
    
    /// Obtiene un glifo por carácter
    pub fn get_glyph(&self, character: char) -> Option<&Glyph> {
        self.glyphs.get(&character)
    }
    
    /// Calcula las dimensiones de un texto con esta fuente
    pub fn measure_text(&self, text: &str) -> (u32, u32) {
        let mut width = 0;
        let mut height = self.line_height;
        
        for c in text.chars() {
            if c == '\n' {
                height += self.line_height;
            } else if let Some(glyph) = self.get_glyph(c) {
                width += glyph.advance as u32;
            }
        }
        
        (width, height)
    }
    
    /// Crea una fuente simple de 8x8 pixeles (monoespaciada)
    pub fn create_simple_font() -> Self {
        // Crear un atlas de fuente 16x16 caracteres (ASCII básico)
        let font_width = 128;
        let font_height = 128;
        let char_width = 8;
        let char_height = 8;
        let mut atlas_data = vec![0u8; (font_width * font_height * 4) as usize];
        
        // Dibujar caracteres ASCII básicos (simplificados como bloques)
        for code in 32..128 {
            let c = code as u8 as char;
            let col = (code - 32) % 16;
            let row = (code - 32) / 16;
            
            let x_start = col * char_width;
            let y_start = row * char_height;
            
            // Dibujar carácter (patrón simple basado en el código ASCII)
            for y in 0..char_height {
                for x in 0..char_width {
                    let pixel_x = x_start + x;
                    let pixel_y = y_start + y;
                    
                    // Índice en el array de datos
                    let idx = ((pixel_y * font_width + pixel_x) * 4) as usize;
                    
                    // Patrón simple basado en código ASCII
                    let pattern = match c {
                        'A'..='Z' | 'a'..='z' => {
                            // Patrones para letras
                            if (x == 0 || x == char_width - 1 || y == 0 || y == char_height / 2) && 
                               !(c == 'C' && x == char_width - 1) {
                                true
                            } else {
                                false
                            }
                        },
                        '0'..='9' => {
                            // Patrones para números
                            x == 0 || x == char_width - 1 || y == 0 || y == char_height - 1
                        },
                        _ => {
                            // Símbolos y otros caracteres
                            (x + y) % 2 == 0
                        }
                    };
                    
                    // Asignar color (blanco o transparente)
                    if pattern {
                        atlas_data[idx] = 255;     // R
                        atlas_data[idx + 1] = 255; // G
                        atlas_data[idx + 2] = 255; // B
                        atlas_data[idx + 3] = 255; // A
                    } else {
                        atlas_data[idx] = 0;     // R
                        atlas_data[idx + 1] = 0; // G
                        atlas_data[idx + 2] = 0; // B
                        atlas_data[idx + 3] = 0; // A (transparente)
                    }
                }
            }
        }
        
        // Crear textura para el atlas
        let atlas = Texture::new(font_width, font_height, atlas_data);
        
        // Crear la estructura de fuente
        let mut font = Font::new(atlas, char_height);
        
        // Añadir glifos para caracteres ASCII (32-127)
        for code in 32..128 {
            let c = code as u8 as char;
            let col = (code - 32) % 16;
            let row = (code - 32) / 16;
            
            // Coordenadas UV normalizadas (0.0-1.0)
            let u1 = col as f32 * char_width as f32 / font_width as f32;
            let v1 = row as f32 * char_height as f32 / font_height as f32;
            let u2 = u1 + char_width as f32 / font_width as f32;
            let v2 = v1 + char_height as f32 / font_height as f32;
            
            let glyph = Glyph {
                uv: [u1, v1, u2, v2],
                size: [char_width, char_height],
                offset: [0, 0],
                advance: char_width as i32,
            };
            
            font.add_glyph(c, glyph);
        }
        
        font
    }
    
    /// Carga una fuente desde un archivo BMFont (.fnt) y su textura preexistente
    pub fn load_bmfont_with_texture(fnt_path: &str, atlas: Texture) -> io::Result<Self> {
        let file = File::open(fnt_path)?;
        let reader = BufReader::new(file);
        
        let mut line_height = 0;
        let mut texture_width = atlas.width;
        let mut texture_height = atlas.height;
        let mut glyphs = HashMap::new();
        
        // Parsear el archivo BMFont
        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            
            if parts.is_empty() {
                continue;
            }
            
            match parts[0] {
                "common" => {
                    // Extraer propiedades comunes
                    for part in &parts[1..] {
                        if let Some(value) = part.strip_prefix("lineHeight=") {
                            line_height = value.parse().unwrap_or(0);
                        } else if let Some(value) = part.strip_prefix("scaleW=") {
                            texture_width = value.parse().unwrap_or(texture_width);
                        } else if let Some(value) = part.strip_prefix("scaleH=") {
                            texture_height = value.parse().unwrap_or(texture_height);
                        }
                    }
                },
                "char" => {
                    // Parsear definición de carácter
                    let mut id = 0;
                    let mut x = 0;
                    let mut y = 0;
                    let mut width = 0;
                    let mut height = 0;
                    let mut xoffset = 0;
                    let mut yoffset = 0;
                    let mut xadvance = 0;
                    
                    for part in &parts[1..] {
                        if let Some(value) = part.strip_prefix("id=") {
                            id = value.parse().unwrap_or(0);
                        } else if let Some(value) = part.strip_prefix("x=") {
                            x = value.parse().unwrap_or(0);
                        } else if let Some(value) = part.strip_prefix("y=") {
                            y = value.parse().unwrap_or(0);
                        } else if let Some(value) = part.strip_prefix("width=") {
                            width = value.parse().unwrap_or(0);
                        } else if let Some(value) = part.strip_prefix("height=") {
                            height = value.parse().unwrap_or(0);
                        } else if let Some(value) = part.strip_prefix("xoffset=") {
                            xoffset = value.parse().unwrap_or(0);
                        } else if let Some(value) = part.strip_prefix("yoffset=") {
                            yoffset = value.parse().unwrap_or(0);
                        } else if let Some(value) = part.strip_prefix("xadvance=") {
                            xadvance = value.parse().unwrap_or(0);
                        }
                    }
                    
                    // Crear el glifo y añadirlo al mapa
                    if id > 0 && width > 0 && height > 0 {
                        // Coordenadas UV normalizadas (0.0-1.0)
                        let u1 = x as f32 / texture_width as f32;
                        let v1 = y as f32 / texture_height as f32;
                        let u2 = (x + width) as f32 / texture_width as f32;
                        let v2 = (y + height) as f32 / texture_height as f32;
                        
                        let glyph = Glyph {
                            uv: [u1, v1, u2, v2],
                            size: [width, height],
                            offset: [xoffset, yoffset],
                            advance: xadvance,
                        };
                        
                        glyphs.insert(char::from_u32(id as u32).unwrap_or('?'), glyph);
                    }
                },
                _ => {} // Ignorar otras líneas
            }
        }
        
        // Crear la fuente
        let mut font = Font::new(atlas, line_height);
        
        // Añadir todos los glifos
        for (character, glyph) in glyphs {
            font.add_glyph(character, glyph);
        }
        
        Ok(font)
    }
    
    /// Carga una fuente desde un archivo binario BMFont (.bin)
    pub fn load_bmfont_binary<P: AsRef<Path>>(bin_path: P, texture_path: Option<P>) -> io::Result<Self> {
        // Implementación para formato binario BMFont
        // Este formato es más complejo y requeriría un parser binario específico
        // Por ahora, devolver un error
        Err(io::Error::new(io::ErrorKind::Unsupported, "BMFont binary format not supported yet"))
    }
}