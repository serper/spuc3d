use std::path::Path;
use std::io;

#[derive(Clone)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub flip_v: bool,
    pub flip_h: bool,
}

impl Texture {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), (width * height * 4) as usize, "Data size does not match texture dimensions");
        Self { width, height, data, flip_v: false, flip_h: false }
    }

    pub fn new_with_flip(width: u32, height: u32, data: Vec<u8>, flip_v: bool, flip_h: bool) -> Self {
        assert_eq!(data.len(), (width * height * 4) as usize, "Data size does not match texture dimensions");
        Self { width, height, data, flip_v, flip_h }
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let index = ((y * self.width + x) * 4) as usize;
        [
            self.data[index],
            self.data[index + 1],
            self.data[index + 2],
            self.data[index + 3],
        ]
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, color: [u8; 4]) {
        let index = ((y * self.width + x) * 4) as usize;
        self.data[index..index + 4].copy_from_slice(&color);
    }    
    
    /// Obtiene un píxel interpolado usando coordenadas de textura normalizadas (0.0 a 1.0)
    pub fn sample(&self, u: f32, v: f32) -> [f32; 4] {
        // Modo mirrored repeat + flip configurable
        let u = mirrored_repeat(u, self.flip_h);
        let v = mirrored_repeat(v, self.flip_v);
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);
        let x = u * (self.width as f32 - 1.0);
        let y = v * (self.height as f32 - 1.0);
        self.sample_bilinear(x, y)
    }
    
    /// Realiza un muestreo con interpolación bilineal
    fn sample_bilinear(&self, x: f32, y: f32) -> [f32; 4] {
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        
        let x_fract = x - x0 as f32;
        let y_fract = y - y0 as f32;
        
        let c00 = self.get_pixel_float(x0, y0);
        let c10 = self.get_pixel_float(x1, y0);
        let c01 = self.get_pixel_float(x0, y1);
        let c11 = self.get_pixel_float(x1, y1);
        
        // Interpolación en X para la fila superior
        let c0 = self.lerp_color(&c00, &c10, x_fract);
        // Interpolación en X para la fila inferior
        let c1 = self.lerp_color(&c01, &c11, x_fract);
        // Interpolación en Y entre los resultados anteriores
        let result = self.lerp_color(&c0, &c1, y_fract);
        
        result
    }
    
    /// Obtiene un píxel como valores flotantes (0.0 a 1.0)
    fn get_pixel_float(&self, x: u32, y: u32) -> [f32; 4] {
        let pixel = self.get_pixel(x, y);
        [
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
            pixel[3] as f32 / 255.0,
        ]
    }
    
    /// Interpola linealmente entre dos colores
    fn lerp_color(&self, a: &[f32; 4], b: &[f32; 4], t: f32) -> [f32; 4] {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        ]
    }
    
    /// Crea una textura vacía de un tamaño determinado
    pub fn create_empty(width: u32, height: u32) -> Self {
        let size = (width * height * 4) as usize;
        let data = vec![0; size];
        Self { width, height, data, flip_v: false, flip_h: false }
    }
    
    /// Crea una textura de prueba con un patrón de tablero de ajedrez
    pub fn create_checkerboard(width: u32, height: u32, cell_size: u32) -> Self {
        let mut texture = Self::create_empty(width, height);
        
        for y in 0..height {
            for x in 0..width {
                let is_white = ((x / cell_size) + (y / cell_size)) % 2 == 0;
                let color = if is_white {
                    [255, 255, 255, 255] // Blanco
                } else {
                    [0, 0, 0, 255]       // Negro
                };
                texture.set_pixel(x, y, color);
            }
        }
        
        texture
    }
    
    /// Carga una textura desde un archivo de imagen
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        // Cargar la imagen con crate image
        let img = image::open(path)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        
        let width = img.width();
        let height = img.height();
        
        // Convertir a formato RGBA
        let rgba_img = img.to_rgba8();
        let pixels = rgba_img.into_raw();
        
        // Crear la textura
        Ok(Self::new(width, height, pixels))
    }
}

fn mirrored_repeat(t: f32, flip: bool) -> f32 {
    let t_calc = if t.floor() % 2.0 == 0.0 {
        t
    } else {
        t.fract()
    };
    if flip {
        1.0 - t_calc.fract()
    } else {
        t_calc.fract()
    }
}