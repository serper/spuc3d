#![feature(portable_simd)]

// Librería de renderización 3D sin GPU en Rust
// Exportación de módulos

pub mod renderer {
    pub mod core {
        pub mod math_cpu;
        pub mod math_simd;
        pub mod pipeline;
        pub mod rasterizer;
        pub mod math;
        pub mod transform;
    }
    
    pub mod ffi;
    pub mod geometry;
    pub mod shader;
    pub mod texture;
}

/// Inicializa la librería y configura el logger
pub fn init() {
    env_logger::init();
    log::info!("Inicializando spuc3d v{}", env!("CARGO_PKG_VERSION"));
}
