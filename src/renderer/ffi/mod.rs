use crate::renderer::core::pipeline::Pipeline;
use crate::renderer::core::rasterizer::Rasterizer;
use crate::renderer::core::transform::Transform;
use crate::renderer::font::Font;
use crate::renderer::geometry::{Mesh, Vertex};
use crate::renderer::shader::{DefaultShader, Shader, VertexShader};
use crate::renderer::texture::Texture;
use crate::renderer::core::pipeline::RenderMode;

use std::ffi::{c_char, CStr};
use std::ptr;
use std::slice;
use std::boxed::Box;

// ... existing create_mesh, destroy_mesh, create_texture, destroy_texture ...

/// Contexto que contiene el Pipeline y el Rasterizer para FFI.
pub struct PipelineContext<'a> {
    rasterizer: Box<Rasterizer>,
    // Usamos un puntero crudo aquí para evitar problemas de lifetime con el rasterizer
    // El lifetime 'a se asocia con el PipelineContext, no directamente con Pipeline<'a>
    pipeline: Box<Pipeline<'a>>,
}

#[no_mangle]
pub extern "C" fn create_pipeline_context(width: u32, height: u32) -> *mut PipelineContext<'static> {
    let rasterizer = Box::new(Rasterizer::new(width, height));
    // Obtenemos un puntero crudo al rasterizer para pasarlo al pipeline.
    let rasterizer_ptr: *mut Rasterizer = Box::into_raw(rasterizer);

    let pipeline = Box::new(Pipeline::new(
        Box::new(DefaultShader::new()),
        unsafe { &mut *rasterizer_ptr }, // Reconstruimos la referencia mutable
    ));

    let context = Box::new(PipelineContext {
        // Recuperamos la propiedad del Box para almacenarlo en el contexto.
        rasterizer: unsafe { Box::from_raw(rasterizer_ptr) },
        pipeline,
    });

    Box::into_raw(context)
}

#[no_mangle]
pub extern "C" fn destroy_pipeline_context(context: *mut PipelineContext) {
    if context.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(context));
    }
}

#[no_mangle]
pub extern "C" fn pipeline_clear(context: *mut PipelineContext, color: u32, depth: f32) {
    if context.is_null() { return; }
    let context = unsafe { &mut *context };
    context.pipeline.rasterizer.clear(color, depth);
}

#[no_mangle]
pub extern "C" fn pipeline_set_render_mode(context: *mut PipelineContext, mode: u32) {
    if context.is_null() { return; }
    let context = unsafe { &mut *context };
    let render_mode = match mode {
        0 => RenderMode::Wireframe,
        1 => RenderMode::Color,
        2 => RenderMode::Texture,
        _ => RenderMode::Texture, // Default a Texture
    };
    context.pipeline.set_render_mode(render_mode);
}

#[no_mangle]
pub extern "C" fn pipeline_look_at(context: *mut PipelineContext, eye: *const f32, target: *const f32, up: *const f32) {
    if context.is_null() { return; }
    let context = unsafe { &mut *context };
    let eye_slice = unsafe { slice::from_raw_parts(eye, 3) };
    let target_slice = unsafe { slice::from_raw_parts(target, 3) };
    let up_slice = unsafe { slice::from_raw_parts(up, 3) };
    context.pipeline.look_at(
        &[eye_slice[0], eye_slice[1], eye_slice[2]],
        &[target_slice[0], target_slice[1], target_slice[2]],
        &[up_slice[0], up_slice[1], up_slice[2]],
    );
}

#[no_mangle]
pub extern "C" fn pipeline_set_perspective(context: *mut PipelineContext, fov_y_radians: f32, aspect: f32, near: f32, far: f32) {
    if context.is_null() { return; }
    let context = unsafe { &mut *context };
    context.pipeline.set_perspective(fov_y_radians, aspect, near, far);
}

#[no_mangle]
pub extern "C" fn pipeline_render_mesh(context: *mut PipelineContext, mesh: *const Mesh, texture: *const Texture) {
    if context.is_null() || mesh.is_null() { return; }
    let context = unsafe { &mut *context };
    let mesh_ref = unsafe { &*mesh };
    let texture_ref = if texture.is_null() { None } else { Some(unsafe { &*texture }) };
    context.pipeline.render_parallel(mesh_ref, texture_ref);
}

#[no_mangle]
pub extern "C" fn pipeline_get_framebuffer(context: *const PipelineContext, width: *mut u32, height: *mut u32) -> *const u32 {
    if context.is_null() {
        unsafe {
            if !width.is_null() { *width = 0; }
            if !height.is_null() { *height = 0; }
        }
        return ptr::null();
    }
    let context = unsafe { &*context };
    unsafe {
        if !width.is_null() { *width = context.rasterizer.width; }
        if !height.is_null() { *height = context.rasterizer.height; }
    }
    context.rasterizer.color_buffer.as_ptr()
}

// --- Font and Text Rendering ---

#[no_mangle]
pub extern "C" fn create_font(fnt_path: *const c_char, png_path: *const c_char) -> *mut Font {
    if fnt_path.is_null() || png_path.is_null() { return ptr::null_mut(); }

    let fnt_cstr = unsafe { CStr::from_ptr(fnt_path) };
    let png_cstr = unsafe { CStr::from_ptr(png_path) };

    let fnt_str = match fnt_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let png_str = match png_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    match Texture::load_from_file(png_str) {
        Ok(atlas_texture) => {
            match Font::load_bmfont_with_texture(fnt_str, atlas_texture) {
                Ok(font) => Box::into_raw(Box::new(font)),
                Err(_) => ptr::null_mut(),
            }
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn destroy_font(font: *mut Font) {
    if font.is_null() { return; }
    unsafe { drop(Box::from_raw(font)); }
}

#[no_mangle]
pub extern "C" fn pipeline_draw_text(
    context: *mut PipelineContext,
    x: i32,
    y: i32,
    text: *const c_char,
    font: *const Font,
    color: u32,
    scale: f32,
    rotation: f32,
    // Añadimos un puntero para la caché, gestionado desde C
    cache_ptr: *mut Option<crate::renderer::core::rasterizer::TextCache>
) {
    if context.is_null() || text.is_null() || font.is_null() || cache_ptr.is_null() { return; }

    let context = unsafe { &mut *context };
    let font_ref = unsafe { &*font };
    let text_cstr = unsafe { CStr::from_ptr(text) };
    let text_str = match text_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return, // No se pudo convertir el texto
    };
    let cache = unsafe { &mut *cache_ptr };

    // Usamos la función con caché
    context.rasterizer.draw_text_matrix_transform_cached(
        x, y, text_str, font_ref, color, scale, rotation, cache
    );
}

// --- Transform (Opcional pero útil) ---

#[no_mangle]
pub extern "C" fn create_transform() -> *mut Transform {
    let rc_refcell = Transform::new();
    let transform = (*rc_refcell.borrow()).clone();
    Box::into_raw(Box::new(transform))
}

#[no_mangle]
pub extern "C" fn destroy_transform(transform: *mut Transform) {
    if transform.is_null() { return; }
    unsafe { drop(Box::from_raw(transform)); }
}

#[no_mangle]
pub extern "C" fn transform_set_position(transform: *mut Transform, x: f32, y: f32, z: f32) {
    if transform.is_null() { return; }
    let transform = unsafe { &mut *transform };
    transform.set_position(crate::renderer::core::math::Vector::new_with_values(x, y, z));
}

#[no_mangle]
pub extern "C" fn transform_set_rotation_euler(transform: *mut Transform, x_radians: f32, y_radians: f32, z_radians: f32) {
    if transform.is_null() { return; }
    let transform = unsafe { &mut *transform };
    let qx = crate::renderer::core::math::Quaternion::from_axis_angle([1.0, 0.0, 0.0], x_radians);
    let qy = crate::renderer::core::math::Quaternion::from_axis_angle([0.0, 1.0, 0.0], y_radians);
    let qz = crate::renderer::core::math::Quaternion::from_axis_angle([0.0, 0.0, 1.0], z_radians);
    transform.set_rotation(qx.multiply(&qy).multiply(&qz));
}

#[no_mangle]
pub extern "C" fn transform_set_uniform_scale(transform: *mut Transform, scale: f32) {
    if transform.is_null() { return; }
    let transform = unsafe { &mut *transform };
    transform.set_uniform_scale(scale);
}

#[no_mangle]
pub extern "C" fn transform_get_matrix(transform: *const Transform, matrix_out: *mut f32) {
    if transform.is_null() || matrix_out.is_null() { return; }
    let transform = unsafe { &*transform };
    let matrix = transform.get_matrix();
    let matrix_slice = unsafe { slice::from_raw_parts_mut(matrix_out, 16) };
    matrix_slice.copy_from_slice(matrix.as_slice());
}