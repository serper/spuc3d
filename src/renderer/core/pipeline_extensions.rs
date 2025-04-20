//--------------------------------------------------------
// EXTENSIONES AL PIPELINE PARA SOPORTE DE SOMBRAS E ILUMINACIÓN AVANZADA
//--------------------------------------------------------

// Añade estos métodos a la implementación de Pipeline:

/// Obtiene la matriz de modelo actual
pub fn get_model_matrix(&self) -> Matrix {
    self.model_matrix
}

/// Obtiene la matriz de vista actual
pub fn get_view_matrix(&self) -> Matrix {
    self.view_matrix
}

/// Obtiene la matriz de proyección actual
pub fn get_projection_matrix(&self) -> Matrix {
    self.projection_matrix
}

/// Obtiene la posición de la cámara (para cálculos de iluminación)
pub fn get_camera_position(&self) -> [f32; 3] {
    // Si estamos en modo quaternion, usar la posición de vista directamente
    if matches!(self.transformation_mode, TransformationMode::Quaternion) {
        return [
            self.view_position.x(),
            self.view_position.y(),
            self.view_position.z()
        ];
    }
    
    // En modo matriz, necesitamos extraer la posición de la cámara de la matriz de vista
    // La matriz de vista es el inverso de la matriz de cámara:
    // View = R * T, donde T = translate(-eye)
    
    // Para obtener eye, necesitamos calcular -R^-1 * t
    let elements = self.view_matrix.elements();
    
    // Extraer la parte de rotación (primeras 3 columnas)
    let rotation_inverse = [
        [elements[0][0], elements[1][0], elements[2][0]],
        [elements[0][1], elements[1][1], elements[2][1]],
        [elements[0][2], elements[1][2], elements[2][2]],
    ];
    
    // Extraer la traslación (última columna)
    let translation = [elements[0][3], elements[1][3], elements[2][3]];
    
    // Calcular -R^-1 * t
    let mut eye = [0.0; 3];
    for i in 0..3 {
        eye[i] = 0.0;
        for j in 0..3 {
            eye[i] -= rotation_inverse[i][j] * translation[j];
        }
    }
    
    eye
}

//--------------------------------------------------------
// IMPLEMENTACIÓN ACTUALIZADA DE process_vertex PARA INCLUIR LA POSICIÓN MUNDIAL
//--------------------------------------------------------

/// Procesa un vértice a través del pipeline de renderizado    
pub fn process_vertex(&self, vertex: &Vertex) -> VertexOutput {
    let world_pos: [f32; 4];
    let view_pos: [f32; 4];
    
    // Aplicar transformación de modelo según el modo seleccionado
    match self.transformation_mode {
        TransformationMode::Matrix => {
            // Método tradicional basado en matrices
            let vertex_position = [vertex.position[0], vertex.position[1], vertex.position[2], 1.0];
            // Transformación al espacio de mundo
            world_pos = self.transform_point(&vertex_position, &self.model_matrix);
        },
        TransformationMode::Quaternion => {
            // Método basado en quaternions para mayor eficiencia en rotaciones
            world_pos = self.transform_point_quaternion_based(&vertex.position);
        }
    }
    
    // Transformación al espacio de cámara (vista) - siempre usando matrices
    view_pos = self.transform_point(&world_pos, &self.view_matrix);
    
    // Transformación al espacio de clip
    let clip_pos = self.transform_point(&view_pos, &self.projection_matrix);
    
    // División de perspectiva para obtener NDC
    let w = clip_pos[3];
    
    // Manejar w cercanos a cero o negativos
    if w.abs() < 0.001 {
        println!("W muy pequeño: {}", w);
        return VertexOutput {
            position: [-999.0, -999.0, 0.0, 1.0], // Fuera de pantalla
            normal: vertex.normal,
            color: vertex.color,
            tex_coords: vertex.tex_coords,
            world_position: [0.0, 0.0, 0.0], // Posición por defecto
        };
    }
    
    let ndc_x = clip_pos[0] / w;
    let ndc_y = clip_pos[1] / w;
    let ndc_z = clip_pos[2] / w;
    
    // Transformación de NDC a coordenadas de pantalla
    let screen_x = (ndc_x + 1.0) * 0.5 * self.rasterizer.width as f32;
    let screen_y = (1.0 - ndc_y) * 0.5 * self.rasterizer.height as f32; // Y invertida para OpenGL
    let screen_z = (ndc_z + 1.0) * 0.5; // Z de [0, 1] para buffer de profundidad
    
    // Extraer la posición mundial para pasar al fragment shader
    let world_position = [world_pos[0], world_pos[1], world_pos[2]];
    
    VertexOutput {
        position: [screen_x, screen_y, screen_z, w],
        normal: vertex.normal,
        color: vertex.color,
        tex_coords: vertex.tex_coords,
        world_position, // Pasar la posición mundial al shader
    }
}

//--------------------------------------------------------
// EJEMPLO DE USO DEL SISTEMA DE SOMBRAS EN LA APLICACIÓN
//--------------------------------------------------------

// Para implementar el renderizado con sombras en tu aplicación main:

fn render_scene_with_shadows() {
    // 1. Configurar tamaño del mapa de sombras
    let shadow_width = 1024;
    let shadow_height = 1024;
    
    // 2. Crear renderer de sombras
    let mut shadow_renderer = ShadowRenderer::new(shadow_width, shadow_height);
    
    // 3. Obtener referencia a las mallas en la escena
    let scene_meshes = vec![&cube_mesh, &ground_mesh]; // Por ejemplo
    
    // 4. Obtener la luz principal (que proyectará sombras)
    let main_light = shader.get_main_light();
    
    // 5. Renderizar el mapa de sombras
    let shadow_map = shadow_renderer.render_shadow_map(&scene_meshes, &main_light);
    
    // 6. Configurar el shader con el mapa de sombras generado
    shader.set_shadow_map(shadow_map);
    
    // 7. Renderizar la escena normalmente - ahora con sombras
    pipeline.render(&cube_mesh, Some(&cube_texture));
    pipeline.render(&ground_mesh, Some(&ground_texture));
}
