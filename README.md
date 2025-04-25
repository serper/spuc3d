# SPUC3D - Motor de Renderizado 3D en Rust

SPUC3D es un motor de renderizado 3D por software escrito completamente en Rust. Está diseñado para ser educativo y demostrar los principios fundamentales del renderizado gráfico sin depender de APIs gráficas de hardware como OpenGL o Vulkan. El renderizado se realiza directamente en el framebuffer de Linux.

## Características Principales

*   **Rasterización por Software**: Implementa un rasterizador completo desde cero, incluyendo:
    *   Dibujo de líneas (Algoritmo de Bresenham).
    *   Dibujo de triángulos (con interpolación baricéntrica).
    *   Z-buffering para la correcta gestión de la profundidad.
*   **Pipeline de Renderizado 3D**:
    *   Transformaciones Modelo-Vista-Proyección (MVP).
    *   Cámara configurable (`look_at`, proyección perspectiva).
    *   Transformaciones jerárquicas de objetos (escena gráfica).
*   **Carga de Modelos**: Soporte para cargar modelos 3D desde archivos `.obj`.
*   **Modos de Renderizado**:
    *   Wireframe (malla de alambre).
    *   Color plano (interpolación de color de vértices).
    *   Texturizado (con carga de texturas desde archivos PNG).
*   **Shading**:
    *   Sistema de shaders intercambiables.
    *   Incluye shaders básicos (Default, Simple con iluminación Phong básica, Multi-shader).
*   **Renderizado de Texto**:
    *   Soporte para fuentes bitmap (`.fnt` y `.png`).
    *   Renderizado de texto 2D en pantalla con escala y rotación.
    *   Sistema de caché de texto para optimizar el rendimiento.
*   **Salida a Framebuffer**: Renderiza directamente al framebuffer de Linux (`/dev/fb0`) usando doble buffering para evitar parpadeos.
*   **Entrada Táctil**: Soporte básico para entrada táctil desde `/dev/input/eventX` para interacción (rotación, zoom).
*   **Paralelización**: Utiliza Rayon para paralelizar partes del proceso de renderizado.

## Requisitos

*   **Rust**: Compilador de Rust y Cargo (generalmente instalados juntos).
*   **Para la biblioteca `spuc3d`**: Ninguno específico del sistema operativo. La biblioteca principal es agnóstica al SO.
*   **Para el ejemplo `rotating_cube` (y otros que usen framebuffer/input directo)**:
    *   **Linux**: El sistema operativo debe ser Linux con acceso al framebuffer (`/dev/fb0`) y a los dispositivos de entrada (`/dev/input/*`).
    *   **Permisos**: Es posible que necesites permisos de superusuario o pertenecer a grupos específicos (`video`, `input`) para acceder al framebuffer y a los dispositivos de entrada.

## Cómo Compilar y Ejecutar

1.  **Clona el repositorio**:
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd spuc3d
    ```
2.  **Compila y ejecuta el ejemplo**:
    El ejemplo principal que demuestra la mayoría de las características es `rotating_cube`.
    ```bash
    cargo run --example rotating_cube
    ```
    *Nota: Puede que necesites ejecutarlo con `sudo` si no tienes los permisos adecuados para el framebuffer.*

## Estructura del Proyecto

*   `src/`: Contiene el código fuente principal del motor de renderizado.
    *   `renderer/`: Módulos principales del renderizador (core, geometry, shader, texture, font).
    *   `lib.rs`: Punto de entrada de la biblioteca.
*   `examples/`: Contiene ejemplos de uso del motor.
    *   `rotating_cube.rs`: Demostración principal con objetos animados, texto y cambio de shaders/modos.
*   `assets/`: Recursos como modelos `.obj`, texturas `.png` y fuentes `.fnt`.

## Controles (Ejemplo `rotating_cube`)

*   **Rotación Automática**: El sistema solar rota automáticamente por defecto.
*   **Interacción Táctil**:
    *   **Arrastrar horizontalmente**: Rota la vista de la cámara alrededor de la escena.
    *   **Arrastrar verticalmente**: Acerca o aleja la cámara (zoom).
    *   La rotación automática se desactiva al interactuar con la pantalla táctil.
*   **Ctrl+C**: Detiene la ejecución del programa.

## Próximos Pasos / Mejoras Posibles

*   Implementar técnicas de antialiasing.
*   Añadir más tipos de luces (puntuales, focos).
*   Optimizar aún más el rasterizador.
*   Soporte para más formatos de modelos 3D.
*   Mejorar el sistema de materiales y shaders.
