use crate::vertex::Vertex;
use crate::framebuffer::Framebuffer;
use crate::rasterizer::Rasterizer;
use nalgebra::{Vector3, Vector4, Matrix4};
use std::sync::{Arc, Mutex};

/// Graphics pipeline state and execution
pub struct Pipeline {
    /// Vertex shader function
    vertex_shader: Option<Box<dyn Fn(Vertex) -> Vertex + Send + Sync>>,
    /// Fragment shader function
    fragment_shader: Option<Box<dyn Fn(Vector3<f32>, Vector3<f32>) -> Vector4<f32> + Send + Sync>>,
    /// Model-View-Projection matrix
    pub mvp_matrix: Matrix4<f32>,
    /// Triangle counter for statistics
    triangle_count: u64,
}

impl Pipeline {
    /// Create a new graphics pipeline
    pub fn new() -> Self {
        Self {
            vertex_shader: None,
            fragment_shader: Some(Box::new(Self::default_fragment_shader)),
            mvp_matrix: Matrix4::identity(),
            triangle_count: 0,
        }
    }

    /// Set the vertex shader
    pub fn set_vertex_shader(&mut self, shader: Box<dyn Fn(Vertex) -> Vertex + Send + Sync>) {
        self.vertex_shader = Some(shader);
    }

    /// Set the fragment shader
    pub fn set_fragment_shader(&mut self, shader: Box<dyn Fn(Vector3<f32>, Vector3<f32>) -> Vector4<f32> + Send + Sync>) {
        self.fragment_shader = Some(shader);
    }

    /// Set the MVP matrix
    pub fn set_mvp_matrix(&mut self, matrix: Matrix4<f32>) {
        self.mvp_matrix = matrix;
    }

    /// Process a triangle through the pipeline
    pub fn process_triangle(&mut self, triangle: [Vertex; 3], framebuffer: &Arc<Mutex<&mut Framebuffer>>) {
        self.triangle_count += 1;

        // Vertex shader stage
        let mut processed_triangle = triangle;
        if let Some(ref vs) = self.vertex_shader {
            for vertex in &mut processed_triangle {
                *vertex = vs(*vertex);
            }
        } else {
            // Default vertex processing: apply MVP matrix
            for vertex in &mut processed_triangle {
                vertex.transform(&self.mvp_matrix);
            }
        }

        // Clipping stage (simplified)
        if Self::should_cull_triangle(&processed_triangle) {
            return;
        }

        // Rasterization and fragment shader stage
        if let Some(ref fs) = self.fragment_shader {
            Rasterizer::rasterize_triangle(processed_triangle, framebuffer, fs.as_ref());
        }
    }

    /// Simple triangle culling (back-face and frustum)
    fn should_cull_triangle(triangle: &[Vertex; 3]) -> bool {
        // Check if all vertices are outside the frustum
        for vertex in triangle {
            let pos = vertex.position;
            if pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0 || pos.z < 0.0 || pos.z > 1.0 {
                continue;
            }
            return false; // At least one vertex is inside
        }
        true // All vertices are outside
    }

    /// Default fragment shader (returns vertex color)
    fn default_fragment_shader(_world_pos: Vector3<f32>, normal: Vector3<f32>) -> Vector4<f32> {
        // Simple lighting calculation
        let light_dir = Vector3::new(0.0, 0.0, 1.0).normalize();
        let diffuse = normal.dot(&light_dir).max(0.0);
        let color = Vector3::new(0.8, 0.8, 0.8) * diffuse + Vector3::new(0.2, 0.2, 0.2);
        Vector4::new(color.x, color.y, color.z, 1.0)
    }

    /// Get triangle count for statistics
    pub fn get_triangle_count(&self) -> u64 {
        self.triangle_count
    }

    /// Reset triangle counter
    pub fn reset_stats(&mut self) {
        self.triangle_count = 0;
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Predefined shader functions
pub mod shaders {
    use super::*;

    /// Simple color fragment shader
    pub fn color_shader(color: Vector4<f32>) -> Box<dyn Fn(Vector3<f32>, Vector3<f32>) -> Vector4<f32> + Send + Sync> {
        Box::new(move |_pos, _normal| color)
    }

    /// Phong lighting fragment shader
    pub fn phong_shader(
        light_pos: Vector3<f32>,
        light_color: Vector3<f32>,
        camera_pos: Vector3<f32>,
    ) -> Box<dyn Fn(Vector3<f32>, Vector3<f32>) -> Vector4<f32> + Send + Sync> {
        Box::new(move |world_pos, normal| {
            let light_dir = (light_pos - world_pos).normalize();
            let view_dir = (camera_pos - world_pos).normalize();
            let reflect_dir = light_dir - 2.0 * normal.dot(&light_dir) * normal; // Manual reflection

            // Ambient
            let ambient = 0.1 * light_color;

            // Diffuse
            let diff = normal.dot(&light_dir).max(0.0);
            let diffuse = diff * light_color;

            // Specular
            let spec = view_dir.dot(&reflect_dir).max(0.0).powi(32);
            let specular = spec * light_color;

            let result = ambient + diffuse + specular;
            Vector4::new(result.x, result.y, result.z, 1.0)
        })
    }

    /// Wireframe shader (for debugging)
    pub fn wireframe_shader() -> Box<dyn Fn(Vector3<f32>, Vector3<f32>) -> Vector4<f32> + Send + Sync> {
        Box::new(|_pos, _normal| Vector4::new(1.0, 1.0, 1.0, 1.0))
    }
}