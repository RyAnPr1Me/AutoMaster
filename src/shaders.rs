use nalgebra::{Vector3, Vector4};

/// Collection of built-in shader functions
pub struct Shaders;

impl Shaders {
    /// Identity vertex shader (no transformation)
    pub fn identity_vertex_shader() -> Box<dyn Fn(crate::vertex::Vertex) -> crate::vertex::Vertex + Send + Sync> {
        Box::new(|vertex| vertex)
    }

    /// Simple MVP vertex shader
    pub fn mvp_vertex_shader(mvp: nalgebra::Matrix4<f32>) -> Box<dyn Fn(crate::vertex::Vertex) -> crate::vertex::Vertex + Send + Sync> {
        Box::new(move |mut vertex| {
            vertex.transform(&mvp);
            vertex
        })
    }

    /// Solid color fragment shader
    pub fn solid_color_fragment_shader(color: Vector4<f32>) -> Box<dyn Fn(Vector3<f32>, Vector3<f32>) -> Vector4<f32> + Send + Sync> {
        Box::new(move |_world_pos, _normal| color)
    }

    /// Simple lighting fragment shader
    pub fn simple_lighting_fragment_shader() -> Box<dyn Fn(Vector3<f32>, Vector3<f32>) -> Vector4<f32> + Send + Sync> {
        Box::new(|_world_pos, normal| {
            let light_dir = Vector3::new(0.0, 0.0, 1.0);
            let intensity = normal.dot(&light_dir).max(0.0);
            let color = Vector3::new(0.8, 0.6, 0.4) * intensity + Vector3::new(0.2, 0.2, 0.2);
            Vector4::new(color.x, color.y, color.z, 1.0)
        })
    }

    /// Normal visualization fragment shader
    pub fn normal_fragment_shader() -> Box<dyn Fn(Vector3<f32>, Vector3<f32>) -> Vector4<f32> + Send + Sync> {
        Box::new(|_world_pos, normal| {
            let color = (normal + Vector3::new(1.0, 1.0, 1.0)) * 0.5;
            Vector4::new(color.x, color.y, color.z, 1.0)
        })
    }

    /// Checkerboard pattern fragment shader
    pub fn checkerboard_fragment_shader(scale: f32) -> Box<dyn Fn(Vector3<f32>, Vector3<f32>) -> Vector4<f32> + Send + Sync> {
        Box::new(move |world_pos, _normal| {
            let x = (world_pos.x * scale).floor() as i32;
            let z = (world_pos.z * scale).floor() as i32;
            let checker = (x + z) % 2;
            let intensity = if checker == 0 { 0.8 } else { 0.2 };
            Vector4::new(intensity, intensity, intensity, 1.0)
        })
    }

    /// Gradient fragment shader
    pub fn gradient_fragment_shader(color1: Vector4<f32>, color2: Vector4<f32>) -> Box<dyn Fn(Vector3<f32>, Vector3<f32>) -> Vector4<f32> + Send + Sync> {
        Box::new(move |world_pos, _normal| {
            let t = (world_pos.y + 1.0) * 0.5; // Assuming Y range [-1, 1]
            color1 + t * (color2 - color1)
        })
    }
}