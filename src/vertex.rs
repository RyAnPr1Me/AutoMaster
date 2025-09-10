use nalgebra::{Vector3, Vector4, Vector2};

/// Vertex data structure for the graphics pipeline
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    /// 3D position in world space
    pub position: Vector3<f32>,
    /// Texture coordinates
    pub texcoords: Vector2<f32>,
    /// Vertex normal for lighting
    pub normal: Vector3<f32>,
    /// Vertex color
    pub color: Vector4<f32>,
}

impl Vertex {
    /// Create a new vertex
    pub fn new(
        position: Vector3<f32>,
        texcoords: Vector2<f32>,
        normal: Vector3<f32>,
        color: Vector4<f32>,
    ) -> Self {
        Self {
            position,
            texcoords,
            normal,
            color,
        }
    }

    /// Create a simple vertex with just position and color
    pub fn simple(position: Vector3<f32>, color: Vector4<f32>) -> Self {
        Self {
            position,
            texcoords: Vector2::new(0.0, 0.0),
            normal: Vector3::new(0.0, 0.0, 1.0),
            color,
        }
    }

    /// Transform vertex position by a matrix
    pub fn transform(&mut self, matrix: &nalgebra::Matrix4<f32>) {
        let pos_homogeneous = matrix * Vector4::new(self.position.x, self.position.y, self.position.z, 1.0);
        self.position = Vector3::new(
            pos_homogeneous.x / pos_homogeneous.w,
            pos_homogeneous.y / pos_homogeneous.w,
            pos_homogeneous.z / pos_homogeneous.w,
        );
    }

    /// Interpolate between two vertices
    pub fn lerp(&self, other: &Vertex, t: f32) -> Vertex {
        Vertex {
            position: self.position + t * (other.position - self.position),
            texcoords: self.texcoords + t * (other.texcoords - self.texcoords),
            normal: (self.normal + t * (other.normal - self.normal)).normalize(),
            color: self.color + t * (other.color - self.color),
        }
    }
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            texcoords: Vector2::zeros(),
            normal: Vector3::new(0.0, 0.0, 1.0),
            color: Vector4::new(1.0, 1.0, 1.0, 1.0),
        }
    }
}