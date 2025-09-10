use nalgebra::{Vector3, Vector4};
use std::sync::{Arc, Mutex};
use log::info;

pub mod vertex;
pub mod rasterizer;
pub mod framebuffer;
pub mod pipeline;
pub mod shaders;
pub mod memory;
pub mod compute;

use vertex::Vertex;
use framebuffer::Framebuffer;
use pipeline::Pipeline;
use memory::GpuMemory;
use compute::ComputeShader;

/// Software GPU implementation that runs entirely on CPU
pub struct SoftwareGpu {
    /// Main framebuffer for rendering
    pub framebuffer: Framebuffer,
    /// GPU memory manager
    pub memory: GpuMemory,
    /// Graphics pipeline state
    pub pipeline: Pipeline,
    /// Compute shader execution engine
    pub compute: ComputeShader,
    /// Number of processing cores to simulate
    pub core_count: usize,
}

impl SoftwareGpu {
    /// Create a new software GPU instance
    pub fn new(width: u32, height: u32) -> Self {
        info!("Initializing Software GPU with resolution {}x{}", width, height);
        
        Self {
            framebuffer: Framebuffer::new(width, height),
            memory: GpuMemory::new(1024 * 1024 * 1024), // 1GB simulated GPU memory
            pipeline: Pipeline::new(),
            compute: ComputeShader::new(),
            core_count: rayon::current_num_threads(),
        }
    }

    /// Clear the framebuffer with a color
    pub fn clear(&mut self, color: [f32; 4]) {
        self.framebuffer.clear(color);
    }

    /// Draw triangles using the graphics pipeline
    pub fn draw_triangles(&mut self, vertices: &[Vertex], indices: &[u32]) {
        info!("Drawing {} triangles", indices.len() / 3);
        
        // Process triangles in parallel
        let triangles: Vec<_> = indices
            .chunks_exact(3)
            .map(|tri| [vertices[tri[0] as usize], vertices[tri[1] as usize], vertices[tri[2] as usize]])
            .collect();

        for triangle in triangles {
            let framebuffer = Arc::new(Mutex::new(&mut self.framebuffer));
            self.pipeline.process_triangle(triangle, &framebuffer);
        }
    }

    /// Execute a compute shader
    pub fn dispatch_compute(&mut self, local_size_x: u32, local_size_y: u32, local_size_z: u32) {
        self.compute.dispatch(local_size_x, local_size_y, local_size_z, &mut self.memory);
    }

    /// Allocate GPU memory buffer
    pub fn allocate_buffer(&mut self, size: usize) -> Result<u32, String> {
        self.memory.allocate(size)
    }

    /// Write data to GPU buffer
    pub fn write_buffer(&mut self, buffer_id: u32, data: &[u8]) -> Result<(), String> {
        self.memory.write(buffer_id, data)
    }

    /// Read data from GPU buffer
    pub fn read_buffer(&self, buffer_id: u32) -> Result<Vec<u8>, String> {
        self.memory.read(buffer_id)
    }

    /// Set vertex shader
    pub fn set_vertex_shader(&mut self, shader: Box<dyn Fn(Vertex) -> Vertex + Send + Sync>) {
        self.pipeline.set_vertex_shader(shader);
    }

    /// Set fragment shader
    pub fn set_fragment_shader(&mut self, shader: Box<dyn Fn(Vector3<f32>, Vector3<f32>) -> Vector4<f32> + Send + Sync>) {
        self.pipeline.set_fragment_shader(shader);
    }

    /// Get framebuffer data as image
    pub fn get_image(&self) -> image::RgbaImage {
        self.framebuffer.to_image()
    }

    /// Save framebuffer to file
    pub fn save_image(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.framebuffer.save(path)
    }

    /// Get GPU statistics
    pub fn get_stats(&self) -> GpuStats {
        GpuStats {
            memory_used: self.memory.get_used_memory(),
            memory_total: self.memory.get_total_memory(),
            core_count: self.core_count,
            triangles_processed: self.pipeline.get_triangle_count(),
        }
    }
}

/// GPU performance and usage statistics
#[derive(Debug, Clone)]
pub struct GpuStats {
    pub memory_used: usize,
    pub memory_total: usize,
    pub core_count: usize,
    pub triangles_processed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_creation() {
        let gpu = SoftwareGpu::new(800, 600);
        assert_eq!(gpu.framebuffer.width, 800);
        assert_eq!(gpu.framebuffer.height, 600);
    }

    #[test]
    fn test_memory_allocation() {
        let mut gpu = SoftwareGpu::new(100, 100);
        let buffer = gpu.allocate_buffer(1024).unwrap();
        assert!(buffer > 0);
    }

    #[test]
    fn test_buffer_operations() {
        let mut gpu = SoftwareGpu::new(100, 100);
        let buffer = gpu.allocate_buffer(1024).unwrap();
        
        let data = vec![1, 2, 3, 4, 5];
        gpu.write_buffer(buffer, &data).unwrap();
        
        let read_data = gpu.read_buffer(buffer).unwrap();
        assert_eq!(data, read_data[..5]);
    }
}