use crate::memory::GpuMemory;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// Compute shader execution engine
pub struct ComputeShader {
    /// Workgroup size for parallel execution
    workgroup_size: (u32, u32, u32),
    /// Shader function
    shader_function: Option<Box<dyn Fn(u32, u32, u32, &mut GpuMemory) + Send + Sync>>,
}

impl ComputeShader {
    /// Create a new compute shader engine
    pub fn new() -> Self {
        Self {
            workgroup_size: (1, 1, 1),
            shader_function: None,
        }
    }

    /// Set the compute shader function
    pub fn set_shader(&mut self, shader: Box<dyn Fn(u32, u32, u32, &mut GpuMemory) + Send + Sync>) {
        self.shader_function = Some(shader);
    }

    /// Set workgroup size
    pub fn set_workgroup_size(&mut self, x: u32, y: u32, z: u32) {
        self.workgroup_size = (x, y, z);
    }

    /// Dispatch compute shader
    pub fn dispatch(&self, groups_x: u32, groups_y: u32, groups_z: u32, memory: &mut GpuMemory) {
        if let Some(ref shader) = self.shader_function {
            let total_threads = (groups_x * self.workgroup_size.0) *
                               (groups_y * self.workgroup_size.1) *
                               (groups_z * self.workgroup_size.2);

            // Execute shader for each thread ID in parallel
            let memory_arc = Arc::new(Mutex::new(memory));
            
            (0..total_threads).into_par_iter().for_each(|thread_id| {
                let z = thread_id / (groups_x * self.workgroup_size.0 * groups_y * self.workgroup_size.1);
                let remainder = thread_id % (groups_x * self.workgroup_size.0 * groups_y * self.workgroup_size.1);
                let y = remainder / (groups_x * self.workgroup_size.0);
                let x = remainder % (groups_x * self.workgroup_size.0);

                let mut mem = memory_arc.lock().unwrap();
                shader(x, y, z, &mut *mem);
            });
        }
    }
}

impl Default for ComputeShader {
    fn default() -> Self {
        Self::new()
    }
}

/// Built-in compute shaders
pub mod compute_shaders {
    use crate::memory::GpuMemory;

    /// Vector addition compute shader
    pub fn vector_add_shader(
        input_a_buffer: u32,
        input_b_buffer: u32,
        output_buffer: u32,
        count: u32,
    ) -> Box<dyn Fn(u32, u32, u32, &mut GpuMemory) + Send + Sync> {
        Box::new(move |x, _y, _z, memory| {
            if x < count {
                let index = x as usize * 4; // Assuming f32 values (4 bytes each)
                
                // Read from input buffers
                if let (Ok(data_a), Ok(data_b)) = (memory.read(input_a_buffer), memory.read(input_b_buffer)) {
                    if index + 4 <= data_a.len() && index + 4 <= data_b.len() {
                        let a = f32::from_le_bytes([data_a[index], data_a[index + 1], data_a[index + 2], data_a[index + 3]]);
                        let b = f32::from_le_bytes([data_b[index], data_b[index + 1], data_b[index + 2], data_b[index + 3]]);
                        let result = a + b;
                        
                        // Write to output buffer
                        let result_bytes = result.to_le_bytes();
                        if let Ok(mut output_data) = memory.read(output_buffer) {
                            if index + 4 <= output_data.len() {
                                output_data[index..index + 4].copy_from_slice(&result_bytes);
                                let _ = memory.write(output_buffer, &output_data);
                            }
                        }
                    }
                }
            }
        })
    }

    /// Matrix multiplication compute shader (simplified)
    pub fn matrix_mul_shader(
        matrix_a_buffer: u32,
        matrix_b_buffer: u32,
        result_buffer: u32,
        size: u32,
    ) -> Box<dyn Fn(u32, u32, u32, &mut GpuMemory) + Send + Sync> {
        Box::new(move |x, y, _z, memory| {
            if x < size && y < size {
                let row = y;
                let col = x;
                
                if let (Ok(data_a), Ok(data_b)) = (memory.read(matrix_a_buffer), memory.read(matrix_b_buffer)) {
                    let mut sum = 0.0f32;
                    
                    for k in 0..size {
                        let a_index = (row * size + k) as usize * 4;
                        let b_index = (k * size + col) as usize * 4;
                        
                        if a_index + 4 <= data_a.len() && b_index + 4 <= data_b.len() {
                            let a_val = f32::from_le_bytes([
                                data_a[a_index], data_a[a_index + 1], 
                                data_a[a_index + 2], data_a[a_index + 3]
                            ]);
                            let b_val = f32::from_le_bytes([
                                data_b[b_index], data_b[b_index + 1], 
                                data_b[b_index + 2], data_b[b_index + 3]
                            ]);
                            sum += a_val * b_val;
                        }
                    }
                    
                    // Write result
                    let result_index = (row * size + col) as usize * 4;
                    let result_bytes = sum.to_le_bytes();
                    if let Ok(mut result_data) = memory.read(result_buffer) {
                        if result_index + 4 <= result_data.len() {
                            result_data[result_index..result_index + 4].copy_from_slice(&result_bytes);
                            let _ = memory.write(result_buffer, &result_data);
                        }
                    }
                }
            }
        })
    }

    /// Image blur compute shader
    pub fn blur_shader(
        input_buffer: u32,
        output_buffer: u32,
        width: u32,
        height: u32,
        kernel_size: u32,
    ) -> Box<dyn Fn(u32, u32, u32, &mut GpuMemory) + Send + Sync> {
        Box::new(move |x, y, _z, memory| {
            if x < width && y < height {
                if let Ok(input_data) = memory.read(input_buffer) {
                    let mut sum = [0.0f32; 4]; // RGBA
                    let mut count = 0;
                    
                    let half_kernel = kernel_size / 2;
                    
                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let sample_x = (x as i32 + kx as i32 - half_kernel as i32).max(0).min(width as i32 - 1) as u32;
                            let sample_y = (y as i32 + ky as i32 - half_kernel as i32).max(0).min(height as i32 - 1) as u32;
                            
                            let pixel_index = (sample_y * width + sample_x) as usize * 4;
                            if pixel_index + 4 <= input_data.len() {
                                sum[0] += input_data[pixel_index] as f32;
                                sum[1] += input_data[pixel_index + 1] as f32;
                                sum[2] += input_data[pixel_index + 2] as f32;
                                sum[3] += input_data[pixel_index + 3] as f32;
                                count += 1;
                            }
                        }
                    }
                    
                    if count > 0 {
                        let output_index = (y * width + x) as usize * 4;
                        let avg = [
                            (sum[0] / count as f32) as u8,
                            (sum[1] / count as f32) as u8,
                            (sum[2] / count as f32) as u8,
                            (sum[3] / count as f32) as u8,
                        ];
                        
                        if let Ok(mut output_data) = memory.read(output_buffer) {
                            if output_index + 4 <= output_data.len() {
                                output_data[output_index..output_index + 4].copy_from_slice(&avg);
                                let _ = memory.write(output_buffer, &output_data);
                            }
                        }
                    }
                }
            }
        })
    }
}