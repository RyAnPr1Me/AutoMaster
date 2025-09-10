use software_gpu::*;
use software_gpu::vertex::Vertex;
use software_gpu::pipeline::shaders;
use nalgebra::{Vector3, Vector4};
use log::info;

fn main() {
    // Initialize logging
    env_logger::init();

    info!("Starting Software GPU Demo");

    // Create a software GPU instance
    let mut gpu = SoftwareGpu::new(800, 600);

    // Demo 1: Simple triangle rendering
    demo_triangle_rendering(&mut gpu);

    // Demo 2: Compute shader usage
    demo_compute_shaders(&mut gpu);

    // Demo 3: Multiple triangles with different shaders
    demo_multiple_triangles(&mut gpu);

    // Print GPU statistics
    let stats = gpu.get_stats();
    println!("GPU Statistics:");
    println!("  Memory Used: {}/{} bytes", stats.memory_used, stats.memory_total);
    println!("  Core Count: {}", stats.core_count);
    println!("  Triangles Processed: {}", stats.triangles_processed);

    info!("Software GPU Demo Complete");
}

fn demo_triangle_rendering(gpu: &mut SoftwareGpu) {
    info!("Demo 1: Triangle Rendering");

    // Clear framebuffer
    gpu.clear([0.1, 0.1, 0.2, 1.0]);

    // Create a simple triangle
    let vertices = vec![
        Vertex::simple(Vector3::new(-0.5, -0.5, 0.0), Vector4::new(1.0, 0.0, 0.0, 1.0)), // Red
        Vertex::simple(Vector3::new(0.5, -0.5, 0.0), Vector4::new(0.0, 1.0, 0.0, 1.0)),  // Green
        Vertex::simple(Vector3::new(0.0, 0.5, 0.0), Vector4::new(0.0, 0.0, 1.0, 1.0)),   // Blue
    ];

    let indices = vec![0, 1, 2];

    // Set a simple color fragment shader
    gpu.set_fragment_shader(shaders::color_shader(Vector4::new(0.8, 0.6, 0.4, 1.0)));

    // Render the triangle
    gpu.draw_triangles(&vertices, &indices);

    // Save the result
    if let Err(e) = gpu.save_image("triangle_demo.png") {
        eprintln!("Failed to save triangle demo: {}", e);
    } else {
        info!("Triangle demo saved to triangle_demo.png");
    }
}

fn demo_compute_shaders(gpu: &mut SoftwareGpu) {
    info!("Demo 2: Compute Shaders");

    // Allocate buffers for vector addition
    let buffer_a = gpu.allocate_buffer(1024).expect("Failed to allocate buffer A");
    let buffer_b = gpu.allocate_buffer(1024).expect("Failed to allocate buffer B");
    let buffer_result = gpu.allocate_buffer(1024).expect("Failed to allocate result buffer");

    // Initialize input data
    let data_a: Vec<u8> = (0..256).flat_map(|i| (i as f32).to_le_bytes().to_vec()).collect();
    let data_b: Vec<u8> = (0..256).flat_map(|i| ((i * 2) as f32).to_le_bytes().to_vec()).collect();

    gpu.write_buffer(buffer_a, &data_a).expect("Failed to write buffer A");
    gpu.write_buffer(buffer_b, &data_b).expect("Failed to write buffer B");

    // Set up compute shader for vector addition
    gpu.compute.set_shader(software_gpu::compute::compute_shaders::vector_add_shader(
        buffer_a,
        buffer_b,
        buffer_result,
        256,
    ));

    // Dispatch compute shader
    gpu.dispatch_compute(256, 1, 1);

    // Read back results
    if let Ok(result_data) = gpu.read_buffer(buffer_result) {
        let results: Vec<f32> = result_data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .take(10) // Show first 10 results
            .collect();
        
        info!("Vector addition results (first 10): {:?}", results);
    }
}

fn demo_multiple_triangles(gpu: &mut SoftwareGpu) {
    info!("Demo 3: Multiple Triangles");

    // Clear framebuffer
    gpu.clear([0.0, 0.0, 0.0, 1.0]);

    // Create multiple triangles
    let vertices = vec![
        // Triangle 1 (left)
        Vertex::simple(Vector3::new(-0.8, -0.3, 0.0), Vector4::new(1.0, 0.0, 0.0, 1.0)),
        Vertex::simple(Vector3::new(-0.4, -0.3, 0.0), Vector4::new(1.0, 0.0, 0.0, 1.0)),
        Vertex::simple(Vector3::new(-0.6, 0.3, 0.0), Vector4::new(1.0, 0.0, 0.0, 1.0)),
        
        // Triangle 2 (right)
        Vertex::simple(Vector3::new(0.4, -0.3, 0.0), Vector4::new(0.0, 1.0, 0.0, 1.0)),
        Vertex::simple(Vector3::new(0.8, -0.3, 0.0), Vector4::new(0.0, 1.0, 0.0, 1.0)),
        Vertex::simple(Vector3::new(0.6, 0.3, 0.0), Vector4::new(0.0, 1.0, 0.0, 1.0)),
        
        // Triangle 3 (center, smaller)
        Vertex::simple(Vector3::new(-0.1, -0.1, 0.1), Vector4::new(0.0, 0.0, 1.0, 1.0)),
        Vertex::simple(Vector3::new(0.1, -0.1, 0.1), Vector4::new(0.0, 0.0, 1.0, 1.0)),
        Vertex::simple(Vector3::new(0.0, 0.1, 0.1), Vector4::new(0.0, 0.0, 1.0, 1.0)),
    ];

    let indices = vec![
        0, 1, 2,  // Triangle 1
        3, 4, 5,  // Triangle 2
        6, 7, 8,  // Triangle 3
    ];

    // Use Phong lighting shader
    gpu.set_fragment_shader(shaders::phong_shader(
        Vector3::new(2.0, 2.0, 2.0),    // Light position
        Vector3::new(1.0, 1.0, 1.0),    // Light color
        Vector3::new(0.0, 0.0, 2.0),    // Camera position
    ));

    // Render triangles
    gpu.draw_triangles(&vertices, &indices);

    // Save the result
    if let Err(e) = gpu.save_image("multiple_triangles_demo.png") {
        eprintln!("Failed to save multiple triangles demo: {}", e);
    } else {
        info!("Multiple triangles demo saved to multiple_triangles_demo.png");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main_demo() {
        // Test that the main demo runs without panicking
        main();
    }

    #[test]
    fn test_triangle_demo() {
        let mut gpu = SoftwareGpu::new(100, 100);
        demo_triangle_rendering(&mut gpu);
        
        let stats = gpu.get_stats();
        assert!(stats.triangles_processed > 0);
    }

    #[test]
    fn test_compute_demo() {
        let mut gpu = SoftwareGpu::new(100, 100);
        demo_compute_shaders(&mut gpu);
        
        let stats = gpu.get_stats();
        assert!(stats.memory_used > 0);
    }
}