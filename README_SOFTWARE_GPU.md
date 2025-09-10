# Software GPU in Rust

A complete software-based GPU implementation that runs entirely on CPU, written in Rust. This project provides a fully functional graphics processing unit emulation with support for both graphics rendering and compute shaders.

## Features

### Graphics Pipeline
- **Vertex Processing**: Configurable vertex shaders with MVP matrix transformations
- **Rasterization**: Triangle rasterization with barycentric coordinate interpolation
- **Fragment Shading**: Customizable fragment shaders with lighting support
- **Depth Testing**: Z-buffer for proper depth handling
- **Parallel Processing**: Multi-threaded rendering using Rayon

### Compute Shaders
- **Parallel Execution**: SIMD-style compute shader dispatch
- **Memory Management**: GPU-style buffer allocation and management
- **Built-in Shaders**: Vector addition, matrix multiplication, image processing

### Memory Management
- **Buffer Allocation**: Dynamic GPU memory management
- **Multiple Buffer Types**: Vertex, Index, Uniform, Texture, Storage buffers
- **Memory Pools**: Efficient allocation for similar-sized objects

## Architecture

```
SoftwareGpu
├── Framebuffer     - RGBA color buffer + depth buffer
├── Pipeline        - Graphics pipeline with vertex/fragment shaders
├── GpuMemory       - Buffer allocation and management
├── ComputeShader   - Compute shader execution engine
└── Rasterizer      - Triangle rasterization and line drawing
```

## Usage

### Basic Triangle Rendering

```rust
use software_gpu::*;
use software_gpu::vertex::Vertex;
use nalgebra::{Vector3, Vector4};

// Create a software GPU instance
let mut gpu = SoftwareGpu::new(800, 600);

// Clear the framebuffer
gpu.clear([0.1, 0.1, 0.2, 1.0]);

// Create triangle vertices
let vertices = vec![
    Vertex::simple(Vector3::new(-0.5, -0.5, 0.0), Vector4::new(1.0, 0.0, 0.0, 1.0)),
    Vertex::simple(Vector3::new(0.5, -0.5, 0.0), Vector4::new(0.0, 1.0, 0.0, 1.0)),
    Vertex::simple(Vector3::new(0.0, 0.5, 0.0), Vector4::new(0.0, 0.0, 1.0, 1.0)),
];

let indices = vec![0, 1, 2];

// Render the triangle
gpu.draw_triangles(&vertices, &indices);

// Save to image
gpu.save_image("output.png").unwrap();
```

### Compute Shader Example

```rust
// Allocate GPU buffers
let buffer_a = gpu.allocate_buffer(1024).unwrap();
let buffer_b = gpu.allocate_buffer(1024).unwrap();
let result_buffer = gpu.allocate_buffer(1024).unwrap();

// Write input data
gpu.write_buffer(buffer_a, &input_data_a).unwrap();
gpu.write_buffer(buffer_b, &input_data_b).unwrap();

// Set up vector addition compute shader
gpu.compute.set_shader(
    compute_shaders::vector_add_shader(buffer_a, buffer_b, result_buffer, 256)
);

// Execute compute shader
gpu.dispatch_compute(256, 1, 1);

// Read results
let results = gpu.read_buffer(result_buffer).unwrap();
```

### Custom Shaders

```rust
// Custom fragment shader with lighting
gpu.set_fragment_shader(Box::new(|world_pos, normal| {
    let light_dir = Vector3::new(0.0, 0.0, 1.0);
    let intensity = normal.dot(&light_dir).max(0.0);
    let color = Vector3::new(0.8, 0.6, 0.4) * intensity;
    Vector4::new(color.x, color.y, color.z, 1.0)
}));

// Custom vertex shader
gpu.set_vertex_shader(Box::new(|mut vertex| {
    // Apply custom transformation
    vertex.position.y += 0.1 * (vertex.position.x * 10.0).sin();
    vertex
}));
```

## Building and Running

```bash
# Build the project
cargo build --release

# Run the demo
cargo run

# Run tests
cargo test
```

## Demo Outputs

The main demo creates several example renders:

1. **triangle_demo.png** - Basic triangle with color interpolation
2. **multiple_triangles_demo.png** - Multiple triangles with Phong lighting
3. **Console output** - Compute shader results and GPU statistics

## Performance

The software GPU is designed for educational purposes and proof-of-concept implementations. Performance characteristics:

- **Parallel Processing**: Utilizes all available CPU cores via Rayon
- **Memory Efficiency**: Custom memory management with pooling
- **Scalable**: Performance scales with CPU core count

## API Reference

### Core Types

- **`SoftwareGpu`** - Main GPU instance
- **`Vertex`** - Vertex data structure with position, normal, texcoords, color
- **`Framebuffer`** - Color and depth buffer management
- **`Pipeline`** - Graphics pipeline configuration
- **`GpuMemory`** - Memory buffer management
- **`ComputeShader`** - Compute shader execution

### Key Methods

- **`new(width, height)`** - Create GPU instance
- **`clear(color)`** - Clear framebuffer
- **`draw_triangles(vertices, indices)`** - Render triangles
- **`dispatch_compute(x, y, z)`** - Execute compute shader
- **`allocate_buffer(size)`** - Allocate GPU memory
- **`save_image(path)`** - Export framebuffer to image

## Technical Details

### Rasterization Algorithm
- Barycentric coordinate-based triangle rasterization
- Sub-pixel precision for anti-aliasing potential
- Parallel per-pixel processing

### Memory Layout
- Contiguous memory buffers for cache efficiency
- Reference counting for buffer management
- Automatic memory defragmentation

### Shader Model
- Function pointer-based shader system
- Full access to vertex attributes and uniforms
- Support for complex lighting calculations

## Dependencies

- **rayon** - Parallel processing
- **nalgebra** - Linear algebra operations
- **image** - Image loading/saving
- **bytemuck** - Safe byte manipulation
- **log** - Logging framework

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Educational Purpose

This software GPU implementation serves as:

1. **Learning Tool** - Understanding GPU architecture and graphics pipelines
2. **Prototyping Platform** - Testing rendering algorithms without hardware constraints
3. **Reference Implementation** - Example of modern Rust graphics programming
4. **Research Base** - Foundation for custom rendering techniques

The implementation prioritizes clarity and educational value over raw performance, making it an excellent resource for understanding how GPUs work internally.