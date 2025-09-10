use nalgebra::Vector4;
use image::{RgbaImage, Rgba};

/// Framebuffer for storing rendered pixels
pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    /// Color buffer (RGBA)
    pub color_buffer: Vec<Vector4<f32>>,
    /// Depth buffer for z-testing
    pub depth_buffer: Vec<f32>,
}

impl Framebuffer {
    /// Create a new framebuffer
    pub fn new(width: u32, height: u32) -> Self {
        let pixel_count = (width * height) as usize;
        Self {
            width,
            height,
            color_buffer: vec![Vector4::new(0.0, 0.0, 0.0, 1.0); pixel_count],
            depth_buffer: vec![f32::INFINITY; pixel_count],
        }
    }

    /// Clear the framebuffer with a color
    pub fn clear(&mut self, color: [f32; 4]) {
        let clear_color = Vector4::new(color[0], color[1], color[2], color[3]);
        self.color_buffer.fill(clear_color);
        self.depth_buffer.fill(f32::INFINITY);
    }

    /// Set a pixel color with depth testing
    pub fn set_pixel(&mut self, x: u32, y: u32, color: Vector4<f32>, depth: f32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }

        let index = (y * self.width + x) as usize;
        
        // Depth test
        if depth < self.depth_buffer[index] {
            self.depth_buffer[index] = depth;
            self.color_buffer[index] = color;
            true
        } else {
            false
        }
    }

    /// Get pixel color at coordinates
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<Vector4<f32>> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let index = (y * self.width + x) as usize;
        Some(self.color_buffer[index])
    }

    /// Convert framebuffer to image
    pub fn to_image(&self) -> RgbaImage {
        let mut image = RgbaImage::new(self.width, self.height);
        
        for y in 0..self.height {
            for x in 0..self.width {
                let index = (y * self.width + x) as usize;
                let color = self.color_buffer[index];
                
                let r = (color.x.clamp(0.0, 1.0) * 255.0) as u8;
                let g = (color.y.clamp(0.0, 1.0) * 255.0) as u8;
                let b = (color.z.clamp(0.0, 1.0) * 255.0) as u8;
                let a = (color.w.clamp(0.0, 1.0) * 255.0) as u8;
                
                image.put_pixel(x, y, Rgba([r, g, b, a]));
            }
        }
        
        image
    }

    /// Save framebuffer to file
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let image = self.to_image();
        image.save(path)?;
        Ok(())
    }

    /// Resize framebuffer
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let pixel_count = (width * height) as usize;
        self.color_buffer.resize(pixel_count, Vector4::new(0.0, 0.0, 0.0, 1.0));
        self.depth_buffer.resize(pixel_count, f32::INFINITY);
    }
}