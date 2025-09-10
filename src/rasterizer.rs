use crate::vertex::Vertex;
use crate::framebuffer::Framebuffer;
use nalgebra::{Vector3, Vector4};
use std::sync::{Arc, Mutex};

/// Triangle rasterizer
pub struct Rasterizer;

impl Rasterizer {
    /// Rasterize a triangle onto the framebuffer
    pub fn rasterize_triangle(
        triangle: [Vertex; 3],
        framebuffer: &Arc<Mutex<&mut Framebuffer>>,
        fragment_shader: &dyn Fn(Vector3<f32>, Vector3<f32>) -> Vector4<f32>,
    ) {
        let [v0, v1, v2] = triangle;

        // Convert to screen space (assuming vertices are already in NDC)
        let fb = framebuffer.lock().unwrap();
        let width = fb.width as f32;
        let height = fb.height as f32;
        drop(fb);

        let screen_v0 = Self::ndc_to_screen(v0.position, width, height);
        let screen_v1 = Self::ndc_to_screen(v1.position, width, height);
        let screen_v2 = Self::ndc_to_screen(v2.position, width, height);

        // Calculate bounding box
        let min_x = (screen_v0.x.min(screen_v1.x).min(screen_v2.x).floor() as i32).max(0);
        let max_x = (screen_v0.x.max(screen_v1.x).max(screen_v2.x).ceil() as i32).min(width as i32 - 1);
        let min_y = (screen_v0.y.min(screen_v1.y).min(screen_v2.y).floor() as i32).max(0);
        let max_y = (screen_v0.y.max(screen_v1.y).max(screen_v2.y).ceil() as i32).min(height as i32 - 1);

        // Rasterize pixels in bounding box
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = Vector3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);
                
                // Calculate barycentric coordinates
                if let Some((u, v, w)) = Self::barycentric_coords(p, screen_v0, screen_v1, screen_v2) {
                    if u >= 0.0 && v >= 0.0 && w >= 0.0 {
                        // Interpolate vertex attributes
                        let depth = u * screen_v0.z + v * screen_v1.z + w * screen_v2.z;
                        let world_pos = u * v0.position + v * v1.position + w * v2.position;
                        let normal = (u * v0.normal + v * v1.normal + w * v2.normal).normalize();

                        // Execute fragment shader
                        let color = fragment_shader(world_pos, normal);

                        // Write pixel with depth test
                        let mut fb = framebuffer.lock().unwrap();
                        fb.set_pixel(x as u32, y as u32, color, depth);
                    }
                }
            }
        }
    }

    /// Convert NDC coordinates to screen space
    fn ndc_to_screen(ndc: Vector3<f32>, width: f32, height: f32) -> Vector3<f32> {
        Vector3::new(
            (ndc.x + 1.0) * 0.5 * width,
            (1.0 - ndc.y) * 0.5 * height,
            ndc.z,
        )
    }

    /// Calculate barycentric coordinates
    fn barycentric_coords(
        p: Vector3<f32>,
        a: Vector3<f32>,
        b: Vector3<f32>,
        c: Vector3<f32>,
    ) -> Option<(f32, f32, f32)> {
        let v0 = c - a;
        let v1 = b - a;
        let v2 = p - a;

        let dot00 = v0.dot(&v0);
        let dot01 = v0.dot(&v1);
        let dot02 = v0.dot(&v2);
        let dot11 = v1.dot(&v1);
        let dot12 = v1.dot(&v2);

        let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
        let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
        let w = 1.0 - u - v;

        Some((w, v, u))
    }

    /// Draw a line between two points (Bresenham's algorithm)
    pub fn draw_line(
        p0: Vector3<f32>,
        p1: Vector3<f32>,
        color: Vector4<f32>,
        framebuffer: &Arc<Mutex<&mut Framebuffer>>,
    ) {
        let fb = framebuffer.lock().unwrap();
        let width = fb.width as f32;
        let height = fb.height as f32;
        drop(fb);

        let screen_p0 = Self::ndc_to_screen(p0, width, height);
        let screen_p1 = Self::ndc_to_screen(p1, width, height);

        let mut x0 = screen_p0.x as i32;
        let mut y0 = screen_p0.y as i32;
        let x1 = screen_p1.x as i32;
        let y1 = screen_p1.y as i32;

        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;

        loop {
            if x0 >= 0 && x0 < width as i32 && y0 >= 0 && y0 < height as i32 {
                let depth = screen_p0.z; // Simplified depth
                let mut fb = framebuffer.lock().unwrap();
                fb.set_pixel(x0 as u32, y0 as u32, color, depth);
            }

            if x0 == x1 && y0 == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x0 += sx;
            }
            if e2 < dx {
                err += dx;
                y0 += sy;
            }
        }
    }
}