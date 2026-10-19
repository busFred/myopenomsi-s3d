//! Text and colour swatches over the picture: the game's HUD overlays (`Scene::overlays`,
//! a texture in a rectangle of window pixels) with the interface's font rasterised on the
//! CPU. Textures are kept by their text and freed together when there are too many.

use omsi_render::{Renderer, Scene, TextureId};
use omsi_ui::{Fonts, Weight};
use std::collections::HashMap;

pub struct Hud {
    fonts: Fonts,
    texts: HashMap<(String, u32, [u8; 4]), (TextureId, u32, u32)>,
    swatches: HashMap<[u8; 4], TextureId>,
    /// Textures made this frame (never freed before the frame is drawn).
    used: Vec<TextureId>,
}

/// Text alignment at the anchor.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Centre,
}

fn rgba(c: [f32; 4]) -> [u8; 4] {
    c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
}

impl Hud {
    pub fn new() -> Hud {
        Hud { fonts: Fonts::new(), texts: HashMap::new(), swatches: HashMap::new(), used: Vec::new() }
    }

    /// A new frame: the overlays of the last one go.
    pub fn begin(&mut self, renderer: &Renderer, scene: &mut Scene) {
        scene.overlays.clear();
        self.used.clear();
        if self.texts.len() > 600 {
            for (_, (t, _, _)) in self.texts.drain() {
                renderer.free_texture(scene, t);
            }
        }
    }

    fn swatch(&mut self, renderer: &Renderer, scene: &mut Scene, colour: [f32; 4]) -> TextureId {
        let k = rgba(colour);
        if let Some(&t) = self.swatches.get(&k) {
            return t;
        }
        let img = omsi_texture::Image { width: 2, height: 2, rgba: k.repeat(4), has_alpha: true };
        let t = renderer.add_texture(scene, &img, false);
        self.swatches.insert(k, t);
        t
    }

    fn text_texture(&mut self, renderer: &Renderer, scene: &mut Scene, text: &str, px: f32, colour: [f32; 4]) -> (TextureId, u32, u32) {
        let k = (text.to_string(), px as u32, rgba(colour));
        if let Some(&t) = self.texts.get(&k) {
            return t;
        }
        let bm = self.fonts.render(text, px, Weight::Medium);
        let (w, h) = (bm.w.max(1), bm.h.max(1));
        let c = rgba(colour);
        let mut data = vec![0u8; (w * h * 4) as usize];
        for (i, a) in bm.alpha.iter().enumerate().take((w * h) as usize) {
            let a = (*a as u32 * c[3] as u32 / 255) as u8;
            data[i * 4..i * 4 + 4].copy_from_slice(&[c[0], c[1], c[2], a]);
        }
        let img = omsi_texture::Image { width: w, height: h, rgba: data, has_alpha: true };
        let t = renderer.add_texture(scene, &img, false);
        self.texts.insert(k, (t, w, h));
        (t, w, h)
    }

    /// A filled rectangle.
    pub fn rect(&mut self, renderer: &Renderer, scene: &mut Scene, x0: f32, y0: f32, x1: f32, y1: f32, colour: [f32; 4]) {
        let t = self.swatch(renderer, scene, colour);
        scene.overlays.push((t, [x0, y0, x1, y1]));
    }

    /// One line of text at (x, y) (its top edge); the width drawn.
    pub fn text(&mut self, renderer: &Renderer, scene: &mut Scene, text: &str, px: f32, x: f32, y: f32, align: Align, colour: [f32; 4]) -> (f32, f32) {
        if text.trim().is_empty() {
            return (0.0, px * 1.3);
        }
        let (t, w, h) = self.text_texture(renderer, scene, text, px, colour);
        self.used.push(t);
        let (w, h) = (w as f32, h as f32);
        let x0 = match align {
            Align::Left => x,
            Align::Centre => x - w * 0.5,
        };
        scene.overlays.push((t, [x0, y, x0 + w, y + h]));
        (w, h)
    }

    /// The size a panel of `lines` takes.
    pub fn panel_size(&self, lines: &[String], px: f32) -> (f32, f32) {
        let pad = px * 0.5;
        let w = lines.iter().map(|l| self.fonts.width(l, px, Weight::Medium)).fold(0.0, f32::max);
        (w + 2.0 * pad, lines.len() as f32 * px * 1.3 + 2.0 * pad)
    }

    /// Lines of text on a dark plate at (x, y); the plate's size.
    pub fn panel(&mut self, renderer: &Renderer, scene: &mut Scene, lines: &[String], px: f32, x: f32, y: f32, colour: [f32; 4]) -> (f32, f32) {
        let pad = px * 0.5;
        let line_h = px * 1.3;
        let mut w = 0.0f32;
        for l in lines {
            w = w.max(self.fonts.width(l, px, Weight::Medium));
        }
        let h = lines.len() as f32 * line_h + 2.0 * pad;
        let w = w + 2.0 * pad;
        self.rect(renderer, scene, x, y, x + w, y + h, [0.04, 0.05, 0.08, 0.72]);
        for (i, l) in lines.iter().enumerate() {
            self.text(renderer, scene, l, px, x + pad, y + pad + i as f32 * line_h, Align::Left, colour);
        }
        (w, h)
    }

    /// A text with a plate behind it centred on (x, y): a label over a vehicle.
    pub fn label(&mut self, renderer: &Renderer, scene: &mut Scene, text: &str, px: f32, x: f32, y: f32, colour: [f32; 4]) {
        let w = self.fonts.width(text, px, Weight::Medium);
        let pad = px * 0.3;
        self.rect(renderer, scene, x - w * 0.5 - pad, y - pad, x + w * 0.5 + pad, y + px * 1.2 + pad, [0.0, 0.0, 0.0, 0.55]);
        self.text(renderer, scene, text, px, x, y, Align::Centre, colour);
    }

    /// A legend: a swatch and a name per row, at (x, y) downwards.
    pub fn legend(&mut self, renderer: &Renderer, scene: &mut Scene, rows: &[([f32; 4], String)], px: f32, x: f32, y: f32) {
        let line_h = px * 1.35;
        let pad = px * 0.5;
        let w = rows.iter().map(|r| self.fonts.width(&r.1, px, Weight::Medium)).fold(0.0, f32::max) + px * 2.2 + 2.0 * pad;
        let h = rows.len() as f32 * line_h + 2.0 * pad;
        self.rect(renderer, scene, x, y, x + w, y + h, [0.04, 0.05, 0.08, 0.80]);
        for (i, (c, name)) in rows.iter().enumerate() {
            let yy = y + pad + i as f32 * line_h;
            let mut c = *c;
            c[3] = (c[3] + 0.4).min(1.0);
            self.rect(renderer, scene, x + pad, yy + px * 0.15, x + pad + px * 1.5, yy + px * 1.05, c);
            self.text(renderer, scene, name, px, x + pad + px * 2.2, yy, Align::Left, [0.92, 0.94, 0.97, 1.0]);
        }
    }
}
