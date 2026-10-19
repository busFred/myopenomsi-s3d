//! The window: the game's renderer drawing the session into a winit window, with the
//! keys and the mouse (see the status line's help, key F1..F4, 1-7, space, L, H, P, N).

use crate::overlay::{Layer, ProfileEdge};
use crate::session::{CamMode, Session};
use crate::timeline::Timeline;
use anyhow::{Context, Result};
use glam::DVec3;
use omsi_render::{Camera, Renderer, SurfaceState};
use openomsi_game::tool::{self, ToolSettings};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// What the window needs before its session exists.
pub struct Pending {
    pub root: PathBuf,
    pub settings: ToolSettings,
    pub timeline: Option<Timeline>,
    pub profile: Vec<ProfileEdge>,
    pub date: Option<i32>,
    pub radius_m: f64,
    pub title: String,
    pub start_time: Option<f64>,
    pub start_mode: CamMode,
}

pub struct Viewer {
    pending: Pending,
    instance: wgpu::Instance,
    window: Option<Arc<Window>>,
    surface: Option<SurfaceState<'static>>,
    session: Option<Session>,
    size: (u32, u32),
    // time
    t: f64,
    playing: bool,
    speed: f64,
    frame: usize,
    last: Instant,
    // camera
    free: Camera,
    mode: CamMode,
    keys: HashSet<KeyCode>,
    looking: bool,
    cursor: Option<(f32, f32)>,
    // state
    hovered: (Option<u64>, Option<usize>),
    pinned: Option<u64>,
    legend: bool,
    help: bool,
    error: Option<String>,
}

impl Viewer {
    pub fn run(pending: Pending) -> Result<()> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        let start_mode = pending.start_mode;
        let mut v = Viewer {
            instance: tool::graphics_instance(),
            pending,
            window: None,
            surface: None,
            session: None,
            size: (1600, 900),
            t: 0.0,
            playing: true,
            speed: 1.0,
            frame: 0,
            last: Instant::now(),
            free: Camera { position: DVec3::ZERO, yaw: 0.0, pitch: -20.0, roll: 0.0, fov_deg: 60.0, near: 0.1, far: 6000.0 },
            mode: start_mode,
            keys: HashSet::new(),
            looking: false,
            cursor: None,
            hovered: (None, None),
            pinned: None,
            legend: false,
            help: true,
            error: None,
        };
        event_loop.run_app(&mut v)?;
        if let Some(e) = v.error {
            anyhow::bail!("{e}");
        }
        Ok(())
    }

    fn open(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let (w, h) = self.pending.settings.resolution.unwrap_or((1600, 900));
        let attrs = Window::default_attributes().with_title(self.pending.title.clone()).with_inner_size(winit::dpi::PhysicalSize::new(w, h));
        let window = Arc::new(event_loop.create_window(attrs).context("opening the window")?);
        let size = window.inner_size();
        self.size = (size.width.max(1), size.height.max(1));
        let options = self.pending.settings.render.clone();
        // the adapter that can show this window
        let renderer = {
            let surface = self.instance.create_surface(window.clone()).context("create_surface")?;
            let mut made: Option<Renderer> = None;
            for adapter in Renderer::adapters_for(&self.instance, &surface) {
                let name = adapter.get_info().name;
                match pollster::block_on(Renderer::new_on(adapter, Some(&surface), None, options.clone())) {
                    Ok(r) => {
                        log::info!("graphics: {name}");
                        made = Some(r);
                        break;
                    }
                    Err(e) => log::warn!("graphics: {name}: {e:#}"),
                }
            }
            match made {
                Some(r) => r,
                None => pollster::block_on(Renderer::new_with(&self.instance, Some(&surface), None, options)).context("no graphics device")?,
            }
        };
        let surface = SurfaceState::new_with(&self.instance, window.clone(), &renderer, self.size.0, self.size.1, self.pending.settings.vsync)?;
        let timeline = self.pending.timeline.take().context("the timeline was taken")?;
        let profile = std::mem::take(&mut self.pending.profile);
        println!("loading the map round the window ({} tiles radius {:.0} m) ...", timeline.map, self.pending.radius_m);
        let session = Session::new(renderer, &self.pending.root, self.pending.settings.clone(), timeline, profile, self.pending.date, self.pending.radius_m)?;
        println!("{} tiles in {:.1} s; {} lanes, {} stop lines, {} kerb spots, {} profile edges", session.tiles_loaded, session.load_secs, session.overlay.counts.lanes, session.overlay.counts.stop_lines, session.overlay.counts.kerb_spots, session.overlay.counts.profile_edges);
        self.free = session.start_camera();
        self.t = self.pending.start_time.unwrap_or(session.timeline.first_time()).clamp(session.timeline.first_time(), session.timeline.last_time());
        self.frame = session.timeline.frame_at(self.t);
        self.window = Some(window);
        self.surface = Some(surface);
        self.session = Some(session);
        self.last = Instant::now();
        Ok(())
    }

    fn key(&mut self, code: KeyCode, pressed: bool) {
        if pressed {
            self.keys.insert(code);
        } else {
            self.keys.remove(&code);
            return;
        }
        let Some(s) = self.session.as_mut() else { return };
        let (t0, t1) = (s.timeline.first_time(), s.timeline.last_time());
        let layer = |k: KeyCode| -> Option<Layer> {
            Some(match k {
                KeyCode::Digit1 => Layer::Footprints,
                KeyCode::Digit2 => Layer::LaneEdges,
                KeyCode::Digit3 => Layer::ProfileEdges,
                KeyCode::Digit4 => Layer::Stops,
                KeyCode::Digit5 => Layer::Vehicles,
                KeyCode::Digit6 => Layer::People,
                KeyCode::Digit7 => Layer::Labels,
                _ => return None,
            })
        };
        match code {
            KeyCode::Space => self.playing = !self.playing,
            KeyCode::Comma => {
                self.playing = false;
                self.frame = self.frame.saturating_sub(1);
                self.t = s.timeline.frames[self.frame].day_time;
            }
            KeyCode::Period => {
                self.playing = false;
                self.frame = (self.frame + 1).min(s.timeline.frames.len() - 1);
                self.t = s.timeline.frames[self.frame].day_time;
            }
            KeyCode::BracketLeft => self.t = (self.t - 5.0).max(t0),
            KeyCode::BracketRight => self.t = (self.t + 5.0).min(t1),
            KeyCode::Home => self.t = t0,
            KeyCode::End => self.t = t1,
            KeyCode::Minus => self.speed = (self.speed * 0.5).max(0.25),
            KeyCode::Equal => self.speed = (self.speed * 2.0).min(4.0),
            KeyCode::KeyL => self.legend = !self.legend,
            KeyCode::KeyH => s.overlay.toggle_heights(&s.renderer, &mut s.scene),
            KeyCode::F5 => self.help = !self.help,
            KeyCode::KeyP => {
                let target = self.hovered.0;
                self.pinned = if self.pinned.is_some() && (target.is_none() || target == self.pinned) { None } else { target.or(self.pinned) };
            }
            KeyCode::KeyN => {
                let ids = s.timeline.vehicle_ids();
                if !ids.is_empty() {
                    let next = match self.pinned.and_then(|p| ids.iter().position(|i| *i == p)) {
                        Some(i) => ids[(i + 1) % ids.len()],
                        None => ids[0],
                    };
                    self.pinned = Some(next);
                    if let CamMode::Follow(_) = self.mode {
                        self.mode = CamMode::Follow(next);
                    }
                }
            }
            KeyCode::F1 => {
                // leaving a placed camera: the hand camera starts where it was
                self.mode = CamMode::Free;
            }
            KeyCode::F2 => self.mode = CamMode::Seat,
            KeyCode::F3 => {
                let id = self.pinned.or(self.hovered.0).or_else(|| s.timeline.frames[self.frame].vehicles.first().map(|v| v.id)).or_else(|| s.timeline.vehicle_ids().first().copied());
                if let Some(id) = id {
                    self.mode = CamMode::Follow(id);
                }
            }
            KeyCode::F4 => {
                if !matches!(self.mode, CamMode::TopDown(_)) {
                    // over where the camera is now, or the window's middle from the seat
                    if matches!(self.mode, CamMode::Seat | CamMode::Follow(_)) {
                        let c = s.timeline.centre;
                        self.free.position = DVec3::new(c.x, c.y, c.z);
                    }
                    self.free.yaw = 0.0;
                    self.mode = CamMode::TopDown(30.0);
                }
            }
            KeyCode::PageUp => {
                if let CamMode::TopDown(h) = &mut self.mode {
                    *h = (*h + 5.0).min(400.0);
                }
            }
            KeyCode::PageDown => {
                if let CamMode::TopDown(h) = &mut self.mode {
                    *h = (*h - 5.0).max(5.0);
                }
            }
            k => {
                if let Some(l) = layer(k) {
                    s.overlay.toggle_layer(&s.renderer, &mut s.scene, l);
                }
            }
        }
    }

    /// Move the hand camera by the keys held.
    fn fly(&mut self, dt: f32) {
        let fast = self.keys.contains(&KeyCode::ShiftLeft) || self.keys.contains(&KeyCode::ShiftRight);
        let slow = self.keys.contains(&KeyCode::ControlLeft) || self.keys.contains(&KeyCode::ControlRight);
        let mut speed: f64 = if fast { 40.0 } else if slow { 1.5 } else { 8.0 };
        if let CamMode::TopDown(h) = self.mode {
            speed = speed.max(h * 0.4);
        }
        let step = speed * dt as f64;
        let (s, c) = (self.free.yaw as f64).to_radians().sin_cos();
        let (fwd, right) = match self.mode {
            CamMode::TopDown(_) => (DVec3::new(0.0, 1.0, 0.0), DVec3::new(1.0, 0.0, 0.0)),
            _ => (DVec3::new(s, c, 0.0), DVec3::new(c, -s, 0.0)),
        };
        let mut d = DVec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) || self.keys.contains(&KeyCode::ArrowUp) {
            d += fwd;
        }
        if self.keys.contains(&KeyCode::KeyS) || self.keys.contains(&KeyCode::ArrowDown) {
            d -= fwd;
        }
        if self.keys.contains(&KeyCode::KeyD) || self.keys.contains(&KeyCode::ArrowRight) {
            d += right;
        }
        if self.keys.contains(&KeyCode::KeyA) || self.keys.contains(&KeyCode::ArrowLeft) {
            d -= right;
        }
        if self.keys.contains(&KeyCode::KeyE) || self.keys.contains(&KeyCode::KeyR) {
            d += DVec3::Z;
        }
        if self.keys.contains(&KeyCode::KeyQ) || self.keys.contains(&KeyCode::KeyF) {
            d -= DVec3::Z;
        }
        if d != DVec3::ZERO {
            self.free.position += d.normalize() * step;
        }
    }

    fn redraw(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32().min(0.25);
        self.last = now;
        self.fly(dt);
        let Some(s) = self.session.as_mut() else { return };
        if self.playing {
            self.t += dt as f64 * self.speed;
            if self.t > s.timeline.last_time() {
                self.t = s.timeline.first_time();
            }
        }
        self.frame = s.timeline.frame_at(self.t);
        s.set_time(self.t);
        let frame = s.timeline.frames[self.frame].clone();
        // leaving a placed camera by hand: the hand camera takes its place
        if self.mode != CamMode::Free && !matches!(self.mode, CamMode::TopDown(_)) {
            let placed = s.camera(self.mode, &frame, &self.free);
            self.free.position = placed.position;
            self.free.yaw = placed.yaw;
        }
        let cam = s.camera(self.mode, &frame, &self.free);
        let (w, h) = (self.size.0 as f32, self.size.1 as f32);
        self.hovered = match self.cursor {
            Some((cx, cy)) if !self.looking => s.pick(&cam, &frame, cx, cy, w, h),
            _ => (None, None),
        };
        s.lay_frame(self.frame, self.hovered.0, self.pinned);
        s.hud.begin(&s.renderer, &mut s.scene);
        s.draw_hud(&cam, self.frame, self.mode, self.speed, self.playing, self.pinned, self.hovered, self.cursor, self.legend, w, h, self.help);
        let (Some(surface), Some(window)) = (self.surface.as_mut(), self.window.as_ref()) else { return };
        match surface.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                let view = frame.texture.create_view(&Default::default());
                let lighting = s.lighting.clone();
                s.renderer.render(&mut s.scene, &view, self.size.0, self.size.1, &cam, &lighting);
                window.pre_present_notify();
                frame.present();
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                let size = window.inner_size();
                self.size = (size.width.max(1), size.height.max(1));
                surface.resize(&s.renderer, self.size.0, self.size.1);
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        if let Err(e) = self.open(event_loop) {
            self.error = Some(format!("{e:#}"));
            event_loop.exit();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.size = (size.width.max(1), size.height.max(1));
                if let (Some(sf), Some(s)) = (self.surface.as_mut(), self.session.as_ref()) {
                    sf.resize(&s.renderer, self.size.0, self.size.1);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if code == KeyCode::Escape && event.state == ElementState::Pressed {
                        event_loop.exit();
                        return;
                    }
                    if event.repeat && event.state == ElementState::Pressed && !matches!(code, KeyCode::Comma | KeyCode::Period | KeyCode::BracketLeft | KeyCode::BracketRight | KeyCode::PageUp | KeyCode::PageDown) {
                        return;
                    }
                    self.key(code, event.state == ElementState::Pressed);
                }
            }
            WindowEvent::CursorMoved { position, .. } => self.cursor = Some((position.x as f32, position.y as f32)),
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Right => {
                    self.looking = state == ElementState::Pressed;
                    if let Some(w) = self.window.as_ref() {
                        w.set_cursor_visible(!self.looking);
                    }
                }
                MouseButton::Left if state == ElementState::Pressed => {
                    let target = self.hovered.0;
                    self.pinned = if target.is_some() && target == self.pinned { None } else { target.or(self.pinned) };
                    if let (CamMode::Follow(_), Some(id)) = (self.mode, self.pinned) {
                        self.mode = CamMode::Follow(id);
                    }
                }
                _ => {}
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                };
                match &mut self.mode {
                    CamMode::TopDown(h) => *h = (*h * (1.0 - 0.1 * steps)).clamp(5.0, 400.0),
                    _ => {
                        let (s, c) = (self.free.yaw as f64).to_radians().sin_cos();
                        let (sp, cp) = (self.free.pitch as f64).to_radians().sin_cos();
                        self.free.position += DVec3::new(s * cp, c * cp, sp) * steps * 3.0;
                    }
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device: winit::event::DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.looking && !matches!(self.mode, CamMode::TopDown(_)) {
                self.free.yaw = (self.free.yaw + delta.0 as f32 * 0.15).rem_euclid(360.0);
                self.free.pitch = (self.free.pitch - delta.1 as f32 * 0.15).clamp(-89.0, 89.0);
                if self.mode != CamMode::Free && !matches!(self.mode, CamMode::Seat) {
                    self.mode = CamMode::Free;
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
        }
    }
}
