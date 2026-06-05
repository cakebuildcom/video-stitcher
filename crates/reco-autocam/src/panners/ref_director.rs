//! Lacrosse ref-based framing director.
//!
//! Tracks red-shirted referees on the field and pans to frame them.
//! Refs are positioned by rules to watch the entire field, so their
//! positions indicate where the action is and where it's heading.
//!
//! Uses lookahead to predict ref movement for smooth pre-framing.

use reco_core::detect::director::ViewportPosition;
use reco_core::detect::panner::{PanContext, Panner};
use reco_core::detect::tracker::{TrackState, TrackedEntity, WorldState};

const LOG_INTERVAL: u64 = 30;

/// Configuration for ref-based director.
#[derive(Debug, Clone)]
pub struct RefDirectorConfig {
    /// Red shirt hue range (center, tolerance in degrees).
    /// Refs wear bright red: H=345°, S=84%, V=73%
    pub red_hue_center: f32,
    pub red_hue_tolerance: f32,
    pub red_sat_min: f32,
    pub red_val_min: f32,
    /// EMA smoothing for ref position tracking.
    pub position_alpha: f32,
    /// Max velocity (radians/frame).
    pub max_velocity_rad_per_sec: f32,
    pub velocity_alpha: f32,
    /// Pitch bias.
    pub pitch_bias: f32,
    /// FOV parameters.
    pub fov_alpha: f32,
    pub fov_tight: f32,
    pub fov_wide: f32,
    pub fov_default: f32,
    pub pitch_near: f32,
    pub pitch_far: f32,
    pub distance_bias_max: f32,
    pub edge_bias_max: f32,
    pub velocity_fov_bias_max: f32,
    /// Lookahead frames: predict ref position N frames ahead.
    pub lookahead_frames: u32,
}

impl Default for RefDirectorConfig {
    fn default() -> Self {
        Self {
            red_hue_center: 345.0,
            red_hue_tolerance: 30.0,
            red_sat_min: 0.70,
            red_val_min: 0.60,
            position_alpha: 0.015,
            max_velocity_rad_per_sec: 0.15,
            velocity_alpha: 0.07,
            pitch_bias: 0.05,
            fov_alpha: 0.01,
            fov_tight: 22.0,
            fov_wide: 58.0,
            fov_default: 40.0,
            pitch_near: -0.05,
            pitch_far: 0.20,
            distance_bias_max: -12.0,
            edge_bias_max: 4.0,
            velocity_fov_bias_max: 10.0,
            lookahead_frames: 5,
        }
    }
}

/// Ref position with lookahead history.
#[derive(Debug, Clone, Copy)]
struct RefTrack {
    yaw: f32,
    pitch: f32,
    confidence: f32,
    age: u32,
}

/// Lacrosse ref-based director: frame where refs are positioned.
pub struct RefDirector {
    config: RefDirectorConfig,
    yaw: f32,
    pitch: f32,
    current_fov: f32,
    ema_yaw: f32,
    ema_pitch: f32,
    ema_initialized: bool,
    velocity_yaw: f32,
    velocity_pitch: f32,
    max_velocity: f32,
    frame_index: u64,
    ref_history: Vec<Vec<RefTrack>>,
}

impl RefDirector {
    /// Create a new ref-based director.
    pub fn new(fps: f32) -> Self {
        Self::with_config(fps, RefDirectorConfig::default())
    }

    /// Create with custom config.
    pub fn with_config(fps: f32, config: RefDirectorConfig) -> Self {
        let fps = fps.clamp(1.0, 1000.0);
        let max_velocity = config.max_velocity_rad_per_sec / fps;
        let fov_default = config.fov_default;
        Self {
            config,
            yaw: 0.0,
            pitch: 0.0,
            current_fov: fov_default,
            ema_yaw: 0.0,
            ema_pitch: 0.0,
            ema_initialized: false,
            velocity_yaw: 0.0,
            velocity_pitch: 0.0,
            max_velocity,
            frame_index: 0,
            ref_history: Vec::new(),
        }
    }

    /// Check if a player's color matches red shirt (ref).
    /// Uses HSV color space: Hue, Saturation, Value.
    fn is_red_shirt(&self, _yaw: f32, _pitch: f32) -> bool {
        // In real implementation, would sample the frame at (yaw, pitch)
        // For now, we rely on the detector already filtering by class_id
        // This is a placeholder for color-based filtering.
        true
    }

    /// Extract ref positions from world state.
    /// Real refs are detected by their class_id from a custom YOLO model
    /// or color-based detection.
    fn extract_refs(&self, players: &[TrackedEntity]) -> Vec<RefTrack> {
        players
            .iter()
            .filter(|p| !matches!(p.state, TrackState::Lost))
            .map(|p| RefTrack {
                yaw: p.yaw,
                pitch: p.pitch,
                confidence: p.confidence,
                age: p.age_frames as u32,
            })
            .collect()
    }

    /// Compute centroid of ref positions (frame refs as a group).
    fn ref_centroid(&self, refs: &[RefTrack]) -> Option<(f32, f32)> {
        if refs.is_empty() {
            return None;
        }

        let total_conf: f32 = refs.iter().map(|r| r.confidence).sum();
        if total_conf <= 0.0 {
            return None;
        }

        let yaw = refs.iter().map(|r| r.yaw * r.confidence).sum::<f32>() / total_conf;
        let pitch = refs.iter().map(|r| r.pitch * r.confidence).sum::<f32>() / total_conf;

        if yaw.is_finite() && pitch.is_finite() {
            Some((yaw, pitch))
        } else {
            None
        }
    }

    /// Predict ref position using lookahead (extrapolate from history).
    fn predict_ref_position(
        &self,
        current_yaw: f32,
        current_pitch: f32,
    ) -> (f32, f32) {
        if self.ref_history.len() < 2 {
            return (current_yaw, current_pitch);
        }

        // Linear extrapolation from last 2 frames
        let hist_len = self.ref_history.len();
        if let (Some(last_refs), Some(prev_refs)) = (
            self.ref_history.get(hist_len - 1),
            self.ref_history.get(hist_len - 2),
        ) {
            if !last_refs.is_empty() && !prev_refs.is_empty() {
                if let (Some((last_y, last_p)), Some((prev_y, prev_p))) = (
                    self.ref_centroid(last_refs),
                    self.ref_centroid(prev_refs),
                ) {
                    let dy = last_y - prev_y;
                    let dp = last_p - prev_p;

                    // Project N frames ahead
                    let lookahead_frames = self.config.lookahead_frames as f32;
                    let predicted_yaw = last_y + dy * lookahead_frames;
                    let predicted_pitch = last_p + dp * lookahead_frames;

                    return (predicted_yaw, predicted_pitch);
                }
            }
        }

        (current_yaw, current_pitch)
    }

    /// Smooth position via EMA.
    fn smooth_position(&mut self, raw_yaw: f32, raw_pitch: f32) -> (f32, f32) {
        if !self.ema_initialized {
            self.ema_yaw = raw_yaw;
            self.ema_pitch = raw_pitch;
            self.ema_initialized = true;
        } else {
            self.ema_yaw += self.config.position_alpha * (raw_yaw - self.ema_yaw);
            self.ema_pitch += self.config.position_alpha * (raw_pitch - self.ema_pitch);
        }
        (self.ema_yaw, self.ema_pitch)
    }

    /// Target FOV from ref spread and distance.
    fn target_fov(&self, spread: f32, pitch: f32, velocity_mag: f32) -> f32 {
        let spread_deg = spread.to_degrees();
        let fov_from_spread = (2.0 * spread_deg).max(self.config.fov_tight);

        let t_dist = ((pitch - self.config.pitch_near)
            / (self.config.pitch_far - self.config.pitch_near))
            .clamp(0.0, 1.0);
        let distance_bias = t_dist * self.config.distance_bias_max;

        let edge_bias = (self.yaw.abs() * 5.0).min(self.config.edge_bias_max);

        let vel_ratio = (velocity_mag / self.max_velocity).clamp(0.0, 1.0);
        let velocity_bias = vel_ratio * self.config.velocity_fov_bias_max;

        let fov = fov_from_spread + distance_bias + edge_bias + velocity_bias;
        fov.clamp(self.config.fov_tight, self.config.fov_wide)
    }

    /// Ref spread (max distance from centroid to any ref).
    fn ref_spread(&self, refs: &[RefTrack], cy: f32, cp: f32) -> f32 {
        refs.iter()
            .map(|r| {
                let dy = r.yaw - cy;
                let dp = r.pitch - cp;
                (dy * dy + dp * dp).sqrt()
            })
            .fold(0.0_f32, f32::max)
    }
}

impl Default for RefDirector {
    fn default() -> Self {
        Self::new(30.0)
    }
}

impl Panner for RefDirector {
    fn decide(&mut self, world: &WorldState, _ctx: &PanContext<'_>) -> ViewportPosition {
        reco_core::profile_scope!("ref_director_decide");

        self.frame_index = self.frame_index.wrapping_add(1);

        let refs = self.extract_refs(&world.players);
        self.ref_history.push(refs.clone());
        if self.ref_history.len() > 10 {
            self.ref_history.remove(0);
        }

        if let Some((raw_yaw, raw_pitch)) = self.ref_centroid(&refs) {
            // Predict where refs will be (lookahead)
            let (predicted_yaw, predicted_pitch) = self.predict_ref_position(raw_yaw, raw_pitch);

            // Smooth the prediction
            let (centroid_yaw, centroid_pitch) = self.smooth_position(predicted_yaw, predicted_pitch);

            let spread = self.ref_spread(&refs, centroid_yaw, centroid_pitch);
            let target_yaw = centroid_yaw;
            let target_pitch = centroid_pitch + self.config.pitch_bias;

            if target_yaw.is_finite() && target_pitch.is_finite() {
                let err_yaw = target_yaw - self.yaw;
                let err_pitch = target_pitch - self.pitch;

                let desired_yaw = err_yaw.clamp(-self.max_velocity, self.max_velocity);
                let desired_pitch = err_pitch.clamp(-self.max_velocity, self.max_velocity);

                self.velocity_yaw +=
                    self.config.velocity_alpha * (desired_yaw - self.velocity_yaw);
                self.velocity_pitch +=
                    self.config.velocity_alpha * (desired_pitch - self.velocity_pitch);

                self.yaw += self.velocity_yaw;
                self.pitch += self.velocity_pitch;
            }

            let vel_mag = (self.velocity_yaw.powi(2) + self.velocity_pitch.powi(2)).sqrt();
            let target_fov = self.target_fov(spread, centroid_pitch, vel_mag);
            if target_fov.is_finite() {
                self.current_fov += self.config.fov_alpha * (target_fov - self.current_fov);
            }

            if self.frame_index.is_multiple_of(LOG_INTERVAL) {
                log::debug!(
                    "RefDirector frame {}: yaw={:.4} pitch={:.4} fov={:.1} refs={} spread={:.3}",
                    self.frame_index,
                    self.yaw,
                    self.pitch,
                    self.current_fov,
                    refs.len(),
                    spread,
                );
            }
        } else {
            log::trace!("RefDirector: no refs detected this frame");
        }

        ViewportPosition {
            yaw: self.yaw,
            pitch: self.pitch,
            fov_degrees: Some(self.current_fov),
        }
    }

    fn debug_event(
        &self,
        _frame_index: u64,
    ) -> Option<reco_core::detect::pipeline_event::PipelineEvent> {
        None
    }
}
