//! Lacrosse-specific panner using player density clustering.
//!
//! # Pipeline (per `decide` call)
//!
//! 1. Divide panorama into 3 zones: left wing, center, right wing.
//! 2. Count live tracked players in each zone.
//! 3. Select the zone with highest player density (the active play zone).
//! 4. Compute centroid of players in that zone.
//! 5. Apply EMA smoothing to prevent jitter during zone transitions.
//! 6. Dynamic FOV from zone spread + velocity.
//!
//! # Why this works for lacrosse
//!
//! - During play: active zone has ≥12 players (7 side + 3+ middies)
//! - Stationary zones have ~7 players
//! - Density automatically identifies the action, no ball tracking needed
//! - Transitions when middies move between zones are smooth via EMA

use reco_core::detect::director::ViewportPosition;
use reco_core::detect::panner::{PanContext, Panner};
use reco_core::detect::tracker::{TrackState, TrackedEntity, WorldState};

const LOG_INTERVAL: u64 = 30;

/// Zone boundaries (yaw angle in radians). Divide [-π, π] into 3 zones.
#[derive(Debug, Clone, Copy)]
pub struct ZoneBoundaries {
    /// Left boundary (yaw < this is left wing).
    pub left_boundary: f32,
    /// Right boundary (yaw > this is right wing, left < yaw < right is center).
    pub right_boundary: f32,
}

impl Default for ZoneBoundaries {
    fn default() -> Self {
        // Panorama is [-π, π]. Divide into thirds:
        // left:   [-π, -π/3)
        // center: [-π/3, π/3)
        // right:  [π/3, π]
        Self {
            left_boundary: -std::f32::consts::PI / 3.0,
            right_boundary: std::f32::consts::PI / 3.0,
        }
    }
}

/// Configuration for the lacrosse panner.
#[derive(Debug, Clone)]
pub struct LacrossePannerConfig {
    /// Zone boundaries (left/center/right).
    pub zones: ZoneBoundaries,
    /// Exponential moving average smoothing for zone transitions.
    pub cluster_alpha: f32,
    /// Max velocity (radians/frame) to prevent jerky pan motion.
    pub max_velocity_rad_per_sec: f32,
    /// Velocity EMA smoothing.
    pub velocity_alpha: f32,
    /// Pitch bias (keep players slightly above center).
    pub pitch_bias: f32,
    /// FOV EMA smoothing.
    pub fov_alpha: f32,
    /// Minimum FOV (degrees).
    pub fov_tight: f32,
    /// Maximum FOV (degrees).
    pub fov_wide: f32,
    /// Default FOV (degrees).
    pub fov_default: f32,
    /// Pitch range for distance estimation.
    pub pitch_near: f32,
    pub pitch_far: f32,
    /// Distance bias for FOV (makes far away play wider).
    pub distance_bias_max: f32,
    /// Edge bias for FOV (wider at panorama edges).
    pub edge_bias_max: f32,
    /// Velocity FOV bias (faster motion = wider FOV).
    pub velocity_fov_bias_max: f32,
}

impl Default for LacrossePannerConfig {
    fn default() -> Self {
        Self {
            zones: ZoneBoundaries::default(),
            cluster_alpha: 0.012,
            max_velocity_rad_per_sec: 0.18,
            velocity_alpha: 0.06,
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
        }
    }
}

/// Zone identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Zone {
    Left,
    Center,
    Right,
}

/// Lacrosse panner: follow the densest player cluster (highest density zone).
pub struct LacrossePanner {
    config: LacrossePannerConfig,
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
    last_debug: Option<LacrossePannerDebug>,
}

#[allow(dead_code)]
struct LacrossePannerDebug {
    left_count: u32,
    center_count: u32,
    right_count: u32,
    active_zone: &'static str,
    centroid_yaw: f32,
    centroid_pitch: f32,
    target_yaw: f32,
    target_pitch: f32,
    fov_target: f32,
}

impl LacrossePanner {
    /// Create a new lacrosse panner with default config.
    pub fn new(fps: f32) -> Self {
        Self::with_config(fps, LacrossePannerConfig::default())
    }

    /// Create with custom config.
    pub fn with_config(fps: f32, config: LacrossePannerConfig) -> Self {
        let fps = fps.clamp(1.0, 1000.0);
        let current_fov = config.fov_default;
        let max_velocity = config.max_velocity_rad_per_sec / fps;
        Self {
            config,
            yaw: 0.0,
            pitch: 0.0,
            current_fov,
            ema_yaw: 0.0,
            ema_pitch: 0.0,
            ema_initialized: false,
            velocity_yaw: 0.0,
            velocity_pitch: 0.0,
            max_velocity,
            frame_index: 0,
            last_debug: None,
        }
    }

    /// Classify a player into a zone based on yaw angle.
    fn classify_zone(&self, yaw: f32) -> Zone {
        if yaw < self.config.zones.left_boundary {
            Zone::Left
        } else if yaw >= self.config.zones.right_boundary {
            Zone::Right
        } else {
            Zone::Center
        }
    }

    /// Count players per zone. Filter out lost entities.
    fn count_zones(&self, players: &[TrackedEntity]) -> (u32, u32, u32) {
        let mut left = 0;
        let mut center = 0;
        let mut right = 0;

        for p in players {
            if matches!(p.state, TrackState::Lost) {
                continue;
            }
            match self.classify_zone(p.yaw) {
                Zone::Left => left += 1,
                Zone::Center => center += 1,
                Zone::Right => right += 1,
            }
        }

        (left, center, right)
    }

    /// Find the zone with highest player count.
    fn find_active_zone(&self, left: u32, center: u32, right: u32) -> Option<Zone> {
        if left == 0 && center == 0 && right == 0 {
            return None;
        }
        let max = left.max(center).max(right);
        if left == max {
            Some(Zone::Left)
        } else if center == max {
            Some(Zone::Center)
        } else {
            Some(Zone::Right)
        }
    }

    /// Compute confidence-weighted centroid of players in a zone.
    fn centroid_in_zone(&self, players: &[TrackedEntity], zone: Zone) -> Option<(f32, f32)> {
        let points: Vec<(f32, f32, f32)> = players
            .iter()
            .filter(|p| !matches!(p.state, TrackState::Lost) && self.classify_zone(p.yaw) == zone)
            .map(|p| (p.yaw, p.pitch, p.confidence))
            .collect();

        if points.is_empty() {
            return None;
        }

        let total_conf: f32 = points.iter().map(|(_, _, c)| c).sum();
        if total_conf <= 0.0 {
            return None;
        }

        let yaw = points.iter().map(|(y, _, c)| y * c).sum::<f32>() / total_conf;
        let pitch = points.iter().map(|(_, p, c)| p * c).sum::<f32>() / total_conf;

        if yaw.is_finite() && pitch.is_finite() {
            Some((yaw, pitch))
        } else {
            None
        }
    }

    /// Smooth centroid via cascaded EMA.
    fn smooth_centroid(&mut self, raw_yaw: f32, raw_pitch: f32) -> (f32, f32) {
        if !self.ema_initialized {
            self.ema_yaw = raw_yaw;
            self.ema_pitch = raw_pitch;
            self.ema_initialized = true;
        } else {
            self.ema_yaw += self.config.cluster_alpha * (raw_yaw - self.ema_yaw);
            self.ema_pitch += self.config.cluster_alpha * (raw_pitch - self.ema_pitch);
        }
        (self.ema_yaw, self.ema_pitch)
    }

    /// Compute zone spread (max distance from centroid to any player in zone).
    fn zone_spread(&self, players: &[TrackedEntity], zone: Zone, cy: f32, cp: f32) -> f32 {
        players
            .iter()
            .filter(|p| !matches!(p.state, TrackState::Lost) && self.classify_zone(p.yaw) == zone)
            .map(|p| {
                let dy = p.yaw - cy;
                let dp = p.pitch - cp;
                (dy * dy + dp * dp).sqrt()
            })
            .fold(0.0_f32, f32::max)
    }

    /// Dynamic FOV: zone spread + distance + edge + velocity biases.
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
}

impl Default for LacrossePanner {
    fn default() -> Self {
        Self::new(30.0)
    }
}

impl Panner for LacrossePanner {
    fn decide(&mut self, world: &WorldState, _ctx: &PanContext<'_>) -> ViewportPosition {
        reco_core::profile_scope!("lacrosse_panner_decide");

        self.frame_index = self.frame_index.wrapping_add(1);

        // Count players in each zone.
        let (left_count, center_count, right_count) = self.count_zones(&world.players);

        // Find the active zone (highest density).
        if let Some(active_zone) = self.find_active_zone(left_count, center_count, right_count) {
            if let Some((raw_yaw, raw_pitch)) = self.centroid_in_zone(&world.players, active_zone)
            {
                let (centroid_yaw, centroid_pitch) = self.smooth_centroid(raw_yaw, raw_pitch);
                let spread = self.zone_spread(&world.players, active_zone, centroid_yaw, centroid_pitch);

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

                let vel_mag =
                    (self.velocity_yaw.powi(2) + self.velocity_pitch.powi(2)).sqrt();
                let target_fov = self.target_fov(spread, centroid_pitch, vel_mag);
                if target_fov.is_finite() {
                    self.current_fov += self.config.fov_alpha * (target_fov - self.current_fov);
                } else {
                    log::warn!(
                        "LacrossePanner: non-finite FOV target ({target_fov}) from \
                         spread={} pitch={}; keeping current_fov={}",
                        spread,
                        centroid_pitch,
                        self.current_fov,
                    );
                }

                let zone_name = match active_zone {
                    Zone::Left => "Left",
                    Zone::Center => "Center",
                    Zone::Right => "Right",
                };

                self.last_debug = Some(LacrossePannerDebug {
                    left_count,
                    center_count,
                    right_count,
                    active_zone: zone_name,
                    centroid_yaw,
                    centroid_pitch,
                    target_yaw,
                    target_pitch,
                    fov_target: target_fov,
                });

                if self.frame_index.is_multiple_of(LOG_INTERVAL) {
                    log::debug!(
                        "LacrossePanner frame {}: zone={} (L={} C={} R={}) yaw={:.4} pitch={:.4} fov={:.1} spread={:.3}",
                        self.frame_index,
                        zone_name,
                        left_count,
                        center_count,
                        right_count,
                        self.yaw,
                        self.pitch,
                        self.current_fov,
                        spread,
                    );
                }
            } else {
                log::trace!(
                    "LacrossePanner: active zone {}, but no players in it",
                    match active_zone {
                        Zone::Left => "Left",
                        Zone::Center => "Center",
                        Zone::Right => "Right",
                    }
                );
            }
        } else {
            self.last_debug = None;
            log::trace!("LacrossePanner: no players detected");
        }

        ViewportPosition {
            yaw: self.yaw,
            pitch: self.pitch,
            fov_degrees: Some(self.current_fov),
        }
    }

    fn debug_event(
        &self,
        frame_index: u64,
    ) -> Option<reco_core::detect::pipeline_event::PipelineEvent> {
        // TODO: extend PipelineEvent to support lacrosse-specific debug info
        // For now, return None until reco-core adds LacrosseDebug variant
        let _ = (frame_index, &self.last_debug);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::detect::detector::CameraId;

    fn player(yaw: f32, pitch: f32, id: u64) -> TrackedEntity {
        TrackedEntity {
            id,
            class_id: 0,
            yaw,
            pitch,
            confidence: 0.9,
            state: TrackState::Tracking,
            age_frames: 5,
            origin: CameraId::Left,
        }
    }

    fn cal() -> reco_core::calibration::MatchCalibration {
        use reco_core::calibration::{CameraParams, MatchCalibration, PlaneLayout};
        MatchCalibration {
            left: CameraParams {
                width: 1920,
                height: 1080,
                fx: 900.0,
                fy: 900.0,
                cx: 960.0,
                cy: 540.0,
                d: [0.0; 4],
            },
            right: CameraParams {
                width: 1920,
                height: 1080,
                fx: 900.0,
                fy: 900.0,
                cx: 960.0,
                cy: 540.0,
                d: [0.0; 4],
            },
            layout: PlaneLayout {
                camera_axis_offset: 0.24,
                intersect: 0.54,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
            },
            rig_tilt: 0.0,
            rig_roll: 0.0,
            sync_offset: 0,
            field_roi: None,
        }
    }

    fn ctx<'a>(
        frame_index: u64,
        cal: &'a reco_core::calibration::MatchCalibration,
    ) -> PanContext<'a> {
        PanContext {
            frame_index,
            timestamp_ms: frame_index as f64 * (1000.0 / 30.0),
            previous_position: ViewportPosition::default(),
            calibration: cal,
        }
    }

    #[test]
    fn follows_densest_zone() {
        let mut p = LacrossePanner::new(30.0);
        let cal = cal();

        // Create 7 players on left, 6 in center, 7 on right.
        // Center should be active (equal to left/right but let's make it clear).
        let mut players = Vec::new();
        // Left zone: 7 players
        for i in 0..7 {
            let yaw = -2.0 - (i as f32) * 0.1;
            players.push(player(yaw, 0.0, i as u64 + 1));
        }
        // Center zone: 8 players (more active)
        for i in 0..8 {
            let yaw = -0.5 + (i as f32) * 0.1;
            players.push(player(yaw, 0.0, i as u64 + 8));
        }
        // Right zone: 7 players
        for i in 0..7 {
            let yaw = 2.0 + (i as f32) * 0.1;
            players.push(player(yaw, 0.0, i as u64 + 16));
        }

        let world = WorldState {
            ball: None,
            players,
        };

        let mut out = p.decide(&world, &ctx(0, &cal));
        for frame in 1..200 {
            out = p.decide(&world, &ctx(frame, &cal));
        }

        // Should pan toward center (yaw ~0)
        assert!(
            out.yaw.abs() < 0.3,
            "should focus on center zone, got yaw={}",
            out.yaw
        );
    }

    #[test]
    fn switches_to_left_when_more_players() {
        let mut p = LacrossePanner::new(30.0);
        let cal = cal();

        let mut players = Vec::new();
        // Left zone: 12 players (active attack)
        for i in 0..12 {
            let yaw = -2.0 - (i as f32) * 0.08;
            players.push(player(yaw, 0.0, i as u64 + 1));
        }
        // Right zone: 7 players (defense)
        for i in 0..7 {
            let yaw = 2.0 + (i as f32) * 0.1;
            players.push(player(yaw, 0.0, i as u64 + 13));
        }

        let world = WorldState {
            ball: None,
            players,
        };

        let mut out = p.decide(&world, &ctx(0, &cal));
        for frame in 1..200 {
            out = p.decide(&world, &ctx(frame, &cal));
        }

        // Should pan toward left (yaw < 0)
        assert!(
            out.yaw < -0.5,
            "should focus on left zone with more players, got yaw={}",
            out.yaw
        );
    }

    #[test]
    fn no_players_holds_position() {
        let mut p = LacrossePanner::new(30.0);
        let cal = cal();
        p.yaw = 0.3;
        p.pitch = 0.05;

        let world = WorldState {
            ball: None,
            players: vec![],
        };

        let out = p.decide(&world, &ctx(0, &cal));
        assert!((out.yaw - 0.3).abs() < 1e-6);
        assert!((out.pitch - 0.05).abs() < 1e-6);
    }

    #[test]
    fn lost_players_excluded() {
        let mut p = LacrossePanner::new(30.0);
        let cal = cal();

        let mut players = Vec::new();
        // Center: 4 live + 1 lost (should count as 4)
        for i in 0..4 {
            let yaw = (i as f32) * 0.1;
            players.push(player(yaw, 0.0, i as u64 + 1));
        }
        let mut lost = player(0.5, 0.0, 5);
        lost.state = TrackState::Lost;
        players.push(lost);

        let world = WorldState {
            ball: None,
            players,
        };

        let out = p.decide(&world, &ctx(0, &cal));
        // Should have focused on center with 4 live players
        assert!(
            out.yaw.abs() < 0.5,
            "lost player must not drag centroid, got yaw={}",
            out.yaw
        );
    }

    #[test]
    fn fov_widens_for_spread_players() {
        let p = LacrossePanner::new(30.0);
        let tight = p.target_fov(0.05, 0.0, 0.0);
        let wide = p.target_fov(0.40, 0.0, 0.0);
        assert!(tight < wide, "tight={tight} wide={wide}");
    }
}
