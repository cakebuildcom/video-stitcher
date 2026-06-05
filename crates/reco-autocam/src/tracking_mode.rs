//! [`TrackingMode`] enum - the one-knob config that selects which
//! tracker(s) and panner setup_autocam wires up.

/// Which automatic-camera strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TrackingMode {
    /// Track player cluster + ball for broadcast-style coverage
    /// (multi-class model). Uses
    /// [`PlayerTracker`](crate::trackers::PlayerTracker),
    /// [`BallTracker`](crate::trackers::BallTracker), and
    /// [`FieldPanner`](crate::panners::FieldPanner).
    /// Ball influence is controlled by `FieldPannerConfig::ball_weight`.
    #[default]
    Field,
    /// Lacrosse-specific tracking: pan to the densest player zone
    /// (left/center/right thirds of panorama).
    /// Uses [`PlayerTracker`](crate::trackers::PlayerTracker) and
    /// [`LacrossePanner`](crate::panners::LacrossePanner).
    /// No ball tracking — follows player density instead.
    Lacrosse,
    /// Lacrosse-specific tracking: pan to frame the refs (red-shirted officials).
    /// Uses [`PlayerTracker`](crate::trackers::PlayerTracker) and
    /// [`RefDirector`](crate::panners::RefDirector).
    /// Refs are positioned by rules to watch the entire field, so their
    /// positions indicate action zones. Includes lookahead prediction
    /// for smooth pre-framing.
    LacrosseRefs,
    /// Ball-only tracking for single-class ball detectors. No player
    /// tracker, no cluster centroid. Uses only
    /// [`BallTracker`](crate::trackers::BallTracker) with higher
    /// confidence threshold and top-1 detection per camera.
    Ball,
    /// Debug mode: slowly pan left-right across the full coverage.
    /// No AI, no tracking. Uses
    /// [`SweepPanner`](crate::panners::SweepPanner).
    Sweep,
}
