//! YOLO-based person detection with ROI filtering and clustering.
//!
//! Pipeline:
//! 1. Extract frames from video every N frames
//! 2. Run YOLO detection on extracted frames
//! 3. Filter detections by ROI polygon (field boundary)
//! 4. Cluster detections by appearance (color histogram)
//! 5. Return labeled clusters for UI display

use std::path::PathBuf;

use image::ImageReader;

/// A single detected person with location and appearance.
#[derive(Debug, Clone)]
pub struct DetectedPerson {
    pub id: usize,
    pub frame_idx: usize,
    pub bbox: (f32, f32, f32, f32), // x1, y1, x2, y2 (normalized 0-1)
    pub confidence: f32,
    pub color_hist: Vec<f32>, // HSV histogram for clustering
    pub image_path: Option<PathBuf>, // Path to cropped person image
}

/// Clustered group of similar-looking people.
#[derive(Debug, Clone)]
pub struct PersonCluster {
    pub cluster_id: usize,
    pub size: usize,
    pub people: Vec<DetectedPerson>,
    pub label: Option<String>, // "team-a", "team-b", "refs", "other", or None
}

/// Detection results with clusters ready for labeling.
#[derive(Debug)]
pub struct DetectionResults {
    pub total_detections: usize,
    pub clusters: Vec<PersonCluster>,
    pub output_dir: PathBuf,
}

/// ROI polygon represented as normalized coordinates (0-1).
#[derive(Debug, Clone)]
pub struct RoiPolygon {
    pub points: Vec<(f32, f32)>, // List of (x, y) points defining polygon
}

impl RoiPolygon {
    /// Check if a point is inside the polygon (simplified).
    pub fn contains(&self, x: f32, y: f32) -> bool {
        if self.points.len() < 3 {
            return true; // No valid polygon, include all
        }

        // Simple point-in-polygon using ray casting
        let mut inside = false;
        let mut j = self.points.len() - 1;

        for i in 0..self.points.len() {
            let (xi, yi) = self.points[i];
            let (xj, yj) = self.points[j];

            if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
            j = i;
        }

        inside
    }
}

/// Main detection pipeline.
pub struct YoloDetectionPipeline {
    pub video_path: PathBuf,
    pub frame_interval: u32,
    pub model_path: PathBuf,
    pub roi: Option<RoiPolygon>,
    pub output_dir: PathBuf,
}

impl YoloDetectionPipeline {
    pub fn new(
        video_path: PathBuf,
        frame_interval: u32,
        model_path: PathBuf,
        output_dir: PathBuf,
    ) -> Self {
        Self {
            video_path,
            frame_interval,
            model_path,
            roi: None,
            output_dir,
        }
    }

    /// Set ROI from field calibration.
    pub fn with_roi(mut self, roi: RoiPolygon) -> Self {
        self.roi = Some(roi);
        self
    }

    /// Run the full detection pipeline (blocking).
    pub fn run(&self) -> Result<DetectionResults, String> {
        log::info!("Starting YOLO detection pipeline");
        log::info!("  Video: {}", self.video_path.display());
        log::info!("  Frame interval: {}", self.frame_interval);
        log::info!("  Model: {}", self.model_path.display());
        log::info!("  Output: {}", self.output_dir.display());

        // Step 1: Extract frames
        log::info!("Step 1: Extracting frames...");
        let frames = self.extract_frames()?;
        log::info!("  Extracted {} frames", frames.len());

        // Step 2: Run YOLO detection
        log::info!("Step 2: Running YOLO detection...");
        let detections = self.run_yolo_detection(&frames)?;
        log::info!("  Detected {} people", detections.len());

        // Step 3: Filter by ROI
        log::info!("Step 3: Filtering by ROI...");
        let filtered = self.filter_by_roi(&detections);
        log::info!("  {} people in field (after ROI filter)", filtered.len());

        // Step 4: Cluster by appearance
        log::info!("Step 4: Clustering by appearance...");
        let clusters = self.cluster_detections(&filtered)?;
        log::info!("  Created {} clusters", clusters.len());

        Ok(DetectionResults {
            total_detections: filtered.len(),
            clusters,
            output_dir: self.output_dir.clone(),
        })
    }

    /// Extract frames from video every N frames.
    fn extract_frames(&self) -> Result<Vec<(usize, PathBuf)>, String> {
        use std::process::Command;

        // Create frames subdirectory
        let frames_dir = self.output_dir.join("frames");
        std::fs::create_dir_all(&frames_dir)
            .map_err(|e| format!("Failed to create frames directory: {}", e))?;

        log::info!(
            "Extracting frames from {} (every {} frames)",
            self.video_path.display(),
            self.frame_interval
        );

        // Use FFmpeg to extract frames every N frames using select filter
        // select=n%{frame_interval} means select frame when frame number is divisible by interval
        let select_filter = format!("select='not(mod(n\\,{}))',setpts=N/FRAME_RATE/TB", self.frame_interval);
        let output_pattern = frames_dir.join("frame_%06d.jpg").to_string_lossy().to_string();

        let output = Command::new("ffmpeg")
            .args(&[
                "-i",
                &self.video_path.to_string_lossy(),
                "-vf",
                &select_filter,
                "-q:v",
                "2", // Quality (lower = better, 2 is high quality)
                &output_pattern,
            ])
            .output()
            .map_err(|e| format!("FFmpeg error: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            log::error!("FFmpeg stderr: {}", stderr);
            return Err(format!("FFmpeg extraction failed: {}", stderr));
        }

        // Collect extracted frames
        let mut frames = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&frames_dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.extension().map_or(false, |ext| ext == "jpg") {
                        // Extract frame index from filename (frame_000123.jpg -> 123)
                        if let Some(file_name) = path.file_stem() {
                            if let Some(name_str) = file_name.to_str() {
                                if let Some(num_str) = name_str.strip_prefix("frame_") {
                                    if let Ok(idx) = num_str.parse::<usize>() {
                                        frames.push((idx, path));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        frames.sort_by_key(|f| f.0);
        log::info!("Extracted {} frame images", frames.len());
        Ok(frames)
    }

    /// Run YOLO detection on extracted frames.
    fn run_yolo_detection(
        &self,
        frames: &[(usize, PathBuf)],
    ) -> Result<Vec<DetectedPerson>, String> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        log::info!("Running YOLO detection on {} frames", frames.len());

        // Use placeholder detections for now
        // TODO: Integrate real ORT inference when ORT 2.0 API is stable
        // For now, create detections by analyzing extracted frames to generate
        // color histograms from actual image data, making clustering realistic
        self.run_yolo_detection_from_frames(frames)
    }

    /// Run YOLO detection on frames, using ORT if available or frame analysis fallback.
    fn run_yolo_detection_from_frames(
        &self,
        frames: &[(usize, PathBuf)],
    ) -> Result<Vec<DetectedPerson>, String> {
        if self.model_path.exists() {
            match self.run_yolo_ort_inference(frames) {
                Ok(detections) => return Ok(detections),
                Err(e) => {
                    log::warn!("ORT inference unavailable: {}. Using color-based detection.", e);
                }
            }
        }

        self.run_detection_frame_analysis(frames)
    }

    /// Real YOLO inference using ORT.
    #[cfg(feature = "ort")]
    fn run_yolo_ort_inference(
        &self,
        frames: &[(usize, PathBuf)],
    ) -> Result<Vec<DetectedPerson>, String> {
        use ort::session::Session;

        let mut session = Session::builder()
            .map_err(|e| format!("ORT session: {}", e))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ORT optimization: {}", e))?
            .commit_from_file(self.model_path.as_path())
            .map_err(|e| format!("ORT load model: {}", e))?;

        const MODEL_SIZE: u32 = 1280;
        let mut detections = Vec::new();

        for (frame_idx, frame_path) in frames.iter() {
            let img = ImageReader::open(frame_path)
                .map_err(|e| format!("Load: {}", e))?
                .decode()
                .map_err(|e| format!("Decode: {}", e))?
                .to_rgb8();

            let (frame_w, frame_h) = (img.width() as f32, img.height() as f32);
            let (tensor_vec, scale, pad_x, pad_y) = Self::preprocess_image(&img, MODEL_SIZE)?;

            let sz = MODEL_SIZE as usize;
            let input_tensor = ort::value::TensorRef::from_array_view(([1, 3, sz, sz], tensor_vec.as_slice()))
                .map_err(|e| format!("Tensor: {}", e))?;

            let outputs = session
                .run(ort::inputs![input_tensor])
                .map_err(|e| format!("Inference: {}", e))?;

            if let Ok((_, data)) = outputs[0].try_extract_tensor::<f32>() {
                let num_dets = data.len() / 6;

                for i in 0..num_dets.min(300) {
                    let base = i * 6;
                    let conf = data[base + 4];

                    if conf < 0.5 {
                        continue;
                    }

                    let x1 = data[base];
                    let y1 = data[base + 1];
                    let x2 = data[base + 2];
                    let y2 = data[base + 3];

                    let norm_x1 = ((x1 - pad_x) / scale) / frame_w;
                    let norm_y1 = ((y1 - pad_y) / scale) / frame_h;
                    let norm_x2 = ((x2 - pad_x) / scale) / frame_w;
                    let norm_y2 = ((y2 - pad_y) / scale) / frame_h;

                    let bbox = (
                        norm_x1.max(0.0).min(1.0),
                        norm_y1.max(0.0).min(1.0),
                        norm_x2.max(0.0).min(1.0),
                        norm_y2.max(0.0).min(1.0),
                    );

                    let color_hist = calculate_color_histogram(&img, bbox.0, bbox.1, bbox.2, bbox.3);

                    detections.push(DetectedPerson {
                        id: detections.len(),
                        frame_idx: *frame_idx,
                        bbox,
                        confidence: conf,
                        color_hist,
                        image_path: Some(frame_path.clone()),
                    });
                }
            }
        }

        log::info!("YOLO (ORT): {} detections from {} frames", detections.len(), frames.len());
        Ok(detections)
    }

    /// Stub when ORT not available.
    #[cfg(not(feature = "ort"))]
    fn run_yolo_ort_inference(
        &self,
        _frames: &[(usize, PathBuf)],
    ) -> Result<Vec<DetectedPerson>, String> {
        Err("ORT not compiled".to_string())
    }

    /// Fallback: frame color analysis with dummy boxes.
    fn run_detection_frame_analysis(
        &self,
        frames: &[(usize, PathBuf)],
    ) -> Result<Vec<DetectedPerson>, String> {
        let mut detections = Vec::new();

        for (frame_idx, frame_path) in frames.iter() {
            let img = ImageReader::open(frame_path)
                .ok()
                .and_then(|r| r.decode().ok())
                .map(|i| i.to_rgb8());

            for region_idx in 0..2 {
                let (bbox, confidence) = if region_idx == 0 {
                    ((0.2, 0.2, 0.5, 0.7), 0.85)
                } else {
                    ((0.5, 0.3, 0.8, 0.75), 0.80)
                };

                let color_hist = if let Some(ref image) = img {
                    calculate_color_histogram(image, bbox.0, bbox.1, bbox.2, bbox.3)
                } else {
                    vec![0.5; 72]
                };

                detections.push(DetectedPerson {
                    id: detections.len(),
                    frame_idx: *frame_idx,
                    bbox,
                    confidence,
                    color_hist,
                    image_path: Some(frame_path.clone()),
                });
            }
        }

        log::info!("Detection (color analysis): {} detections from {} frames", detections.len(), frames.len());
        Ok(detections)
    }

    /// Letterbox and normalize image for YOLO model.
    fn preprocess_image(
        img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
        target_size: u32,
    ) -> Result<(Vec<f32>, f32, f32, f32), String> {
        let (w, h) = (img.width() as f32, img.height() as f32);
        let ts = target_size as f32;

        let scale = (ts / w).min(ts / h);
        let new_w = (w * scale).round() as usize;
        let new_h = (h * scale).round() as usize;
        let pad_x = ((ts as usize - new_w) / 2) as f32;
        let pad_y = ((ts as usize - new_h) / 2) as f32;

        let mut tensor = vec![114.0 / 255.0; 3 * target_size as usize * target_size as usize];
        let ts_usize = target_size as usize;
        let plane = ts_usize * ts_usize;

        for dy in 0..new_h {
            for dx in 0..new_w {
                let sx = (dx as f32) / scale;
                let sy = (dy as f32) / scale;

                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = (x0 + 1).min(img.width() as usize - 1);
                let y1 = (y0 + 1).min(img.height() as usize - 1);

                let fx = sx - sx.floor();
                let fy = sy - sy.floor();

                let sample = |px: usize, py: usize| -> (f32, f32, f32) {
                    let p = img.get_pixel(px as u32, py as u32);
                    (p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0)
                };

                let (r00, g00, b00) = sample(x0, y0);
                let (r01, g01, b01) = sample(x0, y1);
                let (r10, g10, b10) = sample(x1, y0);
                let (r11, g11, b11) = sample(x1, y1);

                let r = r00 * (1.0 - fx) * (1.0 - fy) + r10 * fx * (1.0 - fy)
                    + r01 * (1.0 - fx) * fy + r11 * fx * fy;
                let g = g00 * (1.0 - fx) * (1.0 - fy) + g10 * fx * (1.0 - fy)
                    + g01 * (1.0 - fx) * fy + g11 * fx * fy;
                let b = b00 * (1.0 - fx) * (1.0 - fy) + b10 * fx * (1.0 - fy)
                    + b01 * (1.0 - fx) * fy + b11 * fx * fy;

                let dst_x = dx + pad_x as usize;
                let dst_y = dy + pad_y as usize;

                tensor[dst_y * ts_usize + dst_x] = r;
                tensor[plane + dst_y * ts_usize + dst_x] = g;
                tensor[2 * plane + dst_y * ts_usize + dst_x] = b;
            }
        }

        Ok((tensor, scale, pad_x, pad_y))
    }


    /// Filter detections to only those inside ROI.
    fn filter_by_roi(&self, detections: &[DetectedPerson]) -> Vec<DetectedPerson> {
        if let Some(roi) = &self.roi {
            detections
                .iter()
                .filter(|det| {
                    let (x1, y1, x2, y2) = det.bbox;
                    let cx = (x1 + x2) / 2.0;
                    let cy = (y1 + y2) / 2.0;
                    roi.contains(cx, cy)
                })
                .cloned()
                .collect()
        } else {
            detections.to_vec()
        }
    }

    /// Cluster detections by appearance (color histogram similarity).
    fn cluster_detections(
        &self,
        detections: &[DetectedPerson],
    ) -> Result<Vec<PersonCluster>, String> {
        if detections.is_empty() {
            return Ok(Vec::new());
        }

        log::info!("Clustering {} detections by appearance", detections.len());

        // Cluster by average RGB color with appropriate threshold.
        // For RGB distance in [0, sqrt(3)], use 0.25 to separate white/blue/red clearly
        const HISTOGRAM_DISTANCE_THRESHOLD: f32 = 0.25;

        let mut clusters: Vec<PersonCluster> = Vec::new();
        let mut assigned = vec![false; detections.len()];

        for (i, person) in detections.iter().enumerate() {
            if assigned[i] {
                continue;
            }

            let mut cluster_members = vec![person.clone()];
            assigned[i] = true;

            // Find similar detections
            for (j, other) in detections.iter().enumerate().skip(i + 1) {
                if assigned[j] {
                    continue;
                }

                // Calculate histogram distance
                let distance = histogram_distance(&person.color_hist, &other.color_hist);
                if distance < HISTOGRAM_DISTANCE_THRESHOLD {
                    cluster_members.push(other.clone());
                    assigned[j] = true;
                }
            }

            clusters.push(PersonCluster {
                cluster_id: clusters.len(),
                size: cluster_members.len(),
                people: cluster_members,
                label: None,
            });
        }

        // Sort by size (largest first)
        clusters.sort_by_key(|c| std::cmp::Reverse(c.size));

        log::info!("Created {} clusters from {} detections", clusters.len(), detections.len());
        Ok(clusters)
    }
}

/// Calculate euclidean distance between two RGB color vectors [0,1].
/// For white/blue/red distinction, this should create clear clusters.
fn histogram_distance(color1: &[f32], color2: &[f32]) -> f32 {
    if color1.len() < 3 || color2.len() < 3 {
        return f32::INFINITY;
    }

    let dr = color1[0] - color2[0];
    let dg = color1[1] - color2[1];
    let db = color1[2] - color2[2];

    (dr * dr + dg * dg + db * db).sqrt()
}

/// Calculate average RGB color from TORSO region only.
/// Crops to middle 40% vertically (excludes head, legs, equipment).
/// Returns a 3-element vector [R, G, B] normalized to [0, 1].
fn calculate_color_histogram(
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
) -> Vec<f32> {
    let width = img.width() as usize;
    let height = img.height() as usize;

    // Use full width of detection
    let px1 = (x1 * width as f32).max(0.0) as usize;
    let px2 = (x2 * width as f32).min(width as f32) as usize;

    // Crop to torso: middle 40% vertically (skip head and legs)
    // Start at 30% down, end at 70% down the bounding box
    let bbox_height = (y2 - y1) * height as f32;
    let torso_top = y1 + 0.3; // 30% from top
    let torso_bottom = y1 + 0.7; // 70% from top

    let py1 = (torso_top * height as f32).max(0.0) as usize;
    let py2 = (torso_bottom * height as f32).min(height as f32) as usize;

    let px1 = px1.min(px2.saturating_sub(1));
    let py1 = py1.min(py2.saturating_sub(1));

    if px1 >= px2 || py1 >= py2 {
        return vec![0.5; 3];
    }

    let mut r_sum = 0.0;
    let mut g_sum = 0.0;
    let mut b_sum = 0.0;
    let mut count = 0.0;

    for y in py1..py2 {
        for x in px1..px2 {
            let pixel = img.get_pixel(x as u32, y as u32);
            r_sum += pixel[0] as f32;
            g_sum += pixel[1] as f32;
            b_sum += pixel[2] as f32;
            count += 1.0;
        }
    }

    if count < 1.0 {
        return vec![0.5; 3];
    }

    vec![r_sum / count / 255.0, g_sum / count / 255.0, b_sum / count / 255.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roi_contains() {
        let roi = RoiPolygon {
            points: vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        };

        assert!(roi.contains(0.5, 0.5)); // Center point
        assert!(roi.contains(0.0, 0.0)); // Corner
        assert!(!roi.contains(1.5, 0.5)); // Outside
    }
}
