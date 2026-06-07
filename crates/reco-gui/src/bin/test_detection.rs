fn main() {
    // Initialize logging
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    println!("\n=== Testing YOLO Detection Pipeline ===\n");

    // Get paths relative to app root
    let app_dir = std::path::PathBuf::from(".");
    let left_video = app_dir.join("left.mp4");
    let right_video = app_dir.join("right.mp4");

    println!("App directory: {}", app_dir.display());
    println!("Left video: {} (exists: {})", left_video.display(), left_video.exists());
    println!("Right video: {} (exists: {})", right_video.display(), right_video.exists());

    if !left_video.exists() || !right_video.exists() {
        println!("\n❌ Test FAILED: Missing video files");
        return;
    }

    // Test 1: Check FFmpeg availability
    println!("\n--- Test 1: FFmpeg availability ---");
    match std::process::Command::new("ffmpeg")
        .args(&["-version"])
        .output()
    {
        Ok(output) => {
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout);
                let first_line = version.lines().next().unwrap_or("unknown");
                println!("✓ FFmpeg available: {}", first_line);
            } else {
                println!("❌ FFmpeg not working properly");
                return;
            }
        }
        Err(e) => {
            println!("❌ FFmpeg not found: {}", e);
            return;
        }
    }

    // Test 2: Create detection pipeline and run
    println!("\n--- Test 2: Detection Pipeline ---");
    let output_dir = app_dir.join("detection_output_test");
    let _ = std::fs::remove_dir_all(&output_dir);

    // Test both videos with best model
    let videos = vec![("LEFT", left_video.clone()), ("RIGHT", right_video.clone())];
    let mut left_clusters = Vec::new();
    let mut right_clusters = Vec::new();

    for (side, video_path) in videos {
        println!("\n--- Testing {} video ---", side);

        let pipeline = reco_gui::detection::YoloDetectionPipeline::new(
            video_path,
            30, // frame interval
            app_dir.join("yolo26n.onnx"),
            output_dir.clone(),
        );

        match pipeline.run() {
            Ok(results) => {
                println!("✓ {} video: {} detections in {} clusters",
                    side, results.total_detections, results.clusters.len());
                for (i, cluster) in results.clusters.iter().enumerate() {
                    // Calculate average color for this cluster
                    let mut avg_r = 0.0;
                    let mut avg_g = 0.0;
                    let mut avg_b = 0.0;
                    for person in &cluster.people {
                        if person.color_hist.len() >= 3 {
                            avg_r += person.color_hist[0];
                            avg_g += person.color_hist[1];
                            avg_b += person.color_hist[2];
                        }
                    }
                    avg_r /= cluster.size as f32;
                    avg_g /= cluster.size as f32;
                    avg_b /= cluster.size as f32;

                    // Identify team by color
                    let (team, color) = if avg_r > 0.7 && avg_g > 0.7 && avg_b > 0.7 {
                        ("WHITE", "(White)")
                    } else if avg_b > 0.5 && avg_r < 0.3 && avg_g < 0.3 {
                        ("BLUE", "(Blue)")
                    } else if avg_r > 0.6 && avg_g < 0.3 && avg_b < 0.3 {
                        ("RED", "(Red)")
                    } else {
                        ("OTHER", "(Mixed)")
                    };

                    println!("  - Cluster {}: {} people {} RGB({:.2},{:.2},{:.2})",
                        i, cluster.size, color, avg_r, avg_g, avg_b);
                }

                let cluster_sizes: Vec<usize> = results.clusters.iter().map(|c| c.size).collect();
                if side == "LEFT" {
                    left_clusters = cluster_sizes;
                } else {
                    right_clusters = cluster_sizes;
                }
            }
            Err(e) => {
                println!("❌ {} video failed: {}", side, e);
            }
        }
    }

    // Compare consistency
    println!("\n=== CLUSTER CONSISTENCY CHECK ===");
    if left_clusters.len() == right_clusters.len() {
        println!("✓ Same number of clusters: {}", left_clusters.len());
        for i in 0..left_clusters.len() {
            let diff = (left_clusters[i] as i32 - right_clusters[i] as i32).abs();
            let pct_diff = (diff as f32 / left_clusters[i].max(1) as f32) * 100.0;
            println!("  Cluster {}: LEFT={:3} RIGHT={:3} (diff: {}  {:.1}%)",
                i, left_clusters[i], right_clusters[i], diff, pct_diff);
        }
    } else {
        println!("⚠ Different cluster counts: LEFT={} RIGHT={}",
            left_clusters.len(), right_clusters.len());
    }

    // Test 3: Verify frame extraction
    println!("\n--- Test 3: Frame Extraction ---");
    let frames_dir = output_dir.join("frames");
    if frames_dir.exists() {
        match std::fs::read_dir(&frames_dir) {
            Ok(entries) => {
                let frame_count = entries
                    .filter_map(|e| {
                        e.ok()
                            .and_then(|entry| entry.path().extension().and_then(|ext| {
                                if ext == "jpg" {
                                    Some(())
                                } else {
                                    None
                                }
                            }))
                    })
                    .count();
                println!("✓ Frames extracted: {}", frame_count);
            }
            Err(e) => println!("❌ Could not read frames: {}", e),
        }
    }

    println!("\n=== ALL TESTS COMPLETED ===\n");
}
