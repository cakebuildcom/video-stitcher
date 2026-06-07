#!/usr/bin/env python3
"""
Multi-class detector dataset creation tool.

Extracts frames from left.mp4 and right.mp4, detects people,
clusters by appearance (to find teams), and creates a labeling interface.

Supports labeling: Team A, Team B, Referees, Other
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from collections import defaultdict
import argparse
from scipy.spatial.distance import cdist

# Try importing ultralytics YOLO, fallback to basic detection
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("WARNING: ultralytics not installed. Install with: pip install ultralytics")

class RefDatasetCreator:
    def __init__(self, left_video: str, right_video: str, output_dir: str = "ref_dataset"):
        self.left_video = left_video
        self.right_video = right_video
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)

        self.people_dir = self.output_dir / "people"
        self.people_dir.mkdir(exist_ok=True)

        self.detections = []  # List of {frame, camera, bbox, image_path, color_hist}
        self.clusters = {}    # cluster_id -> list of detection indices
        self.labels = {}      # cluster_id -> "ref" or "player" or None

        # Load YOLO model if available
        self.model = None
        if HAS_YOLO:
            try:
                self.model = YOLO("yolov8n.pt")  # Nano model for speed
                print("Loaded YOLOv8n model (CPU mode for compatibility)")
            except Exception as e:
                print(f"Could not load YOLO model: {e}")

    def extract_frames(self, video_path: str, camera_name: str, frame_interval: int = 30):
        """Extract frames from video every N frames."""
        print(f"\nExtracting frames from {camera_name} ({video_path})...")
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        extracted = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_path = self.frames_dir / f"{camera_name}_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted += 1

            frame_count += 1
            if frame_count % 300 == 0:
                print(f"  Processed {frame_count} frames, extracted {extracted}")

        cap.release()
        print(f"Extracted {extracted} frames from {camera_name}")
        return extracted

    def detect_people(self):
        """Detect people in all extracted frames."""
        if not self.model:
            print("ERROR: YOLO model not loaded. Cannot detect people.")
            return

        print("\nDetecting people in frames...")
        frame_files = sorted(self.frames_dir.glob("*.jpg"))

        for frame_idx, frame_path in enumerate(frame_files):
            frame = cv2.imread(str(frame_path))
            results = self.model(frame, conf=0.5, classes=0, device='cpu')  # class 0 = person, CPU mode

            # Extract bounding boxes
            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    # Crop person from frame
                    person_crop = frame[y1:y2, x1:x2]

                    if person_crop.size == 0:
                        continue

                    person_id = len(self.detections)
                    person_path = self.people_dir / f"person_{person_id:05d}.jpg"
                    cv2.imwrite(str(person_path), person_crop)

                    # Compute HSV color histogram
                    hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [18, 4], [0, 180, 0, 256])
                    hist = cv2.normalize(hist, hist).flatten()

                    self.detections.append({
                        "id": person_id,
                        "frame": frame_path.stem,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "image_path": str(person_path),
                        "color_hist": hist.tolist()
                    })

            if (frame_idx + 1) % 10 == 0:
                print(f"  Processed {frame_idx + 1}/{len(frame_files)} frames, found {len(self.detections)} people")

        print(f"Total people detected: {len(self.detections)}")

    def cluster_by_appearance(self, num_clusters: int = 15):
        """Cluster people by appearance (color + size) to find teams automatically."""
        print(f"\nClustering {len(self.detections)} people by appearance (into ~{num_clusters} groups)...")

        if not self.detections:
            print("No detections to cluster")
            return

        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            # Extract features: color histogram + bounding box size
            features = []
            for d in self.detections:
                hist = np.array(d["color_hist"])
                bbox = d["bbox"]
                width = (bbox[2] - bbox[0]) / 1920.0 if bbox[2] > bbox[0] else 0.1  # normalized
                height = (bbox[3] - bbox[1]) / 1080.0 if bbox[3] > bbox[1] else 0.1  # normalized
                area = width * height

                # Combine: color (weighted heavily) + size (light weight)
                combined = np.concatenate([hist * 2.0, [area]])
                features.append(combined)

            features = np.array(features)

            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # K-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)

            self.clusters = defaultdict(list)
            for person_idx, cluster_id in enumerate(labels):
                self.clusters[int(cluster_id)].append(person_idx)

            # Sort clusters by size (largest first) for easier labeling
            sorted_clusters = sorted(self.clusters.items(), key=lambda x: len(x[1]), reverse=True)
            self.clusters = dict(sorted_clusters)

            print(f"\nCreated {len(self.clusters)} appearance-based clusters:")
            for cluster_id in list(self.clusters.keys())[:10]:  # Show top 10
                size = len(self.clusters[cluster_id])
                sample = self.detections[self.clusters[cluster_id][0]]
                print(f"  Cluster {cluster_id}: {size} people | Color: {[round(v, 2) for v in sample['color_hist'][:3]]}")

        except ImportError as e:
            print(f"ERROR: Missing dependency: {e}")

    def generate_labeling_ui(self):
        """Generate HTML file for labeling clusters as Team A, Team B, Refs, or Other."""
        print("\nGenerating labeling UI...")

        html = """<!DOCTYPE html>
<html>
<head>
    <title>Multi-Class Sport Detector Labeler</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #1a1a1a; color: #fff; }
        .container { max-width: 1400px; margin: 0 auto; }
        .cluster { border: 3px solid #444; margin: 20px 0; padding: 15px; border-radius: 8px; }
        .cluster.team-a { border-color: #0099ff; background: #001a33; }
        .cluster.team-b { border-color: #ff9900; background: #330a00; }
        .cluster.refs { border-color: #00ff00; background: #003300; }
        .cluster.other { border-color: #999; background: #222; }
        .cluster-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .cluster-id { font-size: 18px; font-weight: bold; }
        .label-badge {
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
        }
        .label-badge.team-a { background: #0099ff; }
        .label-badge.team-b { background: #ff9900; }
        .label-badge.refs { background: #00ff00; color: #000; }
        .label-badge.other { background: #666; }
        .sample-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 12px;
            margin: 15px 0;
        }
        .sample-img { width: 100%; height: auto; border: 2px solid #555; border-radius: 4px; cursor: pointer; }
        .sample-img:hover { border-color: #fff; }
        .buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        button {
            padding: 10px 16px;
            font-size: 14px;
            cursor: pointer;
            border: none;
            border-radius: 4px;
            font-weight: bold;
        }
        .team-a-btn { background: #0099ff; color: white; }
        .team-a-btn:hover { background: #00bbff; }
        .team-b-btn { background: #ff9900; color: white; }
        .team-b-btn:hover { background: #ffaa22; }
        .refs-btn { background: #00ff00; color: #000; }
        .refs-btn:hover { background: #33ff33; }
        .other-btn { background: #666; color: white; }
        .other-btn:hover { background: #999; }
        .clear-btn { background: #444; color: white; }
        .clear-btn:hover { background: #666; }
        .stats {
            background: #222;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .stat-item { border-left: 4px solid #666; padding-left: 10px; }
        .stat-item.team-a { border-color: #0099ff; }
        .stat-item.team-b { border-color: #ff9900; }
        .stat-item.refs { border-color: #00ff00; }
        .stat-item.other { border-color: #999; }
        .stat-label { font-size: 12px; color: #aaa; }
        .stat-value { font-size: 24px; font-weight: bold; }
        .progress { margin: 15px 0; }
        .progress-bar { background: #444; height: 24px; border-radius: 4px; overflow: hidden; }
        .progress-fill { background: linear-gradient(90deg, #0099ff, #ff9900, #00ff00); height: 100%; transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: #000; font-weight: bold; }
        .save-section { margin-top: 40px; padding: 20px; background: #222; border-radius: 8px; border: 2px solid #444; }
        h1 { margin-top: 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Multi-Class Sport Detector Labeler</h1>
        <p>Label each cluster as: <strong>Team A</strong> | <strong>Team B</strong> | <strong>Referees</strong> | <strong>Other</strong></p>

        <div class="stats">
            <div class="stat-item team-a">
                <div class="stat-label">Team A</div>
                <div class="stat-value"><span id="team-a-count">0</span></div>
            </div>
            <div class="stat-item team-b">
                <div class="stat-label">Team B</div>
                <div class="stat-value"><span id="team-b-count">0</span></div>
            </div>
            <div class="stat-item refs">
                <div class="stat-label">Referees</div>
                <div class="stat-value"><span id="refs-count">0</span></div>
            </div>
            <div class="stat-item other">
                <div class="stat-label">Other</div>
                <div class="stat-value"><span id="other-count">0</span></div>
            </div>
        </div>

        <div class="progress">
            <strong>Progress:</strong> <span id="progress">0</span>% | <span id="labeled-info">0 of 0 labeled</span>
            <div class="progress-bar">
                <div id="progress-fill" class="progress-fill" style="width: 0%"></div>
            </div>
        </div>

        <div id="clusters-container"></div>

        <div class="save-section">
            <h2>📊 Export Dataset</h2>
            <p>Once you've labeled clusters, click Export to create training dataset.</p>
            <p><strong>Minimum recommended:</strong> 3-5 clusters per class</p>
            <button onclick="exportDataset()" style="background: #0066cc; color: white; padding: 15px 30px; font-size: 16px; cursor: pointer;">
                ⬇️ Export Multi-Class Dataset
            </button>
        </div>
    </div>

    <script>
        const detections = """ + json.dumps(self.detections) + """;
        const clusters = """ + json.dumps(dict(self.clusters)) + """;
        const labels = {};

        const classColors = {
            'team-a': '#0099ff',
            'team-b': '#ff9900',
            'refs': '#00ff00',
            'other': '#666'
        };

        function updateStats() {
            const totalClusters = Object.keys(clusters).length;
            const teamACount = Object.values(labels).filter(l => l === 'team-a').length;
            const teamBCount = Object.values(labels).filter(l => l === 'team-b').length;
            const refsCount = Object.values(labels).filter(l => l === 'refs').length;
            const otherCount = Object.values(labels).filter(l => l === 'other').length;
            const labeled = teamACount + teamBCount + refsCount + otherCount;

            document.getElementById('team-a-count').textContent = teamACount;
            document.getElementById('team-b-count').textContent = teamBCount;
            document.getElementById('refs-count').textContent = refsCount;
            document.getElementById('other-count').textContent = otherCount;
            document.getElementById('labeled-info').textContent = `${labeled} of ${totalClusters} labeled`;

            const progress = Math.round((labeled / totalClusters) * 100);
            document.getElementById('progress').textContent = progress;
            document.getElementById('progress-fill').style.width = progress + '%';
        }

        function setLabel(clusterId, label) {
            labels[clusterId] = label;
            updateStats();
            renderCluster(clusterId);
        }

        function clearLabel(clusterId) {
            delete labels[clusterId];
            updateStats();
            renderCluster(clusterId);
        }

        function renderCluster(clusterId) {
            const personIndices = clusters[clusterId];
            const label = labels[clusterId];

            const html = `
                <div class="cluster ${label || ''}">
                    <div class="cluster-header">
                        <div class="cluster-id">Cluster ${clusterId} — <strong>${personIndices.length} people</strong></div>
                        <div>
                            ${label ? `<span class="label-badge ${label}">${label.toUpperCase().replace('-', ' ')}</span>` : '<span style="color: #999;">UNLABELED</span>'}
                        </div>
                    </div>
                    <div class="sample-grid">
                        ${personIndices.slice(0, 12).map(idx =>
                            `<img src="${detections[idx].image_path}" class="sample-img" title="Person ${idx}">`
                        ).join('')}
                    </div>
                    <div class="buttons">
                        <button class="team-a-btn" onclick="setLabel('${clusterId}', 'team-a')">👕 Team A</button>
                        <button class="team-b-btn" onclick="setLabel('${clusterId}', 'team-b')">👕 Team B</button>
                        <button class="refs-btn" onclick="setLabel('${clusterId}', 'refs')">🏁 Referees</button>
                        <button class="other-btn" onclick="setLabel('${clusterId}', 'other')">❓ Other</button>
                        <button class="clear-btn" onclick="clearLabel('${clusterId}')">✕ Clear</button>
                    </div>
                </div>
            `;

            document.getElementById(`cluster-${clusterId}`).innerHTML = html;
        }

        function initUI() {
            const container = document.getElementById('clusters-container');

            for (const clusterId of Object.keys(clusters)) {
                const div = document.createElement('div');
                div.id = `cluster-${clusterId}`;
                container.appendChild(div);
                renderCluster(clusterId);
            }

            updateStats();
        }

        function exportDataset() {
            const teamAIndices = Object.entries(labels)
                .filter(([_, label]) => label === 'team-a')
                .flatMap(([clusterId, _]) => clusters[clusterId]);

            const teamBIndices = Object.entries(labels)
                .filter(([_, label]) => label === 'team-b')
                .flatMap(([clusterId, _]) => clusters[clusterId]);

            const refsIndices = Object.entries(labels)
                .filter(([_, label]) => label === 'refs')
                .flatMap(([clusterId, _]) => clusters[clusterId]);

            const otherIndices = Object.entries(labels)
                .filter(([_, label]) => label === 'other')
                .flatMap(([clusterId, _]) => clusters[clusterId]);

            const dataset = {
                team_a: teamAIndices.map(idx => detections[idx]),
                team_b: teamBIndices.map(idx => detections[idx]),
                refs: refsIndices.map(idx => detections[idx]),
                other: otherIndices.map(idx => detections[idx]),
                total_team_a: teamAIndices.length,
                total_team_b: teamBIndices.length,
                total_refs: refsIndices.length,
                total_other: otherIndices.length
            };

            const dataStr = JSON.stringify(dataset, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'detector_dataset_labeled.json';
            link.click();

            alert(`Exported!\\nTeam A: ${teamAIndices.length}\\nTeam B: ${teamBIndices.length}\\nRefs: ${refsIndices.length}\\nOther: ${otherIndices.length}`);
        }

        window.onload = initUI;
    </script>
</body>
</html>
"""

        ui_path = self.output_dir / "label.html"
        with open(ui_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Generated labeling UI: {ui_path}")
        return ui_path

    def run(self, frame_interval: int = 5, num_clusters: int = 15):
        """Run full pipeline."""
        print("=" * 60)
        print("MULTI-CLASS SPORT DETECTOR DATASET CREATION")
        print("=" * 60)

        # Extract frames
        self.extract_frames(self.left_video, "left", frame_interval)
        self.extract_frames(self.right_video, "right", frame_interval)

        # Detect people
        self.detect_people()

        # Cluster by appearance
        self.cluster_by_appearance(num_clusters)

        # Generate UI
        ui_path = self.generate_labeling_ui()

        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print(f"1. Open in browser: {ui_path}")
        print("2. Review clusters and label each as:")
        print("   - Team A (home team)")
        print("   - Team B (away team)")
        print("   - Referees (officials)")
        print("   - Other (coaches, staff, etc)")
        print("3. Click 'Export Multi-Class Dataset' to download labeled data")
        print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create multi-class sport detector training dataset")
    parser.add_argument("--left", default="left.mp4", help="Path to left video")
    parser.add_argument("--right", default="right.mp4", help="Path to right video")
    parser.add_argument("--output", default="sport_dataset", help="Output directory")
    parser.add_argument("--interval", type=int, default=5, help="Frame extraction interval (lower = more frames)")
    parser.add_argument("--clusters", type=int, default=15, help="Number of appearance clusters")

    args = parser.parse_args()

    creator = RefDatasetCreator(args.left, args.right, args.output)
    creator.run(args.interval, args.clusters)
