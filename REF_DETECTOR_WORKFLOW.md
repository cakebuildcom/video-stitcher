# Multi-Class Sport Detector Training Workflow

This workflow creates a custom YOLO model to detect:
- **Team A** (home team)
- **Team B** (away team)  
- **Referees** (officials)
- **Other** (coaches, staff, spectators)

Works for any sport (lacrosse, soccer, football, etc.)

## Prerequisites

```bash
pip install -r ref_detector_requirements.txt
```

Or manually:
```bash
pip install ultralytics scikit-learn scipy opencv-python pillow
```

## Step 1: Extract Frames and Cluster by Appearance

This script extracts frames from your videos, detects all people, and **automatically groups similar-looking people together** (teams, refs, etc).

```bash
python create_ref_dataset.py --left target/release/left.mp4 --right target/release/right.mp4 --interval 5 --clusters 15
```

**Options:**
- `--interval 5`: Extract every 5th frame (lower = more data, slower processing)
- `--clusters 15`: Create 15 appearance-based clusters

**What it does:**
1. Extracts frames at regular intervals
2. Detects all people using YOLO
3. Crops each person
4. Clusters by appearance (color + size)
5. Sorts clusters by size (largest first)

**Output:**
- `sport_dataset/frames/` - All extracted frames
- `sport_dataset/people/` - Cropped person images
- `sport_dataset/label.html` - Interactive labeling UI

## Step 2: Label Clusters

1. **Open in browser:** `sport_dataset/label.html`
2. **Review each cluster** (shows 12 sample images per cluster)
3. **Click one label button:**
   - 👕 **Team A** - Primary team uniform
   - 👕 **Team B** - Secondary team uniform
   - 🏁 **Referees** - Officials/referees
   - ❓ **Other** - Coaches, staff, mixed
4. **Click "Export Multi-Class Dataset"** to download `detector_dataset_labeled.json`

**Tips for labeling:**
- Clusters are **automatically sorted by size** (biggest first)
- Big clusters = likely full teams (good candidates)
- Small clusters = mixed groups or outliers
- You only need to label ~10-15 clusters total
- Look at the sample images to understand each cluster's composition

## Step 3: Train Model

Once labeled, train a 4-class YOLO model:

```bash
python train_ref_model.py --dataset detector_dataset_labeled.json --output sport_model --epochs 50
```

**This will:**
1. Create YOLO dataset from labeled images
2. Split: 70% training, 30% validation
3. Train YOLOv8 Nano (fast) for 50 epochs
4. Save best model to `sport_model/train/weights/best.pt`

**Training time:** ~10-20 minutes on GPU, ~1 hour on CPU

## Step 4: Test Model

```bash
python sport_model/test_model.py target/release/left.mp4
```

Should show detections with confidence scores for each class.

## Step 5: Integration (Later)

Once satisfied with results:
1. Convert model to ONNX format
2. Integrate into reco-autocam to use custom classes
3. Update RefDirector to use Team A/Team B/Ref classes

## Expected Results

**Good clustering:** Each cluster is mostly one type (uniform color/appearance)
**Good labeling:** You correctly identify teams and refs
**Good training:** 70%+ accuracy on validation set

## Troubleshooting

### Missing dependencies
```bash
pip install -r ref_detector_requirements.txt
```

### CUDA not available / want CPU training
Edit `train_ref_model.py` line: change `device=0` to `device='cpu'`

### Training too slow
- Use `--epochs 10` instead of 50 (iterate faster)
- Extract fewer frames: use `--interval 10` instead of 5
- Use smaller clusters: `--clusters 10` instead of 15

### Clusters don't make sense
- This means teams have similar-looking uniforms
- Increase `--clusters 20` to split them further
- Or label mixed clusters as "Other"

### Poor detection after training
- Label more clusters (aim for 30+)
- Extract more frames (use `--interval 3`)
- Train longer (use `--epochs 100`)

---

**Questions?** Check script output for detailed logs.
