cd C:\Users\wilso\Desktop\ai\video-stitcher

.\target\release\reco.exe stitch left.mp4 right.mp4 -c match.json -o panorama.mp4 ^
  --tracking field ^
  --model C:\Users\wilso\yolov8m.onnx ^
  --encoder hevc_nvenc ^
  --codec hevc ^
  --quality high ^
  --width 3840 ^
  --height 1080 ^
  --lookahead 1.5 ^
  --detection-interval 2 ^
  --no-zero-copy ^
  --events events.jsonl
  
  
  pause