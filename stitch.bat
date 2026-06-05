@echo off
.\target\release\reco.exe stitch left.mp4 right.mp4 ^
  -c match.json ^
  -o panorama2.mp4 ^
  --tracking field ^
  --model yolov8m.onnx ^
  --encoder hevc_nvenc ^
  --codec hevc ^
  --quality high ^
  --width 1920 ^
  --height 1080 ^
  --lookahead 1.5 ^
  --detection-interval 2 ^
  --no-zero-copy ^
  --events events.jsonl
pause