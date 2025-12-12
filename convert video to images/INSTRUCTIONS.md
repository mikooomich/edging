Use the command to export all frames of a video. 

1. Get FFmpeg somehow.
2. Cd into this folder
3. Run the command with your video file

```bash
ffmpeg -i "<video file path>" frames/out-%03d.png 
```

Use this command to create a 30 fps video from frames (provided you set up the directory correctly)

```bash
ffmpeg -framerate 30 -pattern_type glob -i "*.png" -c:v vp9 -crf 19 -b:v 0 -pix_fmt yuv420p output.mp4
```
