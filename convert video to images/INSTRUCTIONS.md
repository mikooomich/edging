Use the command to export all frames of a video. 

1. Get FFmpeg somehow.
2. Cd into this folder
3. Run the command with your video file

```bash
ffmpeg -i "<video file path>" frames/out-%03d.png 
```
