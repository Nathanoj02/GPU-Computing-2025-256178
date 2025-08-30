# Clustering algorithms on GPUs

## 📥 Download dataset

1. **Download the sample video and image:**
    ```bash
    wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1LP73pllzIlZ45WIV48FdYZwBhqjtLYxp" -O dataset/walking_1080.mp4
    ```
    
    ```bash
    wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1JCUK79T5InpTKnElJkflcmYa9AR7ouX8" -O dataset/frame0.png
    ```


2. **[Optional] Scale at multiple resolutions:**
    
    - 720p:
      ```bash
      ffmpeg -i dataset/walking_1080.mp4 -s 720x1280 -c:a copy dataset/walking_720.mp4
      ```

    - 480p:
      ```bash
      ffmpeg -i dataset/walking_1080.mp4 -s 480x854 -c:a copy dataset/walking_480.mp4
      ```
    
    - 240p:
      ```bash
      ffmpeg -i dataset/walking_1080.mp4 -s 240x426 -c:a copy dataset/walking_240.mp4
      ```