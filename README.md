# Clustering algorithms on GPUs

## 📥 Download dataset

1. **Download the sample video and image:**
    ```bash
    wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1LP73pllzIlZ45WIV48FdYZwBhqjtLYxp" -O dataset/walking_1080.mp4
    ```
    
    ```bash
    wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1JCUK79T5InpTKnElJkflcmYa9AR7ouX8" -O dataset/frame0_1080.png
    ```


2. **[Optional] Scale at multiple resolutions:**

    - 720p:
      ```bash
      ffmpeg -i dataset/walking_1080.mp4 -s 720x1280 -c:a copy dataset/walking_720.mp4
      ```
      ```bash
      convert -resize 720x1280 dataset/frame0_1080.png dataset/frame0_720.png
      ```

    - 480p:
      ```bash
      ffmpeg -i dataset/walking_1080.mp4 -s 480x854 -c:a copy dataset/walking_480.mp4
      ```
      ```bash
      convert -resize 480x854 dataset/frame0_1080.png dataset/frame0_480.png
      ```
    
    - 240p:
      ```bash
      ffmpeg -i dataset/walking_1080.mp4 -s 240x426 -c:a copy dataset/walking_240.mp4
      ```
      ```bash
      convert -resize 240x426 dataset/frame0_1080.png dataset/frame0_240.png
      ```