# Clustering algorithms on GPUs

## 📦 Dependencies
- CUDA Toolkit
- OpenCV for C++
- NVIDIA HPC SDK
- [Optional] Python

## 📥 Download dataset

1. **Download the sample video and image:**
    ```bash
    wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1LP73pllzIlZ45WIV48FdYZwBhqjtLYxp" -O dataset/walking_1080.mp4
    ```
    
    ```bash
    wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1JCUK79T5InpTKnElJkflcmYa9AR7ouX8" -O dataset/frame0_1080.png
    ```

    If the files are not available, you can find them on [Kaggle](https://www.kaggle.com/datasets/sharjeelmazhar/human-activity-recognition-video-dataset?resource=download). Click on the link, then click on the `Walking` folder, then download the `Walking (136).mp4` file
    You can extract the first frame from that video using the following command:
    ```bash
    ffmpeg -i walking_1080.mp4 -vf "select=eq(n\,0)" -q:v 3 frame0_1080.png
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

## 🛠️ Compiling and Running the Programs

### Run on the Cluster
Output files will go in the folder `cluster/`

Metrics analysis (elbow and silhouette methods):
```sh
sbatch run_metrics.sh
```

Baseline CPU algorithm:
```sh
sbatch run_baseline.sh
```

Video GPU algorithm:
```sh
sbatch run_video.sh
```

Video calibration GPU algorithm:
```sh
sbatch run_calibration.sh
```

### Run on local machine
1. **Compile everything**
    ```bash
    make
    ```
    This produces 6 executables, each one specific for a different task

2. **Execute binaries**
    
    Find best `k` with elbow and silhouette methods:
    ```bash
    ./bin/main_metrics -i dataset/frame0_1080.png
    ```
   
    CPU baseline with different types of distance:
    ```bash
    ./bin/main_image -k 3 -i dataset/frame0_1080.png
    ```
    
    GPU implementations on video:
    ```bash
    ./bin/main_video -k 3 -v dataset/walking_1080.mp4
    ```

    GPU video calibration implementation:
    ```bash
    ./bin/main_video_calibration -k 3 -v dataset/walking_1080.mp4
    ```

    Save full clustered video (will run on GPU video calibration implementation for maximum speed):
    ```bash
    ./bin/main_video_showcase -k 3 -v dataset/walking_1080.mp4
    ```

    Most of the script will generate a `.csv` file saving the data for future analysis

    Generate a report with nsys for profiling the (old) GPU implementation:
    ```bash
    nsys profile -o profiling_result bin/main_profile -k 3 -i dataset/frame0_1080.png -alg old
    ```
    Flag alg has the following possible values: `old`, `new` and `reduce`

### Run with multicore CPU
- In the `Makefile` &rarr; `CXXFLAGS` and `CFLAGS` (rows 8-9) replace `-acc -gpu=cc75` flag with `-acc=multicore -fast`
- Compile with `make`
- Run with:
  ```bash
  ACC_NUM_CORES=6 ACC_DEVICE_TYPE=multicore ./bin/main_video -k 3 -v dataset/walking_1080.mp4 -out results/multicore_1080.csv
  ```
  You can change the number of cores to fit your CPU

To have all the plots run also the followings:
```bash
ACC_NUM_CORES=6 ACC_DEVICE_TYPE=multicore ./bin/main_video -k 3 -v dataset/walking_720.mp4 -out results/multicore_720.csv
```
```bash
ACC_NUM_CORES=6 ACC_DEVICE_TYPE=multicore ./bin/main_video -k 3 -v dataset/walking_480.mp4 -out results/multicore_480.csv
```
```bash
ACC_NUM_CORES=6 ACC_DEVICE_TYPE=multicore ./bin/main_video -k 3 -v dataset/walking_240.mp4 -out results/multicore_240.csv
```

## 📊 Generate Plots
In the `plots/` folder there's a Jupyter Notebook that read from the `.csv` files and produces plots comparing the different implementations

1. **[Optional] Use a virtual environment**
    Create the virtualenv
    ```bash
    virtualenv .venv -p python3
    ```

    Activate it
    ```bash
    source .venv/bin/activate
    ```

    Install dependencies
    ```bash
    pip install -r plots/requirements.txt
    ```

2. **Run the Jupyter Notebook**

    You will get the plots in the notebook and also in the `plots/` folder