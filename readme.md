
# Face_recognition_paraLeLe

This project uses OpenCV to perform face detection in images and video, leveraging OpenMP for parallel processing.

## Requirements

- **OpenCV 4** or higher
- **OpenMP** (included in most GCC installations)

### Install Dependencies (Ubuntu)

To install OpenCV and necessary tools, run:

```bash
sudo apt-get update
sudo apt-get install libopencv-dev
```

## Compile and Run Instructions

### 1. Compiling the Video Face Detection Program

The video face detection example captures video from your webcam, detects faces in real-time, and draws rectangles around the detected faces.

To compile the video program:

```bash
cd video
++ -fopenmp -o face_detection_parallel video.cpp `pkg-config --cflags --libs opencv4`  -O2
```

### 2. Compiling the Image Face Detection Program

The image face detection program processes multiple images in parallel using OpenMP and highlights detected faces in each image.

To compile the image program:

```bash
cd photo
g++ main.cpp -o parallel_face_detection `pkg-config --cflags --libs opencv4` -fopenmp
```

## How to Use

### 1. Video Face Detection

1. Make sure your webcam is connected.
2. Run the compiled program:

   ```bash
   ./face_detection
   ```

3. The program will display the video feed with detected faces outlined in green rectangles. Press `q` to quit.

### 2. Image Face Detection

1. Place your images (e.g., `1.jpg`, `2.jpg`, `3.jpg`) in the same directory as the executable or specify their paths in the code.
2. Run the program:

   ```bash
   ./parallel_face_detection
   ```

3. The program will detect faces in each image and save the processed output (e.g., `output1.jpg`, `output2.jpg`, `output3.jpg`).

## Important Notes

- Ensure that the file `haarcascade_frontalface_default.xml` (for face detection) is present in your OpenCV installation or provide the absolute path to it in the code.
- Navigate to the project folder before compiling the code.

---

**Face_recognition_paraLeLe** utilizes both real-time video and static images for efficient face detection using parallel processing.
