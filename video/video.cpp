#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>
#include <omp.h>

// Function to detect faces in a single frame
void detect_faces(cv::Mat& frame, cv::CascadeClassifier& face_cascade) {
    // Convert the frame to grayscale
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Detect faces
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(30, 30));

    // Draw a green rectangle around each detected face
    #pragma omp parallel for
    for (int i = 0; i < faces.size(); i++) {
        cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 2);
    }
}

// Function to apply some additional processing to the frame (example)
void apply_processing(cv::Mat& frame) {
    // Example: Apply Gaussian blur
    cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0);
}

int main() {
    // Load the pre-trained Haar Cascade for face detection
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(cv::samples::findFile("haarcascade_frontalface_default.xml"))) {
        std::cerr << "Error loading Haar cascade file!" << std::endl;
        return -1;
    }

    // Open the default video camera
    cv::VideoCapture cap(0); // 0 is the ID for the default camera
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file!" << std::endl;
        return -1;
    }

    // Main loop to capture and process each frame
    while (true) {
        cv::Mat frame;
        cap >> frame; // Capture the current frame
        if (frame.empty()) break; // Exit the loop if no frame is captured

        // Process the frame using OpenMP
        #pragma omp parallel sections
        {
            // Section 1: Detect faces
            #pragma omp section
            {
                detect_faces(frame, face_cascade);
            }

            // Section 2: Perform other processing
            #pragma omp section
            {
                apply_processing(frame);
            }
        }

        // Display the processed frame
        cv::imshow("Face Detection", frame);

        // Break the loop on 'q' key press
        if (cv::waitKey(1) == 'q') break;
    }

    // Release the video capture object and close all windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}