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
    for (const auto& face : faces) {
        cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
    }
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

        // Process the frame in parallel using OpenMP
        #pragma omp parallel
        {
            #pragma omp single nowait
            detect_faces(frame, face_cascade); // Call the face detection function
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
