#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>
#include <omp.h>

// Function to detect faces and highlight them with green rectangles
void detect_faces(cv::Mat& frame, cv::CascadeClassifier& face_cascade, std::vector<cv::Rect>& faces) {
    // Convert the frame to grayscale
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Detect faces
    face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(30, 30));

    // Draw a green rectangle around each detected face
    #pragma omp parallel for
    for (int i = 0; i < faces.size(); i++) {
        cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 2);
    }
}

// Function to apply processing, preserving green rectangles
void apply_processing(cv::Mat& frame, const std::vector<cv::Rect>& faces) {
    // Apply Gaussian blur
    cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0);

    // Edge detection using Canny
    cv::Mat edges;
    cv::Canny(frame, edges, 100, 200);

    // Convert edges to BGR for display
    cv::Mat edges_bgr;
    cv::cvtColor(edges, edges_bgr, cv::COLOR_GRAY2BGR);

    // Increase brightness and contrast
    cv::Mat brighter_frame;
    frame.convertTo(brighter_frame, -1, 1.2, 30); // Increase contrast by 1.2 and brightness by 30

    // Merge results with some weighted sum for visual effect
    cv::addWeighted(brighter_frame, 0.6, edges_bgr, 0.4, 0, frame);

    // Preserve the green rectangles on the face areas
    #pragma omp parallel for
    for (int i = 0; i < faces.size(); i++) {
        // Extract the face region from the original frame
        cv::Mat face_roi = frame(faces[i]);
        cv::Mat green_face = cv::Mat::zeros(face_roi.size(), face_roi.type());

        // Make the face green while preserving its shape
        green_face.setTo(cv::Scalar(0, 255, 0));
        cv::addWeighted(face_roi, 0.5, green_face, 0.5, 0, face_roi);
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

        std::vector<cv::Rect> faces;

        // Process the frame using OpenMP
        #pragma omp parallel sections
        {
            // Section 1: Detect faces
            #pragma omp section
            {
                detect_faces(frame, face_cascade, faces);
            }

            // Section 2: Perform additional processing
            #pragma omp section
            {
                apply_processing(frame, faces);
            }
        }

        // Display the processed frame
        cv::imshow("Face Detection and Processing", frame);

        // Break the loop on 'q' key press
        if (cv::waitKey(1) == 'q') break;
    }

    // Release the video capture object and close all windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
