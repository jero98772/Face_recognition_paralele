#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>
#include <omp.h>

void detect_faces(const std::string& image_path, const std::string& output_path) {
    // Load the pre-trained Haar Cascade for face detection
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(cv::samples::findFile("haarcascade_frontalface_default.xml"))) {
        std::cerr << "Error loading Haar cascade file!" << std::endl;
        return; // If loading fails, simply return without processing
    }

    // Load the image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Could not open or find the image: " << image_path << std::endl;
        return;
    }

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Detect faces
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(30, 30));

    // Draw a green rectangle around each detected face
    for (const auto& face : faces) {
        cv::rectangle(img, face, cv::Scalar(0, 255, 0), 2);
    }

    // Save the processed image
    cv::imwrite(output_path, img);
}

int main() {
    // Example list of images to process
    std::vector<std::string> image_paths = {"1.jpg", "2.jpg", "3.jpg"}; // Add your image paths here
    std::vector<std::string> output_paths = {"output1.jpg", "output2.jpg", "output3.jpg"};

    // Use OpenMP to process images in parallel
    #pragma omp parallel for
    for (size_t i = 0; i < image_paths.size(); ++i) {
        // Call the face detection function
        detect_faces(image_paths[i], output_paths[i]);
    }

    std::cout << "Face detection completed. Processed images are saved." << std::endl;
    return 0;
}
