#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster.h"
}

int main() {
    // TODO Fix with relative path independent from working directory
    // cv::VideoCapture cap("../dataset/walking.mp4");
    // if (!cap.isOpened()) {
    //     return -1;
    // }

    // cv::Mat frame;

    // while (true) {
    //     cap >> frame;
    //     if (frame.empty()) {
    //         break;
    //     }
    //     cv::imshow("Video Frame", frame);

    //     if (cv::waitKey(30) >= 0) {
    //         break;
    //     }
    // }

    // Load image
    cv::Mat img = cv::imread("../dataset/frame0.png");
    if (img.empty()) {
        return -1;
    }

    size_t img_height = img.rows;
    size_t img_width = img.cols;
    unsigned int dimensions = img.channels(); // Number of channels (e.g., 3 for RGB)

    unsigned int k = 3;
    float stab_error = 1.0f; // Convergence threshold
    int max_iterations = 100;
    cv::Mat clustered_img(img_height, img_width, img.type());
    k_means(clustered_img.data, img.data, img_height, img_width, k, dimensions, stab_error, max_iterations);

    // Display results resized to fit on screen
    cv::resize(img, img, cv::Size(), 0.5, 0.5);
    cv::resize(clustered_img, clustered_img, cv::Size(), 0.5, 0.5);
    cv::imshow("Original Image", img);
    cv::imshow("Clustered Image", clustered_img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}