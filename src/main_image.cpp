#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster.h"
    #include "cluster_acc.h"
}

int main() {
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

    // Time the execution 10 times and mean the result
    double duration = 0.0;
    for (int i = 0; i < 10; i++) {
        double start_time = static_cast<double>(cv::getTickCount());
        k_means_acc(clustered_img.data, img.data, img_height, img_width, k, dimensions, stab_error, max_iterations);
        double end_time = static_cast<double>(cv::getTickCount());

        duration += (end_time - start_time) / cv::getTickFrequency();
    }
    std::cout << "K-means clustering mean time: " << duration / 10 << " seconds." << std::endl;

    // Display results resized to fit on screen
    cv::resize(img, img, cv::Size(), 0.5, 0.5);
    cv::resize(clustered_img, clustered_img, cv::Size(), 0.5, 0.5);
    cv::imshow("Original Image", img);
    cv::imshow("Clustered Image", clustered_img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}