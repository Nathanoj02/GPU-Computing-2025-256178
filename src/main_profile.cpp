#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster_acc.h"
}

int main(int argc, char** argv) {
    // Get k and image name from command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <k> <image_path>" << std::endl;
        return -1;
    }

    unsigned int k = std::stoi(argv[1]);
    std::string image_path = argv[2];

    // Load image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        return -1;
    }

    size_t img_height = img.rows;
    size_t img_width = img.cols;
    unsigned int dimensions = img.channels(); // Number of channels (e.g., 3 for RGB)

    cv::Mat clustered_img(img_height, img_width, img.type());
    float stab_error = 1.0f; // Convergence threshold
    int max_iterations = 100;

    for (int i = 0; i < 10; i++)
    {
        k_means_acc_old(clustered_img.data, img.data, img_height, img_width, k, dimensions, stab_error, max_iterations, 0);
    }

    return 0;
}