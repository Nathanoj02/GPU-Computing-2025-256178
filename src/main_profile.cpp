#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster_acc.h"
    #include "kmeans.h"
}

int main(int argc, char** argv) {
    // Get k and image name from command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <k> <image_path>" << std::endl;
        return -1;
    }

    srand(0);   // Seed for reproducibility

    KMeansParams params;

    params.k = std::stoi(argv[1]);
    std::string image_path = argv[2];

    // Load image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        return -1;
    }

    params.img_height = img.rows;
    params.img_width = img.cols;
    params.dimensions = img.channels(); // Number of channels (e.g., 3 for RGB)

    cv::Mat clustered_img(params.img_height, params.img_width, img.type());
    params.stab_error = 1.0f; // Convergence threshold
    params.max_iterations = 100;

    params.img = img.data;
    params.dst = clustered_img.data;

    for (int i = 0; i < 10; i++)
    {
        k_means_acc(&params);
    }

    return 0;
}