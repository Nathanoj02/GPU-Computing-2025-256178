#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster.h"
    #include "cluster_acc.h"
    #include "distances.h"
    #include "metrics.h"
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

    // Array with all distances functions
    float (*distance_functions[])(const uint8_t*, const uint8_t*, unsigned int, float) = {
        squared_euclidean_distance,
        manhattan_distance,
        chebyshev_distance,
        minkowski_distance,
        cosine_distance
    };

    // Time the execution 10 times and mean the result
    for (int func_idx = 0; func_idx < 5; func_idx++)
    {
        double duration = 0.0;

        for (int i = 0; i < 10; i++)
        {
            double start_time = static_cast<double>(cv::getTickCount());
            k_means(clustered_img.data, img.data, img_height, img_width, k, dimensions, stab_error, max_iterations, distance_functions[func_idx], 1.5f);
            double end_time = static_cast<double>(cv::getTickCount());

            duration += (end_time - start_time) / cv::getTickFrequency();
        }

        std::cout << "K-means clustering mean time with distance function " << func_idx << ": " << duration / 10 << " seconds." << std::endl;

        std::string output_path = "../dataset/output_image_" + std::to_string(func_idx) + ".png";
        cv::imwrite(output_path, clustered_img);
    }

    return 0;
}