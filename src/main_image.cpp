#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster.h"
    #include "cluster_acc.h"
    #include "distances.h"
    #include "metrics.h"
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
            k_means(&params, distance_functions[func_idx], 1.5f);
            double end_time = static_cast<double>(cv::getTickCount());

            duration += (end_time - start_time) / cv::getTickFrequency();
        }

        std::cout << "K-means clustering mean time with distance function " << func_idx << ": " << duration / 10 << " seconds." << std::endl;

        std::string output_path = "../dataset/output_image_" + std::to_string(func_idx) + ".png";
        cv::imwrite(output_path, clustered_img);
    }

    return 0;
}