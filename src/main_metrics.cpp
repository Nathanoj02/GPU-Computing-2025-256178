#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster.h"
    #include "cluster_acc.h"
    #include "distances.h"
    #include "metrics.h"
}

int main(int argc, char** argv) {
    // Get k and image name from command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];

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


    float (*distance_func)(const uint8_t*, const uint8_t*, unsigned int, float) = euclidean_distance;

    for (unsigned int k = 2; k <= 10; k++)
    {
        k_means(clustered_img.data, img.data, img_height, img_width, k, dimensions, stab_error, max_iterations, distance_func, 1.5f);
    
        // Evaluate
        double elbow_err = elbow_method(img.data, clustered_img.data, img_height, img_width, k, dimensions, distance_func, 1.5f);
        std::cout << "Elbow method total squared error for k=" << k << ": " << elbow_err << std::endl;
        std::flush(std::cout);
    
        double silhouette_score = silhouette_method_sampled(img.data, clustered_img.data, img_height, img_width, k, dimensions, distance_func, 1.5f, 5000);
        std::cout << "Silhouette method average score for k=" << k << ": " << silhouette_score << std::endl;
    }

    cv::destroyAllWindows();

    return 0;
}