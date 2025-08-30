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
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    srand(0);   // Seed for reproducibility

    std::string image_path = argv[1];

    // Load image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        return -1;
    }

    KMeansParams params;
    params.img_height = img.rows;
    params.img_width = img.cols;
    params.dimensions = img.channels(); // Number of channels (e.g., 3 for RGB)

    cv::Mat clustered_img(params.img_height, params.img_width, img.type());
    params.stab_error = 1.0f; // Convergence threshold
    params.max_iterations = 100;

    params.img = img.data;
    params.dst = clustered_img.data;

    float (*distance_func)(const uint8_t*, const uint8_t*, unsigned int, float) = euclidean_distance;

    for (unsigned int k = 2; k <= 10; k++)
    {
        params.k = k;

        k_means(&params, distance_func, 1.5f);
    
        // Evaluate
        double elbow_err = elbow_method(params.img, params.dst, params.img_height, params.img_width, k, params.dimensions, distance_func, 1.5f);
        std::cout << "Elbow method total squared error for k=" << k << ": " << elbow_err << std::endl;
        std::flush(std::cout);

        double silhouette_score = silhouette_method_sampled(params.img, params.dst, params.img_height, params.img_width, k, params.dimensions, distance_func, 1.5f, 5000);
        std::cout << "Silhouette method average score for k=" << k << ": " << silhouette_score << std::endl;
    }

    cv::destroyAllWindows();

    return 0;
}