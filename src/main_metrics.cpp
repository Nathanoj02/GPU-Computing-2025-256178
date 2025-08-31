#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster.h"
    #include "cluster_acc.h"
    #include "distances.h"
    #include "metrics.h"
    #include "kmeans.h"
}

#include <fstream>

#define START_K 2
#define END_K 10

int main(int argc, char** argv) {
    srand(0);   // Seed for reproducibility

    KMeansParams params;
    std::string image_path;
    
    // Default values
    params.stab_error = 1.0f;   // Convergence threshold
    params.max_iterations = 100;
    
    // Parse command line arguments
    bool image_set = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-i" && i + 1 < argc) {
            image_path = argv[++i];
            image_set = true;
        }
        else if (arg == "-e" && i + 1 < argc) {
            params.stab_error = std::stof(argv[++i]);
        }
        else if (arg == "-mi" && i + 1 < argc) {
            params.max_iterations = std::stoi(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " -i <image_path> [-e <stab_error>] [-mi <max_iterations>]\n";
            std::cout << "  -i <image_path>    Path to input image (required)\n";
            std::cout << "  -e <stab_error>    Stability error threshold (default: 1.0)\n";
            std::cout << "  -mi <max_iterations> Maximum iterations (default: 100)\n";
            std::cout << "  -h, --help         Show this help message\n";
            return 0;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::cerr << "Use -h or --help for usage information." << std::endl;
            return -1;
        }
    }
    
    // Check required arguments
    if (!image_set) {
        std::cerr << "Error: -i argument is required." << std::endl;
        std::cerr << "Usage: " << argv[0] << " -i <image_path> [-e <stab_error>] [-mi <max_iterations>]" << std::endl;
        return -1;
    }

    // Validate arguments
    if (params.stab_error < 0) {
        std::cerr << "Error: Stability error must be non-negative." << std::endl;
        return -1;
    }
    
    if (params.max_iterations <= 0) {
        std::cerr << "Error: Maximum iterations must be positive." << std::endl;
        return -1;
    }

    // Load image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        return -1;
    }

    params.img_height = img.rows;
    params.img_width = img.cols;
    params.dimensions = img.channels(); // Number of channels (e.g., 3 for RGB)

    cv::Mat clustered_img(params.img_height, params.img_width, img.type());

    params.img = img.data;
    params.dst = clustered_img.data;

    float (*distance_func)(const uint8_t*, const uint8_t*, unsigned int, float) = euclidean_distance;

    // Matrix for timing results
    std::vector<std::vector<double>> data(END_K - START_K + 1, std::vector<double>(2));

    for (unsigned int k = START_K; k <= END_K; k++)
    {
        params.k = k;

        k_means(&params, distance_func, 1.5f);
    
        // Evaluate
        double elbow_err = elbow_method(params.img, params.dst, params.img_height, params.img_width, k, params.dimensions, distance_func, 1.5f);
        std::cout << "Elbow method total squared error for k=" << k << ": " << elbow_err << std::endl;
        std::flush(std::cout);

        double silhouette_score = silhouette_method_sampled(params.img, params.dst, params.img_height, params.img_width, k, params.dimensions, distance_func, 1.5f, 5000);
        std::cout << "Silhouette method average score for k=" << k << ": " << silhouette_score << std::endl;

        data[k - START_K][0] = elbow_err;
        data[k - START_K][1] = silhouette_score;
    }

    cv::destroyAllWindows();

    // Create CSV file
    std::ofstream file("results/metrics.csv");
    
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    // Write header row
    file << "Elbow method total squared error,Silhouette method average score\n";

    // Write data rows
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";  // Add comma between values
            }
        }
        file << "\n";  // New line after each row
    }
    
    file.close();

    return 0;
}