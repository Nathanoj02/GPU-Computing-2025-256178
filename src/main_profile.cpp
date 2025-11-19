#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster_acc.h"
    #include "kmeans.h"
}

int main(int argc, char** argv) {
    srand(0);   // Seed for reproducibility

    KMeansParams params;
    std::string image_path;
    std::string alg = "old"; // Default algorithm

    // Default values
    params.k = 0;  // Will be set as required
    params.stab_error = 1.0f;   // Convergence threshold
    params.max_iterations = 100;
    
    // Parse command line arguments
    bool k_set = false, image_set = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-k" && i + 1 < argc) {
            params.k = std::stoi(argv[++i]);
            k_set = true;
        }
        else if (arg == "-i" && i + 1 < argc) {
            image_path = argv[++i];
            image_set = true;
        }
        else if (arg == "-e" && i + 1 < argc) {
            params.stab_error = std::stof(argv[++i]);
        }
        else if (arg == "-mi" && i + 1 < argc) {
            params.max_iterations = std::stoi(argv[++i]);
        }
        else if (arg == "-alg" && i + 1 < argc) {
            alg = argv[++i];
        }
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " -k <clusters> -i <image_path> [-e <stab_error>] [-mi <max_iterations>] [-alg <old|new|reduce>]\n";
            std::cout << "  -k <clusters>      Number of clusters (required)\n";
            std::cout << "  -i <image_path>    Path to input image (required)\n";
            std::cout << "  -e <stab_error>    Stability error threshold (default: 1.0)\n";
            std::cout << "  -mi <max_iterations> Maximum iterations (default: 100)\n";
            std::cout << "  -alg <old|new|reduce> Algorithm: old, new, or reduce (default: old)\n";
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
    if (!k_set || !image_set) {
        std::cerr << "Error: Both -k and -i arguments are required." << std::endl;
        std::cerr << "Usage: " << argv[0] << " -k <clusters> -i <image_path> [-e <stab_error>] [-mi <max_iterations>] [-alg <old|new|reduce>]" << std::endl;
        return -1;
    }
    
    // Validate arguments
    if (params.k <= 0) {
        std::cerr << "Error: Number of clusters must be positive." << std::endl;
        return -1;
    }
    
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

    // Select implementation based on -alg
    if (alg == "old") {
        k_means_acc_old(&params);
    } 
    else if (alg == "new") {
        k_means_acc(&params);
    } 
    else if (alg == "reduce") {
        k_means_pp_acc_reduce(&params);
    } 
    else {
        std::cerr << "Unknown algorithm: " << alg << std::endl;
        std::cerr << "Valid options: old, new, reduce" << std::endl;
        return -1;
    }

    return 0;
}
