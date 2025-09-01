#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster.h"
    #include "cluster_acc.h"
    #include "distances.h"
    #include "metrics.h"
    #include "kmeans.h"
}

#include <fstream>

#define WARM_UP_RUNS 3
#define MEASURED_RUNS 10

std::string generateOutputPath(const std::string& input_image_path);

int main(int argc, char** argv) {
    srand(0);   // Seed for reproducibility

    KMeansParams params;
    std::string image_path;
    
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
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " -k <clusters> -i <image_path> [-e <stab_error>] [-mi <max_iterations>]\n";
            std::cout << "  -k <clusters>      Number of clusters (required)\n";
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
    if (!k_set || !image_set) {
        std::cerr << "Error: Both -k and -i arguments are required." << std::endl;
        std::cerr << "Usage: " << argv[0] << " -k <clusters> -i <image_path> [-e <stab_error>] [-mi <max_iterations>]" << std::endl;
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

    std::string output_image_path = generateOutputPath(image_path);

    // Array with all distances functions
    float (*distance_functions[])(const uint8_t*, const uint8_t*, unsigned int, float) = {
        squared_euclidean_distance,
        manhattan_distance,
        chebyshev_distance,
        minkowski_distance,
        cosine_distance
    };

    // Matrix for timing results
    std::vector<std::vector<double>> data(MEASURED_RUNS, std::vector<double>(5));

    // Time the execution 10 times and mean the result
    for (int func_idx = 0; func_idx < 5; func_idx++)
    {
        double total_duration = 0.0;

        for (int i = -WARM_UP_RUNS; i < MEASURED_RUNS; i++)
        {
            double start_time = static_cast<double>(cv::getTickCount());
            k_means(&params, distance_functions[func_idx], 1.5f);
            double end_time = static_cast<double>(cv::getTickCount());

            if (i >= 0)
            {
                double duration = (end_time - start_time) / cv::getTickFrequency();
                total_duration += duration;

                data[i][func_idx] = duration;
            }
        }

        std::cout << "K-means clustering mean time with distance function " << func_idx << ": " << total_duration / MEASURED_RUNS << " seconds." << std::endl;

        std::string output_path = output_image_path + "_" + std::to_string(func_idx) + ".png";
        cv::imwrite(output_path, clustered_img);
    }

    // Create CSV file
    std::string csv_name = "results/baseline_" + std::to_string(std::min(params.img_width, params.img_height)) + ".csv";
    std::ofstream file(csv_name);

    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    // Write header row
    file << "Squared euclidean distance,Manhattan distance,Chebyshev distance,Minkowski distance,Cosine distance\n";

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

std::string generateOutputPath(const std::string& input_image_path) {
    size_t last_slash = input_image_path.find_last_of("/\\");
    size_t last_dot = input_image_path.find_last_of(".");
    
    std::string directory;
    std::string filename_without_ext;
    
    if (last_slash != std::string::npos) {
        directory = input_image_path.substr(0, last_slash + 1);
        if (last_dot != std::string::npos && last_dot > last_slash) {
            filename_without_ext = input_image_path.substr(last_slash + 1, last_dot - last_slash - 1);
        } else {
            filename_without_ext = input_image_path.substr(last_slash + 1);
        }
    } else {
        directory = "./";
        if (last_dot != std::string::npos) {
            filename_without_ext = input_image_path.substr(0, last_dot);
        } else {
            filename_without_ext = input_image_path;
        }
    }
    
    return directory + "output_" + filename_without_ext + "_clustered";
}
