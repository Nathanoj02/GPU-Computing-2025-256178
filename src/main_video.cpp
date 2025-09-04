#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster.h"
    #include "cluster_acc.h"
    #include "distances.h"
    #include "kmeans.h"
}

#include <fstream>

#define WARM_UP_FRAMES 5
#define MEASURED_FRAMES 20

int main(int argc, char** argv) {
    srand(0);   // Seed for reproducibility

    KMeansParams params;
    std::string video_path;
    std::string output_csv_path;
    
    // Default values
    params.k = 0;  // Will be set as required
    params.stab_error = 1.0f;   // Convergence threshold
    params.max_iterations = 100;
    
    // Parse command line arguments
    bool k_set = false, video_set = false, out_set = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-k" && i + 1 < argc) {
            params.k = std::stoi(argv[++i]);
            k_set = true;
        }
        else if (arg == "-v" && i + 1 < argc) {
            video_path = argv[++i];
            video_set = true;
        }
        else if (arg == "-e" && i + 1 < argc) {
            params.stab_error = std::stof(argv[++i]);
        }
        else if (arg == "-mi" && i + 1 < argc) {
            params.max_iterations = std::stoi(argv[++i]);
        }
        else if (arg == "-out" && i + 1 < argc) {
            output_csv_path = argv[++i];
            out_set = true;
        }
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " -k <clusters> -v <video_path> [-e <stab_error>] [-mi <max_iterations>]\n";
            std::cout << "  -k <clusters>      Number of clusters (required)\n";
            std::cout << "  -v <video_path>    Path to input video (required)\n";
            std::cout << "  -e <stab_error>    Stability error threshold (default: 1.0)\n";
            std::cout << "  -mi <max_iterations> Maximum iterations (default: 100)\n";
            std::cout << "  -out <output_csv_path> Output csv file path (optional)\n";
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
    if (!k_set || !video_set) {
        std::cerr << "Error: Both -k and -v arguments are required." << std::endl;
        std::cerr << "Usage: " << argv[0] << " -k <clusters> -v <video_path> [-e <stab_error>] [-mi <max_iterations>] [-out <output_csv_path>]" << std::endl;
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

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        return -1;
    }

    cv::Mat frame;

    // Get number of frames from video
    int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    
    // Array with (almost) all kmeans functions
    void (*kmeans_functions[])(KMeansParams*) = {
        k_means_acc_old,
        k_means_acc,
        k_means_pp_acc,
        k_means_pixel_centroid,
        k_means_acc_tiled,
        k_means_pp_acc_tiled
    };
    
    int number_of_functions = 7; // Including k_means_acc_check_conv

    // Matrix for timing results
    std::vector<std::vector<double>> data(MEASURED_FRAMES, std::vector<double>(number_of_functions));
    
    for (int func_idx = 0; func_idx < number_of_functions; func_idx++) {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset to first frame
        double total_duration = 0.0;

        for (int frame_count = 0; ; frame_count++) {
            cap >> frame;
            if (frame.empty()) {
                break;
            }

            // Params initialization
            if (frame_count == 0) {
                params.img_height = frame.rows;
                params.img_width = frame.cols;
                params.dimensions = frame.channels(); // Number of channels (e.g., 3 for RGB)
            }

            cv::Mat clustered_img(params.img_height, params.img_width, frame.type());
            
            params.img = frame.data;
            params.dst = clustered_img.data;

            double start_time, end_time;
            
            if (func_idx < number_of_functions - 1) {
                start_time = static_cast<double>(cv::getTickCount());
                kmeans_functions[func_idx](&params);
                end_time = static_cast<double>(cv::getTickCount());
            }
            else {
                start_time = static_cast<double>(cv::getTickCount());
                k_means_acc_check_conv(&params, 3); // Example, convergence step of 3
                end_time = static_cast<double>(cv::getTickCount());
            }

            if (frame_count >= WARM_UP_FRAMES) {
                double duration = (end_time - start_time) / cv::getTickFrequency();
                total_duration += duration;

                data[frame_count - WARM_UP_FRAMES][func_idx] = duration;
            }

            printf("Processed frame %d of %d (%.2f%%)\r", frame_count + 1, num_frames, static_cast<float>(frame_count) / num_frames * 100);
            std::cout.flush();

            if (frame_count >= WARM_UP_FRAMES + MEASURED_FRAMES - 1 || cv::waitKey(30) >= 0) {
                break;
            }
        }
        
        std::cout << "K-means clustering mean time: " << total_duration / MEASURED_FRAMES << " seconds" << std::endl;
    }


    cap.release();
    cv::destroyAllWindows();

    // Create CSV file
    if (!out_set) {
        output_csv_path = "results/video_" + std::to_string(std::min(params.img_width, params.img_height)) + ".csv";
    }
    std::ofstream file(output_csv_path);

    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    // Write header row
    file << "k-means old, k-means, k-means++, k-means pixel centroids, k-means tiled, k-means++ tiled, k-means skip convergence check\n";

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