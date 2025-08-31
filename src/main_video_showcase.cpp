#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster.h"
    #include "cluster_acc.h"
    #include "distances.h"
    #include "metrics.h"
    #include "kmeans.h"
}

std::string generateOutputPath(const std::string& input_video_path);

int main(int argc, char** argv) {
    srand(0);   // Seed for reproducibility

    KMeansParams params;
    std::string video_path;
    
    // Default values
    params.k = 0;  // Will be set as required
    params.stab_error = 1.0f;   // Convergence threshold
    params.max_iterations = 100;
    int calibration_frames = 10;    // Number of frames to use for calibration
    
    // Parse command line arguments
    bool k_set = false, video_set = false;
    
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
        else if (arg == "-cf" && i + 1 < argc) {
            calibration_frames = std::stoi(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " -k <clusters> -v <video_path> [-e <stab_error>] [-mi <max_iterations>] [-cf <calibration_frames>]\n";
            std::cout << "  -k <clusters>      Number of clusters (required)\n";
            std::cout << "  -v <video_path>    Path to input video (required)\n";
            std::cout << "  -e <stab_error>    Stability error threshold (default: 1.0)\n";
            std::cout << "  -mi <max_iterations> Maximum iterations (default: 100)\n";
            std::cout << "  -cf <calibration_frames> Number of frames for calibration (default: 10)\n";
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
        std::cerr << "Usage: " << argv[0] << " -k <clusters> -v <video_path> [-e <stab_error>] [-mi <max_iterations>]" << std::endl;
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

    if (calibration_frames <= 0) {
        std::cerr << "Error: Calibration frames must be positive." << std::endl;
        return -1;
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        return -1;
    }

    cv::Mat frame;

    // Get number of frames from video
    int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    // Declarations for calibration
    double best_error = FLT_MAX;
    uint8_t* prototypes;
    uint8_t* best_prototypes;

    // Initialize video writer
    cv::VideoWriter writer;

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
            
            // Alloc memory for prototypes
            prototypes = (uint8_t*) malloc (sizeof(uint8_t) * params.k * params.dimensions);
            best_prototypes = (uint8_t*) malloc (sizeof(uint8_t) * params.k * params.dimensions);
        }

        cv::Mat clustered_img(params.img_height, params.img_width, frame.type());

        params.img = frame.data;
        params.dst = clustered_img.data;

        // During calibration
        if (frame_count < calibration_frames) {
            k_means_custom_centroids(&params, prototypes, true);
        
            // Perform sum of squares metric
            double error = elbow_method(frame.data, clustered_img.data, params.img_height, params.img_width, params.k, params.dimensions, euclidean_distance, 0);

            if (error < best_error) {
                best_error = error;
                memcpy(best_prototypes, prototypes, sizeof(uint8_t) * params.k * params.dimensions);
            }
        }
        // After calibration
        else {
            k_means_custom_centroids(&params, best_prototypes, false);
        }

        // Save result into video file
        if (!writer.isOpened()) {
            std::string output_video_path = generateOutputPath(video_path);
            writer.open(output_video_path, cv::VideoWriter::fourcc('m','p','4','v'), 30, clustered_img.size(), true);
            if (!writer.isOpened()) {
                std::cerr << "Could not open the output video for write." << std::endl;
                return -1;
            }
        }
        writer.write(clustered_img);

        printf("Processed frame %d of %d (%.2f%%)\r", frame_count + 1, num_frames, static_cast<float>(frame_count) / num_frames * 100);
        std::cout.flush();
    }

    // Free memory
    free(prototypes);
    free(best_prototypes);

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    return 0;
}

std::string generateOutputPath(const std::string& input_video_path) {
    size_t last_slash = input_video_path.find_last_of("/\\");
    size_t last_dot = input_video_path.find_last_of(".");
    
    std::string directory;
    std::string filename_without_ext;
    std::string extension = ".mp4";
    
    if (last_slash != std::string::npos) {
        directory = input_video_path.substr(0, last_slash + 1);
        if (last_dot != std::string::npos && last_dot > last_slash) {
            filename_without_ext = input_video_path.substr(last_slash + 1, last_dot - last_slash - 1);
        } else {
            filename_without_ext = input_video_path.substr(last_slash + 1);
        }
    } else {
        directory = "./";
        if (last_dot != std::string::npos) {
            filename_without_ext = input_video_path.substr(0, last_dot);
        } else {
            filename_without_ext = input_video_path;
        }
    }
    
    return directory + "output_" + filename_without_ext + "_clustered" + extension;
}
