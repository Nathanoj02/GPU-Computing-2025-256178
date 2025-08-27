#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster.h"
    #include "cluster_acc.h"
    #include "distances.h"
    #include "metrics.h"
}

int main(int argc, char** argv) {
    // Get k and video name from command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <k> <video_path>" << std::endl;
        return -1;
    }

    unsigned int k = std::stoi(argv[1]);
    std::string video_path = argv[2];

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        return -1;
    }

    cv::Mat frame;

    double duration = 0.0;

    // Get number of frames from video
    int frame_count = 0;
    int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    // Declarations for calibration
    double best_error = FLT_MAX;
    uint8_t* prototypes;
    uint8_t* best_prototypes;

    // Initialize video writer
    cv::VideoWriter writer;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        size_t img_height = frame.rows;
        size_t img_width = frame.cols;
        unsigned int dimensions = frame.channels(); // Number of channels (e.g., 3 for RGB)

        unsigned int k = 3;
        float stab_error = 1.0f; // Convergence threshold
        int max_iterations = 100;
        cv::Mat clustered_img(img_height, img_width, frame.type());

        int calibration_frames = 10;    // Number of frames to use for calibration

        // Alloc memory for prototypes
        if (frame_count == 0) {
            prototypes = (uint8_t*) malloc (sizeof(uint8_t) * k * dimensions);
            best_prototypes = (uint8_t*) malloc (sizeof(uint8_t) * k * dimensions);
        }

        // During calibration
        if (frame_count < calibration_frames) {
            k_means_custom_centroids(clustered_img.data, frame.data, img_height, img_width, k, dimensions, prototypes, true, stab_error, max_iterations, 1.0f);
        
            // Perform sum of squares metric
            double error = elbow_method(frame.data, clustered_img.data, img_height, img_width, k, dimensions, euclidean_distance, 0);
        
            if (error < best_error) {
                best_error = error;
                memcpy(best_prototypes, prototypes, sizeof(uint8_t) * k * dimensions);
            }
        }
        // After calibration
        else {
            double start_time = static_cast<double>(cv::getTickCount());
            k_means_custom_centroids(clustered_img.data, frame.data, img_height, img_width, k, dimensions, best_prototypes, false, stab_error, max_iterations, 1.0f);
            double end_time = static_cast<double>(cv::getTickCount());

            duration += (end_time - start_time) / cv::getTickFrequency();
        }
        
        frame_count++;

        // Save result into video file
        if (!writer.isOpened()) {
            writer.open("../dataset/output_video.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 30, clustered_img.size(), true);
            if (!writer.isOpened()) {
                std::cerr << "Could not open the output video for write." << std::endl;
                return -1;
            }
        }
        writer.write(clustered_img);

        printf("Processed frame %d of %d (%.2f%%)\r", frame_count, num_frames, static_cast<float>(frame_count) / num_frames * 100);
        std::cout.flush();
        
        if (frame_count >= 30 || cv::waitKey(30) >= 0) {
            break;
        }
    }

    // Free memory
    free(prototypes);
    free(best_prototypes);

    std::cout << "K-means clustering mean time: " << duration / 30 << " seconds" << std::endl;

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    return 0;
}