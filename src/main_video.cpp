#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster.h"
    #include "cluster_acc.h"
}

int main() {
    // TODO Fix with relative path independent from working directory
    cv::VideoCapture cap("../dataset/walking.mp4");
    if (!cap.isOpened()) {
        return -1;
    }

    cv::Mat frame;

    double duration = 0.0;

    // Get number of frames from video
    int frame_count = 0;
    int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

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
    
        double start_time = static_cast<double>(cv::getTickCount());
        k_means_acc(clustered_img.data, frame.data, img_height, img_width, k, dimensions, stab_error, max_iterations);
        double end_time = static_cast<double>(cv::getTickCount());

        duration += (end_time - start_time) / cv::getTickFrequency();
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
        
        if (cv::waitKey(30) >= 0) {
            break;
        }
    }
    
    std::cout << "K-means clustering mean time: " << duration / num_frames << " seconds" << std::endl;

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    return 0;
}