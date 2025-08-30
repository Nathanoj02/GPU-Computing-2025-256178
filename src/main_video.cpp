#include <opencv2/opencv.hpp>

extern "C" {
    #include "cluster.h"
    #include "cluster_acc.h"
    #include "distances.h"
    #include "kmeans.h"
}

int main(int argc, char** argv) {
    // Get k and video name from command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <k> <video_path>" << std::endl;
        return -1;
    }

    srand(0);   // Seed for reproducibility

    KMeansParams params;

    params.k = std::stoi(argv[1]);
    std::string video_path = argv[2];

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        return -1;
    }

    params.stab_error = 1.0f; // Convergence threshold
    params.max_iterations = 100;

    cv::Mat frame;

    double duration = 0.0;

    // Get number of frames from video
    int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

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
        }

        cv::Mat clustered_img(params.img_height, params.img_width, frame.type());
        
        params.img = frame.data;
        params.dst = clustered_img.data;
        
        double start_time = static_cast<double>(cv::getTickCount());
        k_means_acc(&params);
        double end_time = static_cast<double>(cv::getTickCount());

        duration += (end_time - start_time) / cv::getTickFrequency();

        // Save result into video file
        if (!writer.isOpened()) {
            writer.open("../dataset/output_video.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 30, clustered_img.size(), true);
            if (!writer.isOpened()) {
                std::cerr << "Could not open the output video for write." << std::endl;
                return -1;
            }
        }
        writer.write(clustered_img);

        printf("Processed frame %d of %d (%.2f%%)\r", frame_count + 1, num_frames, static_cast<float>(frame_count) / num_frames * 100);
        std::cout.flush();
        
        if (frame_count >= 30 || cv::waitKey(30) >= 0) {
            break;
        }
    }
    
    std::cout << "K-means clustering mean time: " << duration / 30 << " seconds" << std::endl;

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    return 0;
}