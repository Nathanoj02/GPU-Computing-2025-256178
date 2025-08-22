#include <opencv2/opencv.hpp>

int main() {
    // TODO Fix with relative path independent from working directory
    cv::VideoCapture cap("../dataset/walking.mp4");
    if (!cap.isOpened()) {
        return -1;
    }

    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        cv::imshow("Video Frame", frame);

        if (cv::waitKey(30) >= 0) {
            break;
        }
    }

    return 0;
}