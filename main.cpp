#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <random>
#include "ripple.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // three buffers: current buffer, two time-step before, one time-step before
    float* image_buffer = new float[COL * ROW * DEPTH]();
    float* image_buffer1 = new float[COL * ROW * DEPTH]();
    
    uniform_int_distribution<int> pos_uni(0, COL * ROW);
    uniform_real_distribution<float> amp_uni(0, 10.0);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine random_pos_eng(seed);
    seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine random_amp_eng(seed);

    namedWindow("water surface", WINDOW_FREERATIO); // Create a window for display.

    while (true) {
        // BGR color scheme 
        for (auto i = 0; i < 100; i++) {
            int index = pos_uni(random_pos_eng);
            for (auto channel = 0; channel < DEPTH; channel++) {
                float amp = amp_uni(random_amp_eng);
                image_buffer1[index * DEPTH + channel] = image_buffer1[index * DEPTH + channel] + amp;
            }
        }

        Mat image = Mat(ROW, COL, CV_32FC3, image_buffer);
        imshow("water surface", image); // Show our image inside it.
        waitKey(1);
        
        // serial version
        update_buffer(image_buffer, image_buffer1);

        // OpenCL-parallelized version
    }

    return 0;
}