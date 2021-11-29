#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include "ripple.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // three buffers: current buffer, two time-step before, one time-step before
    float* image_buffer = new float[COL * ROW * DEPTH]();
    float* image_buffer1 = new float[COL * ROW * DEPTH]();
    
    uniform_int_distribution<int> pos_uni(0, COL * ROW);
    uniform_real_distribution<float> amp_uni(0, 5.0);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine random_pos_eng(seed);
    seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine random_amp_eng(seed);

    namedWindow("water surface", WINDOW_FREERATIO); // Create a window for display.

    while (true) {
        // BGR color scheme 
        auto begin = chrono::high_resolution_clock::now();
        generate_raindrops(image_buffer1, pos_uni, amp_uni, random_pos_eng, random_amp_eng);
        auto end = chrono::high_resolution_clock::now();
        cout << "RAINDROP TIME = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "\n";

        begin = chrono::high_resolution_clock::now();
        Mat image = Mat(ROW, COL, CV_32FC3, image_buffer);
        imshow("water surface", image); // Show our image inside it.
        end = chrono::high_resolution_clock::now();
        cout << "RENDER TIME = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "\n";
        waitKey(1);
        
        // serial version
        begin = chrono::high_resolution_clock::now();
        update_buffer(image_buffer, image_buffer1);
        end = chrono::high_resolution_clock::now();
        cout << "CALCULATION TIME = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "\n";

        // OpenCL-parallelized version
    }

    return 0;
}