#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <random>
#define ROW 1080
#define COL 1920

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // three buffers: current buffer, two time-step before, one time-step before
    double* image_buffer = new double[COL * ROW]();
    double* image_buffer1 = new double[COL * ROW]();
    // unsigned char* image_buffer2 = new unsigned char[COL * ROW];
    /*for (auto i = 0; i < COL * ROW; i++) {
        image_buffer1[i] = 0;
        image_buffer[i] = 0;
    }*/

    uniform_int_distribution<int> pos_uni(0, COL * ROW);
    uniform_real_distribution<double> amp_uni(0, 100.0);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine random_pos_eng(seed);
    seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine random_amp_eng(seed);

    namedWindow("water surface", WINDOW_FREERATIO); // Create a window for display.

    while (true) {
        for (auto i = 0; i < 1; i++) {
            int index = pos_uni(random_pos_eng);
            double amp = amp_uni(random_amp_eng);
            image_buffer1[index] = image_buffer1[index] + amp;
        }

        Mat image = Mat(ROW, COL, CV_64F, image_buffer);
        imshow("water surface", image); // Show our image inside it.
        waitKey(1);
        for (auto col = 0; col < COL; col++) {
            for (auto row = 0; row < ROW; row++) {
                int index = row * COL + col;
                int north = (row - 1) * COL + col;
                int south = (row + 1) * COL + col;
                int west = row * COL + col - 1;
                int east = row * COL + col + 1;

                int northeast = (row - 1) * COL + col;
                int southeast = (row + 1) * COL + col;
                int northwest = row * COL + col - 1;
                int southwest = row * COL + col + 1;
                
                if (col == 0) {
                    // continue;
                    // top left
                    if (row == 0) {
                        image_buffer[index] = (image_buffer1[south] + image_buffer1[east])/2 - image_buffer[index];
                    }
                    // bottom left
                    else if (row == ROW - 1) {
                        image_buffer[index] = (image_buffer1[north] + image_buffer1[east])/2 - image_buffer[index];
                    }
                    // on the left wall
                    else {
                        image_buffer[index] = (image_buffer1[north] + image_buffer1[east] + image_buffer1[south])/2
                            - image_buffer[index];
                    }
                }
                else if (col == COL - 1) {
                    // continue;
                    // top right
                    if (row == 0) {
                        image_buffer[index] = (image_buffer1[south] + image_buffer1[west])/2 - image_buffer[index];
                    }
                    // bottom right
                    else if (row == ROW - 1) {
                        image_buffer[index] = (image_buffer1[north] + image_buffer1[west])/2 - image_buffer[index];
                    }
                    // on the right wall
                    else {
                        image_buffer[index] = (image_buffer1[north] + image_buffer1[west] + image_buffer1[south])/2
                            - image_buffer[index];
                    }
                }
                // on the ceiling
                else if (row == 0) {
                    // continue;
                    image_buffer[index] = (image_buffer1[south] + image_buffer1[west] + image_buffer1[east])/2
                        - image_buffer[index];
                }
                // on the floor
                else if (row == ROW - 1) {
                    // continue;
                    image_buffer[index] = (image_buffer1[north] + image_buffer1[west] + image_buffer1[east])/2
                        - image_buffer[index];
                }
                // in the belly
                else {
                    image_buffer[index] = (
                        image_buffer1[north] + 
                        image_buffer1[south] + 
                        image_buffer1[east] + 
                        image_buffer1[west]) / 2.0 - 
                        image_buffer[index];
                }
               
                // damping for 1/16
                image_buffer[index] *= 0.9;
            }
        }

        double* temp = image_buffer1;
        image_buffer1 = image_buffer;
        image_buffer = temp;
    }

    return 0;
}