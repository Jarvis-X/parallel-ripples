#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include "ripple.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // two buffers: current buffer (before going through the current loop, it stores two time-step before), and one time-step before
    float* image_buffer = new float[COL * ROW * DEPTH]();
    float* image_buffer1 = new float[COL * ROW * DEPTH]();
    
    // rng for new raindrops
    uniform_int_distribution<int> pos_uni(0, COL * ROW);
    uniform_real_distribution<float> amp_uni(0, 100.0);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine random_pos_eng(seed);
    seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine random_amp_eng(seed);

    // OpenCV GUI handle
    namedWindow("water surface", WINDOW_FREERATIO);

    // ----------------------------------------------------
    // OpenCL-related initializations
    // Host/device data structures
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    size_t local_size, global_size;
    cl_int i, err;
    cl_uint num_groups;

    // Create a device and context
    device = create_device();
    size_t max_local_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof(max_local_size), &max_local_size, NULL);
    local_size = min(max(COL*ROW*DEPTH / NUM_WORK_GROUP, 1), (int)max_local_size);
    cout << local_size << endl;
    global_size = ceil(ROW * COL * DEPTH * 1.0 / local_size) * local_size;
    cout << global_size << endl;

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    output_msg_and_shut(err, "Couldn't create a context");

    // Create a command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    output_msg_and_shut(err, "Couldn't create a command queue");

    program = build_program(context, device, "kernel.cl");
    cout << "kernel loading done" << endl;
    // now OpenCL kernels are loaded (but not compiled yet)

    // compile update buffer kernel
    cl_kernel update_buffer_kernel = clCreateKernel(program, "update_buffer", &err);
    output_msg_and_shut(err, "Couldn't create a kernel");
    // ----------------------------------------------------

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
        /*
        begin = chrono::high_resolution_clock::now();
        update_buffer_cl(image_buffer, image_buffer1, global_size, local_size, context, queue, update_buffer_kernel, err);
        end = chrono::high_resolution_clock::now();
        cout << "CALCULATION TIME = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "\n";
        */
    }

    return 0;
}