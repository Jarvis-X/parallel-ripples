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
    for (size_t i = 0; i < COL * ROW * DEPTH; i++) {
        image_buffer[i] = INITIAL;
        image_buffer[i] = INITIAL;
    }
    
    // rng for new raindrops
    uniform_int_distribution<int> pos_uni(0, COL * ROW);
    uniform_real_distribution<float> amp_uni(0, 35.0);
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
        setMouseCallback("water surface", CallBackFunc, image_buffer);
        end = chrono::high_resolution_clock::now();
        cout << "RENDER TIME = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "\n";
        waitKey(1);
        
        // serial version
        /*
        begin = chrono::high_resolution_clock::now();
        update_buffer(image_buffer, image_buffer1);
        end = chrono::high_resolution_clock::now();
        cout << "CALCULATION TIME = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "\n";
        */
        // OpenCL-parallelized version
        begin = chrono::high_resolution_clock::now();
        update_buffer_cl(image_buffer, image_buffer1, global_size, local_size, context, queue, update_buffer_kernel, err);
        end = chrono::high_resolution_clock::now();
        cout << "CALCULATION TIME = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "\n";
    }

    return 0;
    /*
    int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;

    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (i = 0; i < platformCount; i++) {

        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++) {

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("%d. Device: %s\n", j + 1, value);
            free(value);

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %d.%d Hardware version: %s\n", j + 1, 1, value);
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf(" %d.%d Software version: %s\n", j + 1, 2, value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf(" %d.%d OpenCL C version: %s\n", j + 1, 3, value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %d.%d Parallel compute units: %d\n", j + 1, 4, maxComputeUnits);

        }

        free(devices);

    }

    free(platforms);
    return 0;
    */
}