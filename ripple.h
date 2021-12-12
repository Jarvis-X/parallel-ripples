#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <vector>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <random>
#include <opencv2/highgui.hpp>
#define N 1000

constexpr auto ROW = 2160;
constexpr auto COL = 3840;
constexpr auto DEPTH = 3;
constexpr auto INITIAL = 0.01;
constexpr auto MAX_SOURCE_SIZE = (0x1000000);
constexpr auto NUM_WORK_GROUP = ROW*20;

cl_device_id create_device();

// Create program from a file and compile it
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);

void output_msg_and_shut(int err, const char* msg);

void update_buffer(float* &image_buffer, float* &image_buffer1);

void generate_raindrops(float*& image_buffer1,  
    std::uniform_int_distribution<int>& pos_uni, 
    std::uniform_real_distribution<float>& amp_uni,
    std::default_random_engine& random_pos_eng,
    std::default_random_engine& random_amp_eng);

void update_buffer_cl(float*& image_buffer, float*& image_buffer1, 
    size_t& global_size, size_t &local_size,
    cl_context context, cl_command_queue queue, cl_kernel update_buffer_kernel, cl_int err);

// interaction in opencv
void CallBackFunc(int event, int x, int y, int flags, void* data);