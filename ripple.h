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

#define ROW 1080
#define COL 1920
#define DEPTH 3


constexpr auto MAX_SOURCE_SIZE = (0x10000000);
constexpr auto NUM_WORK_GROUP = (32);

cl_device_id create_device();

// Create program from a file and compile it
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);

void output_msg_and_shut(int err, const char* msg);

void update_buffer(float* &image_buffer, float* &image_buffer1);