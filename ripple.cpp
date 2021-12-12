#include "ripple.h"
using namespace std;

cl_device_id create_device() {

    cl_platform_id platform;
    cl_device_id dev;
    int err;

    /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0) {
        cerr << "Couldn't identify a platform" << endl;
        exit(1);
    }

    /* Access a device */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        cerr << "Couldn't access GPU devices" << endl;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if (err < 0) {
        cerr << "Couldn't access any devices" << endl;
        exit(1);
    }

    return dev;
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

    cl_program program;
    FILE* program_handle;
    char* program_buffer, * program_log;
    size_t program_size, log_size;
    int err;

    /* Read program file and place content into buffer */
    fopen_s(&program_handle, filename, "rb");
    if (program_handle == NULL) {
        cerr << "Couldn't find the program file" << endl;
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    /* Create program from file */
    program = clCreateProgramWithSource(ctx, 1,
        (const char**)&program_buffer, &program_size, &err);
    if (err < 0) {
        cerr << "Couldn't create the program" << endl;
        exit(1);
    }
    free(program_buffer);

    /* Build program */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
        program_log = (char*)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
        cerr << program_log << endl;
        free(program_log);
        exit(1);
    }
    return program;
}

void output_msg_and_shut(int err, const char* msg) {
    if (err < 0) {
        cerr << msg << ": " << err << endl;
        exit(1);
    };
}

void update_buffer(float* &image_buffer, float* &image_buffer1) {
    for (auto col = 0; col < COL; col++) {
        for (auto row = 0; row < ROW; row++) {
            for (auto channel = 0; channel < DEPTH; channel++) {
                int index = (row * COL + col)*DEPTH + channel;
                int north = ((row - 1) * COL + col)* DEPTH + channel;
                int south = ((row + 1) * COL + col)* DEPTH + channel;
                int west = (row * COL + col - 1)* DEPTH + channel;
                int east = (row * COL + col + 1) * DEPTH + channel;

                int northeast = ((row - 1) * COL + col + 1) * DEPTH + channel;
                int southeast = ((row + 1) * COL + col + 1) * DEPTH + channel;
                int northwest = ((row - 1) * COL + col - 1) * DEPTH + channel;
                int southwest = ((row + 1) * COL + col - 1) * DEPTH + channel;

                if (col == 0) {
                    // continue;
                    // top left
                    if (row == 0) {
                        image_buffer[index] = (image_buffer1[south] + image_buffer1[east]) / 2 - image_buffer[index];
                    }
                    // bottom left
                    else if (row == ROW - 1) {
                        image_buffer[index] = (image_buffer1[north] + image_buffer1[east]) / 2 - image_buffer[index];
                    }
                    // on the left wall
                    else {
                        image_buffer[index] = (image_buffer1[north] + image_buffer1[east] + image_buffer1[south]) / 2
                            - image_buffer[index];
                    }
                }
                else if (col == COL - 1) {
                    // continue;
                    // top right
                    if (row == 0) {
                        image_buffer[index] = (image_buffer1[south] + image_buffer1[west]) / 2 - image_buffer[index];
                    }
                    // bottom right
                    else if (row == ROW - 1) {
                        image_buffer[index] = (image_buffer1[north] + image_buffer1[west]) / 2 - image_buffer[index];
                    }
                    // on the right wall
                    else {
                        image_buffer[index] = (image_buffer1[north] + image_buffer1[west] + image_buffer1[south]) / 2
                            - image_buffer[index];
                    }
                }
                // on the ceiling
                else if (row == 0) {
                    // continue;
                    image_buffer[index] = (image_buffer1[south] + image_buffer1[west] + image_buffer1[east]) / 2
                        - image_buffer[index];
                }
                // on the floor
                else if (row == ROW - 1) {
                    // continue;
                    image_buffer[index] = (image_buffer1[north] + image_buffer1[west] + image_buffer1[east]) / 2
                        - image_buffer[index];
                }
                // in the belly
                else {
                    image_buffer[index] = 0.28 * (
                        image_buffer1[north] +
                        image_buffer1[south] +
                        image_buffer1[east] +
                        image_buffer1[west]) + 0.12 * (
                            image_buffer1[northeast] +
                            image_buffer1[southwest] +
                            image_buffer1[southeast] +
                            image_buffer1[northwest]) -
                        (image_buffer[index] - INITIAL);
                }

                // damping for 1/16
                image_buffer[index] *= 0.9;
                image_buffer[index] += INITIAL;
            }
        }
    }

    float* temp = image_buffer1;
    image_buffer1 = image_buffer;
    image_buffer = temp;
}

void generate_raindrops(float*& image_buffer1, 
    uniform_int_distribution<int> &pos_uni,
    uniform_real_distribution<float> &amp_uni,
    default_random_engine &random_pos_eng,
    default_random_engine &random_amp_eng) {
    for (auto i = 0; i < N; i++) {
        int index = pos_uni(random_pos_eng);
        for (auto channel = 0; channel < DEPTH; channel++) {
            float amp = amp_uni(random_amp_eng);
            image_buffer1[index * DEPTH + channel] = image_buffer1[index * DEPTH + channel] + amp;
        }
    }
}

void update_buffer_cl(float*& image_buffer, float*& image_buffer1,
    size_t &global_size, size_t &local_size,
    cl_context context, cl_command_queue queue, cl_kernel update_buffer_kernel, cl_int err) {

    cl_mem cl_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        ROW * COL * DEPTH * sizeof(float), image_buffer, &err);
    output_msg_and_shut(err, "Couldn't create image buffer");

    cl_mem cl_buffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        ROW * COL * DEPTH * sizeof(float), image_buffer1, &err);
    output_msg_and_shut(err, "Couldn't create image buffer1");

    // err = clEnqueueWriteBuffer(queue, cl_buffer, CL_TRUE, 0,
    //	ROW * COL * DEPTH * sizeof(float), image_buffer, 0, NULL, NULL);
    // err |= clEnqueueWriteBuffer(queue, cl_buffer1, CL_TRUE, 0,
    // 	ROW * COL * DEPTH * sizeof(float), image_buffer1, 0, NULL, NULL);

    /* Create arguments for associate_centers kernel */
    err = clSetKernelArg(update_buffer_kernel, 0, sizeof(cl_mem), &cl_buffer);
    err |= clSetKernelArg(update_buffer_kernel, 1, sizeof(cl_mem), &cl_buffer1);
    // err |= clSetKernelArg(update_buffer_kernel, 2, sizeof(int), NULL);
    output_msg_and_shut(err, "Couldn't set an argument for the [update_buffer_kernel] kernel");

    err = clEnqueueNDRangeKernel(queue, update_buffer_kernel, 1, NULL, &global_size, &local_size,
        0, NULL, NULL);
    output_msg_and_shut(err, "Couldn't enqueue the [update_buffer_kernel] kernel");

    // Wait for the command queue
    clFinish(queue);

    // read from device buffer
    clEnqueueReadBuffer(queue, cl_buffer, CL_TRUE, 0, ROW*COL*DEPTH * sizeof(float), image_buffer, 0, NULL, NULL);

    clReleaseMemObject(cl_buffer);
    clReleaseMemObject(cl_buffer1);
    
    float* temp = image_buffer1;
    image_buffer1 = image_buffer;
    image_buffer = temp;
}

void CallBackFunc(int event, int x, int y, int flags, void* data) {
    float* image = (float*) data;
    if (event == 1) {
        int index = y * COL + x;
        for (auto channel = 0; channel < DEPTH; channel++) {
            image[index * DEPTH + channel] = image[index * DEPTH + channel] + 10000.0;
        }
    }
}