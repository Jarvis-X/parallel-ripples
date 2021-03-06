// TODO: consider using local memory to optimize, ref:
// https://developer.download.nvidia.com/CUDA/training/NVIDIA_GPU_Computing_Webinars_Best_Practises_For_OpenCL_Programming.pdf page 22
// we can avoid the scenario where work groups keep accessing global memory which is slow
#define ROW 2160
#define COL 3840
#define DEPTH 3
#define INITIAL 0.01

__kernel void update_buffer(__global float* image_buffer, __global float* image_buffer1) {
    // local float image_cache[3][COL / 20 * DEPTH];
    // local uchar cached[COL / 20 * DEPTH]; // the flag for whether a local pixel col is cached

    int index = get_global_id(0);
    // int local_index = get_local_id(0);
    // int group_index = get_group_id(0);
  
    if (index < ROW * COL * DEPTH) {
        int channel = index % DEPTH;
        int pixel_index = index / DEPTH;
        int col = pixel_index % COL;
        int row = pixel_index / COL;
        
        int north = ((row - 1) * COL + col) * DEPTH + channel;
        int south = ((row + 1) * COL + col) * DEPTH + channel;
        int west = (row * COL + col - 1) * DEPTH + channel;
        int east = (row * COL + col + 1) * DEPTH + channel;
        
        // printf("pixel (%i, %i) at channel %i has index %i\n", row, col, channel, index);
        // to make the ripples rounder
        int northeast = ((row - 1) * COL + col + 1) * DEPTH + channel;
        int southeast = ((row + 1) * COL + col + 1) * DEPTH + channel;
        int northwest = ((row - 1) * COL + col - 1) * DEPTH + channel;
        int southwest = ((row + 1) * COL + col - 1) * DEPTH + channel;
        /*
        if (cached[local_index] != 100) {
            // cache its own column
            if (row != 0) {
                image_cache[0][local_index] = image_buffer1[north];
            }
            image_cache[1][local_index] = image_buffer1[index];
            if (row != ROW - 1) {
                image_cache[2][local_index] = image_buffer1[south];
            }
            // flag finish
            cached[local_index] = 100;
        }
        
        if (local_index >= DEPTH) {
            // if not left-most, cache its left if not yet
            if (cached[local_index - DEPTH] != 100) {
                // cache the column to the left
                if (row != 0) {
                    image_cache[0][local_index - DEPTH] = image_buffer1[northwest];
                }
                image_cache[1][local_index - DEPTH] = image_buffer1[west];
                if (row != ROW - 1) {
                    image_cache[2][local_index - DEPTH] = image_buffer1[southwest];
                }
                // flag finish
                cached[local_index - DEPTH] = 100;
            }
        }

        if (local_index < (COL / 20 - 1) * DEPTH) {
            // if not right-most, cache its right if not yet
            if (cached[local_index + DEPTH] != 100) {
                // cache the column to the right
                if (row != 0) {
                    image_cache[0][local_index + DEPTH] = image_buffer1[northeast];
                }
                image_cache[1][local_index + DEPTH] = image_buffer1[east];
                if (row != ROW - 1) {
                    image_cache[2][local_index + DEPTH] = image_buffer1[southeast];
                }
                // flag finish
                cached[local_index + DEPTH] = 100;
            }
        }

        if (col == 0) {
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
            image_buffer[index] = (image_buffer1[south] + image_buffer1[west] + image_buffer1[east]) / 2
                - image_buffer[index];
        }
        // on the floor
        else if (row == ROW - 1) {
            image_buffer[index] = (image_buffer1[north] + image_buffer1[west] + image_buffer1[east]) / 2
                - image_buffer[index];
        }
        // in the belly
        else {
            if (local_index >= DEPTH && local_index < (COL / 20 - 1) * DEPTH) {
                image_buffer[index] = 0.28 * (
                    image_cache[0][local_index] +
                    image_cache[2][local_index] +
                    image_cache[1][local_index + DEPTH] +
                    image_cache[1][local_index - DEPTH]) + 0.13 * (
                        image_cache[0][local_index + DEPTH] +
                        image_cache[2][local_index - DEPTH] +
                        image_cache[2][local_index + DEPTH] +
                        image_cache[0][local_index - DEPTH]) -
                    (image_buffer[index] - INITIAL);
            } else {
                image_buffer[index] = 0.28 * (
                    image_buffer1[north] +
                    image_buffer1[south] +
                    image_buffer1[east] +
                    image_buffer1[west]) + 0.13 * (
                        image_buffer1[northeast] +
                        image_buffer1[southwest] +
                        image_buffer1[southeast] +
                        image_buffer1[northwest]) -
                    (image_buffer[index] - INITIAL);
            }
        }*/
        
        
        if (col == 0) {
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
            image_buffer[index] = (image_buffer1[south] + image_buffer1[west] + image_buffer1[east]) / 2
                - image_buffer[index];
        }
        // on the floor
        else if (row == ROW - 1) {
            image_buffer[index] = (image_buffer1[north] + image_buffer1[west] + image_buffer1[east]) / 2
                - image_buffer[index];
        }
        // in the belly
        else {
            image_buffer[index] = 0.28 * (
                image_buffer1[north] +
                image_buffer1[south] +
                image_buffer1[east] +
                image_buffer1[west]) + 0.13 * (
                    image_buffer1[northeast] +
                    image_buffer1[southwest] +
                    image_buffer1[southeast] +
                    image_buffer1[northwest]) -
                (image_buffer[index] - INITIAL);
        }
        
        
        // printf("ROW %i; COL %i; DEPTH %i, index %d\n", ROW, COL, DEPTH, (row * COL + col) * DEPTH + channel);
        // damping for 1/16
        image_buffer[index] *= 0.9;
        image_buffer[index] += INITIAL;
    }
}