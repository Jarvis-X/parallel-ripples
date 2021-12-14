# parallel-ripples
A project for the practice and principle in parallel computing. We aim at generating ripples in a 2D liquid plane and rendering them in real-time under 2K+ resolution. 
The basic pattern is a sparse matrix multiplication with the frame buffer, in a fancy way.
## Technical details:
### Requirements
Obviously, this project was developped in VS 2019. Thus, if intending to work with the .sln solution, one needs to configure OpenCV and OpenCL. 
More specifically, tell the MSVS comipler where to find the header files and the libraries. 
* For OpenCV, I followed https://towardsdatascience.com/install-and-configure-opencv-4-2-0-in-windows-10-vc-d132c52063a1
* For OpenCL, I followed https://medium.com/@pratikone/opencl-on-visual-studio-configuration-tutorial-for-the-confused-3ec1c2b5f0ca
* A Device (preferrably GPU) that supports OpenCL 1.2 or up
* Or, if the developer prefers, one can compile the project with any compiler that supports C++11 or higher using the header file and the cpp files from this repository
on a machine where OpenCV and OpenCL are installed.
### The ripple algorithm
* We improved the algorithm provided by https://github.com/CodingTrain/website/blob/main/CodingChallenges/CC_102_WaterRipples/Processing/CC_102_WaterRipples/CC_102_WaterRipples.pde such that:
  * At every pixel, the water level is collectively obtained from more surrounding pixels, making the ripples more accurate and rounder
  * The program runs faster by using OpenCL parallelism
