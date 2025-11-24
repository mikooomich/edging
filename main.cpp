#include <cassert>
#include <cmath>
#include <iostream>
#include <filesystem>

#include "utils.h"
#include "bitmap.h"
#include "sequential.h"
namespace fs = std::filesystem;



/**
 * Main program entry point
 * @param argc M
 * @param argv
 * @return
 */
int main(int argc, char *argv[]) {
    std::string INDIR = "./data/input";
    std::string OUTDIR = "./data/output";

    // args from program args
    int blur_kernel_size = -1;
    float blur_sigma = -1.0f;

    /**
     *1 = sequential
     *2 = openmp/cude
     *3 = openmpi
     */
    int variant = 1;

    if (argc > 1) {
        variant = std::atoi(argv[1]);
    }
    if (argc > 2) {
        blur_kernel_size = std::atof(argv[2]);
    }
    if (argc > 3) {
        blur_sigma = std::atof(argv[3]);
    }


    if (blur_kernel_size == -1) {
        std::cout << "WARNING: No blur_kernel_size specified. Using default value 11" << std::endl;
        blur_kernel_size = 11;
        std::cout << "Usage: whyareyourunning <blur kernel size> <blur sigma size>" << std::endl;
        std::cout << "Example: whyareyourunning 11 0.2" << std::endl;
    }


    if (blur_sigma == -1.0f) {
        std::cout << "WARNING: No blur_sigma specified. Using default value 0.2" << std::endl;
        blur_sigma = 0.2f;
        std::cout << "Usage: whyareyourunning <blur kernel size> <blur sigma size>" << std::endl;
        std::cout << "Example: whyareyourunning 11 0.2" << std::endl;
    }

    // load kernels, images to process (this step will not be analyzed)
    DataSet s = prepareDataset(blur_kernel_size, blur_sigma, INDIR, false);

    FloatMap gaussianKernel = s.gaussianKernel;
    FloatMap sobelKernelVert = s.sobelKernelVert;
    FloatMap sobelKernelHoriz = s.sobelKernelVert;
    std::vector<BitmapResult> files = s.files;

    std::cout << "\n\n-----------------------" << std::endl;
    std::cout << "Processing images" << std::endl;
    std::cout << "Using blur_kernel_size = " << blur_kernel_size << ", blur_sigma = " << blur_sigma << std::endl;
    std::cout << "-----------------------\n\n" << std::endl;

    // run various versions of the program. 1 = sequential, 2 = openmp. Openmp is in its own file
    long startTime = getSysTime();
    if (variant == 1) {
        runSequential(files, gaussianKernel, sobelKernelVert, sobelKernelHoriz);
    } else {
        std::cout << "not implemented" << std::endl;
    }

    // save results (this step will not be analyzed)
    saveResult(files, OUTDIR, startTime, "variant, blur kernel size, blur sigma\n"+std::to_string(variant) + ","+ std::to_string(blur_kernel_size) + "," + std::to_string(blur_sigma) +"\n");

    return 0;
}
