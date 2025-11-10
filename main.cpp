#include <cassert>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <future>

#include "common/utils.h"
#include "common/bitmap.h"
#include "sequential/sequential.h"
namespace fs = std::filesystem;

// Enable extra debug print
// Uncomment to enable, comment to disable.
#define DEBUG

// Enable saving of debug frames. Useful for checking algorithm correctness, really not useful otherwise
// Uncomment to enable, comment to disable.
// #define SAVE_PROCESS_FRAMES


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

    // create kernels. These are not dependent on any other parameters and are read-only
    assert(blur_kernel_size % 2 == 1 && "Blur kernel size must be odd");
    assert(blur_sigma > 0 && "Blur sigma must be positive");
    FloatMap gaussianKernel = make_gaussian_kernel(blur_kernel_size, blur_sigma);
    FloatMap sobelKernelVert = FloatMap(3, 3);
    FloatMap sobelKernelHoriz = FloatMap(3, 3);

    // vertical
    // 1  2  1
    // 0  0  0
    //-1 -2 -1
    sobelKernelVert.data[0][0] = 1;
    sobelKernelVert.data[0][1] = 2;
    sobelKernelVert.data[0][2] = 1;
    sobelKernelVert.data[1][0] = 0;
    sobelKernelVert.data[1][1] = 0;
    sobelKernelVert.data[1][2] = 0;
    sobelKernelVert.data[2][0] = -1;
    sobelKernelVert.data[2][1] = -2;
    sobelKernelVert.data[2][2] = -1;

    // horizontal
    // 1 0 -1
    // 2 0 -2
    // 1 0 -1
    sobelKernelHoriz.data[0][0] = 1;
    sobelKernelHoriz.data[0][1] = 0;
    sobelKernelHoriz.data[0][2] = -1;
    sobelKernelHoriz.data[1][0] = 2;
    sobelKernelHoriz.data[1][1] = 0;
    sobelKernelHoriz.data[1][2] = -2;
    sobelKernelHoriz.data[2][0] = 1;
    sobelKernelHoriz.data[2][1] = 0;
    sobelKernelHoriz.data[2][2] = -1;


    std::cout << "\n\n-----------------------" << std::endl;
    std::cout << "Loading images" << std::endl;
    std::cout << "-----------------------\n\n" << std::endl;

    // ----- Begin loading images section -----
    // Loading/decoding images is not the focus of the project, so the parallelism in this section is just to make loading large
    // image datasets easier. Do not grade this section.

    // WARNING: All images are loaded as bitmaps into RAM and will stay until the program terminates
    // TODO: figure out how to allocate and deallocate memory for progress frames...

    std::vector<std::future<BitmapResult> > imageLoadingJobs;

    // Collect all the images before from input folder. Hidden files (files starting with ".") are ignored
    for (const auto &entry: fs::directory_iterator(INDIR)) {
        std::string filename = entry.path().filename().string();
        if (filename[0] != '.') {
            std::string fullPath = entry.path().string();
#ifdef DEBUG
            std::cout << "Found: " << filename << " (" << fullPath << ")" << std::endl;
#endif

            imageLoadingJobs.push_back(std::async(std::launch::async, [=]() {
                // Load the input image and convert to grayscale FloatMap
                FloatMap image = load_image_grayscale(fullPath);
                return BitmapResult(filename, 0, image);
            }));
        }
    }

    std::vector<BitmapResult> files;
    files.reserve(imageLoadingJobs.size());
    for (auto &fut: imageLoadingJobs)
        files.emplace_back(fut.get());
    // ----- End loading images section-----


    std::cout << "\n\n-----------------------" << std::endl;
    std::cout << "Processing images" << std::endl;
    std::cout << "Using blur_kernel_size = " << blur_kernel_size << ", blur_sigma = " << blur_sigma << std::endl;
    std::cout << "-----------------------\n\n" << std::endl;


    if (variant == 1) {
        runSequential(files, gaussianKernel, sobelKernelVert, sobelKernelHoriz);
    } else {
        std::cout << "not implemented" << std::endl;
    }

    std::cout << "\n\n-----------------------" << std::endl;
    std::cout << "Saving images" << std::endl;
    std::cout << "-----------------------\n\n" << std::endl;

    // save all bitmaps, then debug frames if debug is enabled
    std::vector<std::future<void> > imageSavingJobs;
    imageSavingJobs.reserve(files.size());
    for (const auto &file: files) {
        imageSavingJobs.push_back(std::async(std::launch::async, [=]() {
            std::cout << "Saving: " << OUTDIR + "/" + file.filename << "\n\tTime taken: " << file.totalRuntime << " ms"
                    << std::endl;
            save_floatmap_as(file.image, OUTDIR + "/" + file.filename);

#ifdef DEBUG
            std::string debugTimePrint = "\tDEBUG: time breakdown: ";
            // save any debug frames
            for (const auto &debugFrame: file.debugFrames) {
#ifdef SAVE_PROCESS_FRAMES
                std::cout << "DEBUG: Saving frame: " << OUTDIR + "/" + debugFrame.filename << std::endl;
                save_floatmap_as(debugFrame.image, OUTDIR + "/" + debugFrame.filename);
#endif
                debugTimePrint.append("/" + std::to_string(debugFrame.totalRuntime));
            }
            std::cout << debugTimePrint << std::endl; // TODO: Unlabeled. label if we keep this...
#endif
        }));
    }

    return 0;
}
