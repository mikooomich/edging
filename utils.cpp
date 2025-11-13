#include "utils.h"

#include <chrono>
#include <future>

#include <cassert>
#include <cmath>
#include <iostream>
#include <filesystem>

#include "bitmap.h"
namespace fs = std::filesystem;

// Enable extra debug print
// Uncomment to enable, comment to disable.
#define DEBUG

// Enable saving of debug frames. Useful for checking algorithm correctness, really not useful otherwise
// Uncomment to enable, comment to disable.
// #define SAVE_PROCESS_FRAMES

long getSysTime() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).
            count();
}

/**
 * Return The required data to setup the algorithms.
 *
 * kernelsOnly = true will only return the kernels
 * kernelsOnly = false will return the kernels PLUS load all the images from disk into memory
 */
DataSet prepareDataset(int blur_kernel_size, float blur_sigma, std::string indir, bool kernelsOnly) {
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

    if (kernelsOnly) {
        return DataSet(gaussianKernel, sobelKernelVert, sobelKernelHoriz);
    }

    // now load all images in the input folder
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
    for (const auto &entry: fs::directory_iterator(indir)) {
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

    return DataSet(files, gaussianKernel, sobelKernelVert, sobelKernelHoriz);
}


void saveResult(std::vector<BitmapResult> &files, std::string outdir) {
    std::cout << "\n\n-----------------------" << std::endl;
    std::cout << "Saving images" << std::endl;
    std::cout << "-----------------------\n\n" << std::endl;

    // save all bitmaps, then debug frames if debug is enabled
    std::vector<std::future<void> > imageSavingJobs;
    imageSavingJobs.reserve(files.size());
    for (const auto &file: files) {
        imageSavingJobs.push_back(std::async(std::launch::async, [=]() {
            std::cout << "Saving: " << outdir + "/" + file.filename << "\n\tTime taken: " << file.totalRuntime << " ms"
                    << std::endl;
            save_floatmap_as(file.image, outdir + "/" + file.filename);

#ifdef DEBUG
            std::string debugTimePrint = "\tDEBUG: time breakdown: ";
            // save any debug frames
            for (const auto &debugFrame: file.debugFrames) {
#ifdef SAVE_PROCESS_FRAMES
            std::cout << "DEBUG: Saving frame: " << outdir + "/" + debugFrame.filename << std::endl;
            save_floatmap_as(debugFrame.image, outdir + "/" + debugFrame.filename);
#endif
            debugTimePrint.append("/" + std::to_string(debugFrame.totalRuntime));
            }
            std::cout << debugTimePrint << std::endl; // TODO: Unlabeled. label if we keep this...
#endif
        }));
    }
}
