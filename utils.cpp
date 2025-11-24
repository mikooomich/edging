#include "utils.h"

#include <chrono>
#include <future>

#include <cassert>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>

#include "bitmap.h"
namespace fs = std::filesystem;

// Enable extra debug print
// Uncomment to enable, comment to disable.
// #define EDGING_DEBUG

// Enable saving of debug frames. Useful for checking algorithm correctness, really not useful otherwise
// Uncomment to enable, comment to disable.
// #define SAVE_PROCESS_FRAMES

/**
 * So apparently windows may use January 1, 1601 as the start of epoch time and not 1970. Bro wtf microsoft
 */
long getSysTime() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
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
#ifdef EDGING_DEBUG
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


void saveResult(std::vector<BitmapResult> &files, const std::string &outdir, long startTime,
                const std::string &infoText) {
    long endTime = getSysTime();
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
        }));
    }


    // save debug frame of last image. used for algorithm debugging
#ifdef SAVE_PROCESS_FRAMES
    BitmapResult debugFrame = files.back();
    std::cout << "DEBUG: Saving frame: " << outdir + "/" + debugFrame.filename << std::endl;
    save_floatmap_as(debugFrame.image, outdir + "/" + debugFrame.filename);
#endif


    // save running time to output file. The time results will be in the order of files in the folder (most likely things are sorted alphanumerically), so file name will not be directly saved
    std::string output;

    output += "Results for run:\n" + std::to_string(startTime)+ "\n" + infoText;
    output += "total run time (ms)\n" + std::to_string(endTime - startTime) + "\n\n\n";

    output += "total, gaussian_blur, extended_blurred_image, sobel_vertical_image, sobel_horizontal_image, magnitude, direction\n";
    for (const auto &file: files) {
        std::string debugTimePrint = "";
        for (const auto &debugFrame: file.debugFrames) {
            debugTimePrint.append(std::to_string(file.totalRuntime) +"," +std::to_string(debugFrame.totalRuntime) + ",");
        }
        debugTimePrint = debugTimePrint.erase(debugTimePrint.size() - 1);
        output += debugTimePrint + "\n";
    }


    // sample printout:
    /*
Results for run:
-1286957056
variant, blur kernel size, blur sigma
1,11,0.200000
total run time (ms)
19189


total, gaussian_blur, extended_blurred_image, sobel_vertical_image, sobel_horizontal_image, magnitude, direction
13044,10317,13044,180,13044,815,13044,821,13044,252,13044,577
909,733,909,10,909,50,909,51,909,17,909,40
26,19,26,0,26,2,26,2,26,0,26,1
917,734,917,11,917,50,917,59,917,15,917,40
921,731,921,10,921,63,921,52,921,16,921,39
3332,2688,3332,38,3332,195,3332,185,3332,61,3332,144
     */


    // std::cout << output << std::endl; // print to terminal
    std::ofstream file(outdir +"/__"+std::to_string(0+startTime) + "  results.txt");
    file << output;

    file.close();
}
