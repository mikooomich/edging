#include "bitmap.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <future>

#include "utils.h"
namespace fs = std::filesystem;

// Enable extra debug print
// Uncomment to enable, comment to disable.
#define DEBUG

// Enable saving of debug frames. Useful for checking algorithm correctness, really not useful otherwise
// Uncomment to enable, comment to disable.
// #define SAVE_PROCESS_FRAMES

// main processing function.
int processImage(BitmapResult *result, int blur_kernel_size, float blur_sigma) {
    assert(blur_kernel_size % 2 == 1 && "Blur kernel size must be odd");
    assert(blur_sigma > 0 && "Blur sigma must be positive");

    //output paths for debugging. Designed for one image only.
    std::string image_original_path = "/image_original.png";
    std::string blurred_image_path = "/blurred_image.png";
    std::string extended_blurred_image_path = "/extended_blurred_image.png";
    std::string sobel_vertical_path = "/sobel_vertical.png";
    std::string sobel_horizontal_path = "/sobel_horizontal.png";
    std::string magnitude_path = "/magnitude.png";
    std::string direction_path = "/direction.png";
    FloatMap dudFloatMap = make_gaussian_kernel(3, 0.1); // for when SAVE_PROCESS_FRAMES is off

    long startTime = getSysTime();
    long t1;
    long t2;

    // The process here will be:
    // 1. Create the image
    // 2. Log how long it takes to complete that step
    // This is repeated for every step after this point

#ifdef DEBUG
    std::cout << "DEBUG: gaussian blur" << std::endl;
#endif

    // Apply Gaussian blur to reduce noise before edge detection
    // Use kernel size 11 and sigma 0.2 for smooth blurring
    t1 = getSysTime();
    FloatMap blurred_image = gaussian_blur(result->image, blur_kernel_size, blur_sigma);
    t2 = getSysTime();

#ifdef SAVE_PROCESS_FRAMES
    result->debugFrames.emplace_back(BitmapResult(blurred_image_path, t2 - t1, result->image, blurred_image));
#else
    result->debugFrames.emplace_back(BitmapResult(blurred_image_path, t2 - t1, dudFloatMap, dudFloatMap));
#endif


#ifdef DEBUG
    std::cout << "DEBUG: extended_blurred_image" << std::endl;
#endif
    // Extend blured image borders
    t1 = getSysTime();
    FloatMap extended_blurred_image = border_extend_floatmap(blurred_image, 1);
    t2 = getSysTime();

#ifdef SAVE_PROCESS_FRAMES
    result->debugFrames.emplace_back(BitmapResult(extended_blurred_image_path, t2 - t1, blurred_image,
                                                  extended_blurred_image));
#else
    result->debugFrames.emplace_back(BitmapResult(extended_blurred_image_path, t2 - t1, dudFloatMap, dudFloatMap));
#endif


#ifdef DEBUG
    std::cout << "DEBUG: sobel_vertical_image" << std::endl;
#endif

    // Apply vertical Sobel operator to detect horizontal edges
    // Convert result to Bitmap for visualization and save
    t1 = getSysTime();
    FloatMap sobel_vertical_kernel = get_sobel_kernel(true);
    FloatMap sobel_vertical_image = apply_kernel_as_sum(extended_blurred_image, sobel_vertical_kernel);
    t2 = getSysTime();

#ifdef SAVE_PROCESS_FRAMES
    result->debugFrames.emplace_back(BitmapResult(sobel_vertical_path, t2 - t1, extended_blurred_image,
                                                  sobel_vertical_image));
#else
    result->debugFrames.emplace_back(BitmapResult(sobel_vertical_path, t2 - t1, dudFloatMap, dudFloatMap));
#endif


#ifdef DEBUG
    std::cout << "DEBUG: sobel_horizontal_image" << std::endl;
#endif

    // Apply horizontal Sobel operator to detect vertical edges
    // Convert result to Bitmap for visualization and save
    t1 = getSysTime();
    FloatMap sobel_horizontal_kernel = get_sobel_kernel(false);
    FloatMap sobel_horizontal_image = apply_kernel_as_sum(extended_blurred_image, sobel_horizontal_kernel);
    t2 = getSysTime();

#ifdef SAVE_PROCESS_FRAMES
    result->debugFrames.emplace_back(BitmapResult(sobel_horizontal_path, t2 - t1, extended_blurred_image,
                                                  sobel_horizontal_image));
#else
    result->debugFrames.emplace_back(BitmapResult(sobel_horizontal_path, t2 - t1, dudFloatMap, dudFloatMap));

#endif


#ifdef DEBUG
    std::cout << "DEBUG: magnitude" << std::endl;
#endif

    // Calculate edge magnitude by combining horizontal and vertical gradients
    // Magnitude represents the strength of edges at each pixel location
    t1 = getSysTime();
    FloatMap magnitude = calculate_magnitude(sobel_horizontal_image, sobel_vertical_image);
    t2 = getSysTime();

    // no support for multi input... we probably wont need it...

#ifdef SAVE_PROCESS_FRAMES
    result->debugFrames.emplace_back(BitmapResult(magnitude_path, t2 - t1, sobel_horizontal_image, magnitude));

#else
    result->debugFrames.emplace_back(BitmapResult(magnitude_path, t2 - t1, dudFloatMap, dudFloatMap));

#endif


#ifdef DEBUG
    std::cout << "DEBUG: direction" << std::endl;
#endif

    // Calculate edge direction using atan2 of gradient components
    // Direction indicates the orientation of edges at each pixel
    // This is not required in sobel. It is required in canny.
    t1 = getSysTime();
    FloatMap direction = calculate_direction(sobel_horizontal_image, sobel_vertical_image);
    t2 = getSysTime();


#ifdef SAVE_PROCESS_FRAMES
    // no support for multi input... we probably wont need it...
    result->debugFrames.emplace_back(BitmapResult(direction_path, t2 - t1, sobel_horizontal_image, direction));

#else
    result->debugFrames.emplace_back(BitmapResult(direction_path, t2 - t1, dudFloatMap, dudFloatMap));

#endif


#ifdef DEBUG
    std::cout << "DEBUG: Create final bitmap" << std::endl;
#endif

    //if we are doing sobel, this is the final output.
    //if we want to go canny, we need to do more processing.
    result->outImage = magnitude;
    result->totalRuntime = getSysTime() - startTime;


#ifdef DEBUG
    // Save the original image for reference
    result->debugFrames.emplace_back(BitmapResult(image_original_path, 0L, result->image, result->image));
#endif


    return 0;
}

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

    if (argc > 1) {
        blur_kernel_size = std::atoi(argv[1]);
    }
    if (argc > 2) {
        blur_sigma = std::atof(argv[2]);
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
                FloatMap imageOutput(image.width, image.height);
                return BitmapResult(filename, 0, image, imageOutput);
            }));
        }
    }

    std::vector<BitmapResult> files;
    for (auto &fut: imageLoadingJobs)
        files.emplace_back(fut.get());
    // ----- End loading images section-----


    std::cout << "\n\n-----------------------" << std::endl;
    std::cout << "Processing images" << std::endl;
    std::cout << "Using blur_kernel_size = " << blur_kernel_size << ", blur_sigma = " << blur_sigma << std::endl;
    std::cout << "-----------------------\n\n" << std::endl;

    // process files
    for (auto &file: files) {
        std::cout << "Processing: " << file.filename << std::endl;
        processImage(&file, blur_kernel_size, blur_sigma);
    }

    std::cout << "\n\n-----------------------" << std::endl;
    std::cout << "Saving images" << std::endl;
    std::cout << "-----------------------\n\n" << std::endl;

    // save all bitmaps, then debug frames if debug is enabled
    std::vector<std::future<void>> imageSavingJobs;
    for (const auto &file: files) {
        imageSavingJobs.push_back(std::async(std::launch::async, [=]() {
            std::cout << "Saving: " << OUTDIR + "/" + file.filename << "\n\tTime taken: " << file.totalRuntime << " ms"
                    << std::endl;
            save_floatmap_as(file.outImage, OUTDIR + "/" + file.filename);

#ifdef DEBUG
            std::string debugTimePrint = "\tDEBUG: time breakdown: ";
            // save any debug frames
            for (const auto &debugFrame: file.debugFrames) {
#ifdef SAVE_PROCESS_FRAMES
                std::cout << "DEBUG: Saving frame: " << OUTDIR + "/" + debugFrame.filename << std::endl;
                save_floatmap_as(debugFrame.outImage, OUTDIR + "/" + debugFrame.filename);
#endif
                debugTimePrint.append("/" + std::to_string(debugFrame.totalRuntime));
            }
            std::cout << debugTimePrint << std::endl; // TODO: Unlabeled. label if we keep this...
#endif
        }));
    }

    return 0;
}
