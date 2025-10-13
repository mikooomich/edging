#include "bitmap.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <filesystem>

#include "utils.h"
namespace fs = std::filesystem;

// Enable extra debug print and image saving.
// Uncomment to enable, comment to disable.
#define DEBUG


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

    result->debugFrames.emplace_back(BitmapResult(blurred_image_path, t2 - t1, result->image, blurred_image));

#ifdef DEBUG
    std::cout << "DEBUG: extended_blurred_image" << std::endl;
#endif
    // Extend blured image borders
    t1 = getSysTime();
    FloatMap extended_blurred_image = border_extend_floatmap(blurred_image, 1);
    t2 = getSysTime();

    result->debugFrames.emplace_back(BitmapResult(extended_blurred_image_path, t2 - t1, blurred_image,
                                                  extended_blurred_image));

#ifdef DEBUG
    std::cout << "DEBUG: sobel_vertical_image" << std::endl;
#endif

    // Apply vertical Sobel operator to detect horizontal edges
    // Convert result to Bitmap for visualization and save
    t1 = getSysTime();
    FloatMap sobel_vertical_kernel = get_sobel_kernel(true);
    FloatMap sobel_vertical_image = apply_kernel_as_sum(extended_blurred_image, sobel_vertical_kernel);
    t2 = getSysTime();

    result->debugFrames.emplace_back(BitmapResult(sobel_vertical_path, t2 - t1, extended_blurred_image,
                                                  sobel_vertical_image));
#ifdef DEBUG
    std::cout << "DEBUG: sobel_horizontal_image" << std::endl;
#endif

    // Apply horizontal Sobel operator to detect vertical edges
    // Convert result to Bitmap for visualization and save
    t1 = getSysTime();
    FloatMap sobel_horizontal_kernel = get_sobel_kernel(false);
    FloatMap sobel_horizontal_image = apply_kernel_as_sum(extended_blurred_image, sobel_horizontal_kernel);
    t2 = getSysTime();

    result->debugFrames.emplace_back(BitmapResult(sobel_horizontal_path, t2 - t1, extended_blurred_image,
                                                  sobel_horizontal_image));

#ifdef DEBUG
    std::cout << "DEBUG: magnitude" << std::endl;
#endif

    // Calculate edge magnitude by combining horizontal and vertical gradients
    // Magnitude represents the strength of edges at each pixel location
    t1 = getSysTime();
    FloatMap magnitude = calculate_magnitude(sobel_horizontal_image, sobel_vertical_image);
    t2 = getSysTime();

    // no support for multi input... we probably wont need it...
    result->debugFrames.emplace_back(BitmapResult(magnitude_path, t2 - t1, sobel_horizontal_image, magnitude));
#ifdef DEBUG
    std::cout << "DEBUG: direction" << std::endl;
#endif

    // Calculate edge direction using atan2 of gradient components
    // Direction indicates the orientation of edges at each pixel
    // This is not required in sobel. It is required in canny.
    t1 = getSysTime();
    FloatMap direction = calculate_direction(sobel_horizontal_image, sobel_vertical_image);
    t2 = getSysTime();

    // no support for multi input... we probably wont need it...
    result->debugFrames.emplace_back(BitmapResult(direction_path, t2 - t1, sobel_horizontal_image, direction));
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
    std::string TMPDIR = "./data/temp";

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

    std::vector<BitmapResult> files;

    // Collect all the images before from input folder. Hidden files (files starting with ".") are ignored
    for (const auto &entry: fs::directory_iterator(INDIR)) {
        std::string filename = entry.path().filename().string();

        if (filename[0] != '.') {
            std::string fullPath = entry.path().string();
            std::cout << "Found: " << filename << " (" << fullPath << ")" << std::endl;


            // Load the input image and convert to grayscale FloatMap
            FloatMap image = load_image_grayscale(fullPath);

            FloatMap imageOutput(image.width, image.height);
            BitmapResult result = BitmapResult(filename, 0, image, imageOutput);
            files.emplace_back(result);
        }
    }

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
    for (const auto &file: files) {
        std::cout << "Saving: " << OUTDIR + "/" + file.filename << "\n\tTime taken: " << file.totalRuntime << " ms" <<
                std::endl;
        save_floatmap_as(file.outImage, OUTDIR + "/" + file.filename);

#ifdef DEBUG
        std::string debugTimePrint = "\tDEBUG: time breakdown: ";
        // save any debug frames
        for (const auto &debugFrame: file.debugFrames) {
            std::cout << "DEBUG: Saving frame: " << OUTDIR + "/" + debugFrame.filename << std::endl;
            save_floatmap_as(debugFrame.outImage, OUTDIR + "/" + debugFrame.filename);
            debugTimePrint.append("/" + std::to_string(debugFrame.totalRuntime));
        }
        std::cout << debugTimePrint << std::endl;
#endif
    }

    return 0;
}
