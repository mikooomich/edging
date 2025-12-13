#include "sequential.h"

#include <iostream>

#include "bitmap.h"
#include "utils.h"
#include "utils.h"

// main processing function.
int processImageNoGauss(BitmapResult *result, const FloatMap &sobelKernelvert, const FloatMap &sobelKernelHoriz) {
    //output paths for debugging. Designed for one image only.
    std::string image_original_path = "/image_original.png";
    std::string blurred_image_path = "/blurred_image.png";
    std::string extended_blurred_image_path = "/extended_blurred_image.png";
    std::string sobel_vertical_path = "/sobel_vertical.png";
    std::string sobel_horizontal_path = "/sobel_horizontal.png";
    std::string magnitude_path = "/magnitude.png";
    std::string direction_path = "/direction.png";
    FloatMap dudFloatMap = FloatMap(0, 0); // for when SAVE_PROCESS_FRAMES is off

    long startTime = getSysTime();
    long t1;
    long t2;

    // The process here will be:
    // 1. Create the image
    // 2. Log how long it takes to complete that step
    // This is repeated for every step after this point


#ifdef EDGING_DEBUG
    std::cout << "DEBUG: extended_blurred_image" << std::endl;
#endif
    // Extend blured image borders
    t1 = getSysTime();
    FloatMap extended_blurred_image = border_extend_floatmap(result->image, 1);
    t2 = getSysTime();

#ifdef SAVE_PROCESS_FRAMES
    result->debugFrames.emplace_back(BitmapResult(extended_blurred_image_path, t2 - t1, extended_blurred_image));
#else
    result->debugFrames.emplace_back(BitmapResult(extended_blurred_image_path, t2 - t1, dudFloatMap));
#endif


#ifdef EDGING_DEBUG
    std::cout << "DEBUG: sobel_vertical_image" << std::endl;
#endif

    // Apply vertical Sobel operator to detect horizontal edges
    // Convert result to Bitmap for visualization and save
    t1 = getSysTime();
    FloatMap sobel_vertical_image = apply_kernel_as_sum(extended_blurred_image, sobelKernelvert);
    t2 = getSysTime();

#ifdef SAVE_PROCESS_FRAMES
    result->debugFrames.emplace_back(BitmapResult(sobel_vertical_path, t2 - t1, sobel_vertical_image));
#else
    result->debugFrames.emplace_back(BitmapResult(sobel_vertical_path, t2 - t1, dudFloatMap));
#endif


#ifdef EDGING_DEBUG
    std::cout << "DEBUG: sobel_horizontal_image" << std::endl;
#endif

    // Apply horizontal Sobel operator to detect vertical edges
    // Convert result to Bitmap for visualization and save
    t1 = getSysTime();
    FloatMap sobel_horizontal_image = apply_kernel_as_sum(extended_blurred_image, sobelKernelHoriz);
    t2 = getSysTime();

#ifdef SAVE_PROCESS_FRAMES
    result->debugFrames.emplace_back(BitmapResult(sobel_horizontal_path, t2 - t1, sobel_horizontal_image));
#else
    result->debugFrames.emplace_back(BitmapResult(sobel_horizontal_path, t2 - t1, dudFloatMap));
#endif


#ifdef EDGING_DEBUG
    std::cout << "DEBUG: magnitude" << std::endl;
#endif

    // Calculate edge magnitude by combining horizontal and vertical gradients
    // Magnitude represents the strength of edges at each pixel location
    t1 = getSysTime();
    FloatMap magnitude = calculate_magnitude(sobel_horizontal_image, sobel_vertical_image);
    t2 = getSysTime();

    // no support for multi input... we probably wont need it...

#ifdef SAVE_PROCESS_FRAMES
    result->debugFrames.emplace_back(BitmapResult(magnitude_path, t2 - t1, magnitude));

#else
    result->debugFrames.emplace_back(BitmapResult(magnitude_path, t2 - t1, dudFloatMap));
#endif


#ifdef EDGING_DEBUG
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
    result->debugFrames.emplace_back(BitmapResult(direction_path, t2 - t1, direction));

#else
    result->debugFrames.emplace_back(BitmapResult(direction_path, t2 - t1, dudFloatMap));
#endif


#ifdef SAVE_PROCESS_FRAMES
    // Save the original image for reference
    result->debugFrames.emplace_back(BitmapResult(image_original_path, 0L, result->image));
#endif

#ifdef EDGING_DEBUG
    std::cout << "DEBUG: Create final bitmap" << std::endl;
#endif

    //if we are doing sobel, this is the final output.
    //if we want to go canny, we need to do more processing.
    // reuse the input image to save the output. It is assumed both are the same size, or else something is wrong with the algorithm
    for (int y = 0; y < result->image.height; y++) {
        for (int x = 0; x < result->image.width; x++) {
            result->image.data[y][x] = magnitude.data[y][x];
        }
    }
    result->totalRuntime += getSysTime() - startTime;

    return 0;
}

void processGaussianBlur(BitmapResult *result, const FloatMap &gaussianKernel) {
    //output paths for debugging. Designed for one image only.
    std::string blurred_image_path = "/blurred_image.png";
    FloatMap dudFloatMap = FloatMap(0, 0); // for when SAVE_PROCESS_FRAMES is off

    long t1;
    long t2;

    // The process here will be:
    // 1. Create the image
    // 2. Log how long it takes to complete that step
    // This is repeated for every step after this point

#ifdef EDGING_DEBUG
    std::cout << "DEBUG: gaussian blur" << std::endl;
#endif

    // Apply Gaussian blur to reduce noise before edge detection
    // Use kernel size 11 and sigma 0.2 for smooth blurring
    t1 = getSysTime();
    FloatMap blurred_image = gaussian_blur(result->image, gaussianKernel);
    t2 = getSysTime();

#ifdef SAVE_PROCESS_FRAMES
    result->debugFrames.emplace_back(BitmapResult(blurred_image_path, t2 - t1, blurred_image));
#else
    result->debugFrames.emplace_back(BitmapResult(blurred_image_path, t2 - t1, dudFloatMap));
#endif

     result->totalRuntime = t2 - t1;
}

/**
 * Running time recorded with sequential
 *
 * frame time: ALl kernel applications are done sequentially. The runtime for each step (aka frame) is saved separately under debugFrames (see BitmapResult)
 *
 *
 *
 * totalRuntime: The total runtime is the time elapsed to process a single image, and save it to data back to the input vector
 * Includes any overhead.
 *
 *overheadTime: The overhead time is totalRuntime - the sum of the frame times
 *
 */
void runSequential(std::vector<BitmapResult> &files, const FloatMap &gaussianKernel, const FloatMap &sobelKernelVert,
                   const FloatMap &sobelKernelHoriz) {
    int completed = 0;
    // process files
    for (auto &file: files) {
        std::cout << "Processing: " << file.filename << std::endl;
        processGaussianBlur(&file, gaussianKernel);
        processImageNoGauss(&file, sobelKernelVert, sobelKernelHoriz);

        long frameRuntime = 0L;
        for (const auto& frame: file.debugFrames) {
            frameRuntime += frame.totalRuntime;
        }
        file.overheadTime = file.totalRuntime - frameRuntime;

        completed++;
        if (completed % 5 == 0) {
            // printf("[MASTER]: Finished processing %d of %llu\n", completed, files.size());
            printf("[MASTER]: Finished processing %d of %lu\n", completed, files.size());
        }
    }
}
