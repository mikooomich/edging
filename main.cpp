#include "bitmap.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <filesystem>

#include "utils.h"
namespace fs = std::filesystem;

#define DEBUG = true;


// main processing function.
// Loads image, processes it, and then saves it
int processImage(BitmapResult *result, int blur_kernel_size, float blur_sigma) {
    assert(blur_kernel_size % 2 == 1 && "Blur kernel size must be odd");
    assert(blur_sigma > 0 && "Blur sigma must be positive");

#ifdef DEBUG
    //output paths for debugging. Designed for one image only.
    std::string image_original_path = "/image_original.png";
    std::string blurred_image_path = "/blurred_image.png";
    std::string sobel_vertical_path = "/sobel_vertical.png";
    std::string sobel_horizontal_path = "/sobel_horizontal.png";
    std::string magnitude_path = "/magnitude.png";
    std::string direction_path = "/direction.png";
#endif

    long startTime = getSysTime();
    long t1;
    long t2;


#ifdef DEBUG
    // Save the original image for reference
    t1 = getSysTime();
    Bitmap image_bitmap(result->image.width, result->image.height);
    create_bitmap_from_floatmap(result->image, image_bitmap);
    t2 = getSysTime();

    result->debugFrames.emplace_back(BitmapResult(image_original_path, t2 - t1, result->image, image_bitmap));

    std::cout << "DEBUG: gaussian blur" << std::endl;
#endif

    // Apply Gaussian blur to reduce noise before edge detection
    // Use kernel size 11 and sigma 0.2 for smooth blurring
    t1 = getSysTime();
    FloatMap blurred_image = gaussian_blur(result->image, blur_kernel_size, blur_sigma);
    Bitmap blurred_image_bitmap(blurred_image.width, blurred_image.height);
    create_bitmap_from_floatmap(blurred_image, blurred_image_bitmap);
    t2 = getSysTime();
#ifdef DEBUG
    result->debugFrames.emplace_back(BitmapResult(blurred_image_path, t2 - t1, result->image, blurred_image_bitmap));
    std::cout << "DEBUG: sobel_vertical_image" << std::endl;
#endif
    FloatMap extended_blurred_image = border_extend_floatmap(blurred_image, 1);

    // Apply vertical Sobel operator to detect horizontal edges
    // Convert result to Bitmap for visualization and save
    t1 = getSysTime();
    FloatMap sobel_vertical_kernel = get_sobel_kernel(true);
    FloatMap sobel_vertical_image = apply_kernel_as_sum(extended_blurred_image, sobel_vertical_kernel);
    Bitmap sobel_vertical_image_bitmap(sobel_vertical_image.width, sobel_vertical_image.height);
    create_bitmap_from_floatmap(sobel_vertical_image, sobel_vertical_image_bitmap);
    t2 = getSysTime();
#ifdef DEBUG
    result->debugFrames.emplace_back(BitmapResult(sobel_vertical_path, t2 - t1, result->image,
                                                  sobel_vertical_image_bitmap));
    std::cout << "DEBUG: sobel_horizontal_image" << std::endl;
#endif

    // Apply horizontal Sobel operator to detect vertical edges
    // Convert result to Bitmap for visualization and save
    t1 = getSysTime();
    FloatMap sobel_horizontal_kernel = get_sobel_kernel(false);
    FloatMap sobel_horizontal_image = apply_kernel_as_sum(extended_blurred_image, sobel_horizontal_kernel);
    Bitmap sobel_horizontal_image_bitmap(sobel_horizontal_image.width, sobel_horizontal_image.height);
    create_bitmap_from_floatmap(sobel_horizontal_image, sobel_horizontal_image_bitmap);
    t2 = getSysTime();
#ifdef DEBUG
    result->debugFrames.emplace_back(BitmapResult(sobel_horizontal_path, t2 - t1, result->image,
                                                  sobel_vertical_image_bitmap));
    std::cout << "DEBUG: magnitude" << std::endl;
#endif

    // Calculate edge magnitude by combining horizontal and vertical gradients
    // Magnitude represents the strength of edges at each pixel location
    t1 = getSysTime();
    FloatMap magnitude = calculate_magnitude(sobel_horizontal_image, sobel_vertical_image);
    Bitmap magnitude_bitmap(magnitude.width, magnitude.height);
    create_bitmap_from_floatmap(magnitude, magnitude_bitmap);
    t2 = getSysTime();
#ifdef DEBUG
    result->debugFrames.emplace_back(BitmapResult(magnitude_path, t2 - t1, result->image, magnitude_bitmap));
    std::cout << "DEBUG: direction" << std::endl;
#endif

    // Calculate edge direction using atan2 of gradient components
    // Direction indicates the orientation of edges at each pixel
    // This is not required in sobel. It is required in canny.
    t1 = getSysTime();
    FloatMap direction = calculate_direction(sobel_horizontal_image, sobel_vertical_image);
    Bitmap direction_bitmap(direction.width, direction.height);
    create_bitmap_from_floatmap(direction, direction_bitmap);
    t2 = getSysTime();

#ifdef DEBUG
    result->debugFrames.emplace_back(BitmapResult(direction_path, t2 - t1, result->image, direction_bitmap));
    std::cout << "DEBUG: Create final bitmap" << std::endl;
#endif

    //if we are doing sobel, this is the final output.
    //if we want to go canny, we need to do more processing.
    create_bitmap_from_floatmap(magnitude, result->outBitmap);
    result->totalRuntime = getSysTime() - startTime;

    return 0;
}

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
    }


    if (blur_sigma == -1.0f) {
        std::cout << "WARNING: No blur_sigma specified. Using default value 0.2" << std::endl;
        blur_sigma = 0.2f;
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

            Bitmap image_bitmap(image.width, image.height);
            BitmapResult result = BitmapResult(filename, 0, image, image_bitmap);
            files.emplace_back(result);
        }
    }

    std::cout << "\n\n-----------------------" << std::endl;
    std::cout << "Processing images" << std::endl;
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
        save_bitmap_as(file.outBitmap, OUTDIR + "/" + file.filename);

#ifdef DEBUG
        std::string debugTimePrint = "\tDEBUG: time breakdown: ";
        // save any debug frames
        for (const auto &debugFrame: file.debugFrames) {
            std::cout << "DEBUG: Saving frame: " << OUTDIR + "/" + debugFrame.filename << std::endl;
            save_bitmap_as(debugFrame.outBitmap, OUTDIR + "/" + debugFrame.filename);
            debugTimePrint.append("/" + std::to_string(debugFrame.totalRuntime));
        }
        std::cout << debugTimePrint << std::endl;
#endif
    }

    return 0;
}
