#include "bitmap.h"
#include <cmath>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

// main processing function.
// Loads image, processes it, and then saves it
int processImage(const std::string& filename, const std::string& inputDir, const std::string& outputDir) {

    //compute required file paths
    std::string image_path = inputDir + "/" + filename;

    //we can keep these in config file later if you hate hardcoding filenames so much.
    std::string image_original_path = outputDir + "/image_original.png";
    std::string blurred_image_path = outputDir + "/blurred_image.png";
    std::string sobel_vertical_path = outputDir + "/sobel_vertical.png";
    std::string sobel_horizontal_path = outputDir + "/sobel_horizontal.png";
    std::string magnitude_path = outputDir + "/magnitude.png";
    std::string direction_path = outputDir + "/direction.png";

    std::string final_output_path = outputDir + "/result_" + filename;

    // int q = "2";
    
    std::cout << "Trying to detect edges" << std::endl;
    // Load the input image and convert to grayscale FloatMap
    // Save the original image for reference
    FloatMap image = load_image_grayscale(image_path);
    save_bitmap_as(image, image_original_path);
    // Apply Gaussian blur to reduce noise before edge detection
    // Use kernel size 11 and sigma 0.2 for smooth blurring
    FloatMap blurred_image = gaussian_blur(image, 11, 0.2f);

    std::cout << "Saving blurred image" << std::endl;
    save_bitmap_as(blurred_image, blurred_image_path);
    
    
    // Apply vertical Sobel operator to detect horizontal edges
    // Convert result to Bitmap for visualization and save
    FloatMap sobel_vertical_kernel = get_sobel_kernel(true);
    FloatMap sobel_vertical_image = apply_kernel_as_sum(image, sobel_vertical_kernel);
    std::cout << "Saving sobel vertical image" << std::endl;
    Bitmap vertical_bitmap = create_bitmap_from_floatmap(sobel_vertical_image);
    save_bitmap_as(vertical_bitmap, sobel_vertical_path);

    // Apply horizontal Sobel operator to detect vertical edges
    // Convert result to Bitmap for visualization and save
    FloatMap sobel_horizontal_kernel = get_sobel_kernel(false);
    FloatMap sobel_horizontal_image = apply_kernel_as_sum(image, sobel_horizontal_kernel);
    std::cout << "Saving sobel horizontal image" << std::endl;
    Bitmap horizontal_bitmap = create_bitmap_from_floatmap(sobel_horizontal_image);
    save_bitmap_as(horizontal_bitmap, sobel_horizontal_path);


    // Calculate edge magnitude by combining horizontal and vertical gradients
    // Magnitude represents the strength of edges at each pixel location
    FloatMap magnitude = calculate_magnitude(sobel_horizontal_image, sobel_vertical_image);
    std::cout << "Saving magnitude image" << std::endl;
    Bitmap magnitude_bitmap = create_bitmap_from_floatmap(magnitude);
    save_bitmap_as(magnitude_bitmap, magnitude_path);

    // Calculate edge direction using atan2 of gradient components
    // Direction indicates the orientation of edges at each pixel
    FloatMap direction = FloatMap(sobel_vertical_image.width, sobel_vertical_image.height);
    for (int y = 0; y < direction.height; y++) {
        for (int x = 0; x < direction.width; x++) {
            float gx = sobel_horizontal_image.data[y][x];
            float gy = sobel_vertical_image.data[y][x];
            direction.data[y][x] = std::atan2(gy, gx);
        }
    }
    std::cout << "Saving direction image" << std::endl;
    Bitmap direction_bitmap = create_bitmap_from_floatmap(direction);
    save_bitmap_as(direction_bitmap, direction_path);


    Bitmap final_output_bitmap = create_bitmap_from_floatmap(magnitude);
    std::cout << "Saving final output image" << std::endl;
    save_bitmap_as(final_output_bitmap, final_output_path);

    std::cout << "Saved everything to " + outputDir + "/" << std::endl;


    return 0;
}

int main() {
    std::string indir = "./data/input";
    std::string outdir = "./data/output";
    std::string tmpdir = "./data/temp";

    // process all image from input folder
    for (const auto & entry : fs::directory_iterator(indir)) {
        std::cout << "Processing: " << entry.path() << std::endl;
        std::string filename = entry.path().filename().string();
        std::cout << "Processing: " << filename << std::endl;
        processImage(filename, indir, outdir);
    }

    return 0;
}
