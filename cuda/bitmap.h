#ifndef BITMAP_H
#define BITMAP_H

#include <utility>
#include <vector>
#include <cstdint>
#include <string>
// #include <functional>

struct Bitmap {
    int width;
    int height;
    std::vector<std::vector<uint8_t>> data;
    
    Bitmap(int w, int h);
};

// For storing convolution kernels with negative values (e.g., Sobel)
struct FloatMap {
    int width;
    int height;
    std::vector<std::vector<float>> data;
    
    FloatMap(int w, int h);
};

struct BitmapResult {
    std::string filename;
    long totalRuntime;
    FloatMap image; // input image
    FloatMap outImage; // output image

    std::vector<BitmapResult> debugFrames; // for debug only

    BitmapResult(std::string  filename, long initialTime, FloatMap input, FloatMap output);
};


inline BitmapResult::BitmapResult(std::string  filename, long initialTime, FloatMap input, FloatMap output):
  filename(std::move(filename)), totalRuntime(initialTime), image(std::move(input)), outImage(std::move(output)) {}


bool is_valid_floatmap(const FloatMap& floatmap, int max_output_lines = 10);
FloatMap load_image_grayscale(const std::string& filename);
FloatMap border_extend_floatmap(const FloatMap& floatmap, int padding);
// void display_bitmap(const Bitmap& bitmap);
void save_bitmap_as(const Bitmap& bitmap, const std::string& filename);
void save_floatmap_as(const FloatMap& floatmap, const std::string& filename);
FloatMap apply_kernel_as_weighted_average(const FloatMap& floatmap, const FloatMap& kernel);
FloatMap apply_kernel_as_sum(const FloatMap& floatmap, const FloatMap& kernel);
FloatMap make_gaussian_kernel(int size, float sigma = 1.0f);
FloatMap gaussian_blur(const FloatMap& floatmap, const int kernel_size, const float sigma);
FloatMap get_sobel_kernel(const bool vertical = true);
FloatMap calculate_magnitude(const FloatMap& sobel_horizontal, const FloatMap& sobel_vertical);
Bitmap create_bitmap_from_floatmap(const FloatMap& floatmap);
FloatMap calculate_direction(const FloatMap& sobel_horizontal_image, const FloatMap& sobel_vertical_image);



#endif