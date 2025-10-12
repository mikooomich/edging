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

    bool operator==(const Bitmap& other) const;
    bool operator!=(const Bitmap& other) const;

    
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
    Bitmap outBitmap; // output image

    std::vector<BitmapResult> debugFrames; // for debug only

    BitmapResult(std::string  filename, long initialTime, FloatMap input, Bitmap output);

    /**
     * for debug progress bitmaps
     */
    BitmapResult(std::string filename, FloatMap input, Bitmap output);
};


inline BitmapResult::BitmapResult(std::string  filename, long initialTime, FloatMap input, Bitmap output):
  filename(std::move(filename)), totalRuntime(initialTime), image(std::move(input)), outBitmap(std::move(output)) {}

// for debug and progress bitmaps
inline BitmapResult:: BitmapResult(std::string  filename, FloatMap input, Bitmap output): filename(std::move(filename)), totalRuntime(0L), image(std::move(input)), outBitmap(std::move(output)) {}

bool is_valid_floatmap(const FloatMap& floatmap, int max_output_lines = 10);
FloatMap load_image_grayscale(const std::string& filename);
FloatMap border_extend_floatmap(const FloatMap& floatmap, int padding);
// void display_bitmap(const Bitmap& bitmap);
void save_bitmap_as(const Bitmap& bitmap, const std::string& filename);
void save_bitmap_as(const FloatMap& floatmap, const std::string& filename);
FloatMap apply_kernel(const Bitmap& bitmap, const FloatMap& kernel); 
FloatMap apply_kernel_as_weighted_average(const FloatMap& floatmap, const FloatMap& kernel);
FloatMap apply_kernel_as_sum(const FloatMap& floatmap, const FloatMap& kernel);
FloatMap make_gaussian_kernel(int size, float sigma = 1.0f);
FloatMap gaussian_blur(const FloatMap& floatmap, const int kernel_size, const float sigma);
FloatMap get_sobel_kernel(const bool vertical = true);
FloatMap calculate_magnitude(const FloatMap& sobel_horizontal, const FloatMap& sobel_vertical);
Bitmap create_bitmap_from_floatmap(const FloatMap& floatmap, Bitmap& bitmap);
FloatMap calculate_direction(const FloatMap& sobel_horizontal_image, const FloatMap& sobel_vertical_image);



#endif