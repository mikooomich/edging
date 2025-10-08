#ifndef BITMAP_H
#define BITMAP_H

#include <vector>
#include <cstdint>
#include <string>
#include <functional>

struct Bitmap {
    int width;
    int height;
    std::vector<std::vector<uint8_t>> data;
    
    Bitmap(int w, int h);
};

// 用于存储含负数的卷积核（如 Sobel）
struct FloatMap {
    int width;
    int height;
    std::vector<std::vector<float>> data;
    
    FloatMap(int w, int h);
};

// 声明所有函数
Bitmap generate_noise(int width = 50, int height = 100);
Bitmap load_image_grayscale(const std::string& filename);
Bitmap border_extend(const Bitmap& bitmap, int padding);
// void display_bitmap(const Bitmap& bitmap);
void save_bitmap(const Bitmap& bitmap, const std::string& filename);
Bitmap apply_kernel(const Bitmap& bitmap, const Bitmap& kernel);
FloatMap apply_kernel(const Bitmap& bitmap, const FloatMap& kernel);  // 支持负数 kernel
Bitmap make_gaussian_kernel(int size, float sigma = 1.0f);
Bitmap gaussian_blur(const Bitmap& bitmap, const int kernel_size, const float sigma);
FloatMap get_sobel_kernel(const bool vertical = true);
Bitmap create_bitmap_from_floatmap(const FloatMap& floatmap);

// 对 bitmap 的每个像素应用函数（返回新 bitmap，非 inplace）
Bitmap apply(const Bitmap& bitmap, std::function<uint8_t(uint8_t)> func);
// 对 bitmap 的每个像素应用函数（inplace 修改原 bitmap）
void apply_inplace(Bitmap& bitmap, std::function<uint8_t(uint8_t)> func);

// 对 floatmap 的每个元素应用函数（返回新 floatmap，非 inplace）
FloatMap apply(const FloatMap& floatmap, std::function<float(float)> func);
// 对 floatmap 的每个元素应用函数（inplace 修改原 floatmap）
void apply_inplace(FloatMap& floatmap, std::function<float(float)> func);

#endif