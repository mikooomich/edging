// #include <functional>
#include <utility>
#include <vector>
#include <cstdint>
#include <string>
#include <cstdio>
#include <algorithm>
#include <cassert>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>

#include "stb_image_write.h"

#include "bitmap.h"

// implement Bitmap constructor
Bitmap::Bitmap(int w, int h) 
    : width(w), height(h), data(h, std::vector<uint8_t>(w, 0)) {}

// implement Bitmap equal operator
bool Bitmap::operator==(const Bitmap &other) const {
    if (width != other.width || height != other.height) {
        return false;
    }
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (data[y][x] != other.data[y][x]) {
                return false;
            }
        }
    }
    return true;
}

// implement Bitmap not equal operator
bool Bitmap::operator!=(const Bitmap &other) const {
    return !(*this == other);
}

// implement FloatMap constructor
FloatMap::FloatMap(int w, int h) 
    : width(w), height(h), data(h, std::vector<float>(w, 0.0f)) {}


bool is_valid_floatmap(const FloatMap &floatmap, int max_output_lines) {
    bool valid = true;
    int output_count = 0;
    for (int y = 0; y < floatmap.height; y++) {
        for (int x = 0; x < floatmap.width; x++) {
            if (floatmap.data[y][x] < 0.0f || floatmap.data[y][x] > 1.0f) {
                if (output_count < max_output_lines) {
                    printf("(%d, %d): %.6f\n", x, y, floatmap.data[y][x]);
                    output_count++;
                }
                valid = false;
            }
        }
    }
    if (!valid && output_count >= max_output_lines) {
        printf("... (more out-of-range pixels exist, but output truncated)\n");
    }
    return valid;
}

FloatMap load_image_grayscale(const std::string &filename) {
    int width, height, channels;

    // Load image (force convert to grayscale, 1 channel)
    unsigned char *img_data = stbi_load(filename.c_str(), &width, &height, &channels, 1);

    if (!img_data) {
        fprintf(stderr, "Error: Failed to load image '%s'\n", filename.c_str());
        exit(1);
    }

    // Create Bitmap object
    Bitmap bitmap(width, height);

    // Convert 1D array to 2D array (data[y][x])
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            bitmap.data[y][x] = img_data[y * width + x];
        }
    }

    // Free memory allocated by stb_image
    stbi_image_free(img_data);

    // Convert Bitmap to FloatMap (normalize to 0-1 range)
    FloatMap floatmap(width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            floatmap.data[y][x] = bitmap.data[y][x] / 255.0f;
        }
    }

    return floatmap;
}


void save_bitmap_as(const Bitmap &bitmap, const std::string &filename) {
    // Convert 2D array to 1D array
    std::vector<uint8_t> pixels;
    pixels.reserve(bitmap.width * bitmap.height);

    for (int y = 0; y < bitmap.height; y++) {
        for (int x = 0; x < bitmap.width; x++) {
            pixels.push_back(bitmap.data[y][x]);
        }
    }

    // Write to temporary file first (avoid browser reading incomplete file)
    std::string temp_filename = filename + ".tmp";
    stbi_write_png(temp_filename.c_str(), bitmap.width, bitmap.height, 1,
                   pixels.data(), bitmap.width);

    // Atomically rename to overwrite target file (rename is atomic)
    std::rename(temp_filename.c_str(), filename.c_str());
    // Note: after successful rename, temp file no longer exists (renamed to target file)
    // So no need to delete temp file separately
}

void save_bitmap_as(const FloatMap &floatmap, const std::string &filename) {
    // Check FloatMap validity
    if (!is_valid_floatmap(floatmap)) {
        fprintf(stderr, "Error: FloatMap contains values outside [0, 1] range\n");
        exit(1);
    }

    // Convert FloatMap to Bitmap
    Bitmap bitmap(floatmap.width, floatmap.height);
    for (int y = 0; y < floatmap.height; y++) {
        for (int x = 0; x < floatmap.width; x++) {
            bitmap.data[y][x] = static_cast<uint8_t>(floatmap.data[y][x] * 255.0f);
        }
    }

    // Call the base save_bitmap function
    save_bitmap_as(bitmap, filename);
}


FloatMap border_extend_floatmap(const FloatMap &floatmap, int padding) {
    // Create extended floatmap (width and height increased by 2*padding)
    FloatMap extended(floatmap.width + 2 * padding, floatmap.height + 2 * padding);

    for (int y = 0; y < extended.height; y++) {
        for (int x = 0; x < extended.width; x++) {
            // Calculate corresponding coordinates in original image (considering padding offset)
            int src_x = x - padding;
            int src_y = y - padding;

            // Clamp to original image boundaries
            int clamped_x = std::max(0, std::min(src_x, floatmap.width - 1));
            int clamped_y = std::max(0, std::min(src_y, floatmap.height - 1));

            // Copy pixel value
            extended.data[y][x] = floatmap.data[clamped_y][clamped_x];
        }
    }

    return extended;
}

FloatMap get_sobel_kernel(const bool vertical) {
    FloatMap kernel(3, 3);

    if (vertical) {
        // Sobel vertical edge detection (detects horizontal edges)
        kernel.data[0][0] = 1;
        kernel.data[0][1] = 2;
        kernel.data[0][2] = 1;
        kernel.data[1][0] = 0;
        kernel.data[1][1] = 0;
        kernel.data[1][2] = 0;
        kernel.data[2][0] = -1;
        kernel.data[2][1] = -2;
        kernel.data[2][2] = -1;
    } else {
        // Sobel horizontal edge detection (detects vertical edges)
        kernel.data[0][0] = 1;
        kernel.data[0][1] = 0;
        kernel.data[0][2] = -1;
        kernel.data[1][0] = 2;
        kernel.data[1][1] = 0;
        kernel.data[1][2] = -2;
        kernel.data[2][0] = 1;
        kernel.data[2][1] = 0;
        kernel.data[2][2] = -1;
    }
    return kernel;
}

int save_bitmap(const Bitmap &bitmap, const std::string &filename) {
    // Convert 2D array to 1D array
    std::vector<uint8_t> pixels;
    pixels.reserve(bitmap.width * bitmap.height);

    for (int y = 0; y < bitmap.height; y++) {
        for (int x = 0; x < bitmap.width; x++) {
            pixels.push_back(bitmap.data[y][x]);
        }
    }

    // Write to temporary file first (avoid browser reading incomplete file)
    std::string temp_filename = filename + ".tmp";
    stbi_write_png(temp_filename.c_str(), bitmap.width, bitmap.height, 1,
                   pixels.data(), bitmap.width);

    // Atomically rename to overwrite target file (rename is atomic)
    std::rename(temp_filename.c_str(), filename.c_str());
    // Note: after successful rename, temp file no longer exists (renamed to target file)
    // So no need to delete temp file separately

    std::cout << "Saved image as " << filename << std::endl;
    return 0;
}

Bitmap apply_kernel(const Bitmap &bitmap, const Bitmap &kernel) {
    assert(kernel.width == kernel.height && "Kernel must be square");
    assert(kernel.width % 2 == 1 && "Kernel width must be odd");

    // Calculate kernel sum (for normalization)
    float kernel_sum = 0;
    for (int ky = 0; ky < kernel.height; ky++) {
        for (int kx = 0; kx < kernel.width; kx++) {
            kernel_sum += kernel.data[ky][kx];
        }
    }

    /**
    when bitmap is x*y, kernel is k*k, then the blurred bitmap is (x-k+1)*(y-k+1)
    */
    Bitmap blurred(bitmap.width - kernel.width + 1, bitmap.height - kernel.height + 1);

    // Python: blurred[y][x] = sum(bitmap[y+ky][x+kx] * kernel[ky][kx]
    //                              for ky in range(k) for kx in range(k)) / kernel_sum
    for (int y = 0; y < blurred.height; y++) {
        for (int x = 0; x < blurred.width; x++) {
            float sum = 0;
            for (int ky = 0; ky < kernel.height; ky++) {
                for (int kx = 0; kx < kernel.width; kx++) {
                    sum += bitmap.data[y + ky][x + kx] * kernel.data[ky][kx];
                }
            }
            // Normalize and clamp to 0-255
            blurred.data[y][x] = static_cast<uint8_t>(std::min(255.0f, sum / kernel_sum));
        }
    }
    return blurred;
}

// Support kernels with negative values (e.g. Sobel edge detection)
FloatMap apply_kernel(const Bitmap &bitmap, const FloatMap &kernel) {
    assert(kernel.width == kernel.height && "Kernel must be square");
    assert(kernel.width % 2 == 1 && "Kernel width must be odd");


    /**
    when bitmap is x*y, kernel is k*k, then the result is (x-k+1)*(y-k+1)
    */
    FloatMap result(bitmap.width - kernel.width + 1, bitmap.height - kernel.height + 1);

    for (int y = 0; y < result.height; y++) {
        for (int x = 0; x < result.width; x++) {
            float sum = 0;
            for (int ky = 0; ky < kernel.height; ky++) {
                for (int kx = 0; kx < kernel.width; kx++) {
                    sum += (float) bitmap.data[y + ky][x + kx] * kernel.data[ky][kx];
                }
            }
            result.data[y][x] = sum;
        }
    }
    return result;
}

FloatMap apply_kernel_as_weighted_average(const FloatMap &floatmap, const FloatMap &kernel) {
    assert(kernel.width == kernel.height && "Kernel must be square");
    assert(kernel.width % 2 == 1 && "Kernel width must be odd");

    /**
    when floatmap is x*y, kernel is k*k, then the result is (x-k+1)*(y-k+1)
    */
    FloatMap result(floatmap.width - kernel.width + 1, floatmap.height - kernel.height + 1);

    // Calculate kernel sum for normalization
    float kernel_sum = 0;
    for (int ky = 0; ky < kernel.height; ky++) {
        for (int kx = 0; kx < kernel.width; kx++) {
            kernel_sum += kernel.data[ky][kx];
        }
    }

    for (int y = 0; y < result.height; y++) {
        for (int x = 0; x < result.width; x++) {
            float sum = 0;
            for (int ky = 0; ky < kernel.height; ky++) {
                for (int kx = 0; kx < kernel.width; kx++) {
                    sum += floatmap.data[y + ky][x + kx] * kernel.data[ky][kx];
                }
            }
            result.data[y][x] = sum / kernel_sum; // Normalize by kernel sum
        }
    }
    return result;
}

FloatMap apply_kernel_as_sum(const FloatMap &floatmap, const FloatMap &kernel) {
    assert(kernel.width == kernel.height && "Kernel must be square");
    assert(kernel.width % 2 == 1 && "Kernel width must be odd");

    /**
    when floatmap is x*y, kernel is k*k, then the result is (x-k+1)*(y-k+1)
    */
    FloatMap result(floatmap.width - kernel.width + 1, floatmap.height - kernel.height + 1);

    for (int y = 0; y < result.height; y++) {
        for (int x = 0; x < result.width; x++) {
            float sum = 0;
            for (int ky = 0; ky < kernel.height; ky++) {
                for (int kx = 0; kx < kernel.width; kx++) {
                    sum += floatmap.data[y + ky][x + kx] * kernel.data[ky][kx];
                }
            }
            result.data[y][x] = sum; // No normalization - just sum
        }
    }
    return result;
}

FloatMap make_gaussian_kernel(int size, float sigma) {
    assert(size % 2 == 1 && "Kernel size must be odd");

    FloatMap kernel(size, size);
    int center = size / 2;

    // Generate Gaussian weights
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int dx = x - center;
            int dy = y - center;

            // 2D Gaussian function: G(x,y) = exp(-(x²+y²)/(2σ²))
            float value = std::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));

            // Store as float (normalized)
            kernel.data[y][x] = value;
        }
    }

    return kernel;
}

FloatMap gaussian_blur(const FloatMap &floatmap, const int kernel_size, const float sigma) {
    FloatMap result = border_extend_floatmap(floatmap, kernel_size / 2);

    // Create Gaussian kernel
    FloatMap kernel = make_gaussian_kernel(kernel_size, sigma);

    // Apply Gaussian blur
    result = apply_kernel_as_weighted_average(result, kernel);
    return result;
}

// This implementation is not very elegant
float find_max_in_floatmap(const FloatMap &floatmap) {
    // nested loop
    float max_val = floatmap.data[0][0];
    for (int y = 0; y < floatmap.height; y++) {
        for (int x = 0; x < floatmap.width; x++) {
            if (floatmap.data[y][x] > max_val) {
                max_val = floatmap.data[y][x];
            }
        }
    }
    return max_val;
}

float find_min_in_floatmap(const FloatMap &floatmap) {
    // nested loop
    float min_val = floatmap.data[0][0];
    for (int y = 0; y < floatmap.height; y++) {
        for (int x = 0; x < floatmap.width; x++) {
            if (floatmap.data[y][x] < min_val) {
                min_val = floatmap.data[y][x];
            }
        }
    }
    return min_val;
}

FloatMap calculate_magnitude(const FloatMap &sobel_horizontal, const FloatMap &sobel_vertical) {
    FloatMap magnitude(sobel_vertical.width, sobel_vertical.height);
    for (int y = 0; y < magnitude.height; y++) {
        for (int x = 0; x < magnitude.width; x++) {
            float gx = sobel_horizontal.data[y][x];
            float gy = sobel_vertical.data[y][x];
            magnitude.data[y][x] = std::sqrt(gx * gx + gy * gy);
        }
    }
    return magnitude;
}

FloatMap calculate_direction(const FloatMap &sobel_horizontal_image, const FloatMap &sobel_vertical_image) {
    FloatMap direction(sobel_vertical_image.width, sobel_vertical_image.height);
    for (int y = 0; y < direction.height; y++) {
        for (int x = 0; x < direction.width; x++) {
            float gx = sobel_horizontal_image.data[y][x];
            float gy = sobel_vertical_image.data[y][x];
            direction.data[y][x] = std::atan2(gy, gx);
        }
    }
    return direction;
}

Bitmap create_bitmap_from_floatmap(const FloatMap &floatmap, Bitmap &bitmap) {
    /*
    Find max/min in floatmap
    Create linear map from max/min to 0/255
    Apply linear map to floatmap
    Return bitmap
     */
    float max_val = find_max_in_floatmap(floatmap);
    float min_val = find_min_in_floatmap(floatmap);
    float range = max_val - min_val;
    for (int y = 0; y < floatmap.height; y++) {
        for (int x = 0; x < floatmap.width; x++) {
            bitmap.data[y][x] = static_cast<uint8_t>((floatmap.data[y][x] - min_val) / range * 255);
        }
    }
    return bitmap;
}
