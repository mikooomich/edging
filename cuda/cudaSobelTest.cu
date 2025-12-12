#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include "../bitmap.h"
#include "cudaSobel.cu"  // 包含算法和数据结构

// 编译命令（需要链接 bitmap.cpp）:
// nvcc local_workspace/cudaSobelTest.cu local_workspace/cudaSobel.cu bitmap.cpp -o cuso_test && ./cuso_test

/**
 * @brief Compare two FloatMap objects with tolerance
 * 
 * @param expected Expected FloatMap (ground truth)
 * @param product Actual FloatMap (to be tested)
 * @param tolerance Tolerance for floating point comparison
 * @return bool True if all pixels match within tolerance, false otherwise
 * @throws std::runtime_error If dimensions don't match
 * 
 * @note If test fails, error messages are printed to stdout
 * @note If test passes, nothing is printed (caller decides whether to report)
 */
bool compareFloatMaps(
    const FloatMap& expected,
    const FloatMap& product,
    float tolerance = 1e-5f
) {
    // 检查尺寸是否匹配
    if (expected.width != product.width || expected.height != product.height) {
        fprintf(stderr, "Error: FloatMap dimensions don't match: "
                "expected (%d x %d), got (%d x %d)\n",
                expected.width, expected.height,
                product.width, product.height);
        throw std::runtime_error("FloatMap dimensions don't match");
    }
    
    int total_pixels = expected.width * expected.height;
    int mismatch_count = 0;
    const int max_mismatches_to_report = 10;
    
    // 逐像素比较
    for (int y = 0; y < expected.height; y++) {
        for (int x = 0; x < expected.width; x++) {
            float expected_val = expected.data[y][x];
            float product_val = product.data[y][x];
            float diff = std::abs(expected_val - product_val);
            
            if (diff > tolerance) {
                if (mismatch_count < max_mismatches_to_report) {
                    printf("Mismatch at (%d, %d): expected=%.6f, got=%.6f, diff=%.6f\n",
                           x, y, expected_val, product_val, diff);
                }
                mismatch_count++;
            }
        }
    }
    
    // 如果测试失败，输出统计信息
    if (mismatch_count > 0) {
        printf("✗ FAIL: Found %d mismatches out of %d pixels\n",
               mismatch_count, total_pixels);
        printf("  Tolerance: %.2e\n", tolerance);
        if (mismatch_count >= max_mismatches_to_report) {
            printf("  (Only first %d mismatches shown)\n", max_mismatches_to_report);
        }
        return false;
    }
    
    // 测试通过，不输出任何信息
    return true;
}

int bitmap_test(){
    printf("=== Testing bitmap.cpp I/O functionality ===\n\n");
    
    // ========== 使用 bitmap.cpp 创建和保存图像 ==========
    
    // 1. 创建一个简单的测试图像 (200x200)
    int width = 200;
    int height = 200;
    FloatMap testImage(width, height);
    
    printf("Creating %dx%d test image...\n", width, height);
    
    // 2. 填充测试数据：创建一个简单的渐变图案
    // 从左上角(0,0)到右下角(1,1)的线性渐变
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 计算渐变值：x方向从0到1，y方向从0到1
            float value = (float)(x + y) / (width + height);
            // 确保值在 [0, 1] 范围内
            testImage.data[y][x] = value;
        }
    }
    
    printf("Image data filled with gradient pattern.\n");
    
    // 3. 保存图像到 local_workspace 文件夹
    std::string outputPath = "test_image";  // 会保存为 test_image.png
    printf("Saving image to: %s.png\n", outputPath.c_str());
    
    save_floatmap_as(testImage, outputPath);
    
    printf("\n=== I/O test completed successfully! ===\n");
    printf("Check 'test_image.png' in local_workspace folder.\n");
    
    return 0;
}

int GPUImage_lifecycle_test(const std::string& image_path) {
    printf("=== Loading image to GPU ===\n\n");
    
    // ========== 1. 使用 bitmap 库加载图像 ==========
    printf("Loading image from: %s\n", image_path.c_str());
    FloatMap* image = new FloatMap(load_image_grayscale(image_path));
    
    printf("Image loaded successfully!\n");
    printf("  Dimensions: %d x %d\n", image->width, image->height);
    printf("  Total pixels: %d\n", image->width * image->height);
    
    // 保存 FloatMap 的原始信息用于后续检查
    int original_width = image->width;
    int original_height = image->height;
    FloatMap* original_floatmap_ptr = image;
    
    // ========== 2. 使用作用域块管理 GPUImage 生命周期 ==========
    printf("\n=== Entering scope block for GPUImage ===\n");
    
    // 在作用域块开始前检查 FloatMap
    printf("\nBefore GPUImage creation:\n");
    if (image != NULL && image->width == original_width && 
        image->height == original_height && !image->data.empty()) {
        printf("  FloatMap object is valid\n");
        printf("  FloatMap pointer: %p\n", (void*)image);
        printf("  FloatMap dimensions: %d x %d\n", image->width, image->height);
    } else {
        printf("  Warning: FloatMap object is invalid\n");
    }
    
    {
        // ========== 3. 使用 GPUImage 构造函数准备数据 ==========
        printf("\nCreating GPUImage object (using constructor)...\n");
        try {
            GPUImage gpu_img(image);  // 构造函数自动完成：分配内存、展平数据、复制到GPU
            // 注意：GPUImage 现在拥有 FloatMap 的所有权
            // 如果构造失败，会抛出异常
            
            printf("  GPUImage created successfully!\n");
            printf("  GPU memory allocated: %.2f MB\n", 
                   gpu_img.size_bytes / (1024.0f * 1024.0f));
            printf("  GPU pointer: %p\n", (void*)gpu_img.device_data);
            printf("  GPU memory size: %zu bytes\n", gpu_img.size_bytes);
        
            // 检查 FloatMap 对象是否仍然有效（在 GPUImage 创建后，作用域块内）
            printf("\nInside scope block (GPUImage exists):\n");
            if (image != NULL && image->width == original_width && 
                image->height == original_height && !image->data.empty()) {
                printf("  FloatMap object is still valid (owned by GPUImage)\n");
                printf("  FloatMap pointer: %p\n", (void*)image);
                printf("  FloatMap dimensions: %d x %d\n", image->width, image->height);
            } else {
                printf("  Warning: FloatMap object may have been modified\n");
            }
            
            printf("\n=== Image loaded to GPU successfully! ===\n");
            
            // ========== 4. 等待用户输入确认 ==========
            printf("\nPress Enter to continue and release GPU memory...\n");
            std::cin.get();  // 等待用户按回车
            
            // 在作用域块结束前检查 FloatMap
            printf("\nBefore scope block ends (GPUImage still exists):\n");
            if (image != NULL && image->width == original_width && 
                image->height == original_height && !image->data.empty()) {
                printf("  FloatMap object is still valid\n");
                printf("  FloatMap pointer: %p\n", (void*)image);
            } else {
                printf("  Warning: FloatMap object may have been modified\n");
            }
            
            printf("\nLeaving scope block (GPUImage destructor will be called)...\n");
        } catch (const GPUImageException& e) {
            fprintf(stderr, "Error: Failed to create GPUImage: %s\n", e.what());
            delete image;  // 清理 FloatMap
            return 1;
        }
    }  // GPUImage 析构函数在这里被调用，FloatMap 也会被销毁
    
    // ========== 5. 检查 FloatMap 对象是否已销毁（作用域块结束后）==========
    printf("\nAfter scope block ends (GPUImage destroyed):\n");
    printf("Checking if FloatMap object is still accessible...\n");
    
    // 检查指针是否仍然指向同一个地址（对象已被销毁，但指针可能仍然指向该地址）
    if (image == original_floatmap_ptr) {
        printf("  FloatMap pointer still points to same address: %p\n", (void*)image);
        printf("  WARNING: FloatMap object has been destroyed by GPUImage destructor\n");
        printf("  WARNING: Accessing this pointer is undefined behavior\n");
        printf("  FloatMap is NO LONGER AVAILABLE (destroyed)\n");
        
        // 将指针设为 NULL，明确表示对象已销毁且不可用
        image = NULL;
        printf("  FloatMap pointer set to NULL (object is destroyed and no longer available)\n");
    } else {
        printf("  FloatMap pointer has changed or object has been destroyed\n");
    }
    
    // 验证 FloatMap 确实不再可用
    if (image == NULL) {
        printf("  Confirmed: FloatMap pointer is NULL - object is destroyed and no longer available\n");
    } else {
        printf("  WARNING: FloatMap pointer is not NULL, but object may be destroyed\n");
    }
    
    printf("\n=== Cleanup completed ===\n");
    printf("Note: GPUImage destructor has been called automatically\n");
    printf("Note: FloatMap object has been destroyed by GPUImage destructor\n");
    printf("Note: FloatMap is no longer accessible after GPUImage destruction\n");
    
    return 0;
}

/**
 * @brief Test function for GPU Gaussian blur
 * 
 * Tests the CUDA implementation of convolution with weighted average
 * by comparing it with the CPU version.
 */
int test_gpu_gaussian_blur(const std::string& image_path) {
    printf("=== Testing GPU Gaussian Blur ===\n\n");
    
    // 1. 加载图像
    printf("Loading image from: %s\n", image_path.c_str());
    FloatMap* input_floatmap = new FloatMap(load_image_grayscale(image_path));
    
    printf("Image loaded successfully!\n");
    printf("  Dimensions: %d x %d\n", input_floatmap->width, input_floatmap->height);
    printf("  Total pixels: %d\n", input_floatmap->width * input_floatmap->height);
    
    // 2. 创建输入 GPUImage
    printf("\nCreating input GPUImage...\n");
    try {
        GPUImage input_img(input_floatmap);
        // 如果构造失败，会抛出异常
        
        printf("  Input GPUImage created successfully\n");
        printf("  GPU memory allocated: %.2f MB\n", 
               input_img.size_bytes / (1024.0f * 1024.0f));
        
        // 3. 生成高斯核
    int kernel_size = 11;
    float sigma = 0.2f;
    printf("\nGenerating Gaussian kernel...\n");
    printf("  Kernel size: %d x %d\n", kernel_size, kernel_size);
    printf("  Sigma: %.2f\n", sigma);
    
    FloatMap kernel = make_gaussian_kernel(kernel_size, sigma);
    
    // 计算 kernel_sum
    float kernel_sum = 0.0f;
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            kernel_sum += kernel.data[y][x];
        }
    }
    printf("  Kernel sum: %.6f\n", kernel_sum);
    
    // 4. 在 CPU 上执行（用于对比）
    printf("\n=== CPU Version (for comparison) ===\n");
    // FloatMap cpu_result = gaussian_blur(*input_floatmap, kernel_size, sigma);
    FloatMap ker = make_gaussian_kernel(kernel_size, sigma);
    // Apply Gaussian blur
    FloatMap cpu_result = apply_kernel_as_weighted_average(*input_floatmap, kernel);
    printf("CPU convolution completed\n");
    printf("  Output dimensions: %d x %d\n", cpu_result.width, cpu_result.height);
    
    // 5. 在 GPU 上执行
    printf("\n=== GPU Version ===\n");
    GPUImage output_img;  // 创建空对象（默认构造函数，device_data 为 NULL）
    // 函数内部会根据计算出的尺寸构造新的 GPUImage 对象
    
    cudaError_t err = apply_kernel_as_weighted_average_cuda(
        input_img, &output_img, kernel, kernel_sum
    );
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: GPU convolution failed: %s\n", 
                cudaGetErrorString(err));
        return 1;
    }
    
    printf("GPU convolution completed\n");
    printf("  Output dimensions: %d x %d\n", output_img.width, output_img.height);
    
    // 6. 将 GPU 结果复制回 CPU 并比较
    printf("\n=== Comparing Results ===\n");
    
    try {
        // 使用 makeFloatMap() 成员函数将 GPU 数据转换为 FloatMap
        FloatMap gpu_result = output_img.makeFloatMap();
        
        // 使用 compareFloatMaps() 函数比较结果
        float tolerance = 1e-5f;
        bool test_passed = compareFloatMaps(cpu_result, gpu_result, tolerance);
        
        // 8. 报告结果
        printf("\n=== Test Results ===\n");
        if (test_passed) {
            printf("✓ PASS: GPU result matches CPU result exactly (tolerance: %.2e)\n", 
                   tolerance);
            printf("  Total pixels compared: %d\n", 
                   cpu_result.width * cpu_result.height);
        }
        // 如果测试失败，compareFloatMaps 已经输出了错误信息
        
        printf("\n=== Test completed ===\n");
        
        return test_passed ? 0 : 1;
    } catch (const GPUImageException& e) {
        fprintf(stderr, "Error: Failed to copy GPU data to FloatMap: %s\n", e.what());
        return 1;
    } catch (const std::runtime_error& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    } catch (const GPUImageException& e) {
        fprintf(stderr, "Error: Failed to create input GPUImage: %s\n", e.what());
        delete input_floatmap;  // 清理 FloatMap
        return 1;
    }
}

/**
 * @brief Test function for GPU Sobel vertical kernel (apply_kernel_as_sum)
 * 
 * Tests the CUDA implementation of convolution without normalization
 * by comparing it with the CPU version using vertical Sobel kernel.
 */
int test_gpu_sobel_vertical(const std::string& image_path) {
    printf("=== Testing GPU Sobel Vertical (apply_kernel_as_sum) ===\n\n");
    
    // 1. 加载图像
    printf("Loading image from: %s\n", image_path.c_str());
    FloatMap* input_floatmap = new FloatMap(load_image_grayscale(image_path));
    
    printf("Image loaded successfully!\n");
    printf("  Dimensions: %d x %d\n", input_floatmap->width, input_floatmap->height);
    printf("  Total pixels: %d\n", input_floatmap->width * input_floatmap->height);
    
    // 2. 创建输入 GPUImage
    printf("\nCreating input GPUImage...\n");
    try {
        GPUImage input_img(input_floatmap);
        // 如果构造失败，会抛出异常
        
        printf("  Input GPUImage created successfully\n");
        printf("  GPU memory allocated: %.2f MB\n", 
               input_img.size_bytes / (1024.0f * 1024.0f));
        
        // 3. 生成 vertical Sobel 核
        printf("\nGenerating vertical Sobel kernel...\n");
        FloatMap sobel_vertical_kernel = get_sobel_kernel(true);  // true = vertical
        printf("  Kernel size: %d x %d\n", sobel_vertical_kernel.width, sobel_vertical_kernel.height);
        
        // 4. 在 CPU 上执行（用于对比）
        printf("\n=== CPU Version (for comparison) ===\n");
        FloatMap cpu_result = apply_kernel_as_sum(*input_floatmap, sobel_vertical_kernel);
        printf("CPU convolution completed\n");
        printf("  Output dimensions: %d x %d\n", cpu_result.width, cpu_result.height);
        
        // 5. 在 GPU 上执行
        printf("\n=== GPU Version ===\n");
        GPUImage output_img;  // 创建空对象（默认构造函数，device_data 为 NULL）
        // 函数内部会根据计算出的尺寸构造新的 GPUImage 对象
        
        cudaError_t err = apply_kernel_as_sum_cuda(
            input_img, &output_img, sobel_vertical_kernel
        );
        
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: GPU convolution failed: %s\n", 
                    cudaGetErrorString(err));
            return 1;
        }
        
        printf("GPU convolution completed\n");
        printf("  Output dimensions: %d x %d\n", output_img.width, output_img.height);
        
        // 6. 将 GPU 结果复制回 CPU 并比较
        printf("\n=== Comparing Results ===\n");
        
        try {
            // 使用 makeFloatMap() 成员函数将 GPU 数据转换为 FloatMap
            FloatMap gpu_result = output_img.makeFloatMap();
            
            // 使用 compareFloatMaps() 函数比较结果
            float tolerance = 1e-5f;
            bool test_passed = compareFloatMaps(cpu_result, gpu_result, tolerance);
            
            // 8. 报告结果
            printf("\n=== Test Results ===\n");
            if (test_passed) {
                printf("✓ PASS: GPU result matches CPU result exactly (tolerance: %.2e)\n", 
                       tolerance);
                printf("  Total pixels compared: %d\n", 
                       cpu_result.width * cpu_result.height);
            }
            // 如果测试失败，compareFloatMaps 已经输出了错误信息
            
            printf("\n=== Test completed ===\n");
            
            return test_passed ? 0 : 1;
        } catch (const GPUImageException& e) {
            fprintf(stderr, "Error: Failed to copy GPU data to FloatMap: %s\n", e.what());
            return 1;
        } catch (const std::runtime_error& e) {
            fprintf(stderr, "Error: %s\n", e.what());
            return 1;
        }
    } catch (const GPUImageException& e) {
        fprintf(stderr, "Error: Failed to create input GPUImage: %s\n", e.what());
        delete input_floatmap;  // 清理 FloatMap
        return 1;
    }
}

int test_gpu_sobel_magnitude(const std::string& image_path) {
    return 0;
}

/**
 * @brief Test function for GPU calculate_magnitude
 * 
 * Tests the CUDA implementation of magnitude calculation
 * by comparing it with the CPU version.
 * 
 * Note: The input gradients (sobel_horizontal and sobel_vertical) are assumed
 * to be correctly computed. This test only verifies the magnitude calculation.
 */
int test_gpu_calculate_magnitude(const std::string& image_path) {
    printf("=== Testing GPU Calculate Magnitude ===\n\n");
    
    // 1. 加载图像（假设前面的步骤都成功，这里只是为了获取测试数据）
    printf("Loading image from: %s\n", image_path.c_str());
    FloatMap* input_floatmap = new FloatMap(load_image_grayscale(image_path));
    
    printf("Image loaded successfully!\n");
    printf("  Dimensions: %d x %d\n", input_floatmap->width, input_floatmap->height);
    
    // 2. 生成测试用的梯度数据（使用 CPU 版本生成，假设这些步骤都正确）
    printf("\nGenerating test gradients (using CPU version)...\n");
    
    // 扩展边界
    FloatMap extended = border_extend_floatmap(*input_floatmap, 1);
    
    // 生成 Sobel 核
    FloatMap sobel_horizontal_kernel = get_sobel_kernel(false);  // horizontal
    FloatMap sobel_vertical_kernel = get_sobel_kernel(true);      // vertical
    
    // 计算梯度（CPU 版本，假设正确）
    FloatMap sobel_horizontal_cpu = apply_kernel_as_sum(extended, sobel_horizontal_kernel);
    FloatMap sobel_vertical_cpu = apply_kernel_as_sum(extended, sobel_vertical_kernel);
    
    printf("  Horizontal gradient dimensions: %d x %d\n", 
           sobel_horizontal_cpu.width, sobel_horizontal_cpu.height);
    printf("  Vertical gradient dimensions: %d x %d\n", 
           sobel_vertical_cpu.width, sobel_vertical_cpu.height);
    
    // 3. 在 CPU 上计算 magnitude（用于对比）
    printf("\n=== CPU Version (for comparison) ===\n");
    FloatMap magnitude_cpu = calculate_magnitude(sobel_horizontal_cpu, sobel_vertical_cpu);
    printf("CPU magnitude calculation completed\n");
    printf("  Output dimensions: %d x %d\n", magnitude_cpu.width, magnitude_cpu.height);
    
    // 4. 在 GPU 上计算 magnitude
    printf("\n=== GPU Version ===\n");
    try {
        // 创建 GPUImage 对象用于梯度数据
        FloatMap* sobel_h_fm = new FloatMap(sobel_horizontal_cpu);
        FloatMap* sobel_v_fm = new FloatMap(sobel_vertical_cpu);
        
        GPUImage sobel_horizontal_gpu(sobel_h_fm);
        GPUImage sobel_vertical_gpu(sobel_v_fm);
        
        printf("  Gradient GPUImages created successfully\n");
        
        // 创建空的输出对象
        GPUImage magnitude_gpu;  // 创建空对象（默认构造函数，device_data 为 NULL）
        
        cudaError_t err = calculate_magnitude_cuda(
            sobel_horizontal_gpu, sobel_vertical_gpu, &magnitude_gpu
        );
        
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: GPU magnitude calculation failed: %s\n", 
                    cudaGetErrorString(err));
            delete input_floatmap;
            return 1;
        }
        
        printf("GPU magnitude calculation completed\n");
        printf("  Output dimensions: %d x %d\n", 
               magnitude_gpu.width, magnitude_gpu.height);
        
        // 5. 将 GPU 结果复制回 CPU 并比较
        printf("\n=== Comparing Results ===\n");
        
        try {
            // 使用 makeFloatMap() 成员函数将 GPU 数据转换为 FloatMap
            FloatMap magnitude_gpu_result = magnitude_gpu.makeFloatMap();
            
            // 使用 compareFloatMaps() 函数比较结果
            float tolerance = 1e-5f;
            bool test_passed = compareFloatMaps(magnitude_cpu, magnitude_gpu_result, tolerance);
            
            // 6. 报告结果
            printf("\n=== Test Results ===\n");
            if (test_passed) {
                printf("✓ PASS: GPU result matches CPU result exactly (tolerance: %.2e)\n", 
                       tolerance);
                printf("  Total pixels compared: %d\n", 
                       magnitude_cpu.width * magnitude_cpu.height);
            }
            // 如果测试失败，compareFloatMaps 已经输出了错误信息
            
            printf("\n=== Test completed ===\n");
            
            delete input_floatmap;
            return test_passed ? 0 : 1;
        } catch (const GPUImageException& e) {
            fprintf(stderr, "Error: Failed to copy GPU data to FloatMap: %s\n", e.what());
            delete input_floatmap;
            return 1;
        } catch (const std::runtime_error& e) {
            fprintf(stderr, "Error: %s\n", e.what());
            delete input_floatmap;
            return 1;
        }
    } catch (const GPUImageException& e) {
        fprintf(stderr, "Error: Failed to create gradient GPUImages: %s\n", e.what());
        delete input_floatmap;
        return 1;
    }
}

int main() {
    // bitmap_test();
    // GPUImage_lifecycle_test("test_image.png");
    // test_gpu_gaussian_blur("test_image.png");
    // test_gpu_sobel_vertical("test_image.png");
    test_gpu_calculate_magnitude("test_image.png");
    return 0;
}

