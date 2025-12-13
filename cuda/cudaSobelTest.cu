#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include "bitmap.h"
#include "cudaSobel.cu"


// nvcc cuda/cudaSobelTest.cu cuda/cudaSobel.cu bitmap.cpp -o temp/cuso_test && ./temp/cuso_test

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
    // Check if dimensions match
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
    
    // Compare pixel by pixel
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
    
    // If test fails, output statistics
    if (mismatch_count > 0) {
        printf("✗ FAIL: Found %d mismatches out of %d pixels\n",
               mismatch_count, total_pixels);
        printf("  Tolerance: %.2e\n", tolerance);
        if (mismatch_count >= max_mismatches_to_report) {
            printf("  (Only first %d mismatches shown)\n", max_mismatches_to_report);
        }
        return false;
    }
    
    // Test passed, no output
    return true;
}

int bitmap_test(){
    printf("=== Testing bitmap.cpp I/O functionality ===\n\n");
    
    // ========== Create and save image using bitmap.cpp ==========
    
    // 1. Create a simple test image (200x200)
    int width = 200;
    int height = 200;
    FloatMap testImage(width, height);
    
    printf("Creating %dx%d test image...\n", width, height);
    
    // 2. Fill test data: create a simple gradient pattern
    // Linear gradient from top-left (0,0) to bottom-right (1,1)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Calculate gradient value: x direction from 0 to 1, y direction from 0 to 1
            float value = (float)(x + y) / (width + height);
            // Ensure value is in [0, 1] range
            testImage.data[y][x] = value;
        }
    }
    
    printf("Image data filled with gradient pattern.\n");
    
    // 3. Save image to temp folder
    std::string outputPath = "temp/test_image";  // Will be saved as test_image.png
    printf("Saving image to: %s.png\n", outputPath.c_str());
    
    save_floatmap_as(testImage, outputPath);
    
    printf("\n=== I/O test completed successfully! ===\n");
    printf("Check 'test_image.png' in temp folder.\n");
    
    return 0;
}

int GPUImage_lifecycle_test(const std::string& image_path) {
    printf("=== Loading image to GPU ===\n\n");
    
    // ========== 1. Load image using bitmap library ==========
    printf("Loading image from: %s\n", image_path.c_str());
    FloatMap* image = new FloatMap(load_image_grayscale(image_path));
    
    printf("Image loaded successfully!\n");
    printf("  Dimensions: %d x %d\n", image->width, image->height);
    printf("  Total pixels: %d\n", image->width * image->height);
    
    // Save original FloatMap information for later checks
    int original_width = image->width;
    int original_height = image->height;
    FloatMap* original_floatmap_ptr = image;
    
    // ========== 2. Use scope block to manage GPUImage lifecycle ==========
    printf("\n=== Entering scope block for GPUImage ===\n");
    
    // Check FloatMap before scope block starts
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
        // ========== 3. Use GPUImage constructor to prepare data ==========
        printf("\nCreating GPUImage object (using constructor)...\n");
        try {
            GPUImage gpu_img(image);  // Constructor automatically: allocates memory, flattens data, copies to GPU
            // Note: GPUImage now owns the FloatMap
            // If construction fails, an exception will be thrown
            
            printf("  GPUImage created successfully!\n");
            printf("  GPU memory allocated: %.2f MB\n", 
                   gpu_img.size_bytes / (1024.0f * 1024.0f));
            printf("  GPU pointer: %p\n", (void*)gpu_img.device_data);
            printf("  GPU memory size: %zu bytes\n", gpu_img.size_bytes);
        
            // Check if FloatMap object is still valid (after GPUImage creation, inside scope block)
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
            
            // ========== 4. Wait for user input confirmation ==========
            printf("\nPress Enter to continue and release GPU memory...\n");
            std::cin.get();  // Wait for user to press Enter
            
            // Check FloatMap before scope block ends
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
            delete image;  // Clean up FloatMap
            return 1;
        }
    }  // GPUImage destructor is called here, FloatMap will also be destroyed
    
    // ========== 5. Check if FloatMap object has been destroyed (after scope block ends) ==========
    printf("\nAfter scope block ends (GPUImage destroyed):\n");
    printf("Checking if FloatMap object is still accessible...\n");
    
    // Check if pointer still points to the same address (object has been destroyed, but pointer may still point to that address)
    if (image == original_floatmap_ptr) {
        printf("  FloatMap pointer still points to same address: %p\n", (void*)image);
        printf("  WARNING: FloatMap object has been destroyed by GPUImage destructor\n");
        printf("  WARNING: Accessing this pointer is undefined behavior\n");
        printf("  FloatMap is NO LONGER AVAILABLE (destroyed)\n");
        
        // Set pointer to NULL to clearly indicate object is destroyed and no longer available
        image = NULL;
        printf("  FloatMap pointer set to NULL (object is destroyed and no longer available)\n");
    } else {
        printf("  FloatMap pointer has changed or object has been destroyed\n");
    }
    
    // Verify FloatMap is indeed no longer available
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
    
    // 1. Load image
    printf("Loading image from: %s\n", image_path.c_str());
    FloatMap* input_floatmap = new FloatMap(load_image_grayscale(image_path));
    
    printf("Image loaded successfully!\n");
    printf("  Dimensions: %d x %d\n", input_floatmap->width, input_floatmap->height);
    printf("  Total pixels: %d\n", input_floatmap->width * input_floatmap->height);
    
    // 2. Create input GPUImage
    printf("\nCreating input GPUImage...\n");
    try {
        GPUImage input_img(input_floatmap);
        // If construction fails, an exception will be thrown
        
        printf("  Input GPUImage created successfully\n");
        printf("  GPU memory allocated: %.2f MB\n", 
               input_img.size_bytes / (1024.0f * 1024.0f));
        
        // 3. Generate Gaussian kernel
    int kernel_size = 11;
    float sigma = 0.2f;
    printf("\nGenerating Gaussian kernel...\n");
    printf("  Kernel size: %d x %d\n", kernel_size, kernel_size);
    printf("  Sigma: %.2f\n", sigma);
    
    FloatMap kernel = make_gaussian_kernel(kernel_size, sigma);
    
    // Calculate kernel_sum
    float kernel_sum = 0.0f;
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            kernel_sum += kernel.data[y][x];
        }
    }
    printf("  Kernel sum: %.6f\n", kernel_sum);
    
    // 4. Execute on CPU (for comparison)
    printf("\n=== CPU Version (for comparison) ===\n");
    // FloatMap cpu_result = gaussian_blur(*input_floatmap, kernel_size, sigma);
    FloatMap ker = make_gaussian_kernel(kernel_size, sigma);
    // Apply Gaussian blur
    FloatMap cpu_result = apply_kernel_as_weighted_average(*input_floatmap, kernel);
    printf("CPU convolution completed\n");
    printf("  Output dimensions: %d x %d\n", cpu_result.width, cpu_result.height);
    
    // 5. Execute on GPU
    printf("\n=== GPU Version ===\n");
    GPUImage output_img;  // Create empty object (default constructor, device_data is NULL)
    // Function will construct new GPUImage object based on calculated dimensions
    
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
    
    // 6. Copy GPU result back to CPU and compare
    printf("\n=== Comparing Results ===\n");
    
    try {
        // Use makeFloatMap() member function to convert GPU data to FloatMap
        FloatMap gpu_result = output_img.makeFloatMap();
        
        // Use compareFloatMaps() function to compare results
        float tolerance = 1e-5f;
        bool test_passed = compareFloatMaps(cpu_result, gpu_result, tolerance);
        
        // 8. Report results
        printf("\n=== Test Results ===\n");
        if (test_passed) {
            printf("✓ PASS: GPU result matches CPU result exactly (tolerance: %.2e)\n", 
                   tolerance);
            printf("  Total pixels compared: %d\n", 
                   cpu_result.width * cpu_result.height);
        }
        // If test fails, compareFloatMaps has already output error information
        
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
        delete input_floatmap;  // Clean up FloatMap
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
    
    // 1. Load image
    printf("Loading image from: %s\n", image_path.c_str());
    FloatMap* input_floatmap = new FloatMap(load_image_grayscale(image_path));
    
    printf("Image loaded successfully!\n");
    printf("  Dimensions: %d x %d\n", input_floatmap->width, input_floatmap->height);
    printf("  Total pixels: %d\n", input_floatmap->width * input_floatmap->height);
    
    // 2. Create input GPUImage
    printf("\nCreating input GPUImage...\n");
    try {
        GPUImage input_img(input_floatmap);
        // If construction fails, an exception will be thrown
        
        printf("  Input GPUImage created successfully\n");
        printf("  GPU memory allocated: %.2f MB\n", 
               input_img.size_bytes / (1024.0f * 1024.0f));
        
        // 3. Generate vertical Sobel kernel
        printf("\nGenerating vertical Sobel kernel...\n");
        FloatMap sobel_vertical_kernel = get_sobel_kernel(true);  // true = vertical
        printf("  Kernel size: %d x %d\n", sobel_vertical_kernel.width, sobel_vertical_kernel.height);
        
        // 4. Execute on CPU (for comparison)
        printf("\n=== CPU Version (for comparison) ===\n");
        FloatMap cpu_result = apply_kernel_as_sum(*input_floatmap, sobel_vertical_kernel);
        printf("CPU convolution completed\n");
        printf("  Output dimensions: %d x %d\n", cpu_result.width, cpu_result.height);
        
        // 5. Execute on GPU
        printf("\n=== GPU Version ===\n");
        GPUImage output_img;  // Create empty object (default constructor, device_data is NULL)
        // Function will construct new GPUImage object based on calculated dimensions
        
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
        
        // 6. Copy GPU result back to CPU and compare
        printf("\n=== Comparing Results ===\n");
        
        try {
            // Use makeFloatMap() member function to convert GPU data to FloatMap
            FloatMap gpu_result = output_img.makeFloatMap();
            
            // Use compareFloatMaps() function to compare results
            float tolerance = 1e-5f;
            bool test_passed = compareFloatMaps(cpu_result, gpu_result, tolerance);
            
            // 8. Report results
            printf("\n=== Test Results ===\n");
            if (test_passed) {
                printf("✓ PASS: GPU result matches CPU result exactly (tolerance: %.2e)\n", 
                       tolerance);
                printf("  Total pixels compared: %d\n", 
                       cpu_result.width * cpu_result.height);
            }
            // If test fails, compareFloatMaps has already output error information
            
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
        delete input_floatmap;  // Clean up FloatMap
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
    
    // 1. Load image (assuming previous steps succeeded, this is just to get test data)
    printf("Loading image from: %s\n", image_path.c_str());
    FloatMap* input_floatmap = new FloatMap(load_image_grayscale(image_path));
    
    printf("Image loaded successfully!\n");
    printf("  Dimensions: %d x %d\n", input_floatmap->width, input_floatmap->height);
    
    // 2. Generate test gradient data (using CPU version, assuming these steps are correct)
    printf("\nGenerating test gradients (using CPU version)...\n");
    
    // Extend borders
    FloatMap extended = border_extend_floatmap(*input_floatmap, 1);
    
    // Generate Sobel kernels
    FloatMap sobel_horizontal_kernel = get_sobel_kernel(false);  // horizontal
    FloatMap sobel_vertical_kernel = get_sobel_kernel(true);      // vertical
    
    // Calculate gradients (CPU version, assumed correct)
    FloatMap sobel_horizontal_cpu = apply_kernel_as_sum(extended, sobel_horizontal_kernel);
    FloatMap sobel_vertical_cpu = apply_kernel_as_sum(extended, sobel_vertical_kernel);
    
    printf("  Horizontal gradient dimensions: %d x %d\n", 
           sobel_horizontal_cpu.width, sobel_horizontal_cpu.height);
    printf("  Vertical gradient dimensions: %d x %d\n", 
           sobel_vertical_cpu.width, sobel_vertical_cpu.height);
    
    // 3. Calculate magnitude on CPU (for comparison)
    printf("\n=== CPU Version (for comparison) ===\n");
    FloatMap magnitude_cpu = calculate_magnitude(sobel_horizontal_cpu, sobel_vertical_cpu);
    printf("CPU magnitude calculation completed\n");
    printf("  Output dimensions: %d x %d\n", magnitude_cpu.width, magnitude_cpu.height);
    
    // 4. Calculate magnitude on GPU
    printf("\n=== GPU Version ===\n");
    try {
        // Create GPUImage objects for gradient data
        FloatMap* sobel_h_fm = new FloatMap(sobel_horizontal_cpu);
        FloatMap* sobel_v_fm = new FloatMap(sobel_vertical_cpu);
        
        GPUImage sobel_horizontal_gpu(sobel_h_fm);
        GPUImage sobel_vertical_gpu(sobel_v_fm);
        
        printf("  Gradient GPUImages created successfully\n");
        
        // Create empty output object
        GPUImage magnitude_gpu;  // Create empty object (default constructor, device_data is NULL)
        
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
        
        // 5. Copy GPU result back to CPU and compare
        printf("\n=== Comparing Results ===\n");
        
        try {
            // Use makeFloatMap() member function to convert GPU data to FloatMap
            FloatMap magnitude_gpu_result = magnitude_gpu.makeFloatMap();
            
            // Use compareFloatMaps() function to compare results
            float tolerance = 1e-5f;
            bool test_passed = compareFloatMaps(magnitude_cpu, magnitude_gpu_result, tolerance);
            
            // 6. Report results
            printf("\n=== Test Results ===\n");
            if (test_passed) {
                printf("✓ PASS: GPU result matches CPU result exactly (tolerance: %.2e)\n", 
                       tolerance);
                printf("  Total pixels compared: %d\n", 
                       magnitude_cpu.width * magnitude_cpu.height);
            }
            // If test fails, compareFloatMaps has already output error information
            
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
    std::string image_path = "data/input/image.png";
    bitmap_test();
    GPUImage_lifecycle_test(image_path);
    test_gpu_gaussian_blur(image_path);
    test_gpu_sobel_vertical(image_path);
    test_gpu_calculate_magnitude(image_path);
    return 0;
}

