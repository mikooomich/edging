#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string>
#include "bitmap.h"

// Custom exception class: used for GPUImage construction failures
class GPUImageException : public std::runtime_error {
public:
    GPUImageException(const std::string& msg) : std::runtime_error(msg) {}
};

//nvcc cuda_helloworld.cu -o cullo && ./cullo
// Compilation command (needs to link bitmap.cpp):
// nvcc local_workspace/cudaSobel.cu bitmap.cpp -o cuso && ./cuso
// 
// Note: This file contains algorithms and data structures, test code is in cudaSobelTest.cu

struct GPUImage{

    float* host_data;
    float* device_data;
    int width;
    int height;
    size_t size_bytes;
    int pixel_count;
    FloatMap* floatmap_ptr;  // Owns the FloatMap object

    // Default constructor: creates an empty GPUImage object
    // Used as an output receiver, memory will be allocated inside functions
    GPUImage() 
        : host_data(NULL),
          device_data(NULL),
          width(0),
          height(0),
          size_bytes(0),
          pixel_count(0),
          floatmap_ptr(NULL) {
        // Empty GPUImage, does not allocate any resources
    }

    // Constructor: allocates GPU memory based on dimensions (does not copy data)
    // Used to create output GPUImage objects inside functions
    // Throws GPUImageException if construction fails
    GPUImage(int width, int height)
        : host_data(NULL),
          device_data(NULL),
          width(width),
          height(height),
          size_bytes(width * height * sizeof(float)),
          pixel_count(width * height),
          floatmap_ptr(NULL) {
        
        if (width <= 0 || height <= 0) {
            throw GPUImageException("Invalid dimensions for GPUImage");
        }
        
        // Only allocate GPU memory, do not allocate CPU memory, do not copy data
        cudaError_t err = cudaMalloc(&device_data, size_bytes);
        if (err != cudaSuccess) {
            throw GPUImageException("Failed to allocate GPU memory: " + 
                                   std::string(cudaGetErrorString(err)));
        }
    }

    // Constructor: creates GPUImage from FloatMap, ready to enter CUDA processing pipeline
    // Note: This constructor takes ownership of FloatMap, FloatMap will be destroyed when GPUImage is destructed
    // Throws GPUImageException if construction fails
    GPUImage(FloatMap* image_ptr) 
        : host_data(NULL),
          device_data(NULL),
          width(image_ptr ? image_ptr->width : 0),
          height(image_ptr ? image_ptr->height : 0),
          pixel_count(image_ptr ? image_ptr->width * image_ptr->height : 0),
          size_bytes(image_ptr ? image_ptr->width * image_ptr->height * sizeof(float) : 0),
          floatmap_ptr(image_ptr) {
        
        if (image_ptr == NULL) {
            throw GPUImageException("Null FloatMap pointer");
        }
        
        // 1. Allocate CPU memory
        host_data = (float*)malloc(size_bytes);
        if (host_data == NULL) {
            throw GPUImageException("Failed to allocate host memory");
        }
        
        // 2. Flatten 2D data into 1D array (row-major order)
        try {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    host_data[y * width + x] = image_ptr->data[y][x];
                }
            }
        } catch (...) {
            free(host_data);
            host_data = NULL;
            throw;  // Re-throw exception
        }
        
        // 3. Allocate GPU memory
        cudaError_t err = cudaMalloc(&device_data, size_bytes);
        if (err != cudaSuccess) {
            free(host_data);
            host_data = NULL;
            throw GPUImageException("Failed to allocate GPU memory: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // 4. Copy data from CPU to GPU
        err = cudaMemcpy(device_data, host_data, size_bytes,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(device_data);
            free(host_data);
            device_data = NULL;
            host_data = NULL;
            throw GPUImageException("Failed to copy data to GPU: " + 
                                   std::string(cudaGetErrorString(err)));
        }
    }

    // Move assignment operator: supports *output = GPUImage(...) usage
    GPUImage& operator=(GPUImage&& other) noexcept {
        if (this != &other) {
            // Release current object's resources
            if (device_data != NULL) {
                cudaFree(device_data);
            }
            if (host_data != NULL) {
                free(host_data);
            }
            if (floatmap_ptr != NULL) {
                delete floatmap_ptr;
            }
            
            // Move resources
            device_data = other.device_data;
            host_data = other.host_data;
            width = other.width;
            height = other.height;
            size_bytes = other.size_bytes;
            pixel_count = other.pixel_count;
            floatmap_ptr = other.floatmap_ptr;
            
            // Clear source object
            other.device_data = NULL;
            other.host_data = NULL;
            other.width = 0;
            other.height = 0;
            other.size_bytes = 0;
            other.pixel_count = 0;
            other.floatmap_ptr = NULL;
        }
        return *this;
    }

    // Member function: copies GPU data to FloatMap object
    // Returns an independent FloatMap object with separate lifetime management
    // Throws GPUImageException if it fails
    FloatMap makeFloatMap() const {
        if (device_data == NULL) {
            throw GPUImageException("GPUImage has no device data");
        }
        
        if (width <= 0 || height <= 0) {
            throw GPUImageException("GPUImage has invalid dimensions");
        }
        
        // Create FloatMap object
        FloatMap result(width, height);
        
        // Allocate temporary CPU memory
        float* temp_cpu = (float*)malloc(size_bytes);
        if (temp_cpu == NULL) {
            throw GPUImageException("Failed to allocate temporary CPU memory");
        }
        
        // Copy from GPU to CPU
        cudaError_t err = cudaMemcpy(temp_cpu, device_data, size_bytes,
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            free(temp_cpu);
            throw GPUImageException("Failed to copy data from GPU: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // Convert 1D array to 2D FloatMap (row-major order)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                result.data[y][x] = temp_cpu[y * width + x];
            }
        }
        
        free(temp_cpu);
        return result;  // Return independent object with separate lifetime management
    }

    // Destructor: releases GPU memory, CPU memory, and destroys FloatMap object
    ~GPUImage() {
        // Release GPU memory
        if (device_data != NULL) {
            cudaError_t err = cudaFree(device_data);
            if (err != cudaSuccess) {
                fprintf(stderr, "Warning: Failed to free GPU memory: %s\n",
                        cudaGetErrorString(err));
            }
            device_data = NULL;
        }
        
        // Release CPU memory
        if (host_data != NULL) {
            free(host_data);
            host_data = NULL;
        }
        
        // Destroy FloatMap object (ensures FloatMap is no longer available)
        if (floatmap_ptr != NULL) {
            delete floatmap_ptr;
            floatmap_ptr = NULL;
            // printf("  FloatMap object has been destroyed in GPUImage destructor\n");
        }
    }

};

/**
 * @brief CUDA kernel for applying convolution with weighted average normalization
 * 
 * Each thread processes one output pixel by convolving the input image with the kernel.
 * 
 * @param input Input image data (GPU memory, row-major order)
 * @param output Output image data (GPU memory, row-major order)
 * @param kernel Convolution kernel data (GPU memory, row-major order, flattened 1D array)
 * @param input_width Width of input image
 * @param input_height Height of input image
 * @param kernel_size Size of convolution kernel (assumed to be square)
 * @param kernel_sum Sum of all kernel weights (for normalization)
 */
__global__ void apply_kernel_as_weighted_average(
    const float* input,      // Input image data (GPU memory)
    float* output,           // Output image data (GPU memory)
    const float* kernel,     // Convolution kernel data (GPU memory, flattened)
    int input_width,         // Width of input image
    int input_height,        // Height of input image
    int kernel_size,         // Size of convolution kernel (square kernel)
    float kernel_sum         // Sum of all kernel weights (for normalization)
) {
    // Calculate output pixel coordinates for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate output image dimensions
    int output_width = input_width - kernel_size + 1;
    int output_height = input_height - kernel_size + 1;
    
    // Boundary check: only process valid output pixels
    if (x >= output_width || y >= output_height) {
        return;
    }
    
    // Perform convolution: sum(input * kernel) / kernel_sum
    float sum = 0.0f;
    
    // Iterate over kernel
    for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
            // Calculate input pixel coordinates
            int input_x = x + kx;
            int input_y = y + ky;
            
            // Get input pixel value (row-major order: data[y * width + x])
            float input_value = input[input_y * input_width + input_x];
            
            // Get kernel weight (row-major order: kernel[ky * kernel_size + kx])
            float kernel_weight = kernel[ky * kernel_size + kx];
            
            // Accumulate weighted sum
            sum += input_value * kernel_weight;
        }
    }
    
    // Normalize by kernel sum and store result
    // Output is also row-major order: output[y * output_width + x]
    output[y * output_width + x] = sum / kernel_sum;
}

/**
 * @brief Host wrapper function for applying convolution with weighted average
 * 
 * This function provides an object-oriented interface that accepts GPUImage objects,
 * extracts the necessary information, and calls the actual CUDA kernel.
 * 
 * @param input Input GPUImage (must have valid device_data)
 * @param output Output GPUImage pointer (must point to nullptr, will be constructed inside the function)
 * @param kernel Convolution kernel (CPU FloatMap, will be copied to GPU)
 * @param kernel_sum Sum of all kernel weights (for normalization)
 * @return cudaError_t CUDA error code (cudaSuccess if successful)
 */
cudaError_t apply_kernel_as_weighted_average_cuda(
    GPUImage& input,
    GPUImage* output,
    const FloatMap& kernel,
    float kernel_sum
) {
    // 1. Validate input
    if (input.device_data == NULL) {
        fprintf(stderr, "Error: Input GPUImage has no device data\n");
        return cudaErrorInvalidValue;
    }
    
    if (output == NULL) {
        fprintf(stderr, "Error: Output GPUImage pointer is NULL\n");
        return cudaErrorInvalidValue;
    }
    
    // output must point to an empty object (created via default constructor)
    if (output->device_data != NULL) {
        fprintf(stderr, "Error: Output GPUImage must be empty (device_data must be NULL)\n");
        return cudaErrorInvalidValue;
    }
    
    if (kernel.width != kernel.height) {
        fprintf(stderr, "Error: Kernel must be square\n");
        return cudaErrorInvalidValue;
    }
    
    int kernel_size = kernel.width;
    
    // 2. Calculate output dimensions
    int output_width = input.width - kernel_size + 1;
    int output_height = input.height - kernel_size + 1;
    
    if (output_width <= 0 || output_height <= 0) {
        fprintf(stderr, "Error: Output dimensions invalid (%d x %d)\n", 
                output_width, output_height);
        return cudaErrorInvalidValue;
    }
    
    // 3. Construct output GPUImage object (using constructor with dimensions)
    try {
        *output = GPUImage(output_width, output_height);
    } catch (const GPUImageException& e) {
        fprintf(stderr, "Error: Failed to create output GPUImage: %s\n", e.what());
        return cudaErrorMemoryAllocation;
    }
    
    // 4. Copy kernel to GPU using GPUImage object
    // Create a copy of kernel (GPUImage constructor requires non-const pointer and takes ownership)
    FloatMap* kernel_copy = new FloatMap(kernel.width, kernel.height);
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            kernel_copy->data[y][x] = kernel.data[y][x];
        }
    }
    
    // Use GPUImage to manage kernel's GPU memory (automatically handles flattening and copying)
    // GPUImage will take ownership of kernel_copy and automatically release it on destruction
    // If construction fails, an exception will be thrown
    try {
        GPUImage kernel_gpu(kernel_copy);
        // Note: If construction fails, exception will be thrown, kernel_copy will be automatically deleted by GPUImage destructor (RAII)
        
        // 5. Configure kernel launch parameters
        dim3 blockSize(16, 16);  // 16x16 threads per block
        dim3 gridSize(
            (output_width + blockSize.x - 1) / blockSize.x,
            (output_height + blockSize.y - 1) / blockSize.y
        );
        
        // 6. Launch CUDA kernel
        apply_kernel_as_weighted_average<<<gridSize, blockSize>>>(
            input.device_data,          // Extract from GPUImage
            output->device_data,        // Extract from GPUImage pointer
            kernel_gpu.device_data,     // Extract from GPUImage (kernel data)
            input.width,                // Extract from GPUImage
            input.height,               // Extract from GPUImage
            kernel_size,                // Extract from kernel
            kernel_sum                  // Parameter passed in
        );
        
        // 7. Check kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: CUDA kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return err;
        }
        
        // 8. Wait for kernel to complete
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: CUDA kernel execution failed: %s\n",
                    cudaGetErrorString(err));
            return err;
        }
        
        // 9. kernel_gpu object will be automatically destructed at function end, releasing GPU memory
        
        return cudaSuccess;
    } catch (const GPUImageException& e) {
        fprintf(stderr, "Error: Failed to create GPUImage for kernel: %s\n", e.what());
        // kernel_copy will be automatically deleted by GPUImage destructor (RAII), but if construction fails it won't be deleted
        // Need to manually delete kernel_copy
        delete kernel_copy;
        return cudaErrorMemoryAllocation;
    }
}

/**
 * @brief CUDA kernel for applying convolution without normalization (direct sum)
 * 
 * Each thread processes one output pixel by convolving the input image with the kernel.
 * Unlike apply_kernel_as_weighted_average, this kernel does NOT normalize by kernel sum.
 * 
 * @note This is an internal implementation function, not meant for direct external access.
 * Use apply_kernel_as_sum_cuda() instead.
 * 
 * @param input Input image data (GPU memory, row-major order)
 * @param output Output image data (GPU memory, row-major order)
 * @param kernel Convolution kernel data (GPU memory, row-major order, flattened 1D array)
 * @param input_width Width of input image
 * @param input_height Height of input image
 * @param kernel_size Size of convolution kernel (assumed to be square)
 */
__global__ void _apply_kernel_as_sum_kernel(
    const float* input,      // Input image data (GPU memory)
    float* output,           // Output image data (GPU memory)
    const float* kernel,     // Convolution kernel data (GPU memory, flattened)
    int input_width,         // Width of input image
    int input_height,        // Height of input image
    int kernel_size          // Size of convolution kernel (square kernel)
) {
    // Calculate output pixel coordinates for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate output image dimensions
    int output_width = input_width - kernel_size + 1;
    int output_height = input_height - kernel_size + 1;
    
    // Boundary check: only process valid output pixels
    if (x >= output_width || y >= output_height) {
        return;
    }
    
    // Perform convolution: sum(input * kernel) without normalization
    float sum = 0.0f;
    
    // Iterate over kernel
    for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
            // Calculate input pixel coordinates
            int input_x = x + kx;
            int input_y = y + ky;
            
            // Get input pixel value (row-major order: data[y * width + x])
            float input_value = input[input_y * input_width + input_x];
            
            // Get kernel weight (row-major order: kernel[ky * kernel_size + kx])
            float kernel_weight = kernel[ky * kernel_size + kx];
            
            // Accumulate weighted sum
            sum += input_value * kernel_weight;
        }
    }
    
    // Store result without normalization
    // Output is also row-major order: output[y * output_width + x]
    output[y * output_width + x] = sum;
}

/**
 * @brief Host wrapper function for applying convolution without normalization
 * 
 * This function provides an object-oriented interface that accepts GPUImage objects,
 * extracts the necessary information, and calls the actual CUDA kernel.
 * 
 * @param input Input GPUImage (must have valid device_data)
 * @param output Output GPUImage pointer (must point to empty object, will be constructed inside the function)
 * @param kernel Convolution kernel (CPU FloatMap, will be copied to GPU)
 * @return cudaError_t CUDA error code (cudaSuccess if successful)
 */
cudaError_t apply_kernel_as_sum_cuda(
    GPUImage& input,
    GPUImage* output,
    const FloatMap& kernel
) {
    // 1. Validate input
    if (input.device_data == NULL) {
        fprintf(stderr, "Error: Input GPUImage has no device data\n");
        return cudaErrorInvalidValue;
    }
    
    if (output == NULL) {
        fprintf(stderr, "Error: Output GPUImage pointer is NULL\n");
        return cudaErrorInvalidValue;
    }
    
    // output must point to an empty object (created via default constructor)
    if (output->device_data != NULL) {
        fprintf(stderr, "Error: Output GPUImage must be empty (device_data must be NULL)\n");
        return cudaErrorInvalidValue;
    }
    
    if (kernel.width != kernel.height) {
        fprintf(stderr, "Error: Kernel must be square\n");
        return cudaErrorInvalidValue;
    }
    
    int kernel_size = kernel.width;
    
    // 2. Calculate output dimensions (calculated inside function based on input object)
    int output_width = input.width - kernel_size + 1;
    int output_height = input.height - kernel_size + 1;
    
    if (output_width <= 0 || output_height <= 0) {
        fprintf(stderr, "Error: Output dimensions invalid (%d x %d)\n", 
                output_width, output_height);
        return cudaErrorInvalidValue;
    }
    
    // 3. Construct output GPUImage object (using constructor with dimensions)
    try {
        *output = GPUImage(output_width, output_height);
    } catch (const GPUImageException& e) {
        fprintf(stderr, "Error: Failed to create output GPUImage: %s\n", e.what());
        return cudaErrorMemoryAllocation;
    }
    
    // 4. Copy kernel to GPU using GPUImage object
    // Create a copy of kernel (GPUImage constructor requires non-const pointer and takes ownership)
    FloatMap* kernel_copy = new FloatMap(kernel.width, kernel.height);
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            kernel_copy->data[y][x] = kernel.data[y][x];
        }
    }
    
    // Use GPUImage to manage kernel's GPU memory (automatically handles flattening and copying)
    // GPUImage will take ownership of kernel_copy and automatically release it on destruction
    // If construction fails, an exception will be thrown
    try {
        GPUImage kernel_gpu(kernel_copy);
        // Note: If construction fails, exception will be thrown, kernel_copy will be automatically deleted by GPUImage destructor (RAII)
        
        // 5. Configure kernel launch parameters
        dim3 blockSize(16, 16);  // 16x16 threads per block
        dim3 gridSize(
            (output_width + blockSize.x - 1) / blockSize.x,
            (output_height + blockSize.y - 1) / blockSize.y
        );
        
        // 6. Launch CUDA kernel
        _apply_kernel_as_sum_kernel<<<gridSize, blockSize>>>(
            input.device_data,          // Extract from GPUImage
            output->device_data,        // Extract from GPUImage pointer
            kernel_gpu.device_data,     // Extract from GPUImage (kernel data)
            input.width,                // Extract from GPUImage
            input.height,               // Extract from GPUImage
            kernel_size                 // Extract from kernel
        );
        
        // 7. Check kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: CUDA kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return err;
        }
        
        // 8. Wait for kernel to complete
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: CUDA kernel execution failed: %s\n",
                    cudaGetErrorString(err));
            return err;
        }
        
        // 9. kernel_gpu object will be automatically destructed at function end, releasing GPU memory
        
        return cudaSuccess;
    } catch (const GPUImageException& e) {
        fprintf(stderr, "Error: Failed to create GPUImage for kernel: %s\n", e.what());
        // kernel_copy will be automatically deleted by GPUImage destructor (RAII), but if construction fails it won't be deleted
        // Need to manually delete kernel_copy
        delete kernel_copy;
        return cudaErrorMemoryAllocation;
    }
}

/**
 * @brief CUDA kernel for calculating edge magnitude from Sobel gradients
 * 
 * Each thread processes one pixel by computing: magnitude = sqrt(gx² + gy²)
 * 
 * @note This is an internal implementation function, not meant for direct external access.
 * Use calculate_magnitude_cuda() instead.
 * 
 * @param sobel_horizontal Horizontal gradient data (GPU memory, row-major order)
 * @param sobel_vertical Vertical gradient data (GPU memory, row-major order)
 * @param magnitude Output magnitude data (GPU memory, row-major order)
 * @param width Width of the images
 * @param height Height of the images
 */
__global__ void _calculate_magnitude_kernel(
    const float* sobel_horizontal,  // Horizontal gradient (Gx)
    const float* sobel_vertical,     // Vertical gradient (Gy)
    float* magnitude,                // Output magnitude
    int width,                       // Width of the images
    int height                       // Height of the images
) {
    // Calculate pixel coordinates for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check: only process valid pixels
    if (x >= width || y >= height) {
        return;
    }
    
    // Get gradient values (row-major order: data[y * width + x])
    float gx = sobel_horizontal[y * width + x];
    float gy = sobel_vertical[y * width + x];
    
    // Calculate magnitude: sqrt(gx² + gy²)
    magnitude[y * width + x] = sqrtf(gx * gx + gy * gy);
}

/**
 * @brief Host wrapper function for calculating edge magnitude
 * 
 * This function provides an object-oriented interface that accepts GPUImage objects,
 * extracts the necessary information, and calls the actual CUDA kernel.
 * 
 * @param sobel_horizontal Horizontal gradient GPUImage (must have valid device_data)
 * @param sobel_vertical Vertical gradient GPUImage (must have valid device_data)
 * @param magnitude Output magnitude GPUImage pointer (must point to empty object, will be constructed inside the function)
 * @return cudaError_t CUDA error code (cudaSuccess if successful)
 */
cudaError_t calculate_magnitude_cuda(
    GPUImage& sobel_horizontal,
    GPUImage& sobel_vertical,
    GPUImage* magnitude
) {
    // 1. Validate input
    if (sobel_horizontal.device_data == NULL) {
        fprintf(stderr, "Error: Horizontal gradient GPUImage has no device data\n");
        return cudaErrorInvalidValue;
    }
    
    if (sobel_vertical.device_data == NULL) {
        fprintf(stderr, "Error: Vertical gradient GPUImage has no device data\n");
        return cudaErrorInvalidValue;
    }
    
    if (magnitude == NULL) {
        fprintf(stderr, "Error: Magnitude GPUImage pointer is NULL\n");
        return cudaErrorInvalidValue;
    }
    
    // magnitude must point to an empty object (created via default constructor)
    if (magnitude->device_data != NULL) {
        fprintf(stderr, "Error: Magnitude GPUImage must be empty (device_data must be NULL)\n");
        return cudaErrorInvalidValue;
    }
    
    // 2. Check if input dimensions match
    if (sobel_horizontal.width != sobel_vertical.width || 
        sobel_horizontal.height != sobel_vertical.height) {
        fprintf(stderr, "Error: Gradient images have different dimensions: "
                "horizontal (%d x %d), vertical (%d x %d)\n",
                sobel_horizontal.width, sobel_horizontal.height,
                sobel_vertical.width, sobel_vertical.height);
        return cudaErrorInvalidValue;
    }
    
    int width = sobel_horizontal.width;
    int height = sobel_horizontal.height;
    
    // 3. Construct output GPUImage object (using constructor with dimensions)
    try {
        *magnitude = GPUImage(width, height);
    } catch (const GPUImageException& e) {
        fprintf(stderr, "Error: Failed to create magnitude GPUImage: %s\n", e.what());
        return cudaErrorMemoryAllocation;
    }
    
    // 4. Configure kernel launch parameters
    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    // 5. Launch CUDA kernel
    _calculate_magnitude_kernel<<<gridSize, blockSize>>>(
        sobel_horizontal.device_data,  // Extract from GPUImage
        sobel_vertical.device_data,    // Extract from GPUImage
        magnitude->device_data,        // Extract from GPUImage pointer
        width,                         // Extract from GPUImage
        height                         // Extract from GPUImage
    );
    
    // 6. Check kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: CUDA kernel launch failed: %s\n",
                cudaGetErrorString(err));
            return err;
        }
        
        // 7. Wait for kernel to complete
        err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: CUDA kernel execution failed: %s\n",
                cudaGetErrorString(err));
        return err;
    }
    
    return cudaSuccess;
}

