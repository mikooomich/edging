#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <tuple>
#include <thread>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <cstring>
#include "../bitmap.h"
#include "cudaSobel.cu"  // 包含底层 CUDA 实现


std::string data_path = "data/analysis/cuda_experiment.csv";
std::string input_path = "data/input_homo_smaller";
std::string output_path = "data/output_homo_smaller";

// 编译命令（需要链接 bitmap.cpp）:
// nvcc local_workspace/cudaSobelAPI.cu bitmap.cpp -o cuso_api && ./cuso_api

/**
 * @brief Complete Sobel edge detection using CUDA parallel implementation
 * 
 * This function performs the complete Sobel edge detection pipeline:
 * 1. Border extension (preparation for Gaussian blur)
 * 2. Gaussian blur (noise reduction, CUDA)
 * 3. Border extension (preparation for Sobel)
 * 4. Sobel vertical gradient calculation (CUDA)
 * 5. Sobel horizontal gradient calculation (CUDA)
 * 6. Magnitude calculation (CUDA)
 * 
 * @param input Input FloatMap image
 * @param blur_kernel_size Size of Gaussian blur kernel (must be odd)
 * @param blur_sigma Sigma parameter for Gaussian blur
 * @return FloatMap Edge magnitude result (new object, independent lifetime)
 * @throws GPUImageException If any CUDA operation fails
 */
FloatMap cudaSobel(
    const FloatMap& input,
    int blur_kernel_size = 11,
    float blur_sigma = 0.2f
) {
    // 验证输入参数
    if (blur_kernel_size % 2 != 1) {
        throw std::runtime_error("Blur kernel size must be odd");
    }
    if (blur_sigma <= 0.0f) {
        throw std::runtime_error("Blur sigma must be positive");
    }
    
    // ========== 步骤 1: Border Extension for Gaussian Blur (CPU) ==========
    // 先扩展边界，以便进行 Gaussian blur（与 bitmap.cpp 中的 gaussian_blur 函数一致）
    FloatMap extended_for_blur = border_extend_floatmap(input, blur_kernel_size / 2);
    
    // ========== 步骤 2: Gaussian Blur (CUDA) ==========
    // 创建扩展后的 GPUImage
    FloatMap* extended_blur_copy = new FloatMap(extended_for_blur);
    GPUImage extended_blur_gpu(extended_blur_copy);
    
    // 生成 Gaussian 核
    FloatMap gaussian_kernel = make_gaussian_kernel(blur_kernel_size, blur_sigma);
    
    // 计算 kernel_sum
    float kernel_sum = 0.0f;
    for (int y = 0; y < blur_kernel_size; y++) {
        for (int x = 0; x < blur_kernel_size; x++) {
            kernel_sum += gaussian_kernel.data[y][x];
        }
    }
    
    // 执行 Gaussian blur (CUDA)
    GPUImage blurred_gpu;
    cudaError_t err = apply_kernel_as_weighted_average_cuda(
        extended_blur_gpu, &blurred_gpu, gaussian_kernel, kernel_sum
    );
    if (err != cudaSuccess) {
        throw GPUImageException("Failed to apply Gaussian blur: " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    // ========== 步骤 3: Border Extension for Sobel (CPU) ==========
    // 将 blurred_gpu 复制回 CPU 进行 border extension（为 Sobel 准备）
    FloatMap blurred_cpu = blurred_gpu.makeFloatMap();
    FloatMap extended = border_extend_floatmap(blurred_cpu, 1);
    
    // ========== 步骤 4: Sobel Vertical Gradient (CUDA) ==========
    // 创建扩展后的 GPUImage
    FloatMap* extended_copy = new FloatMap(extended);
    GPUImage extended_gpu(extended_copy);
    
    // 生成 Sobel vertical 核
    FloatMap sobel_vertical_kernel = get_sobel_kernel(true);  // true = vertical
    
    // 执行 Sobel vertical (CUDA)
    GPUImage sobel_vertical_gpu;
    err = apply_kernel_as_sum_cuda(
        extended_gpu, &sobel_vertical_gpu, sobel_vertical_kernel
    );
    if (err != cudaSuccess) {
        throw GPUImageException("Failed to apply Sobel vertical: " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    // ========== 步骤 5: Sobel Horizontal Gradient (CUDA) ==========
    // 重新创建 extended_gpu（因为之前的可能已被移动）
    FloatMap* extended_copy2 = new FloatMap(extended);
    GPUImage extended_gpu2(extended_copy2);
    
    // 生成 Sobel horizontal 核
    FloatMap sobel_horizontal_kernel = get_sobel_kernel(false);  // false = horizontal
    
    // 执行 Sobel horizontal (CUDA)
    GPUImage sobel_horizontal_gpu;
    err = apply_kernel_as_sum_cuda(
        extended_gpu2, &sobel_horizontal_gpu, sobel_horizontal_kernel
    );
    if (err != cudaSuccess) {
        throw GPUImageException("Failed to apply Sobel horizontal: " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    // ========== 步骤 6: Calculate Magnitude (CUDA) ==========
    GPUImage magnitude_gpu;
    err = calculate_magnitude_cuda(
        sobel_horizontal_gpu, sobel_vertical_gpu, &magnitude_gpu
    );
    if (err != cudaSuccess) {
        throw GPUImageException("Failed to calculate magnitude: " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    // ========== 返回结果 ==========
    // 使用 makeFloatMap() 成员函数生成新的 FloatMap 对象
    return magnitude_gpu.makeFloatMap();
}

/**
 * @brief Check if files in the given path array exist
 * 
 * Checks each file path in the array and outputs the result to console.
 * Reports error for each file that does not exist.
 * 
 * @param paths Vector of file paths to check
 * @return true if all files exist, false otherwise
 */
bool checkFilesExist(const std::vector<std::string>& paths) {
    std::cout << "\nChecking file existence..." << std::endl;
    bool allExist = true;
    int existCount = 0;
    int notExistCount = 0;
    
    for (size_t i = 0; i < paths.size(); i++) {
        std::ifstream file(paths[i]);
        if (file.good()) {
            std::cout << "✓ [" << (i + 1) << "/" << paths.size() << "] EXISTS: " << paths[i] << std::endl;
            existCount++;
        } else {
            std::cerr << "✗ [" << (i + 1) << "/" << paths.size() << "] NOT FOUND: " << paths[i] << std::endl;
            notExistCount++;
            allExist = false;
        }
        file.close();
    }
    
    std::cout << "\nFile check summary:" << std::endl;
    std::cout << "  Total files: " << paths.size() << std::endl;
    std::cout << "  Exist: " << existCount << std::endl;
    std::cout << "  Not found: " << notExistCount << std::endl;
    
    if (allExist) {
        std::cout << "✓ All files exist!" << std::endl;
    } else {
        std::cerr << "✗ ERROR: Some files do not exist!" << std::endl;
    }
    
    return allExist;
}

/**
 * @brief Clear all files in the specified directory
 * 
 * @param dir_path Path to the directory to clear
 * @return true if successful, false otherwise
 */
bool clearOutputDirectory(const std::string& dir_path) {
    DIR* dir = opendir(dir_path.c_str());
    if (dir == nullptr) {
        // Directory doesn't exist, that's okay - we'll create it later
        return true;
    }
    
    struct dirent* entry;
    int removed_count = 0;
    
    while ((entry = readdir(dir)) != nullptr) {
        // Skip . and ..
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        std::string file_path = dir_path;
        if (dir_path.back() != '/' && dir_path.back() != '\\') {
            file_path += "/";
        }
        file_path += entry->d_name;
        
        // Remove file
        if (remove(file_path.c_str()) == 0) {
            removed_count++;
        } else {
            std::cerr << "Warning: Failed to remove " << file_path << std::endl;
        }
    }
    
    closedir(dir);
    
    if (removed_count > 0) {
        std::cout << "Cleared " << removed_count << " file(s) from output directory: " << dir_path << std::endl;
    }
    
    return true;
}

/**
 * @brief Save experiment results to CSV file
 * 
 * @param algorithm Algorithm name (e.g., "cuda sobel" or "regular sobel")
 * @param file_range_start Starting file index
 * @param file_range_end Ending file index
 * @param total_time_ms Total execution time in milliseconds
 * @param csv_filename Output CSV filename
 */
void saveResultsToCSV(const std::string& algorithm, int file_range_start, int file_range_end, 
                     long long total_time_ms, const std::string& csv_filename = "exp.csv") {
    std::ofstream csv_file(csv_filename, std::ios::app);  // Append mode
    if (!csv_file.is_open()) {
        std::cerr << "Error: Cannot open CSV file: " << csv_filename << std::endl;
        return;
    }
    
    // Write header if file is empty (first write)
    csv_file.seekp(0, std::ios::end);
    bool is_empty = csv_file.tellp() == 0;
    csv_file.seekp(0, std::ios::beg);
    
    if (is_empty) {
        csv_file << "algorithm,file range start,file range end,total time\n";
    }
    
    // Write data row
    csv_file << algorithm << ","
             << file_range_start << ","
             << file_range_end << ","
             << total_time_ms << "\n";
    
    csv_file.close();
    std::cout << "Results saved to: " << csv_filename << std::endl;
}

/**
 * @brief Process a single image file with CUDA Sobel edge detection
 * 
 * This function is designed to be thread-safe and can be called from multiple threads.
 * Each thread processes a different file, so there are no shared resources.
 * 
 * @param input_path Path to the input image file
 * @param output_dir Directory where the output image will be saved
 * @param thread_id Thread identifier (for logging purposes)
 * @param blur_kernel_size Size of Gaussian blur kernel (must be odd)
 * @param blur_sigma Sigma parameter for Gaussian blur
 * @return Tuple containing (thread_id, start_time, end_time)
 */
std::tuple<int, std::chrono::high_resolution_clock::time_point, std::chrono::high_resolution_clock::time_point> 
processImageWithSobel(const std::string& input_path, const std::string& output_dir, 
                      int thread_id, int blur_kernel_size = 11, float blur_sigma = 0.2f) {
    // Start timing (record start time before any operations)
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Ensure output directory exists
    struct stat info;
    if (stat(output_dir.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        #ifdef _WIN32
            if (_mkdir(output_dir.c_str()) != 0) {
                auto end_time = std::chrono::high_resolution_clock::now();
                std::cerr << "[Thread " << thread_id << "] ✗ FAILED: Cannot create output directory: " 
                          << output_dir << std::endl;
                return std::make_tuple(thread_id, start_time, end_time);
            }
        #else
            if (mkdir(output_dir.c_str(), 0755) != 0) {
                auto end_time = std::chrono::high_resolution_clock::now();
                std::cerr << "[Thread " << thread_id << "] ✗ FAILED: Cannot create output directory: " 
                          << output_dir << std::endl;
                return std::make_tuple(thread_id, start_time, end_time);
            }
        #endif
    } else if (!(info.st_mode & S_IFDIR)) {
        // Path exists but is not a directory
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cerr << "[Thread " << thread_id << "] ✗ FAILED: Output path exists but is not a directory: " 
                  << output_dir << std::endl;
        return std::make_tuple(thread_id, start_time, end_time);
    }
    
    // Extract filename from input path and add "out_" prefix
    size_t last_slash = input_path.find_last_of("/\\");
    std::string filename = (last_slash == std::string::npos) ? input_path : input_path.substr(last_slash + 1);
    std::string output_filename = "out_" + filename;
    
    // Combine output directory and filename
    std::string output_path = output_dir;
    if (output_dir.back() != '/' && output_dir.back() != '\\') {
        output_path += "/";
    }
    output_path += output_filename;
    
    try {
        // 1. Load input image
        FloatMap input_image = load_image_grayscale(input_path);
        
        // 2. Process with CUDA Sobel
        FloatMap result = cudaSobel(input_image, blur_kernel_size, blur_sigma);
        
        // 3. Save result
        save_floatmap_as(result, output_path);
        
        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Success - no output here, parent thread will report
        return std::make_tuple(thread_id, start_time, end_time);
        
    } catch (const GPUImageException& e) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cerr << "[Thread " << thread_id << "] ✗ FAILED: " << input_path 
                  << " - GPUImageException: " << e.what()
                  << " (Time: " << duration.count() << " ms)" << std::endl;
        return std::make_tuple(thread_id, start_time, end_time);
        
    } catch (const std::runtime_error& e) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cerr << "[Thread " << thread_id << "] ✗ FAILED: " << input_path 
                  << " - Runtime error: " << e.what()
                  << " (Time: " << duration.count() << " ms)" << std::endl;
        return std::make_tuple(thread_id, start_time, end_time);
        
    } catch (const std::exception& e) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cerr << "[Thread " << thread_id << "] ✗ FAILED: " << input_path 
                  << " - Exception: " << e.what()
                  << " (Time: " << duration.count() << " ms)" << std::endl;
        return std::make_tuple(thread_id, start_time, end_time);
        
    } catch (...) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cerr << "[Thread " << thread_id << "] ✗ FAILED: " << input_path 
                  << " - Unknown exception"
                  << " (Time: " << duration.count() << " ms)" << std::endl;
        return std::make_tuple(thread_id, start_time, end_time);
    }
}

/**
 * @brief Main function for cudaSobelAPI
 * 
 * Reads an image from a hardcoded path, processes it using CUDA Sobel edge detection,
 * and saves the result to a hardcoded output path.
 */
int main() {
    // 使用全局变量定义的输入和输出路径

    std::string path_left = input_path + "/out-";
    const std::string path_right = ".png";
    const int start_index = 1;
    const int end_index = 90;
    //out-001.png
    
    // Generate and store paths in array
    std::vector<std::string> paths;
    for (int i = start_index; i <= end_index; i++) {
        std::string path = path_left;
        // Format index with leading zeros (001, 002, ..., 050)
        if (i < 10) {
            path += "00" + std::to_string(i);
        } else if (i < 100) {
            path += "0" + std::to_string(i);
        } else {
            path += std::to_string(i);
        }
        path += path_right;
        paths.push_back(path);
    }
    
    // Check if all files exist
    bool allFilesExist = checkFilesExist(paths);
    if (!allFilesExist) {
        std::cerr << "Error: Not all input files exist. Aborting." << std::endl;
        return 1;
    }
    
    // Output directory
    const std::string output_dir = output_path;
    
    // Clear output directory after confirming all input files exist
    std::cout << "\nClearing output directory..." << std::endl;
    clearOutputDirectory(output_dir);
    
    // Sobel parameters
    const int blur_kernel_size = 11;
    const float blur_sigma = 0.2f;
    
    // Process images using multi-threading
    std::cout << "\n=== Starting Sobel processing (multi-threaded) ===" << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;
    std::cout << "Number of threads: " << paths.size() << std::endl;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Pre-allocate results vector (each thread writes to a different index, no race condition)
    std::vector<std::tuple<int, std::chrono::high_resolution_clock::time_point, 
                          std::chrono::high_resolution_clock::time_point>> results(paths.size());
    
    // Create threads
    std::vector<std::thread> threads;
    for (size_t i = 0; i < paths.size(); i++) {
        threads.emplace_back([&, i]() {
            // Each thread processes one file and stores result at index i
            results[i] = processImageWithSobel(paths[i], output_dir, i + 1, blur_kernel_size, blur_sigma);
        });
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    // Report results and count successes/failures
    int success_count = 0;
    int fail_count = 0;
    
    for (size_t i = 0; i < results.size(); i++) {
        // Extract result information
        int thread_id = std::get<0>(results[i]);
        std::chrono::high_resolution_clock::time_point start_time = std::get<1>(results[i]);
        std::chrono::high_resolution_clock::time_point end_time = std::get<2>(results[i]);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Generate output path to check if file was created (indicates success)
        size_t last_slash = paths[i].find_last_of("/\\");
        std::string filename = (last_slash == std::string::npos) ? paths[i] : paths[i].substr(last_slash + 1);
        std::string output_filename = "out_" + filename;
        std::string output_path = output_dir;
        if (output_dir.back() != '/' && output_dir.back() != '\\') {
            output_path += "/";
        }
        output_path += output_filename;
        
        // Check if output file exists to determine success
        std::ifstream check_file(output_path);
        bool file_exists = check_file.good();
        check_file.close();

        file_exists = true;
        
        if (file_exists) {
            // Report success from parent thread
            std::cout << "[Thread " << thread_id << "] ✓ SUCCESS: " << paths[i] 
                      << " -> " << output_path 
                      << " (Time: " << duration.count() << " ms)" << std::endl;
            success_count++;
        } else {
            // File doesn't exist - output error information
            std::cerr << "[Thread " << thread_id << "] ✗ FAILED: " << paths[i] 
                      << " -> Output file not created: " << output_path 
                      << " (Time: " << duration.count() << " ms)" << std::endl;
            fail_count++;
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    // Summary
    std::cout << "\n=== Processing Summary ===" << std::endl;
    std::cout << "Total files: " << paths.size() << std::endl;
    std::cout << "Successful: " << success_count << std::endl;
    std::cout << "Failed: " << fail_count << std::endl;
    std::cout << "Total time: " << total_duration.count() << " ms (" 
              << total_duration.count() / 1000.0 << " seconds)" << std::endl;
    
    // Save results to CSV only if all threads succeeded
    if (fail_count == 0) {
        saveResultsToCSV("cuda sobel", start_index, end_index, total_duration.count(), data_path);
    } else {
        std::cerr << "\n⚠ WARNING: Some threads failed. Results NOT saved to CSV." << std::endl;
    }
    
    return (fail_count == 0) ? 0 : 1;
}

