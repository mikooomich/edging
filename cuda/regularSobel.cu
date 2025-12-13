#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstring>
#include "bitmap.h"

std::string data_path = "cuda/analysis/cuda_experiment.csv";
std::string input_path = "cuda/input_homo_smaller";
std::string output_path = "cuda/output_homo_smaller";

// Compilation command:
// nvcc cuda/regularSobel.cu cuda/bitmap.cpp -o temp/regular_sobel && ./temp/regular_sobel

/**
 * @brief Check if all files in the given paths exist
 * 
 * @param paths Vector of file paths to check
 */
void checkFilesExist(const std::vector<std::string>& paths) {
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
    // First check if file exists and is empty
    std::ifstream check_file(csv_filename);
    bool file_exists = check_file.good();
    bool is_empty = true;
    if (file_exists) {
        check_file.seekg(0, std::ios::end);
        is_empty = (check_file.tellg() == 0);
    }
    check_file.close();
    
    // Open in append mode
    std::ofstream csv_file(csv_filename, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Cannot open CSV file: " << csv_filename << std::endl;
        return;
    }
    
    // Write header if file is empty (first write)
    if (is_empty) {
        csv_file << "algorithm,file range start,file range end,total time\n";
    }
    
    // Write data row (will be appended to end of file)
    csv_file << algorithm << ","
             << file_range_start << ","
             << file_range_end << ","
             << total_time_ms << "\n";
    
    csv_file.close();
    std::cout << "Results saved to: " << csv_filename << std::endl;
}

/**
 * @brief Process a single image file with regular (CPU) Sobel edge detection
 * 
 * This function performs Sobel edge detection using CPU-based FloatMap operations.
 * It is NOT thread-safe and should be called sequentially.
 * 
 * @param input_path Path to the input image file
 * @param output_path Path where the output image will be saved
 * @param blur_kernel_size Size of Gaussian blur kernel (must be odd)
 * @param blur_sigma Sigma parameter for Gaussian blur
 * @return true if processing succeeded, false otherwise
 */
bool regularSobel(const std::string& input_path, const std::string& output_path,
                  int blur_kernel_size = 11, float blur_sigma = 0.2f) {
    try {
        // 1. Load input image
        FloatMap input_image = load_image_grayscale(input_path);
        
        // 2. Apply Gaussian blur to reduce noise
        FloatMap blurred_image = gaussian_blur(input_image, blur_kernel_size, blur_sigma);
        
        // 3. Extend borders for convolution (padding = 1 for 3x3 Sobel kernel)
        FloatMap extended_blurred_image = border_extend_floatmap(blurred_image, 1);
        
        // 4. Apply vertical Sobel operator (detects horizontal edges)
        FloatMap sobel_vertical_kernel = get_sobel_kernel(true);
        FloatMap sobel_vertical_image = apply_kernel_as_sum(extended_blurred_image, sobel_vertical_kernel);
        
        // 5. Apply horizontal Sobel operator (detects vertical edges)
        FloatMap sobel_horizontal_kernel = get_sobel_kernel(false);
        FloatMap sobel_horizontal_image = apply_kernel_as_sum(extended_blurred_image, sobel_horizontal_kernel);
        
        // 6. Calculate edge magnitude
        FloatMap magnitude = calculate_magnitude(sobel_horizontal_image, sobel_vertical_image);
        
        // 7. Save result
        save_floatmap_as(magnitude, output_path);
        
        return true;
        
    } catch (const std::runtime_error& e) {
        std::cerr << "✗ FAILED: " << input_path 
                  << " - Runtime error: " << e.what() << std::endl;
        return false;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ FAILED: " << input_path 
                  << " - Exception: " << e.what() << std::endl;
        return false;
        
    } catch (...) {
        std::cerr << "✗ FAILED: " << input_path 
                  << " - Unknown exception" << std::endl;
        return false;
    }
}

/**
 * @brief Main function for regularSobel
 * 
 * Processes multiple images using regular (CPU) Sobel edge detection
 * and saves results to CSV file.
 */
int main() {
    // Use global variables to define input and output paths
    std::string path_left = input_path + "/out-";
    const std::string path_right = ".png";
    const int start_index = 1;
    const int end_index = 20;
    
    // Generate and store paths in array
    std::vector<std::string> paths;
    for (int i = start_index; i <= end_index; i++) {
        std::string path = path_left;
        // Format index with leading zeros (001, 002, ..., 100)
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
    checkFilesExist(paths);
    
    // Sobel parameters
    const int blur_kernel_size = 11;
    const float blur_sigma = 0.2f;
    
    // Output directory
    const std::string output_dir = output_path;
    
    // Ensure output directory exists
    struct stat info;
    if (stat(output_dir.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        #ifdef _WIN32
            if (_mkdir(output_dir.c_str()) != 0) {
                std::cerr << "✗ FAILED: Cannot create output directory: " << output_dir << std::endl;
                return 1;
            }
        #else
            if (mkdir(output_dir.c_str(), 0755) != 0) {
                std::cerr << "✗ FAILED: Cannot create output directory: " << output_dir << std::endl;
                return 1;
            }
        #endif
    } else if (!(info.st_mode & S_IFDIR)) {
        // Path exists but is not a directory
        std::cerr << "✗ FAILED: Output path exists but is not a directory: " << output_dir << std::endl;
        return 1;
    }
    
    // Process images sequentially (single-threaded)
    std::cout << "\n=== Starting Sobel processing (regular CPU, single-threaded) ===" << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    int success_count = 0;
    int fail_count = 0;
    
    for (size_t i = 0; i < paths.size(); i++) {
        // Generate output path
        size_t last_slash = paths[i].find_last_of("/\\");
        std::string filename = (last_slash == std::string::npos) ? paths[i] : paths[i].substr(last_slash + 1);
        std::string output_filename = "out_" + filename;
        std::string output_path = output_dir;
        if (output_dir.back() != '/' && output_dir.back() != '\\') {
            output_path += "/";
        }
        output_path += output_filename;
        
        // Process image
        auto start_time = std::chrono::high_resolution_clock::now();
        bool success = regularSobel(paths[i], output_path, blur_kernel_size, blur_sigma);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (success) {
            std::cout << "[" << (i + 1) << "/" << paths.size() << "] ✓ SUCCESS: " << paths[i] 
                      << " -> " << output_path 
                      << " (Time: " << duration.count() << " ms)" << std::endl;
            success_count++;
        } else {
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
    
    // Save results to CSV
    saveResultsToCSV("regular sobel", start_index, end_index, total_duration.count(), data_path);
    
    return (fail_count == 0) ? 0 : 1;
}

