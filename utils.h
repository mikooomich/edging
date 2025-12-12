#ifndef EDGING_UTILS_H
#define EDGING_UTILS_H
#include <vector>

#include "bitmap.h"


// Enable extra debug print
// Uncomment to enable, comment to disable.
#define EDGING_DEBUG

// Enable saving of debug frames. Useful for checking algorithm correctness, really not useful otherwise
// Uncomment to enable, comment to disable.
// #define SAVE_PROCESS_FRAMES

// Enable saving final images to disk
// Uncomment to enable, comment to disable.
#define SAVE_FINAL_RESULT

struct DataSet {
    std::vector<BitmapResult> files;

    FloatMap gaussianKernel;
    FloatMap sobelKernelVert;
    FloatMap sobelKernelHoriz;

    // Full dataset with images
    DataSet(const std::vector<BitmapResult> &files, const FloatMap &gaussianKernel ,const FloatMap &sobelKernelVert ,const FloatMap &sobelKernelHoriz):
    files(files),gaussianKernel(gaussianKernel),sobelKernelVert(sobelKernelVert),sobelKernelHoriz(sobelKernelHoriz) {}

    // kernels only dataset
    DataSet(const FloatMap &gaussianKernel ,const FloatMap &sobelKernelVert ,const FloatMap &sobelKernelHoriz):
    gaussianKernel(gaussianKernel),sobelKernelVert(sobelKernelVert),sobelKernelHoriz(sobelKernelHoriz) {}
};

long getSysTime();

DataSet prepareDataset(int blur_kernel_size, float blur_sigma , std::string indir, bool kernelsOnly);


void saveResult(std::vector<BitmapResult> &files, const std::string& outdir, long startTime,  const std::string& infoText);

#endif //EDGING_UTILS_H
