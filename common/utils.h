#ifndef EDGING_UTILS_H
#define EDGING_UTILS_H
#include <vector>

#include "bitmap.h"

struct DataSet {
    std::vector<BitmapResult> files;

    FloatMap gaussianKernel ;
    FloatMap sobelKernelVert ;
    FloatMap sobelKernelHoriz ;

    DataSet(const std::vector<BitmapResult> &files, const FloatMap &gaussianKernel ,
    const FloatMap &sobelKernelVert ,
    const FloatMap &sobelKernelHoriz): files(files),
    gaussianKernel(gaussianKernel),
    sobelKernelVert(sobelKernelVert),
    sobelKernelHoriz(sobelKernelHoriz) {}
};

long getSysTime();

DataSet prepareDataset(int blur_kernel_size, float blur_sigma , std::string indir, std::string outdir);


void saveResult(  std::vector<BitmapResult> files, std::string outdir);

#endif //EDGING_UTILS_H
