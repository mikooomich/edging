
#ifndef EDGING_SEQUENTIAL_H
#define EDGING_SEQUENTIAL_H
#include "bitmap.h"

int processImage(BitmapResult *result, const FloatMap &gaussianKernel, const FloatMap &sobelKernelvert,
                 const FloatMap &sobelKernelHoriz);


void runSequential( std::vector<BitmapResult>& files,  const FloatMap &gaussianKernel, const FloatMap &sobelKernelVert, const FloatMap &sobelKernelHoriz);

#endif //EDGING_SEQUENTIAL_H