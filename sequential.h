
#ifndef EDGING_SEQUENTIAL_H
#define EDGING_SEQUENTIAL_H
#include "bitmap.h"


int processImageNoGauss(BitmapResult *result, const FloatMap &sobelKernelvert, const FloatMap &sobelKernelHoriz);

void processGaussianBlur(BitmapResult *result, const FloatMap &gaussianKernel) ;

void runSequential( std::vector<BitmapResult>& files,  const FloatMap &gaussianKernel, const FloatMap &sobelKernelVert, const FloatMap &sobelKernelHoriz);

#endif //EDGING_SEQUENTIAL_H