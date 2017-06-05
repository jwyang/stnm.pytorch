// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

int BilinearSamplerBHWD_updateOutput_cuda(THCudaTensor *canvas, THCudaTensor *inputImages, THCudaTensor *grids,
                                        THCudaTensor *masks, THCudaTensor *output);

int BilinearSamplerBHWD_updateGradInput_cuda(THCudaTensor *canvas, THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *masks,
                                        THCudaTensor *gradCanvas, THCudaTensor *gradInputImages,
                                        THCudaTensor *gradGrids, THCudaTensor *gradMasks, THCudaTensor *gradOutput);

int BilinearSamplerBHWD_updateGradInputOnlyGrid_cuda(THCudaTensor *inputImages, THCudaTensor *grids,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput);
