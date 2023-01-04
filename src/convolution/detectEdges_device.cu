#include <stdio.h>
#include <stdint.h>
#include "../library.h"

__global__ void detectEdges_kernel1(uint8_t * inPixels, int width, int height, 
        float * filter, int filterWidth, uint8_t * outPixels) {

	  int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width) { 
        uint8_t outPixel = 0;
				for (int filterR = 0; filterR < filterWidth; filterR++) {
					for (int filterC = 0; filterC < filterWidth; filterC++) {
						float filterVal = filter[filterR*filterWidth + filterC];
						int inPixelsR = r - filterWidth/2 + filterR;
						int inPixelsC = c - filterWidth/2 + filterC;

						inPixelsR = min(max(0, inPixelsR), height - 1);
						inPixelsC = min(max(0, inPixelsC), width - 1);

						outPixel += filterVal * inPixels[inPixelsR*width + inPixelsC];
					}
        }
        outPixels[r*width+c] = outPixel;
    }    
}

__global__ void detectEdges_kernel2(uint8_t * inPixels, int width, int height, 
        float * filter, int filterWidth, uint8_t * outPixels) {
	extern __shared__ uint8_t s_inPixels[];

  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

	int share_width = blockDim.x + filterWidth - 1;
	float share_length = (blockDim.x + filterWidth - 1) * (blockDim.y + filterWidth - 1);

	int nPixelsPerThread = ceil(share_length / (blockDim.x * blockDim.y));
	int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

	int firstR = blockIdx.y * blockDim.y - filterWidth / 2;
	int firstC = blockIdx.x * blockDim.x - filterWidth / 2;

	for (int i = 0; i < nPixelsPerThread; i++) {
		int pos = threadIndex * nPixelsPerThread + i;
		if (pos >= share_length) break;

		int inPixelR = pos / share_width + firstR;
		int inPixelC = pos % share_width + firstC;
		inPixelR = min(max(0, inPixelR), height - 1);
		inPixelC = min(max(0, inPixelC), width - 1);

		s_inPixels[pos] = inPixels[inPixelR * width + inPixelC];
	}
	__syncthreads();

    if (r < height && c < width) {
		uint8_t outPixel = 0;
		for (int filterR = 0; filterR < filterWidth; filterR++)
		{
			for (int filterC = 0; filterC < filterWidth; filterC++)
			{
				float filterVal = filter[filterR * filterWidth + filterC];
				int inPixelR = threadIdx.y + filterR;
				int inPixelC = threadIdx.x + filterC;

				outPixel += filterVal * s_inPixels[inPixelR * share_width + inPixelC];
			}
		}
		outPixels[r * width + c] = outPixel;
	}
}

__global__ void detectEdges_kernel3(uint8_t * inPixels, int width, int height, 
        int filterWidth, uint8_t * outPixels) {
	extern __shared__ uint8_t s_inPixels[];

  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

	int share_width = blockDim.x + filterWidth - 1;
	float share_length = (blockDim.x + filterWidth - 1) * (blockDim.y + filterWidth - 1);

	int nPixelsPerThread = ceil(share_length / (blockDim.x * blockDim.y));
	int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

	int firstR = blockIdx.y * blockDim.y - filterWidth / 2;
	int firstC = blockIdx.x * blockDim.x - filterWidth / 2;

	for (int i = 0; i < nPixelsPerThread; i++)
	{
		int pos = threadIndex * nPixelsPerThread + i;
		if (pos >= share_length) break;

		int inPixelR = pos / share_width + firstR;
		int inPixelC = pos % share_width + firstC;
		inPixelR = min(max(0, inPixelR), height - 1);
		inPixelC = min(max(0, inPixelC), width - 1);

		s_inPixels[pos] = inPixels[inPixelR * width + inPixelC];
	}
	__syncthreads();

    if (r < height && c < width)
	{
		uint8_t outPixel = 0;
		for (int filterR = 0; filterR < filterWidth; filterR++)
		{
			for (int filterC = 0; filterC < filterWidth; filterC++)
			{
				float filterVal = dc_filter[filterR * filterWidth + filterC];
				int inPixelR = threadIdx.y + filterR;
				int inPixelC = threadIdx.x + filterC;

				outPixel += filterVal * s_inPixels[inPixelR * share_width + inPixelC];
			}
		}
		outPixels[r * width + c] = outPixel;
	}
}						