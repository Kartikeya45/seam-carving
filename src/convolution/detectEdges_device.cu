#include <stdio.h>
#include <stdint.h>

__global__ void detectEdges_device(uint8_t * inPixels, int width, int height, 
        float * filter, int filterWidth, uint8 * outPixels) {
	  int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width) { 
        float outPixel = 0;
				for (int filterR = 0; filterR < filterWidth; filterR++)
				{
					for (int filterC = 0; filterC < filterWidth; filterC++)
					{
						float filterVal = filter[filterR*filterWidth + filterC];
						int inPixelsR = r - filterWidth/2 + filterR;
						int inPixelsC = c - filterWidth/2 + filterC;
						inPixelsR = min(max(0, inPixelsR), height - 1);
						inPixelsC = min(max(0, inPixelsC), width - 1);
						uint8_t inPixel = inPixels[inPixelsR*width + inPixelsC];
						outPixel += filterVal * inPixel;
					}
        }
        outPixels[r*width+c] = outPixel;
    }    
}