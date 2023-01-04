#include <stdio.h>
#include <stdint.h>

void detectEdges_host(uint8_t * inPixels, int width, int height, float * filter, int filterWidth, uint8_t * outPixels) {
	for (int outPixelsR = 0; outPixelsR < height; outPixelsR++) 	{
			for (int outPixelsC = 0; outPixelsC < width; outPixelsC++) {
				float outPixel = 0;
				for (int filterR = 0; filterR < filterWidth; filterR++) {
					for (int filterC = 0; filterC < filterWidth; filterC++) {
						float filterVal = filter[filterR*filterWidth + filterC];
						int inPixelsR = outPixelsR - filterWidth/2 + filterR;
						int inPixelsC = outPixelsC - filterWidth/2 + filterC;
						inPixelsR = min(max(0, inPixelsR), height - 1);
						inPixelsC = min(max(0, inPixelsC), width - 1);

						outPixel += filterVal * inPixels[inPixelsR*width + inPixelsC];
					}
				}
				outPixels[outPixelsR*width + outPixelsC] = outPixel; 
			}
		}
}