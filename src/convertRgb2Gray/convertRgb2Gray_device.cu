#include <stdio.h>
#include <stdint.h>

__global__ void convertRgb2Gray_device(uint8_t * inPixels, int width, int height, uint8_t * outPixels) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width) { 
      int i = r * width + c;
      uint8_t red = inPixels[3 * i];
      uint8_t green = inPixels[3 * i + 1];
      uint8_t blue = inPixels[3 * i + 2];

      outPixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
    }
}