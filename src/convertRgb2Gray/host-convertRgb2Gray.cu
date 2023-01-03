#include <stdio.h>

void convertRgb2Gray_host(uint8_t * inPixels, int width, int height, uint8_t * outPixels) {
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      int i = r * width + c;
      uint8_t red = inPixels[3 * i];
      uint8_t green = inPixels[3 * i + 1];
      uint8_t blue = inPixels[3 * i + 2];

      outPixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
    }
  }
}