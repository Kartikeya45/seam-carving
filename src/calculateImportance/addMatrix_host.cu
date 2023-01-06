#include <stdio.h>
#include <stdint.h>

void addMatrix_host(uint8_t * in1, uint8_t * in2, int height, int width, uint8_t * out) {
	for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
        int i = r * width + c;
        out[i] = in1[i] + in2[i];
    }
  }
} 