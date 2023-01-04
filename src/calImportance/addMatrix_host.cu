#include <stdio.h>
#include <stdint.h>

void addMatrix_host(uint8_t *in1, uint8_t *in2, int nRows, int nCols, uint8_t *out) {
	for (int r = 0; r < nRows; r++) {
    for (int c = 0; c < nCols; c++) {
        int i = r * nCols + c;
        out[i] = in1[i] + in2[i];
    }
  }
} 