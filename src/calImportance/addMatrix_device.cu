#include <stdio.h>
#include <stdint.h>

__global__ void addMatrix_kernel(uint8_t *in1, uint8_t *in2, int nRows, int nCols, uint8_t *out) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < nRows && c < nCols) { 
        int i = r * nCols + c;
        out[i] = in1[i] + in2[i];
    }
}