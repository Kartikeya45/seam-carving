#include <stdio.h>
#include <stdint.h>

void deepcopy_host(uint8_t * in, int n, uint8_t * outPixels) {
    outPixels = (uint8_t *)malloc(n);
    for (int i = 0; i < n; i++) 
        out[i] = in[i];
}

/**
 * @param inPixels: we will compute on this input.
 * Hence, it's must be a copy.
 * The original one will be reused.
 */
void importanceToTheEnd_host(uint8_t * inPixels, int width, int height, uint8_t * outPixels) {

    deepcopy_host(inPixels, width * height, outPixels);

    for (int row = height - 2; row >= 0; row--) {
        for (int col = 0; col < width; col++) {

            // get 3 values belows & get min
            uint8_t val1 = outPixels[(row + 1) * width + max(0, col - 1)];
            uint8_t val2 = outPixels[(row + 1) * width + col];
            uint8_t val3 = outPixels[(row + 1) * width + min(width - 1, col + 1)];

            uint8_t min = min(min(val1, val2), val3);

            // add to current number
            outPixels[row * width + col] += min;
        }
    }
}    