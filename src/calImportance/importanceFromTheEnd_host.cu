#include <stdio.h>
#include <stdint.h>

/**
 * @param inPixels: we will compute on this input.
 * Hence, it's must be a cop.
 * The original one will be reused.
 */
void importanceFromTheEnd(uint8_t * inPixels, int width, int height) {
    for (int row = height - 2; row >= 0; row--) {
        for (int col = 0; col < width; col++) {

            // get 3 values belows & get min
            uint8_t val1 = result.get((row + 1) * width + Math.max(0, col - 1));
            uint8_t val2 = result.get((row + 1) * width + col);
            uint8_t val3 = result.get((row + 1) * width + Math.min(width - 1, col + 1));

            uint8_t min = Math.min(Math.min(val1, val2), val3);

            // add to current number
            inPixels[row * width + col] += min;
        }
}    