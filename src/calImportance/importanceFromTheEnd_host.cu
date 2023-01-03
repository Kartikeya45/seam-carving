#include <stdio.h>

/**
 * @param inPixels: we will compute on this input.
 * Hence, it's must be a cop.
 * The original one will be reused.
 */
void importanceFromTheEnd(uchar3 * inPixels, int width, int height) {
    for (int row = height - 2; row >= 0; row--) {
        for (int col = 0; col < width; col++) {

            // get 3 values belows & get min
            int val1 = result.get((row + 1) * width + Math.max(0, col - 1));
            int val2 = result.get((row + 1) * width + col);
            int val3 = result.get((row + 1) * width + Math.min(width - 1, col + 1));

            int min = Math.min(Math.min(val1, val2), val3);

            // add to current number
            inPixels[row * width + col] += min;
        }
}    