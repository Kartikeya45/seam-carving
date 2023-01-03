#include <stdio.h>

float computeError(uint8_t * a1, uint8_t * a2, int n) {
	float err = 0;
	for (int i = 0; i < n; i++)
		err += abs((int)a1[i] - (int)a2[i]);
	err /= n;
	return err;
}