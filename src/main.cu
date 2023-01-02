#include <stdio.h>
#include "library.h"

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

/**
 * @param argc[1] name of the input file (.pmn)
 * @param argc[2] name of output file with no extension, created by using host & device
 * @param argc[3] - optional - default(32): blocksize.x
 * @param argc[4] - optional - default(32): blocksize.x
 * @see HW1_P2
 */
int main(int argc, char ** argv) {
    if (argc != 3 && argc != 5) {
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}
	
	printDeviceInfo();
	
	// Read input image file
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("Image size (width x height): %i x %i\n\n", width, height);
	
	// Write results to files
	writePnm(inPixels, width, height, concatStr(argv[2], "_host.pnm"));
// 	writePnm(hostOutPixels, width, height, concatStr(outFileNameBase, "_host.pnm"));
// 	writePnm(deviceOutPixels, width, height, concatStr(outFileNameBase, "_device.pnm"));
    
    // Free memories
    free(inPixels);
}