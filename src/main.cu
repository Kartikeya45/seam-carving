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
 * @param argc[3] horizontal of image you want to resize 
 * @param argc[4] - optional - default(32): blocksize.x
 * @param argc[5] - optional - default(32): blocksize.y
 * @see HW1_P2
 */
int main(int argc, char ** argv) {
    if (argc != 4 && argc != 6) {
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}
	
	printDeviceInfo();
	
  // Read user's args
  dim3 blockSize(32, 32); // Default
  int resize = argv[3];
  if (argc == 5) {
		blockSize.x = atoi(argv[4]);
		blockSize.y = atoi(argv[5]);
	}

	// Read input image file
  int numChannels, width, height;
  uint8_t * inPixels;
	readPnm(argv[1], numChannels, width, height, inPixels);

  if (numChannels != 3) {
    printf("Number of channels is not 3, input image must be RGB!");
		return EXIT_FAILURE; 
  }
	printf("Image size (width x height): %i x %i\n\n", width, height);

  // Convert RGB to Grayscale
  uint8_t * correctOutPixels= (uint8_t *)malloc(width * height);
	convertRgb2Gray(inPixels, width, height, correctOutPixels);

  uint8_t * outPixels= (uint8_t *)malloc(width * height);
  convertRgb2Gray(inPixels, width, height, outPixels, true, blockSize);
	
  float err = computeError(outPixels, correctOutPixels, width * height);
	printf("Error after convert RGB to Grayscale: %f\n", err);

  // Write the Grayscale image
	writePnm_uint8_t(correctOutPixels, 1, width, height, concatStr(argv[2], "_host.pnm"));
	writePnm_uint8_t(outPixels, 1, width, height, concatStr(argv[2], "_device.pnm"));

  // Seam carving


	// Write results to files
	//writePnm(inPixels, width, height, concatStr(argv[2], "_host.pnm"));
// 	writePnm(hostOutPixels, width, height, concatStr(outFileNameBase, "_host.pnm"));
// 	writePnm(deviceOutPixels, width, height, concatStr(outFileNameBase, "_device.pnm"));
    
  // Free memories
  free(inPixels);
  free(outPixels);
}