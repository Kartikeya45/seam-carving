#include <stdio.h>
#include <stdint.h>
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
	
    // Get block Size
    dim3 blockSize(32, 32); // Default

    if (argc == 5) {
		blockSize.x = atoi(argv[4]);
		blockSize.y = atoi(argv[5]);
	}

	// Read input image file 
    int width, height;
    uint8_t * inPixels;
    readPnm(argv[1], width, height, inPixels);

    // Check if we can resize the width to be smaller or not
    int newWidthSize = atoi(argv[3]);
    if (newWidthSize >= width || newWidthSize <= 0) {
        printf("The size you've given is wrong!\n",);
		exit(EXIT_FAILURE);
    }

    // Step 1: Convert RGB to Grayscale
    uint8_t * correctOutPixels= (uint8_t *)malloc(width * height);
	convertRgb2Gray(inPixels, width, height, correctOutPixels);

    uint8_t * outPixels= (uint8_t *)malloc(width * height);
    convertRgb2Gray(inPixels, width, height, outPixels, true, blockSize);
	
    float err = computeError(outPixels, correctOutPixels, width * height);
	printf("Error after convert RGB to Grayscale: %f\n", err);

    // Write the Grayscale images 
	writePnm(correctOutPixels, width, height, concatStr(argv[2], "_grayscale_host.pnm"));
	writePnm(outPixels, width, height, concatStr(argv[2], "_grayscale_device.pnm"));

    // Free memories
    free(inPixels);
    free(outPixels);

    //Read grayscale image
    uint8_t * in_Pixels;
    readPnm(concatStr(argv[2], "_grayscale_host.pnm"), width, height, in_Pixels);

    // Read x-sobel & y-sobel
    int filterWidth, _filterWidth;
    float * x_sobel = readFilter("./fileHandler/filter/x-sobel.txt", filterWidth);
    float * y_sobel = readFilter("./fileHandler/filter/y-sobel.txt", _filterWidth);
    
    // if x-sobel & y-sobel doesn't have the same width or their width is even:
    if (filterWidth != _filterWidth || filterWidth % 2 == 0) {
        printf("Filters are not suitable!\n",);
		exit(EXIT_FAILURE);
    }

    // apply using host
    uint8_t * correctOutPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t)); 
	convolution(in_Pixels, width, height, x_sobel, filterWidth, correctOutPixels);

    // apply using device
    uchar3 * outPixels1 = (uchar3 *)malloc(width * height * sizeof(uchar3));
	blurImg(inPixels, width, height, filter, filterWidth, outPixels1, true, blockSize, 1);
	printError(outPixels1, correctOutPixels, width, height);
	
	// Blur input image using device, kernel 2
	uchar3 * outPixels2 = (uchar3 *)malloc(width * height * sizeof(uchar3));
	blurImg(inPixels, width, height, filter, filterWidth, outPixels2, true, blockSize, 2);
	printError(outPixels2, correctOutPixels, width, height);

	// Blur input image using device, kernel 3
	uchar3 * outPixels3 = (uchar3 *)malloc(width * height * sizeof(uchar3));
	blurImg(inPixels, width, height, filter, filterWidth, outPixels3, true, blockSize, 3);
	printError(outPixels3, correctOutPixels, width, height);

    // Calculate the importance & add them together

    // Calculate the importance from the end

    // Find & Erase seam


	// Write results to files
	writePnm(in_Pixels, width, height, concatStr(argv[2], "_resize_host.pnm"));
    // writePnm(hostOutPixels, width, height, concatStr(outFileNameBase, "_host.pnm"));
    // writePnm(deviceOutPixels, width, height, concatStr(outFileNameBase, "_device.pnm"));
    
}