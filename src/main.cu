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

void step0_checkInput(int argc, char ** argv, dim3 &blockSize) {
    if (argc != 4 && argc != 6) {
		printf("The number of arguments is invalid\n");
		exit(EXIT_FAILURE);
	}

    int width, height;
    uint8_t * inPixels;

    readPnm(argv[1], width, height, inPixels);
	
    // Check if we can resize the width to be smaller or not
    int newWidthSize = atoi(argv[3]);
    if (newWidthSize >= width || newWidthSize <= 0) {
        printf("The size you've given is wrong!\n");
		exit(EXIT_FAILURE);
    }

    // Get block Size
    if (argc == 5) {
		blockSize.x = atoi(argv[4]);
		blockSize.y = atoi(argv[5]);
	}

    // Check if GPU is still working
    printDeviceInfo();
}


void step1_convertRgb2Gray(char * inFileName, char * outFileName, dim3 blockSize, int version) {
    int width, height;
    uint8_t * inPixels;

    readPnm(inFileName, width, height, inPixels);

    // host
    uint8_t * outPixels_host = (uint8_t *)malloc(width * height);
	convertRgb2Gray(inPixels, width, height, outPixels_host);

    // device (version 1)
    uint8_t * outPixels = (uint8_t *)malloc(width * height);
    convertRgb2Gray(inPixels, width, height, outPixels, true, blockSize);
    
    // Error
    printError("RGB TO GRAYSCALE - Error ", outPixels, outPixels_host, width, height);

    // Write the Grayscale images 
	writePnm(outPixels_host, width, height, outFileName);
	writePnm(outPixels, width, height, outFileName);

    // Free memory
    free(inPixels);
    free(outPixels_host);
    free(outPixels);
}


void step2_detectEdges(char * inFileName, char * outFileName = "", dim3 blockSize, int version, char * filterName) {
    // Read x-sobel & y-sobel
    int filterWidth;
    float * filter;

    if (strcmp(filterName, "x") == 0)
        filter = readFilter("../filter/x-sobel.txt", filterWidth);
    else if (strcmp(filterName, "y") == 0) 
        filter = readFilter("../filter/y-sobel.txt", filterWidth);
    
    // if x-sobel & y-sobel doesn't have the same width or their width is even:
    if (filterWidth % 2 == 0) {
        printf("Filter's width is not even!\n");
		exit(EXIT_FAILURE);
    }

    //Read grayscale image
    int width, height;
    uint8_t * inPixels;
    readPnm(inFileName, width, height, inPixels);

    // apply using host
    uint8_t * outPixels_host = (uint8_t *)malloc(width * height * sizeof(uint8_t)); 
	convolution(inPixels, width, height, filter, filterWidth, outPixels_host);
    
    uint8_t * outPixels_device = (uint8_t *)malloc(width * height * sizeof(uint8_t));

    convolution(inPixels, width, height, filter, filterWidth, outPixels_device, true, blockSize, version);
    printf("Version %d", version);
    printError("CONVOLUTION - Error", outPixels_device, outPixels_host, width, height);
    
    // write file
    if (strcmp(outFileName, "") != 0)
        //writePnm(outPixels_host, width, height, outFileName); // write using host - INCORRECT
        writePnm(outPixels_device, width, height, outFileName); // write using device - CORRECT

    // free memory
    free(inPixels);
    free(outPixels_host);
    free(outPixels_device);
}

/**
 * @param inFileName: user input (= argv[2])
 */
void step3_calculateImportance(char * inFileName, dim3 blockSize, int version) {

    int width, height, _width, _height;
    uint8_t * in1, * in2;

    // Read detected edges file
    readPnm(concatStr(inFileName, "_x_edge.pnm"), width, height, in1);
    readPnm(concatStr(inFileName, "_y_edge.pnm"), _width, _height, in2);

    // Check if inFileName is still intact or not
    if (width != _width || height != _height) {
        printf("Your edge detection files may be wrong!\n");
		exit(EXIT_FAILURE);
    }

    // Host
    uint8_t * outPixels_host = (uint8_t *)malloc(width * height);
    calImportance(in1, in2, height, width, outPixels_host);

    // Device
    uint8_t * outPixels_device = (uint8_t *)malloc(width * height);
    calImportance(in1, in2, height, width, outPixels_host, true, blockSize);

    printError("ADD MATRIX - Error", outPixels_device, outPixels_host, width, height);

    // Free memory
    free(in1);
    free(in2);

    
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

    // STEP 0
    dim3 blockSize(32, 32);
    step0_checkInput(argc, argv, blockSize);

    // STEP 1
    char * grayscalePath = concatStr(argv[2], "_grayscale.pnm");
    step1_convertRgb2Gray(argv[1], grayscalePath, blockSize, 1);

    // STEP 2
    step2_detectEdges(grayscalePath, concatStr(argv[2], "_x_edge.pnm"), blockSize, 1, "x");
    step2_detectEdges(grayscalePath, blockSize, 2, "x");
    step2_detectEdges(grayscalePath, blockSize, 3, "x");

    step2_detectEdges(grayscalePath, concatStr(argv[2], "_y_edge.pnm"), blockSize, 1, "y");
    step2_detectEdges(grayscalePath, blockSize, 2, "y");
    step2_detectEdges(grayscalePath, blockSize, 3, "y");

    // STEP 3
    step3_calculateImportance(argv[2], blockSize, 1);

    // Calculate the importance from the end

    // Find & Erase seam

    
}