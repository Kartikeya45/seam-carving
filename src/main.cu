#include <stdio.h>
#include <stdint.h>
#include "library.h"


int height, width, filterWidth;
float * x_sobel, * y_sobel;
dim3 blockSize(32, 32);


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

void step0_checkInput(int argc, char ** argv, uint8_t * rgbPic) {

    if (argc != 4 && argc != 6) {
		printf("The number of arguments is invalid\n");
		exit(EXIT_FAILURE);
	}

    readPnm(argv[1], width, height, rgbPic);
	
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

    // Check filter
    int _filterWidth;
    x_sobel = readFilter("../filter/x-sobel.txt", filterWidth);
    y_sobel = readFilter("../filter/y-sobel.txt", _filterWidth);
    
    if (filterWidth != _filterWidth || filterWidth % 2 == 0) {
        printf("Filters width is not equal or not even!\n");
		exit(EXIT_FAILURE);
    }

    // Check if GPU is still working
    printDeviceInfo();
}


uint8_t * grayPixels step1_convertRgb2Gray(uint8_t * rgbPic, char * outFileName, int version) {

    uint8_t * out = (uint8_t *)malloc(width * height);

    if (version == 0) {
        convertRgb2Gray(rgbPic, width, height, out);
    } else {
        convertRgb2Gray(rgbPic, width, height, out, true, blockSize);
    }
    //printError("RGB TO GRAYSCALE - Error ", outPixels_device, outPixels_host, width, height);

    // Write the Grayscale images 
    if (version == 0) {
        writePnm(out, width, height, outFileName);
    }

    return out;
}


uint8_t * step2_detectEdges(uint8_t * grayPic, float * filter, int version) {

    uint8_t * out = (uint8_t *)malloc(width * height);

    if (version == 0) {
        convolution(grayPic, width, height, filter, filterWidth, out);
    } else {
        convolution(grayPic, width, height, filter, filterWidth, out, true, blocksize, version);
        writePnm(out, width, height, )
    }
    return out;
    
    // printf("Version %d", version);
    // printError("CONVOLUTION - Error", outPixels_device, outPixels_host, width, height);
}

/**
 * 
 */
 uint8_t * step3_calculateImportance(uint8_t * xEdge, uint8_t * yEdge, int version) {
     
    uint8_t * out = (uint8_t *)malloc(width * height);

    if (version == 0) {
        calImportance(xEdge, yEdge, height, width, out);
    }
    else {
        calImportance(xEdge, yEdge, height, width, out, true, blockSize);
    }

    return out;
}


// void resizing(char * fileName, char * outFileName = "", dim3 blockSize, int version) {

// }


/**
 * @param argc[1] name of the input file (.pmn)
 * @param argc[2] name of output file with no extension, created by using host & device
 * @param argc[3] horizontal of image you want to resize 
 * @param argc[4] - optional - default(32): blocksize.x
 * @param argc[5] - optional - default(32): blocksize.y
 * @see HW1_P2
 */
int main(int argc, char ** argv) {

    uint8_t * rgbPic;
    // int width, height, filterWidth;
    //float * x_sobel, * y_sobel;
    //dim3 blockSize(32, 32);
    
    char * grayscalePath = concatStr(argv[2], "_grayscale.pnm");
    int newWidth = atoi(argv[3]);
    
    
    step0_checkInput(argc, argv, rgbPic);
    
    
    for (int ver = 0; ver <= 3; ver++) {
        uint8_t * grayPixels = step1_convertRgb2Gray(argv[1], grayscalePath, ver);
            
        while (newWidth < width) {
            uint8_t xEdge = step2_detectEdges(grayPixels, x_sobel, 0);
            uint8_t yEdge = step2_detectEdges(grayPixels, y_sobel, 0);
            
            step3_calculateImportance(xEdge, yEdge, version);
        }
    }

    free(rgbPic);
    free(x_sobel);
    free(y_sobel);
}