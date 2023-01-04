#include <stdio.h>
#include <stdint.h>
__constant__ float dc_filter[9];

/**
 * Working with files
 * File types:
 * - .pnm file with "P2" or "P3" header
 * - filter (plain text)
 */
void readPnm(char * fileName, int &width, int &height, uint8_t * &pixels);
void writePnm(uint8_t * pixels, int width, int height, char * fileName);

float *  readFilter(char * fileName, int &filterWidth);


/**
 * convert RGB to Grayscale
 */
void convertRgb2Gray(uint8_t * inPixels, int width, int height, uint8_t * outPixels, bool useDevice=false, dim3 blockSize=dim3(1));
void convertRgb2Gray_host(uint8_t * inPixels, int width, int height, uint8_t * outPixels);
__global__ void convertRgb2Gray_device(uint8_t * inPixels, int width, int height, uint8_t * outPixels);


/**
 * Convolution
 */
void convolution(uint8_t * inPixels, int width, int height, float * filter, int filterWidth, 
  uint8_t * outPixels, bool useDevice=false, dim3 blockSize=dim3(1, 1), int kernelType=1);

void detectEdges_host(uint8_t * inPixels, int width, int height, float * filter, int filterWidth, uint8_t * outPixels);

__global__ void detectEdges_kernel1(uint8_t * inPixels, int width, int height, float * filter, int filterWidth, uint8_t * outPixels);
__global__ void detectEdges_kernel2(uint8_t * inPixels, int width, int height, float * filter, int filterWidth, uint8_t * outPixels);
__global__ void detectEdges_kernel3(uint8_t * inPixels, int width, int height, int filterWidth, uint8_t * outPixels);


/**
 * Calculation the importance of pixels from the end
 */
void callImportance(uint8_t *in1, uint8_t *in2, int nRows, int nCols, uint8_t *out, bool useDevice=false, dim3 blockSize=dim3(1));
void addMatrix_host(uint8_t *in1, uint8_t *in2, int nRows, int nCols, uint8_t *out);
__global__ void addMatrix_kernel(uint8_t *in1, uint8_t *in2, int nRows, int nCols, uint8_t *out);


void importanceFromTheEnd(uchar3 * inPixels, int width, int height);//unfinished

/**
 * Another functions
 */
void printDeviceInfo();
char * concatStr(const char * s1, const char * s2);
float computeError(uint8_t * a1, uint8_t * a2, int n);


/**
 * Time counter
 */
struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop() {
        cudaEventRecord(stop, 0);
    }

    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};