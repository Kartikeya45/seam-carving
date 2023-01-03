#include <stdio.h>
/**
 * Working with files
 * File types:
 * - .pnm file with "P2" or "P3" header
 * - filter (plain text)
 */
void readPnm_uint8_t(char * fileName, int &numChannels, int &width, int &height, uint8_t * &pixels);
void writePnm_uint8_t(uint8_t * pixels, int numChannels, int width, int height, char * fileName);
void readPnm_uchar3(char * fileName, int &width, int &height, uchar3 * &pixels);
void writePnm_uchar3(uchar3 * pixels, int width, int height, char * fileName);

void readFilter(char * fileName, int &filterWidth, float * filter);


/**
 * convert RGB to Grayscale
 */
void convertRgb2Gray(uint8_t * inPixels, int width, int height, uint8_t * outPixels, bool useDevice=false, dim3 blockSize=dim3(1));
void convertRgb2Gray_host(uint8_t * inPixels, int width, int height, uint8_t * outPixels);//host
__global__ void convertRgb2GrayKernel(uint8_t * inPixels, int width, int height, uint8_t * outPixels);//device

/**
 * Convolution
 *
 */


/**
 * Calculation the importance of pixels from the end
 */
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