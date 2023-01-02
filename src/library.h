/**
 * Once declare default args in this file, DO NOT declare anywhere else
 */ 


/**
 * Host (./host/*)
 */
void printDeviceInfo();
void readFilter(char * fileName, int &filterWidth, float * filter);
void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels);
void writePnm(uchar3 * pixels, int width, int height, char * fileName);
char * concatStr(const char * s1, const char * s2);

void importanceFromTheEnd(uchar3 * inPixels, int width, int height);


//_____________________________________________________________________________
/**
 * Device (./device/*)
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

//Hàm dưới chỉ để tham khảo
// __global__ void reduceBlksKernel1(int * in, int n, int * out);
// __global__ void reduceBlksKernel2(int * in, int n, int * out);
// __global__ void reduceBlksKernel3(int * in, int n, int * out);