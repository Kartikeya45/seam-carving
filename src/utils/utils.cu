#include <stdio.h>
#include <stdint.h>
#include "../library.h"

void printDeviceInfo() {
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("_____________GPU info_____________\n");
    printf("|Name:                   %s|\n", devProv.name);
    printf("|Compute capability:          %d.%d|\n", devProv.major, devProv.minor);
    printf("|Num SMs:                      %d|\n", devProv.multiProcessorCount);
    printf("|Max num threads per SM:     %d|\n", devProv.maxThreadsPerMultiProcessor); 
    printf("|Max num warps per SM:         %d|\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("|GMEM:           %zu byte|\n", devProv.totalGlobalMem);
    printf("|SMEM per SM:          %zu byte|\n", devProv.sharedMemPerMultiprocessor);
    printf("|SMEM per block:       %zu byte|\n", devProv.sharedMemPerBlock);
    printf("|________________________________|\n");
}

char * concatStr(const char * s1, const char * s2) {
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}
