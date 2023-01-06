#include <stdio.h>
#include <stdint.h>
#include "../library.h"


/**
 * @param useDevice = false (default)
 * @param blockSize = dim3(32,32) (default)
 */
void importanceToTheEnd(uint8_t * inPixels, int height, int width, uint8_t * outPixels, bool useDevice, dim3 blockSize) {
    GpuTimer timer;
	timer.Start();

    if (useDevice == false) 
        outPixels = importanceToTheEnd_host(inPixels, width, height, outPixels);
	else { 
        // Allocate device memories
        uint8_t * d_inPixels, * d_outPixels; // d_outPixels is empty. No need to cudaMemcpy
        size_t nBytes = height * width * sizeof(uint8_t);
        CHECK(cudaMalloc(&d_inPixels, nBytes));
        CHECK(cudaMalloc(&d_outPixels, nBytes));

        // Copy data to device memories
        CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes, cudaMemcpyHostToDevice));

        // Set grid size and call kernel
        dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
        // importanceToTheEnd_device<<<gridSize, blockSize>>>(d_inPixels, width, height, d_outPixels);

        // Copy result from device memory
        //CHECK(cudaMemcpy(outPixels, d_outPixels, nBytes, cudaMemcpyDeviceToHost));

        // Free device memories
        CHECK(cudaFree(d_inPixels));
        CHECK(cudaFree(d_outPixels));
    }

    timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", useDevice == true? "use device" : "use host", time);
}