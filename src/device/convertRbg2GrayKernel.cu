__global__ void convertRgb2GrayKernel(uint8_t * inPixels, int width, int height, 
		uint8_t * outPixels)
{
	// TODO
    // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width)
    { 
		int i = r * width + c;
		uint8_t red = inPixels[3 * i];
		uint8_t green = inPixels[3 * i + 1];
		uint8_t blue = inPixels[3 * i + 2];
		outPixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
    }
}

void convertRgb2Gray(uint8_t * inPixels, int width, int height,
		uint8_t * outPixels, 
		bool useDevice=false, dim3 blockSize=dim3(1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
        // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                int i = r * width + c;
                uint8_t red = inPixels[3 * i];
                uint8_t green = inPixels[3 * i + 1];
                uint8_t blue = inPixels[3 * i + 2];
                outPixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
            }
        }
	}
	else // use device
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

		// TODO: Allocate device memories
        uint8_t * d_inPixels;
        uint8_t * d_outPixels;
        size_t nBytes = width * height * sizeof(int);
        CHECK(cudaMalloc(&d_inPixels, nBytes));
        CHECK(cudaMalloc(&d_outPixels, nBytes));

		// TODO: Copy data to device memories
        CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_outPixels, outPixels, nBytes, cudaMemcpyHostToDevice));

		// TODO: Set grid size and call kernel (remember to check kernel error)
        dim3 gridSize((width - 1) / blockSize.x + 1,
                (height - 1) / blockSize.y + 1);
        convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_outPixels);

		cudaError_t errSync  = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess) 
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

		// TODO: Copy result from device memories
        CHECK(cudaMemcpy(outPixels, d_outPixels, nBytes, cudaMemcpyDeviceToHost));

		// TODO: Free device memories
        CHECK(cudaFree(d_inPixels));
        CHECK(cudaFree(d_outPixels));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", 
			useDevice == true? "use device" : "use host", time);
}