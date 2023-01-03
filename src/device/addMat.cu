__global__ void addMatKernel(int *in1, int *in2, int nRows, int nCols, 
        int *out)
{
    // TODO
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < nRows && c < nCols)
    { 
        int i = r * nCols + c;
        out[i] = in1[i] + in2[i];
    }
}

void addMat(int *in1, int *in2, int nRows, int nCols, 
        int *out, 
        bool useDevice=false, dim3 blockSize=dim3(1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
        // TODO
        for (int r = 0; r < nRows; r++)
        {
            for (int c = 0; c < nCols; c++)
            {
                int i = r * nCols + c;
                out[i] = in1[i] + in2[i];
            }
        }
	}
	else // Use device
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

		// TODO: Allocate device memories
        int * d_in1;
        int * d_in2;
        int * d_out;
        size_t nBytes = nRows * nCols * sizeof(int);
        CHECK(cudaMalloc(&d_in1, nBytes));
        CHECK(cudaMalloc(&d_in2, nBytes));
        CHECK(cudaMalloc(&d_out, nBytes));

		// TODO: Copy data to device memories
        CHECK(cudaMemcpy(d_in1, in1, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_in2, in2, nBytes, cudaMemcpyHostToDevice));

		// TODO: Set grid size and call kernel
        dim3 gridSize((nCols - 1) / blockSize.x + 1, 
                (nRows - 1) / blockSize.y + 1);
        addMatKernel<<<gridSize, blockSize>>>(d_in1, d_in2, nRows, nCols, d_out);

		// TODO: Copy result from device memory
        CHECK(cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost));

		// TODO: Free device memories
        CHECK(cudaFree(d_in1));
        CHECK(cudaFree(d_in2));
        CHECK(cudaFree(d_out));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", 
			useDevice == true? "use device" : "use host", time);
}