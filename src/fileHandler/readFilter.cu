#include <stdio.h>

/**
 * @param fileName: ./host/filter/*
 * @param filterWidth should be 3 when using x-sobel or y-sobel
 * @return: float *  
 * read the first line to see how many lines we are going to read
 */
float * readFilter(char * fileName, int &filterWidth) {
	FILE * f = fopen(fileName, "r");

	if (f == NULL) {
		printf("Cannot read fiilter %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fscanf(f, "%d", &filterWidth);
	float * filter;
	filter = (float *)malloc(filterWidth * filterWidth * sizeof(float));
	int i = 0;
	while(!feof(f)) {
		fscanf(f,"%f",&filter[i]);
		i++;
	}
	fclose(f);
	return filter;
}