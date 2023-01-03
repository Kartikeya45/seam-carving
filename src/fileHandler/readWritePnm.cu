#include <stdio.h>
#include <stdint.h>

/**
 * Type: uint8_t 
 */
void readPnm(char * fileName, int &width, int &height, uint8_t * &pixels) {
	FILE * f = fopen(fileName, "r");
	if (f == NULL) {
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

  int numChannels;
	char type[3];
	fscanf(f, "%s", type);
	if (strcmp(type, "P2") == 0)
		numChannels = 1;
	else if (strcmp(type, "P3") == 0)
		numChannels = 3;
	else { // we don't touch other types
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);

	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) { // Assuming 1 byte per value
	
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uint8_t *)malloc(width * height * numChannels);
	for (int i = 0; i < width * height * numChannels; i++)
		fscanf(f, "%hhu", &pixels[i]);

	fclose(f);
}

void writePnm(uint8_t * pixels, int width, int height, char * fileName) {
	FILE * f = fopen(fileName, "w");
	if (f == NULL) {
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

  fprintf(f, "P2\n");//1 channel only
	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}
