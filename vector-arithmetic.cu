#include<stdio.h>
#include<stdlib.h>

int main(void){
	//declaration of host variables
	const int num = 50;
	float *z, *x, *y;
	
	float A = 34;
	
	//initialising host variables
	z = (float*) malloc(num * sizeof(float));
	x = (float*) malloc(num * sizeof(float));
	y = (float*) malloc(num * sizeof(float));
	
	for(int i = 0; i < num; i++){
		x[i] = i;
		y[i] = 7 * i;
	}

	//task to be performed by GPU
	for(int i = 0; i < num; i++){
		z[i] = A * x[i] + y[i];
	}
	
	//postprocessing: output to terminal
	for(int j = 0; j < num; j++){
		printf("%f ", z[j]);
	}
	
	return 0;
}
