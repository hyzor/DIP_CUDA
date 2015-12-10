/*
 * Implementation of DES Algorithm.
 * Author      			: Minsu Kim
 * CUDA modification 	: Jesper Hansson Falkenby
 * Description 			: This code implements DES Algorithm.
 */

#include "des.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define THREADS_PER_BLOCK 1024

__device__ uint32_t IP_cu[] = {58, 50, 42, 34, 26, 18, 10, 2,
		60, 52, 44, 36, 28, 20, 12, 4,
		62, 54, 46, 38, 30, 22, 14, 6,
		64, 56, 48, 40, 32, 24, 16, 8,
		57, 49, 41, 33, 25, 17,  9, 1,
		59, 51, 43, 35, 27, 19, 11, 3,
		61, 53, 45, 37, 29, 21, 13, 5,
		63, 55, 47, 39, 31, 23, 15, 7};

__device__ uint32_t S_cu[8][64] =
{{14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7,
		 0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8,
	 4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0,
	15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13},

 {15,  1,  8, 14,  6, 11,  3,  4,  9,  7,  2, 13, 12,  0,  5, 10,
	 3, 13,  4,  7, 15,  2,  8, 14, 12,  0,  1, 10,  6,  9, 11,  5,
	 0, 14,  7, 11, 10,  4, 13,  1,  5,  8, 12,  6,  9,  3,  2, 15,
	13,  8, 10,  1,  3, 15,  4,  2, 11,  6,  7, 12,  0,  5, 14,  9},

 {10,  0,  9, 14,  6,  3, 15,  5,  1, 13, 12,  7, 11,  4,  2,  8,
	13,  7,  0,  9,  3,  4,  6, 10,  2,  8,  5, 14, 12, 11, 15,  1,
	13,  6,  4,  9,  8, 15,  3,  0, 11,  1,  2, 12,  5, 10, 14,  7,
	 1, 10, 13,  0,  6,  9,  8,  7,  4, 15, 14,  3, 11,  5,  2, 12},

 { 7, 13, 14,  3,  0,  6,  9, 10,  1,  2,  8,  5, 11, 12,  4, 15,
	13,  8, 11,  5,  6, 15,  0,  3,  4,  7,  2, 12,  1, 10, 14,  9,
	10,  6,  9,  0, 12, 11,  7, 13, 15,  1,  3, 14,  5,  2,  8,  4,
	 3, 15,  0,  6, 10,  1, 13,  8,  9,  4,  5, 11, 12,  7,  2, 14},

 { 2, 12,  4,  1,  7, 10, 11,  6,  8,  5,  3, 15, 13,  0, 14,  9,
	14, 11,  2, 12,  4,  7, 13,  1,  5,  0, 15, 10,  3,  9,  8,  6,
	 4,  2,  1, 11, 10, 13,  7,  8, 15,  9, 12,  5,  6,  3,  0, 14,
	11,  8, 12,  7,  1, 14,  2, 13,  6, 15,  0,  9, 10,  4,  5,  3},

 {12,  1, 10, 15,  9,  2,  6,  8,  0, 13,  3,  4, 14,  7,  5, 11,
	10, 15,  4,  2,  7, 12,  9,  5,  6,  1, 13, 14,  0, 11,  3,  8,
	 9, 14, 15,  5,  2,  8, 12,  3,  7,  0,  4, 10,  1, 13, 11,  6,
	 4,  3,  2, 12,  9,  5, 15, 10, 11, 14,  1,  7,  6,  0,  8, 13},

 { 4, 11,  2, 14, 15,  0,  8, 13,  3, 12,  9,  7,  5, 10,  6,  1,
	13,  0, 11,  7,  4,  9,  1, 10, 14,  3,  5, 12,  2, 15,  8,  6,
	 1,  4, 11, 13, 12,  3,  7, 14, 10, 15,  6,  8,  0,  5,  9,  2,
	 6, 11, 13,  8,  1,  4, 10,  7,  9,  5,  0, 15, 14,  2,  3, 12},

 {13,  2,  8,  4,  6, 15, 11,  1, 10,  9,  3, 14,  5,  0, 12,  7,
	 1, 15, 13,  8, 10,  3,  7,  4, 12,  5,  6, 11,  0, 14,  9,  2,
	 7, 11,  4,  1,  9, 12, 14,  2,  0,  6, 10, 13, 15,  3,  5,  8,
	 2,  1, 14,  7,  4, 10,  8, 13, 15, 12,  9,  0,  3,  5,  6, 11}};

__device__ uint32_t IP_inv_cu[] = {40, 8, 48, 16, 56, 24, 64, 32,
								39, 7, 47, 15, 55, 23, 63, 31,
								38, 6, 46, 14, 54, 22, 62, 30,
								37, 5, 45, 13, 53, 21, 61, 29,
								36, 4, 44, 12, 52, 20, 60, 28,
								35, 3, 43, 11, 51, 19, 59, 27,
								34, 2, 42, 10, 50, 18, 58, 26,
								33, 1, 41,  9, 49, 17, 57, 25};

__device__ uint32_t E_cu[] = {32,  1,  2,  3,  4,  5,
					  4,  5,  6,  7,  8,  9,
					  8,  9, 10, 11, 12, 13,
					 12, 13, 14, 15, 16, 17,
					 16, 17, 18, 19, 20, 21,
					 20, 21, 22, 23, 24, 25,
					 24, 25, 26, 27, 28, 29,
					 28, 29, 30, 31, 32,  1};

__device__ uint32_t P_cu[] = {16,  7, 20, 21,
					 29, 12, 28, 17,
					  1, 15, 23, 26,
						5, 18, 31, 10,
						2,  8, 24, 14,
					 32, 27,  3,  9,
					 19, 13, 30,  6,
					 22, 11,  4, 25};

extern "C" void printBinary(long long int num) {
	int i;
	for(i = 0; i < 64; ++i) {
		printf("%lld", LAST(num >> (63-i), 1));
		if(i%4 == 3) printf(" ");
	}
	printf("\n");
}

extern "C" void keyGen(char* key) {
	long long int newkey = 0;
	int i;
	FILE* keyFile = fopen(key, "w");

	if(!keyFile) {
		printf("File Error\n");
		return;
	}

	/* Simply generate a random key of size 64 */
	srand(time(NULL));
	for(i = 0; i < 8; ++i)
		newkey |= LAST(rand(), 8) << (i * 8);
	fwrite(&newkey, sizeof(long long int), 1, keyFile);

	fclose(keyFile);
}

extern "C" inline long long int rotLeft28(long int in, int shift) {
	return LAST((in << shift) | (in >> (28-shift)), 28);
}

extern "C" long long int* keySchedule(long long int key, mode mode) {
	/* key : 64 bits, return keys : 48 bits each */

	int i;
	long long int temp = 0;
	long long int rotated = 0;
	long long int* keys = (long long int*)
												malloc(sizeof(long long int) * 16);

	memset(keys, 0, sizeof(long long int) * 16);

	// Pass PC-1
	PERMUTATE(temp, key, PC_1, 56);

	// Generate 16 subkeys
	for(i = 0; i < 16; ++i) {
		// Rotate upper half and lower half, respectively
		rotated =
		          rotLeft28(LAST(temp >> 28, 28), KS[i]) << 28
		            | rotLeft28(LAST(temp, 28), KS[i]);

		// Pass PC-2
		if(mode == enc) {
			PERMUTATE(keys[i], rotated, PC_2, 48);
		}
		else { // mode == dec : reverse
			PERMUTATE(keys[15-i], rotated, PC_2, 48);
		}
		
		// Propagate
		temp = rotated;
	}

	return keys;
}

__device__ unsigned int F_cu(unsigned int c, long long int key) {
	/* c : 32 bits, key : 48 bits, return : 32 bits */
	int i;

	// Expansion
	long long int expanded = 0;
	PERMUTATE(expanded, (long long int) c, E_cu, 48);

	// XOR
	long long int xored = expanded ^ key;

	// S Boxing
	unsigned int sBoxed = 0;
	for(i = 0; i < 8; ++i) {
		unsigned char x = LAST(xored >> ((7-i)*6), 6);
		sBoxed |= ((long long int) sBox_cu(S_cu[i], x)) << ((7-i)*4);
	}

	// Permutation
	long long int result = 0;
	PERMUTATE(result, sBoxed, P_cu, 32);

	return result;
}

__global__ void DES_cu(long long int* results, long long int *MD, long long int *keys, unsigned int num_bytes) {
	// Find global thread ID
	int t_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (t_id >= num_bytes)
		return;

	// Initial permutation
	long long int permutated = 0;
	PERMUTATE(permutated, MD[t_id], IP_cu, 64);

	// XOR iteration
	int i;
	long long int prev = permutated;
	long long int curr = 0;
	for(i = 0; i < 16; ++i) {
		curr = 0;
        curr |= LAST(prev, 32) << 32;
        curr |= LAST(prev >> 32, 32) ^ F_cu(LAST(prev, 32), keys[i]);
		prev = curr;
	}

	// Irregular swap
	long long int swapped =
			((curr << 32) & 0xffffffff00000000ULL)
				| ((curr >> 32) & 0x00000000ffffffffULL);

	// Inverse permutation
	long long int result = 0;
	PERMUTATE(result, swapped, IP_inv_cu, 64);
	results[t_id] = result;
}

__device__ unsigned char sBox_cu(uint32_t table[], unsigned char x) {
	/* x : 6 bits, return : 4 bits */
	unsigned char row = ((x & 0x20UL) >> 4) | (x & 1UL);
	unsigned char col = LAST(x >> 1UL, 4);
	return table[row * 16 + col];
}

extern "C" void run_cu(FILE* inFile, FILE* outFile, FILE* keyFile, mode mode) {
	cudaError_t cuda_err;

	long long int mainkey;
	fread(&mainkey, sizeof(long long int), 1, keyFile);
	long long int *subkeys = keySchedule(mainkey, mode);

	// Read input data stream
	fseek(inFile, 0, SEEK_END);
	int inputBytes = ftell(inFile) - ((mode == dec) ? sizeof(int) : 0);
	rewind(inFile);

	int mallocSize = inputBytes + (SIZE_LLI - (inputBytes % SIZE_LLI)) % 8;
	long long int *inputStream = (long long int*) malloc(mallocSize);
	memset(inputStream, 0, mallocSize);
	fread(inputStream, 1, inputBytes, inFile);

    if(mode == dec) // Read size of plain text at the end of enc file
        fread(&inputBytes, sizeof(int), 1, inFile);

	int numChars = mallocSize / SIZE_LLI;
	long long int* results_cu;
	cuda_err = cudaMalloc((void**)&results_cu, mallocSize);

    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate results_cu (error code %s)!\n", cudaGetErrorString(cuda_err));
        exit(EXIT_FAILURE);
    }

    long long int* inputStream_cu;
    cuda_err = cudaMalloc((void**)&inputStream_cu, mallocSize);

    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate inputStream_cu (error code %s)!\n", cudaGetErrorString(cuda_err));
        exit(EXIT_FAILURE);
    }

    size_t subkeys_size = sizeof(long long int) * 16;

    long long int* subkeys_cu;
    cuda_err = cudaMalloc(&subkeys_cu, subkeys_size);

    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate subkeys_cu (error code %s)!\n", cudaGetErrorString(cuda_err));
        exit(EXIT_FAILURE);
    }

    cuda_err = cudaMemcpy(inputStream_cu, inputStream, mallocSize, cudaMemcpyHostToDevice);

    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy inputStream from host to device (error code %s)!\n", cudaGetErrorString(cuda_err));
        exit(EXIT_FAILURE);
    }

    cuda_err = cudaMemcpy(subkeys_cu, subkeys, subkeys_size, cudaMemcpyHostToDevice);

    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy subkeys from host to device (error code %s)!\n", cudaGetErrorString(cuda_err));
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock;
    int gridSize;

    if (numChars > THREADS_PER_BLOCK)
    {
    	threadsPerBlock = THREADS_PER_BLOCK;
    	gridSize = (int)ceil((float)numChars / THREADS_PER_BLOCK);
    }
    else {
    	threadsPerBlock = numChars;
    	gridSize = 1;
    }

	DES_cu<<<gridSize, threadsPerBlock>>>(results_cu, inputStream_cu, subkeys_cu, numChars);

	long long int* results = (long long int*) malloc(mallocSize);
	cuda_err = cudaMemcpy(results, results_cu, mallocSize, cudaMemcpyDeviceToHost);

    if (cuda_err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n", cudaGetErrorString(cuda_err));
        exit(EXIT_FAILURE);
    }

    // Write out
    fwrite(results, 1, mode == enc ? mallocSize : inputBytes, outFile);

    if(mode == enc) // At the end of encryption, record size of plain text
        fwrite(&inputBytes, sizeof(int), 1, outFile);

	free(subkeys);
	free(inputStream);
	free(results);

	cudaFree(results_cu);
}

extern "C" void encryption_cu(char *in, char *out, char *key) {
	FILE* inFile = fopen(in, "r");
	FILE* outFile = fopen(out, "w");
	FILE* keyFile = fopen(key, "r");

	if(!inFile || !outFile || !keyFile) {
		printf("File Error\n");
		return;
	}

	run_cu(inFile, outFile, keyFile, enc);

	fclose(inFile);
	fclose(outFile);
	fclose(keyFile);
}

extern "C" void decryption_cu(char *in, char *out, char *key) {
	FILE* inFile = fopen(in, "r");
	FILE* outFile = fopen(out, "w");
	FILE* keyFile = fopen(key, "r");

	if(!inFile || !outFile || !keyFile) {
		printf("File Error\n");
		return;
	}

	run_cu(inFile, outFile, keyFile, dec);

	fclose(inFile);
	fclose(outFile);
	fclose(keyFile);
}
