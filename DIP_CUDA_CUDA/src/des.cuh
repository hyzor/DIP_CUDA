/*
 * Header file for implementation of DES Algorithm.
 * Author      			: Minsu Kim
 * CUDA modification 	: Jesper Hansson Falkenby
 * Description 			: This code implements DES Algorithm.
 */

#ifndef __DES_CUH__
#define __DES_CUH__

#include <stdio.h>
#include <stdint.h>

#define SIZE_LLI sizeof(long long int)
#define LAST(num, bits) ((num) % (1UL << (bits)))
#define PERMUTATE(dest, src, table, count) \
	do { \
		int _idx; \
		for(_idx = 0; _idx < (count); ++_idx) { \
			(dest) |= (((src)>>(table[_idx]-1)) & 0x1ULL) << (_idx); \
		} \
	} while(0)

typedef enum {enc, dec} mode;
extern "C" void keyGen(char *key);
extern "C" long long int* keySchedule(long long int key, mode mode);
extern "C" unsigned int F(unsigned int c, long long int key);
extern "C" unsigned char sBox(int table[], unsigned char x);

// CUDA equivalent functions
__device__ unsigned int F_cu(unsigned int c, long long int key);
__global__ void DES_cu(long long int *MD, long long int *keys, unsigned int num_bytes);
__device__ unsigned char sBox_cu(uint32_t table[], unsigned char x);
extern "C" void run_cu(FILE* inFile, FILE* outFile, FILE* keyFile, mode mode);
extern "C" void encryption_cu(char *in, char *out, char *key);
extern "C" void decryption_cu(char *in, char *out, char *key);

/* Tables */
static int PC_1[] = {57, 49, 41, 33, 25, 17,  9,
							 1, 58, 50, 42, 34, 26, 18,
							10,  2, 59, 51, 43, 35, 27,
							19, 11,  3, 60, 52, 44, 36,
							63, 55, 47, 39, 31, 23, 15,
							 7, 62, 54, 46, 38, 30, 22,
							14,  6, 61, 53, 45, 37, 29,
							21, 13,  5, 28, 20, 12,  4};

static int PC_2[] = {14, 17, 11, 24,  1,  5,
							 3, 28, 15,  6, 21, 10,
							23, 19, 12,  4, 26,  8,
							16,  7, 27, 20, 13,  2,
							41, 52, 31, 37, 47, 55,
							30, 40, 51, 45, 33, 48,
							44, 49, 39, 56, 34, 53,
							46, 42, 50, 36, 29, 32};

static int KS[] = {1, 1, 2, 2, 2, 2, 2, 2, 2,
						1, 2, 2, 2, 2, 2, 2, 2, 1};


#endif // ndef __DES_CUH__
