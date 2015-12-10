#include "des.cuh"
#include <stdio.h>

void printUsage() {
	printf("Usage : ./main [k|e|d] key (input output)\n");
	printf("\t./main k key\n");
  printf("\t./main e key plain-text encrypted-text\n");
  printf("\t./main d key encrypted-text decrypted-text\n");
}

int main(int argc, char* argv[]) {
	bool invalid = false;

	switch(argc) {
		case 3: {
			if(*argv[1] == 'k') {
				/* key geneartion */
				keyGen(argv[2]);
			}
			else
				invalid = true;
			break;
		}
		case 5: {
			if(*argv[1] == 'e') {
				/* encryption */
				encryption_cu(argv[3], argv[4], argv[2]);
			}
			else if(*argv[1] == 'd') {
				/* decryption */
				decryption_cu(argv[3], argv[4], argv[2]);
			}
			else
				invalid = true;
			break;
		}
		default:
			invalid = true;
	}

	if(invalid)
		printUsage();

	return 0;
}
