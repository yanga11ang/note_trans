#ifndef SETTING_H
#define SETTING_H
#define INT long long
#define REAL float
#include <string.h>
#include <stdio.h>
//#include <string>
char inPath[1000] = "../data/FB15K/";
char outPath[1000] = "../data/FB15K/";

extern "C"
void setInPath(char *path) {
	INT len = strlen(path);
	inPath[0] = '\0';
	strcat(inPath, path);
	printf("Input Files Path : %s\n", inPath);
}

extern "C"
void setOutPath(char *path) {
	INT len = strlen(path);
	outPath[0] = '\0';
	strcat(outPath, path);
	printf("Output Files Path : %s\n", outPath);
}

/*
============================================================
*/

INT workThreads = 1;

extern "C"
void setWorkThreads(INT threads) {
	workThreads = threads;
}

extern "C"
INT getWorkThreads() {
	return workThreads;
}

/*
============================================================
*/

INT relationTotal = 0;
INT entityTotal = 0;
INT tripleTotal = 0;
INT testTotal = 0;
INT trainTotal = 0;
INT validTotal = 0;

extern "C"
INT getEntityTotal() {
	return entityTotal;
}

extern "C"
INT getRelationTotal() {
	return relationTotal;
}

extern "C"
INT getTripleTotal() {
	return tripleTotal;
}

extern "C"
INT getTrainTotal() {
	return trainTotal;
}

extern "C"
INT getTestTotal() {
	return testTotal;
}

extern "C"
INT getValidTotal() {
	return validTotal;
}
/*
============================================================
*/

INT bernFlag = 0;

extern "C"
void setBern(INT con) {
	bernFlag = con;
}

#endif
