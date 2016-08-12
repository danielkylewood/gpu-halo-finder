#include <cstdio>
#include <vector>
#include <cstdlib>
#include <float.h>
#include <string>
#include <iostream>
#include <fstream>
#include <time.h>
#include <Windows.h>
#include "kd-tree.h"
#include "cuda-kd-tree.h"


// Container for each particle read from the data file
struct ParticleData
{
	float mass;
	float pos[3];
	float vel[3];
	float eps;
	float phi;
};

// Container for simulation data contained at the head of the data file
struct ParticleCountData
{
	double time;
	int nbodies;
	int ndim;
	int nsph;
	int ndark;
	int nstar;
};

int ReadFileData(char* fileName, kdNode **dataArray);
void InitialiseResultArray(int* resultArray, int nParticles);
void FindParticles(int* clean, int* processed, int* parent, int currentPart, int increment);
void ComputeResults(int* rollingHaloMemberCounterArray, int* particlesProcessedArray, int* resultArray, int minimumHaloGroupCount, int nParticles);

int main() 
{		
	float linkingLength = 0.001;
	int minimumHaloGroupCount = 20;
	char* fileName = "data.bin";

	// Read file data into kd-node array
	kdNode *dataArray;
	int nParticles = ReadFileData(fileName, &dataArray);

	// Initialise result array
	int* resultArray = new int[nParticles];
	InitialiseResultArray(resultArray, nParticles);

	// Construct the kd-tree on CPU (kd-tree is built on the data array)
	int kdTreeRoot = ConstructTree(dataArray, dataArray, nParticles, 0, 3, -1);
		
	// Copy data and result arrays to GPU
	CopyDataToDevice(dataArray, resultArray, nParticles);
	
	printf("Computing halo-finding for linking length %f\n", linkingLength);

	// Calls device to perform the range queries on the kd-tree
	ComputeResultArray(linkingLength, nParticles, kdTreeRoot);

	// Fetches the result array from device
	FetchDeviceResultArray(resultArray, nParticles);

	// Zeroed helper arrays for evaluating results 
	int* haloParticleCounterArray = new int[nParticles]{};
	int* particlesProcessedArray = new int[nParticles]{};
	
	// Process result array
	ComputeResults(haloParticleCounterArray, particlesProcessedArray, resultArray, minimumHaloGroupCount, nParticles);

	//Free arrays
	ReleaseDeviceMemory();
	delete(dataArray);
	delete(resultArray);
	delete(haloParticleCounterArray);
	delete(particlesProcessedArray);
	return 0;
}

// Read file data
int ReadFileData(char* fileName, kdNode** dataArray)
{
	// Structs for chunking
	struct ParticleCountData dataDump;
	struct ParticleData particle;

	FILE *file = fopen(fileName, "rb");
	if (file == NULL) {
		printf("File not found!");
		exit(1);
	}

	// Header chunk
	fread(&dataDump, sizeof(ParticleCountData), 1, file);

	// File header data
	int nParticles = dataDump.nbodies;
	int nDark = dataDump.ndark;
	int nGas = dataDump.nsph;
	int nStar = dataDump.nstar;
	int nActive = 0;
	if (nDark) nActive += nDark;
	if (nGas) nActive += nGas;
	if (nStar) nActive += nStar;

	// Allocate memory for kd-node array
	*dataArray = new kdNode[nParticles];

	// Read each data chunk and add to kd-node array 	
	for (unsigned int i = 0; i < nParticles; i++) {
		fread(&particle, sizeof(ParticleData), 1, file);
		if (nDark) {
			(*dataArray)[i].coords[0] = particle.pos[0];
			(*dataArray)[i].coords[1] = particle.pos[1];
			(*dataArray)[i].coords[2] = particle.pos[2];
		}	
	}
	fclose(file);
	return nParticles;
}

// Initialise resultant array
void InitialiseResultArray(int* resultArray, int nParticles)
{
	// Each index represents a particle, and it's contents represent which particle it is linked to
	for (int i = 0; i < nParticles; i++)
		resultArray[i] = i;
}

// Computes results returned from device by counting the number of particles belonging 
// to seperate halos, and which of these halos exceed the minimum particles per halo count
void ComputeResults(int* haloMemberCounterArray, int* particlesProcessedArray, int* resultArray, int minimumHaloGroupCount, int nParticles)
{
	// Iterate through result array and find particles on particles that have not been processed
	for (int currentParticle = 0; currentParticle < nParticles; currentParticle++) 
	{
		if (particlesProcessedArray[currentParticle] == 0) 
			FindParticles(haloMemberCounterArray, particlesProcessedArray, resultArray, currentParticle, 0);
	}

	// Get the number of particles in each halo
	int possibleHalo = 0, definiteHalo = 0;
	for (int i = 0; i < nParticles; i++) 
	{
		if (haloMemberCounterArray[i] > 0)
			possibleHalo++;
		if (haloMemberCounterArray[i] >= minimumHaloGroupCount)
			definiteHalo++;		
	}
	printf("%d initial groups, %d actual halos\n", possibleHalo, definiteHalo);
}
 
// Determines how many particles belong to a particle at the end of the chain
void FindParticles(int* haloMemberCounterArray, int* particlesProcessedArray, int* resultArray, int currentParticle, int particleCounter) 
{
	// Check that the current particle has not been processed
	if (particlesProcessedArray[currentParticle] == 0)
	{
		particleCounter++;
		particlesProcessedArray[currentParticle] = 1;
	}

	// If the current particle is the end of the chain, record the number of particles in the chain
	if (resultArray[currentParticle] == currentParticle) 
		haloMemberCounterArray[currentParticle] = haloMemberCounterArray[currentParticle] + particleCounter;
	else 
		// Continue down the chain
		FindParticles(haloMemberCounterArray, particlesProcessedArray, resultArray, resultArray[currentParticle], particleCounter);
}

