#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <time.h>
#include <cuda.h>
#include "kd-tree.h"
#include "cuda-kd-tree.h"

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
		}																	\
}

//The necessary device arrays
kdNode *d_dataArray;
int *d_resultArray;

// Computes the time difference between two time values
double TimeDiff(timeval t1, timeval t2) 
{
	double time;
	time = (t2.tv_sec - t1.tv_sec) * 1000.0; // sec to ms
	time += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
	return time;
}

// Computes the Euclidean distance between two particles
__device__ float Distance(const kdNode &a, const kdNode &b) 
{
	float dist = 0;
	for (int i = 0; i < 3; i++) {
		float d = a.coords[i] - b.coords[i];
		dist += d * d;
	}
	return sqrt(dist);
}

// Links together particles that are found within linking length range of one another
__device__ void EvaluateParticlePairsWithinLinkingLength(int *resultArray, int queryIndex, int target) 
{
	int targetCur, targetBack;
	int selfCur, selfBack;
	targetCur = target;
	targetBack = resultArray[target];
	selfCur = queryIndex;
	selfBack = resultArray[queryIndex];
	while (selfCur != selfBack || targetCur != targetBack) {
		targetCur = targetBack;
		targetBack = resultArray[targetCur];
		selfCur = selfBack;
		selfBack = resultArray[selfCur];
	}
	if (selfBack != targetBack) {
		if (selfBack < targetBack) {
			resultArray[targetBack] = selfBack;
		}
		else if (selfBack > targetBack) {
			resultArray[selfBack] = targetBack;
		}
	}
}



// Iteratively searches the kd-tree - this method is prefered as it uses no extra memory at only a small hit in run-time
__device__ void SearchKdTreeIteratively(const kdNode *dataArray, const kdNode &queryPoint, int *resultArray, float linkingLength, int kdRoot, int queryIndex) 
{
	int previousNode = kdRoot;
	int currentNode = kdRoot;
	bool goingDown = true;
	int splitAxis = dataArray[currentNode].splitDim;
	float coordinateValueCurrentNode = dataArray[currentNode].coords[splitAxis];
	float coordinateValueQueryNode = queryPoint.coords[splitAxis];

	float calculatedDistance = Distance(queryPoint, dataArray[currentNode]);
	if (calculatedDistance <= linkingLength && calculatedDistance != 0)
		EvaluateParticlePairsWithinLinkingLength(resultArray, queryIndex, currentNode);

	bool possibleNodeLeft = coordinateValueCurrentNode > (coordinateValueQueryNode - linkingLength);
	bool possibleNodeRight = coordinateValueCurrentNode < (coordinateValueQueryNode + linkingLength);

	if (!possibleNodeLeft && !possibleNodeRight)
		return;
	if (possibleNodeLeft)
		currentNode = dataArray[currentNode].left;
	else
		currentNode = dataArray[currentNode].right;

	while (!(currentNode == kdRoot && previousNode == dataArray[currentNode].right)) 
	{
		splitAxis = dataArray[currentNode].splitDim;
		coordinateValueCurrentNode = dataArray[currentNode].coords[splitAxis];
		coordinateValueQueryNode = queryPoint.coords[splitAxis];
		int currentLeft = dataArray[currentNode].left;
		int currentRight = dataArray[currentNode].right;

		if (goingDown) 
		{
			possibleNodeLeft = coordinateValueCurrentNode > (coordinateValueQueryNode - linkingLength);;
			possibleNodeRight = coordinateValueCurrentNode < (coordinateValueQueryNode + linkingLength);

			calculatedDistance = Distance(queryPoint, dataArray[currentNode]);
			if (calculatedDistance <= linkingLength && calculatedDistance != 0)
				EvaluateParticlePairsWithinLinkingLength(resultArray, queryIndex, currentNode);

			if (possibleNodeLeft && currentLeft != -1) 
			{
				previousNode = currentNode;
				currentNode = currentLeft;
			}
			else if (possibleNodeRight && currentRight != -1) 
			{
				previousNode = currentNode;
				currentNode = currentRight;
			}
			else 
			{
				previousNode = currentNode;
				currentNode = dataArray[previousNode].parent;
				goingDown = false;
			}
		}
		else 
		{
			possibleNodeRight = coordinateValueCurrentNode < (coordinateValueQueryNode + linkingLength);
			if (previousNode == currentLeft) {
				if (possibleNodeRight && currentRight != -1) 
				{
					previousNode = currentNode;
					currentNode = currentRight;
					goingDown = true;
				}
				else 
					previousNode = currentRight;
				
			}
			else if (previousNode == currentRight) 
			{
				previousNode = currentNode;
				currentNode = dataArray[previousNode].parent;
			}
		}
	}
}

// Recursively searches the kd-tree - this method is not prefered as it does not run on devices supporting compute <2.0
__device__ void SearchKDTreeRecursively(const kdNode *dataArray, const kdNode &queryPoint, int *resultArray, float linkingLength, int kdRoot, int queryIndex)
{
	int splitAxis = dataArray[kdRoot].splitDim;
	float distance = Distance(queryPoint, dataArray[kdRoot]);
	if (distance <= linkingLength && distance != 0)
		EvaluateParticlePairsWithinLinkingLength(resultArray, queryIndex, kdRoot);
	if (dataArray[kdRoot].left != -1)
		if (dataArray[kdRoot].coords[splitAxis] > (queryPoint.coords[splitAxis] - linkingLength))
			SearchKDTreeRecursively(dataArray, queryPoint, resultArray, linkingLength, dataArray[kdRoot].left, queryIndex);
	if (dataArray[kdRoot].right != -1)
		if (dataArray[kdRoot].coords[splitAxis] < (queryPoint.coords[splitAxis] + linkingLength))
			SearchKDTreeRecursively(dataArray, queryPoint, resultArray, linkingLength, dataArray[kdRoot].right, queryIndex);
}

// Recursively searches the kd-tree - this method is not prefered as recursion is simulated and uses a lot of additional memory
__device__ void SearchKdTreeRecursivelySimulated(const kdNode *dataArray, const kdNode &queryPoint, int *resultArray, float linkingLength, int kdRoot, int queryIndex) 
{
	float distance;
	int recurseIndex;
	int splitAxis, count = 0;
	int to_visit[500];
	to_visit[count] = kdRoot;
	while (count > -1) 
	{
		recurseIndex = to_visit[count];
		count--;
		splitAxis = dataArray[recurseIndex].splitDim;
		distance = Distance(queryPoint, dataArray[recurseIndex]);
		if (distance <= linkingLength && distance != 0) 
			EvaluateParticlePairsWithinLinkingLength(resultArray, queryIndex, recurseIndex);		
		if (dataArray[recurseIndex].left != -1) 
			if (dataArray[recurseIndex].coords[splitAxis] > (queryPoint.coords[splitAxis] - linkingLength)) 
			{
				count++;
				to_visit[count] = dataArray[recurseIndex].left;
			}		
		if (dataArray[recurseIndex].right != -1) 
			if (dataArray[recurseIndex].coords[splitAxis] < (queryPoint.coords[splitAxis] + linkingLength)) 
			{
				count++;
				to_visit[count] = dataArray[recurseIndex].right;
			}		
	}
}

// Initalises threads and assigns indexes
__global__ void BeginRangeQuery(const kdNode *dataArray, int *resultArray, const float linkingLength, const int nParticles, const int kdRoot)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nParticles)
		return;
	SearchKdTreeIteratively(dataArray, dataArray[idx], resultArray, linkingLength, kdRoot, idx);
}

// Copies data to device and initialises memory
void CopyDataToDevice(kdNode *dataArray, int *resultArray, int nParticles) 
{
	// Initialise memory for data and result arrays
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_resultArray, sizeof(int) * nParticles));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_dataArray, sizeof(kdNode) * nParticles));

	float memoryAllocated = ((float)sizeof(d_dataArray[0]) + (float)sizeof(d_resultArray[0])) / 1000000 * nParticles;
	printf("Allocated %fMB of device memory...\n", memoryAllocated);

	//Copy data and result arrays to device
	CUDA_CHECK_RETURN(cudaMemcpy(d_resultArray, resultArray, sizeof(int) * nParticles, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_dataArray, dataArray, sizeof(kdNode) * nParticles, cudaMemcpyHostToDevice));

	printf("Copied all data to device...\n");
}

// Sets up and invokes kernel
void PerformRangeQuery(float linkingLength, int nParticles, int kdRoot) 
{
	int threads = 256;
	int blocks = nParticles / threads + ((nParticles % threads) ? 1 : 0);

	BeginRangeQuery <<<blocks, threads>>>(d_dataArray, d_resultArray, linkingLength, nParticles, kdRoot);
	cudaThreadSynchronize();
}

// Fetches the result array from device and copies it back to host
void FetchDeviceResultArray(int *resultArray, int nParticles) 
{
	CUDA_CHECK_RETURN(cudaMemcpy(resultArray, d_resultArray, sizeof(int) * nParticles, cudaMemcpyDeviceToHost));
}

// Returns the result array to its initial values in case we want to run on another linking length
void RefreshDeviceResultArray(int *resultArray, int nParticles) 
{
	for (int i = 0; i < nParticles; i++) {
		resultArray[i] = i;
	}
	CUDA_CHECK_RETURN(cudaMemcpy(d_resultArray, resultArray, sizeof(int) * nParticles, cudaMemcpyHostToDevice));
}

// Releases all device memory
void ReleaseDeviceMemory() 
{
	cudaFree(d_resultArray);
	cudaFree(d_dataArray);
}
