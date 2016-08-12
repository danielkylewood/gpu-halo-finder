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

// The necessary device arrays
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
	int targetCurrent, targetSink, selfCurrent, selfSink;
	targetCurrent = target;
	targetSink = resultArray[target];
	selfCurrent = queryIndex;
	selfSink = resultArray[queryIndex];

	// Need to find the current sink particle for the self halo or the target halo
	while (selfCurrent != selfSink || targetCurrent != targetSink)
	{
		targetCurrent = targetSink;
		targetSink = resultArray[targetCurrent];
		selfCurrent = selfSink;
		selfSink = resultArray[selfCurrent];
	}

	// If they are not currently part of the same halo, connect them using some simple conditional statements to avoid race conditions	
	bool updateSinkParticle = false;
	while ((selfSink != targetSink) && (updateSinkParticle == false))
	{
		int valueChange; 
		if (selfSink < targetSink)
		{
			valueChange = atomicCAS(&resultArray[targetSink], targetSink, selfSink);
			if (valueChange == selfSink)
				updateSinkParticle = true;
			else
				targetSink = valueChange;
		}
		else if (selfSink > targetSink)
		{
			valueChange = atomicCAS(&resultArray[selfSink], selfSink, targetSink);
			if (valueChange == selfSink)
				updateSinkParticle = true;			
			else			
				selfSink = valueChange;
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

	// Find the distance between these two particles and attempt to link them if they are within linking length distance
	float calculatedDistance = Distance(queryPoint, dataArray[currentNode]);
	if (calculatedDistance <= linkingLength && calculatedDistance != 0)
		EvaluateParticlePairsWithinLinkingLength(resultArray, queryIndex, currentNode);

	// Check if there are possible particles within linking length distance down the left or right of the tree
	bool possibleNodeLeft = coordinateValueCurrentNode > (coordinateValueQueryNode - linkingLength);
	bool possibleNodeRight = coordinateValueCurrentNode < (coordinateValueQueryNode + linkingLength);
		
	if (!possibleNodeLeft && !possibleNodeRight)
		return;
	if (possibleNodeLeft)
		currentNode = dataArray[currentNode].left;
	else
		currentNode = dataArray[currentNode].right;

	// Check that we have not finished traversing the entire tree already
	while (!(currentNode == kdRoot && previousNode == dataArray[currentNode].right)) 
	{
		splitAxis = dataArray[currentNode].splitDim;
		coordinateValueCurrentNode = dataArray[currentNode].coords[splitAxis];
		coordinateValueQueryNode = queryPoint.coords[splitAxis];
		int currentLeft = dataArray[currentNode].left;
		int currentRight = dataArray[currentNode].right;

		// If we are currently traversing down the tree
		if (goingDown) 
		{
			// Find the distance between these two particles and attempt to link them if they are within linking length distance
			calculatedDistance = Distance(queryPoint, dataArray[currentNode]);
			if (calculatedDistance <= linkingLength && calculatedDistance != 0)
				EvaluateParticlePairsWithinLinkingLength(resultArray, queryIndex, currentNode);

			// Check if there are possible particles within linking length distance down the left or right of the tree
			possibleNodeLeft = coordinateValueCurrentNode > (coordinateValueQueryNode - linkingLength);;
			possibleNodeRight = coordinateValueCurrentNode < (coordinateValueQueryNode + linkingLength);

			// If there is a possible node to the left and the node is not empty
			if (possibleNodeLeft && currentLeft != -1) 
			{
				// Go down this branch
				previousNode = currentNode;
				currentNode = currentLeft;
			}
			// If there is a possible node to the right and the node is not empty
			else if (possibleNodeRight && currentRight != -1) 
			{

				// Go down this branch
				previousNode = currentNode;
				currentNode = currentRight;
			}
			else 
			{
				// We are at a leaf node or there are no further particles of interest and should head back up the tree
				previousNode = currentNode;
				currentNode = dataArray[previousNode].parent;
				goingDown = false;
			}
		}
		// If we are currently traversing up the tree
		else 
		{
			// Check if there are possible particles within linking length distance down the right of the tree
			possibleNodeRight = coordinateValueCurrentNode < (coordinateValueQueryNode + linkingLength);

			// If we came to this node from a left child node
			if (previousNode == currentLeft) 
			{
				// If there is a possible node to the right and it is not empty
				if (possibleNodeRight && currentRight != -1) 
				{
					// Go down this branch
					previousNode = currentNode;
					currentNode = currentRight;
					goingDown = true;
				}
				else 
					// Pretend we are coming back up from this branch
					previousNode = currentRight;				
			}
			// If we came to this node from a right child node
			else if (previousNode == currentRight) 
			{
				// We are finished here and should head back to parent
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

	// If the current node is within linking length distance of the query particle
	if (distance <= linkingLength && distance != 0)
		EvaluateParticlePairsWithinLinkingLength(resultArray, queryIndex, kdRoot);

	// If there exists a node to the left and there is possibly a node within linking length distance down that branch
	if (dataArray[kdRoot].left != -1)
		if (dataArray[kdRoot].coords[splitAxis] > (queryPoint.coords[splitAxis] - linkingLength))
			SearchKDTreeRecursively(dataArray, queryPoint, resultArray, linkingLength, dataArray[kdRoot].left, queryIndex);

	// If there exists a node to the right and there is possibly a node within linking length distance down that branch
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

	// Set up the recursive stack 

	int recuriveStack[500];
	recuriveStack[count] = kdRoot;

	// While there are nodes in the recursive stack
	while (count > -1) 
	{
		recurseIndex = recuriveStack[count];
		count--;
		splitAxis = dataArray[recurseIndex].splitDim;
		distance = Distance(queryPoint, dataArray[recurseIndex]);

		// If the current node is within linking length distance of the query particle
		if (distance <= linkingLength && distance != 0) 
			EvaluateParticlePairsWithinLinkingLength(resultArray, queryIndex, recurseIndex);		

		// If there exists a node to the left and there is possibly a node within linking length distance down that branch
		if (dataArray[recurseIndex].left != -1) 
			if (dataArray[recurseIndex].coords[splitAxis] > (queryPoint.coords[splitAxis] - linkingLength)) 
			{
				count++;
				recuriveStack[count] = dataArray[recurseIndex].left;
			}		

		// If there exists a node to the right and there is possibly a node within linking length distance down that branch
		if (dataArray[recurseIndex].right != -1) 
			if (dataArray[recurseIndex].coords[splitAxis] < (queryPoint.coords[splitAxis] + linkingLength)) 
			{
				count++;
				recuriveStack[count] = dataArray[recurseIndex].right;
			}		
	}
}

// Initalises threads and assigns indexes
__global__ void BeginRangeQuery(const kdNode *dataArray, int *resultArray, const float linkingLength, const int nParticles, const int kdRoot)
{
	// Set up the thread indexes
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nParticles)
		return;

	// Select tree traversal method
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
void ComputeResultArray(float linkingLength, int nParticles, int kdRoot) 
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
