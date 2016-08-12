#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "kd-tree.h"

/*
* This kd-tree construction was modified from that found at rosettacode.org/wiki/k-d_tree#C
*/

// Swaps two kd-nodes
void SwapNodes(struct kdNode *x, struct kdNode *y) {
	float coords[3];
	int tempId;
	memcpy(coords, x->coords, sizeof(coords));
	memcpy(x->coords, y->coords, sizeof(coords));
	memcpy(y->coords, coords, sizeof(coords));
}

// Finds the median for the specified range of kd-nodes with quick select
kdNode* FindMedian(struct kdNode *start, struct kdNode *end, int idx)
{
	if (end <= start) return NULL;
	if (end == start + 1)
		return start;

	struct kdNode *p, *store, *mid = start + (end - start) / 2;
	float pivot;

	while (1) {
		pivot = mid->coords[idx];
		SwapNodes(mid, end - 1);
		for (store = p = start; p < end; p++) {
			if (p->coords[idx] < pivot) {
				if (p != store)
					SwapNodes(p, store);
				store++;
			}
		}
		SwapNodes(store, end - 1);

		if (store->coords[idx] == mid->coords[idx])
			return mid;

		if (store > mid)
			end = store;
		else
			start = store;
	}
}

//Constructs the kd-tree recursively overlayed on the data array and returns the array index of the root node
int ConstructTree(kdNode *arrayBeginning, kdNode* startPointer, int length, int curDimension, int dim, int parent)
{	
	kdNode *midPointer;
	
	int index;

	midPointer = FindMedian(startPointer, startPointer + length, curDimension);
	if (midPointer) {
		index = midPointer - arrayBeginning;
		midPointer->splitDim = curDimension;
		curDimension = (curDimension + 1) % dim;
		midPointer->parent = parent;
		midPointer->left = ConstructTree(arrayBeginning, startPointer, midPointer - startPointer, curDimension, dim, index);
		midPointer->right = ConstructTree(arrayBeginning, midPointer + 1, startPointer + length - (midPointer + 1), curDimension, dim, index);
		return index;
	}
	return -1;
}
