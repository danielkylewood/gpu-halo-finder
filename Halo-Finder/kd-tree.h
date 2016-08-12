struct kdNode {
	float coords[3];
	int splitDim, left, right, parent;
};

int ConstructTree(kdNode *arrayBeginning, kdNode* startPointer, int length, int curDimension, int dim, int parent);
