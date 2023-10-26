#include "hySortOD_lib.h"

// Import dataset
__host__ int importDataset(char *fname, int N, double *dataset, int DIM) {

    FILE *fp = fopen(fname, "r");

    if (!fp) {
        printf("Unable to open file\n");
        return (1);
    }

    char buf[4096];
    int count = 0;
    int totalElements = N * DIM;
    while (fgets(buf, 4096, fp) && count < totalElements) {
        int colCnt = 0;
        char *field = strtok(buf, ",");
        double tmp;
        sscanf(field, "%lf", &tmp);
        dataset[count] = tmp;

        while (field && (colCnt < DIM - 1)) {
            colCnt++;
            count++;
            field = strtok(NULL, ",");

            if (field != NULL) {
                double tmp;
                sscanf(field, "%lf", &tmp);
                dataset[count] = tmp;
            }
        }
        count++;
    }

    fclose(fp);
    return 0;
}

// Normalize dataset
__host__ void normalizeDataset(double *dataset, int N, int DIM) {
    double minValue[DIM];
    double maxValue[DIM];

    int totalElements = N * DIM;

    for (int i = 0; i < DIM; i++) {
        minValue[i] = dataset[i];
        maxValue[i] = dataset[i];
        for (int j = i; j < totalElements; j = j + DIM) {
            if (dataset[j] > maxValue[i]) {
                maxValue[i] = dataset[j];
            }

            if (dataset[j] < minValue[i]) {
                minValue[i] = dataset[j];
            }
        }
    }

    for (int i = 0, j = 0; i < totalElements; i++, j++) {
        j = j % DIM;
        if (maxValue[j] != minValue[j]) {
            dataset[i] =
                    (double) ((dataset[i] - minValue[j]) / (maxValue[j] - minValue[j]));
        } else {
            dataset[i] = 0;
        }
    }

    return;
}

// Encode current hypercube coordinates
__inline__ __device__ void encodeHypercube(MY_DATATYPE *curHypercube, int *hypercube,
                                int DIM, int index, int encodeBlockSize,
                                int k) {
    int idx = 0;
    int dimPerBlock = floor((double) (sizeof(MY_DATATYPE) * 8) / (double) k);

    for (int i = 0; i < DIM; i++) {

        if (i % dimPerBlock == 0) {
            idx = i / dimPerBlock + index * encodeBlockSize;
        }
        curHypercube[idx] = curHypercube[idx] << k | hypercube[i];
    }
    return;
}

// Map points to hypercube and encode
__global__ void createHypercube(MY_DATATYPE *hypercube, double *dataset, int N,
                                int DIM, int BIN, int encodeBlockSize, int k) {
    int totalElements = N * DIM;
    int totalElementsPerBlock = blockDim.x / DIM;
    int threadId = blockIdx.x * totalElementsPerBlock * DIM + threadIdx.x;

    double length = (double) (1) / (double) BIN;

    extern __shared__ int tempHypercube[];

    // Map points to hypercube and store them in a temp array
    if (threadId < totalElements && threadIdx.x < totalElementsPerBlock * DIM) {
        tempHypercube[threadIdx.x] = (int) floor(dataset[threadId] / length);
    }

    __syncthreads();

    int hypercubeIndex = (threadIdx.x / DIM) + blockIdx.x * totalElementsPerBlock;

    // encode hypercubes
    if ((threadIdx.x % DIM == 0) && hypercubeIndex < N &&
        hypercubeIndex < (blockIdx.x + 1) * totalElementsPerBlock) {
        encodeHypercube(hypercube, tempHypercube + threadIdx.x, DIM, hypercubeIndex,
                        encodeBlockSize, k);
    }

    return;
}

// Return number with k bits set to 1
__host__ __device__ int setBitsTo1(int k) { return pow(2, k) - 1; }

// Decode encoded hypercube into hypercube array
__inline__ __device__ void decodeHypercube(MY_DATATYPE *hypercubeEncodedArray,
                                int *hypercube, int DIM, int hypercubeCount,
                                int threadId, int encodeBlockSize, int k) {

    int lastIndex = encodeBlockSize - 1;
    int elementsPerBlock = floor((double) (sizeof(MY_DATATYPE) * 8) / (double) k);
    int elementsPerLastBlock = DIM % (elementsPerBlock);
    int iterationCount = elementsPerLastBlock;
    int kBitsTo1 = setBitsTo1(k);

    if (elementsPerLastBlock == 0) {
        iterationCount = elementsPerBlock;
    }

    // Start from the last index
    for (int i = (DIM - 1); i >= 0; i--) {

        if (hypercubeEncodedArray[lastIndex] == 0 && iterationCount == 0) {
            lastIndex--;
            iterationCount = elementsPerBlock;
        }

        iterationCount--;

        // Get the last k bits
        hypercube[threadId * DIM + i] = hypercubeEncodedArray[lastIndex] & kBitsTo1;
        // Right shift by k bits
        hypercubeEncodedArray[lastIndex] = hypercubeEncodedArray[lastIndex] >> k;
    }

    return;
}

// Decode encoded hypercube and build hypercube array
__global__ void buildHypercubeArray(MY_DATATYPE *hypercube, int *hypercubeArray,
                                    int hypercubeCount, int DIM,
                                    int encodeBlockSize, int k) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadId < hypercubeCount) {
        decodeHypercube(hypercube + threadId * encodeBlockSize, hypercubeArray, DIM,
                        hypercubeCount, threadId, encodeBlockSize, k);
    }
    return;
}

// Supporting function to add node to a tree
__host__ void appendNode(treeNode **rootNode, int startIndex, int endIndex,
                         int coordinate, int parentIndex, int *curCount) {

    treeNode *curNode = (treeNode *) malloc(sizeof(treeNode));
    curNode->startIndex = startIndex;
    curNode->endIndex = endIndex;
    curNode->coordinate = coordinate;
    curNode->parentIndex = parentIndex;
    curNode->nextChildIndex = NONE;
    curNode->nextSiblingIndex = NONE;

    // Allocate memory for one nod
    if ((*curCount) == 0) {
        (*rootNode) = (treeNode *) malloc(sizeof(treeNode) * ((*curCount) + 1));
    } else {
        (*rootNode) =
                (treeNode *) realloc((*rootNode), sizeof(treeNode) * ((*curCount) + 1));
    }

    (*rootNode)[(*curCount)] = (*curNode);
    (*curCount)++;
    return;
}

// Build simple tree
__host__ void buildLinearTree(int *hypercube, treeNode **linearTree,
                              int *childCount, int *dimStart, int *curCount,
                              int curDim, int N, int MINSPLIT, int DIM) {
    if (curDim == DIM) {
        return;
    }

    int curDimElementCount;

    // Keep track of the nodes in current dimension
    if (curDim - 1 >= 0) {
        curDimElementCount = childCount[curDim - 1];
    } else {
        curDimElementCount = 1;
    }

    // Find current dimension starting and ending indexes
    int dimensionStart = dimStart[curDim];
    int dimensionEnd = curDimElementCount + dimensionStart;

    // Iterate from dim start to dim end
    for (int i = dimensionStart; i < dimensionEnd; i++) {

        if (curDim + 1 < DIM) {
            dimStart[curDim + 1] = i + 1;
        }

        // Find the starting and ending coordinates of curren node
        int startIndex = (*linearTree)[i].startIndex;
        int endIndex = (*linearTree)[i].endIndex;

        // Skip if it doesn't meet the threshold
        if (endIndex - startIndex < MINSPLIT) {
            continue;
        }

        int curValue = hypercube[startIndex * DIM + curDim];
        int curChildCount = 0;

        // Append nodes by mapping further
        for (int j = startIndex; j <= endIndex; j++) {

            if (hypercube[j * DIM + curDim] > curValue) {

                if (curChildCount > 0) {
                    (*linearTree)[*curCount - 1].nextSiblingIndex = *curCount;
                } else {
                    (*linearTree)[i].nextChildIndex = *curCount;
                }
                appendNode(linearTree, startIndex, j - 1, curValue, i, curCount);
                curChildCount++;
                childCount[curDim]++;

                startIndex = j;

                curValue = hypercube[j * DIM + curDim];
            }

            // Edge case
            if (j == endIndex) {
                if (curChildCount > 0) {
                    (*linearTree)[*curCount - 1].nextSiblingIndex = *curCount;
                } else {
                    (*linearTree)[i].nextChildIndex = *curCount;
                }

                appendNode(linearTree, startIndex, j, curValue, i, curCount);
                curChildCount++;
                childCount[curDim]++;
            }
        }
    }

    // Append nodes in the next dimension
    buildLinearTree(hypercube, linearTree, childCount, dimStart, curCount,
                    curDim + 1, N, MINSPLIT, DIM);
    return;
}

// Supporting function to check if the difference between current coordinates is
// less than 1
__inline__ __device__ int checkNeighbor(int index, int *hypercube, treeNode *linearTree,
                             int curIndex, int curDim) {
    if (curDim == 0) { return 1;}
    int curDif = abs(hypercube[index] - linearTree[curIndex].coordinate);
    if (curDif <= 1) {
        return 1;
    }
    return 0;
}

// Supporting function to check if current hypercube is an immediate neighbor
__inline__ __device__ int checkImmediateNeighbor(int *hypercubeA, int *hypercubeB,
                                      int hypercubeCount, int DIM, int curDim) {
    
    curDim = (curDim > 0) ? curDim - 1 : curDim;

    for (int i = curDim; i < DIM; i++) {
        if (abs(hypercubeA[i] - hypercubeB[i]) > 1) {
            return 0;
        }
    }
    return 1;
}

// Neighborhood density of subtree - locality optimized
__device__ int neighborDensitySubTree(int *hypercube, treeNode *linearTree,
                                      int hypercubeIndex, int *childCount,
                                      int *instancesCount, int parentIndex,
                                      int curDim, int N, int DIM) {

    int curDensity = 0;
    int curHypercubeIndex = parentIndex;
    int index;
    // Set end index as the sibling index of starting index
    int endIndex = linearTree[curHypercubeIndex].nextSiblingIndex;

    // Traverse from current index to end index
    while (curHypercubeIndex != NONE && curHypercubeIndex != endIndex) {

        index = hypercubeIndex * DIM + (curDim - 1);

        // Check if the current node is a leaf node
        if (linearTree[curHypercubeIndex].nextChildIndex == NONE) {
            // Iterate through all the hypercubes in the current node
            for (int i = linearTree[curHypercubeIndex].startIndex;
                 i <= linearTree[curHypercubeIndex].endIndex; i++) {

                // Check if current hypercube is an immediate neighbor
                if (checkImmediateNeighbor(hypercube + hypercubeIndex * DIM,
                                           hypercube + i * DIM, N, DIM, curDim)) {
                    curDensity += instancesCount[i];
                }
            }
        }

        // Move to child node if present and satisfies the condition for neighbor
        if ((linearTree[curHypercubeIndex].nextChildIndex != NONE &&
             checkNeighbor(index, hypercube, linearTree, curHypercubeIndex, curDim))) {

            int tempIndex = linearTree[curHypercubeIndex].nextChildIndex;

            while (tempIndex != NONE &&
                   (!checkNeighbor(index + 1, hypercube, linearTree, tempIndex, curDim))) {
                tempIndex = linearTree[tempIndex].nextSiblingIndex;
            }

            if (tempIndex != NONE) {
                curHypercubeIndex = tempIndex;
                curDim++;
            } else {
                while (curHypercubeIndex != NONE) {
                    if (linearTree[curHypercubeIndex].nextSiblingIndex != NONE) {
                        curHypercubeIndex = linearTree[curHypercubeIndex].nextSiblingIndex;
                        break;
                    }
                    curHypercubeIndex = linearTree[curHypercubeIndex].parentIndex;
                    curDim--;
                }
            }

        }
            // Move to sibling node if present
        else if (linearTree[curHypercubeIndex].nextSiblingIndex != NONE) {
            curHypercubeIndex = linearTree[curHypercubeIndex].nextSiblingIndex;
        }
            // backtrack to parent node until the current node has a sibling
        else {
            while (curHypercubeIndex != NONE) {
                curHypercubeIndex = linearTree[curHypercubeIndex].parentIndex;
                curDim--;
                if (curHypercubeIndex != NONE &&
                    linearTree[curHypercubeIndex].nextSiblingIndex != NONE) {

                    curHypercubeIndex = linearTree[curHypercubeIndex].nextSiblingIndex;
                    break;
                }
            }
        }
    }

    return curDensity;
}

// Calculate neighborhood density using locality optimized tree
__global__ void neighborhoodDensity(int *density, int *instancesCount,
                                    treeNode *linearTree, int *hypercubeArray,
                                    int *childCount, int DIM,
                                    int hypercubeCount, int *dimNodes) {

    // Set thread indexes
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int initialDimNodeCount = childCount[0];
    int totalNodes = hypercubeCount * initialDimNodeCount;
    // If  root node has no children
    // If  root node has no children
    if (initialDimNodeCount == 0) {
        // Assign one thread per distinct hypercube
        if (threadId < hypercubeCount) {
            int hypercubeIndex = threadId;
            int parentIndex = 0;
            // Calculate neighborhood density
            int curThreadDensity = neighborDensitySubTree(
                    hypercubeArray, linearTree, hypercubeIndex, childCount,
                    instancesCount, parentIndex, 1, hypercubeCount, DIM);

            atomicAdd(density + hypercubeIndex, curThreadDensity);
        }

    } else {
        // Parallelize tree traversal further if root node has children
        if (threadId < totalNodes) {
            int hypercubeIndex = threadId / initialDimNodeCount;
            // Map multiple threads to children nodes of root
            int parentIndex = dimNodes[threadId % initialDimNodeCount];

            // Calculate neighborhood density by traversing a sub section of tree
            int curThreadDensity = neighborDensitySubTree(
                    hypercubeArray, linearTree, hypercubeIndex, childCount,
                    instancesCount, parentIndex, 1, hypercubeCount, DIM);

            // use atomic addition to update neighborhood density of one hypercube
            atomicAdd(density + hypercubeIndex, curThreadDensity);
        }
    }

    return;
}

// Calculate neighborhood density using simple tree
__global__ void simpleNeighborhoodDensity(int *density, int *instancesCount,
                                          treeNode *linearTree,
                                          int *hypercubeArray, int *childCount,
                                          int DIM, int hypercubeCount) {

    // Set thread indexes
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int initialDimNodeCount = childCount[0];
    int totalNodes = hypercubeCount * initialDimNodeCount;

    // If  root node has no children
    if (initialDimNodeCount == 0) {
        // Assign one thread per distinct hypercube
        if (threadId < hypercubeCount) {
            int hypercubeIndex = threadId;
            int parentIndex = 0;
            // Calculate neighborhood density
            int curThreadDensity = neighborDensitySubTree(
                    hypercubeArray, linearTree, hypercubeIndex, childCount,
                    instancesCount, parentIndex, 1, hypercubeCount, DIM);

            atomicAdd(density + hypercubeIndex, curThreadDensity);
        }

    } else {
        // Parallelize tree traversal further if root node has children
        if (threadId < totalNodes) {
            int hypercubeIndex = threadId / initialDimNodeCount;
            // Map multiple threads to children nodes of root
            int parentIndex = (threadId % initialDimNodeCount) + 1;

            // Calculate neighborhood density by traversing a sub section of tree
            int curThreadDensity = neighborDensitySubTree(
                    hypercubeArray, linearTree, hypercubeIndex, childCount,
                    instancesCount, parentIndex, 1, hypercubeCount, DIM);
            // use atomic addition to update neighborhood density of one hypercube
            atomicAdd(density + hypercubeIndex, curThreadDensity);
        }
    }

    return;
}

// Supporting function to copy node contents
__host__ void copyContents(treeNode *linearCurNode, treeNode *optimCurNode,
                           int parentIndex, int siblingIndex, int childIndex) {
    (*optimCurNode).startIndex = (*linearCurNode).startIndex;
    (*optimCurNode).endIndex = (*linearCurNode).endIndex;
    (*optimCurNode).coordinate = (*linearCurNode).coordinate;
    (*optimCurNode).parentIndex = parentIndex;
    (*optimCurNode).nextChildIndex = childIndex;
    (*optimCurNode).nextSiblingIndex = siblingIndex;
    return;
}

// build locality optimized tree
void buildOptimizedLinearTree(treeNode *linearTree,
                              treeNode *optimizedLinearTree, int *dimNodes) {
    int curHypercubeIndex = 1;
    int optimTreeCount = 0;

    // Check if root node has children
    if (linearTree[0].nextChildIndex == NONE) {
        copyContents(linearTree, optimizedLinearTree, NONE, NONE, NONE);
        optimTreeCount++;
        return;
    }

    // Copy root node and first child of root node
    copyContents(linearTree, optimizedLinearTree, NONE, NONE, 1);
    optimTreeCount++;
    copyContents(linearTree + 1, optimizedLinearTree + 1, 0, NONE, NONE);
    optimTreeCount++;

    // Supporting variable declaration
    int backTrackIndex = 1;
    int flag = 0;
    int curDim = 1;
    int count = 0;

    // Traverse until current index is NULL
    while (curHypercubeIndex != NONE) {

        // Keep track of node positions in first dimension
        if (curDim == 1) {
            dimNodes[count] = optimTreeCount - 1;
            count++;
        }

        // Move to child node if present
        if (linearTree[curHypercubeIndex].nextChildIndex != NONE) {

            curHypercubeIndex = linearTree[curHypercubeIndex].nextChildIndex;
            curDim++;

            // Copy contents and update mapping
            if (flag == 1) {
                optimizedLinearTree[backTrackIndex].nextChildIndex = optimTreeCount;
                copyContents(linearTree + curHypercubeIndex,
                             optimizedLinearTree + optimTreeCount, backTrackIndex, NONE,
                             NONE);
            } else {

                optimizedLinearTree[optimTreeCount - 1].nextChildIndex = optimTreeCount;
                copyContents(linearTree + curHypercubeIndex,
                             optimizedLinearTree + optimTreeCount, optimTreeCount - 1,
                             NONE, NONE);
            }

            backTrackIndex = optimTreeCount;
            optimTreeCount++;
            flag = 0;

        }
            // Move to sibling index if there is no child node
        else if (linearTree[curHypercubeIndex].nextSiblingIndex != NONE) {

            // Copy contents and update mapping
            optimizedLinearTree[backTrackIndex].nextSiblingIndex = optimTreeCount;
            curHypercubeIndex = linearTree[curHypercubeIndex].nextSiblingIndex;

            int temp = optimizedLinearTree[backTrackIndex].parentIndex;
            backTrackIndex = optimTreeCount;

            copyContents(linearTree + curHypercubeIndex,
                         optimizedLinearTree + optimTreeCount, temp, NONE, NONE);
            optimTreeCount++;
        }
            // backtrack to parent node until the current node has a sibling
        else {

            while (curHypercubeIndex != NONE) {
                curHypercubeIndex = linearTree[curHypercubeIndex].parentIndex;
                backTrackIndex = optimizedLinearTree[backTrackIndex].parentIndex;
                curDim--;

                if (curHypercubeIndex != NONE &&
                    linearTree[curHypercubeIndex].nextSiblingIndex != NONE) {
                    flag = 1;
                    // Copy contents and update mapping
                    curHypercubeIndex = linearTree[curHypercubeIndex].nextSiblingIndex;
                    int temp = optimizedLinearTree[backTrackIndex].parentIndex;

                    optimizedLinearTree[backTrackIndex].nextSiblingIndex = optimTreeCount;

                    backTrackIndex = optimTreeCount;
                    copyContents(linearTree + curHypercubeIndex,
                                 optimizedLinearTree + optimTreeCount, temp, NONE, NONE);
                    optimTreeCount++;
                    break;
                }
            }
        }
    }

    return;
}

// Supporting function
__inline__ __device__ int optimCheckNeighbor(int index, int *hypercube,
                                  optimTreeNode *linearTree, int curIndex, int curDim) {
    if (curDim == 0) { return 1; }
    int curDif = abs(hypercube[index] - linearTree[curIndex].coordinate);
    if (curDif <= 1) {
        return 1;
    }
    return 0;
}

// Calculate neighborhood density of subtree
__inline__ __device__ int optimNeighborDensitySubTree(int *hypercube,
                                           optimTreeNode *linearTree,
                                           int hypercubeIndex,
                                           int *instancesCount, int parentIndex,
                                           int N, int DIM) {
    int curDensity = 0;
    int index;
    int curHypercubeIndex = parentIndex;
    int endIndex = linearTree[parentIndex].nextBreakIndex;
    int curDim;

    // traverse from starting index till ending index
    while (curHypercubeIndex != NONE && curHypercubeIndex != endIndex) {
        curDim = linearTree[curHypercubeIndex].curDim;
        index = hypercubeIndex * DIM + (curDim - 1);

        // Check if current node is a leaf node
        if (linearTree[curHypercubeIndex].nextChildIndex == NONE) {
            for (int i = linearTree[curHypercubeIndex].startIndex;
                 i <= linearTree[curHypercubeIndex].endIndex; i++) {

                // Check if current node is an immediate neighbor
                if (checkImmediateNeighbor(hypercube + hypercubeIndex * DIM,
                                           hypercube + i * DIM, N, DIM, curDim)) {
                    curDensity += instancesCount[i];
                }
            }
        }

        // Move to child node if present
        if (linearTree[curHypercubeIndex].nextChildIndex != NONE &&
            optimCheckNeighbor(index, hypercube, linearTree, curHypercubeIndex, curDim)) {
            curHypercubeIndex = linearTree[curHypercubeIndex].nextChildIndex;
        }
            // Move to node pointed by break index if there is no child node
                else {
            curHypercubeIndex = linearTree[curHypercubeIndex].nextBreakIndex;
        }
    }

    // Return current density of subtree
    return curDensity;
}

// Calculate neighborhood density using naive approach
__global__ void naiveNeighborhoodDensity(int *density, int *instancesCount,
                                         int DIM, int hypercubeCount,
                                         int *hypercube, int SPLIT) {
    // Set thread indexes
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int totalThreads = hypercubeCount * SPLIT;

    // Assign multiple threads to traverse hypercube array
    if (threadId < totalThreads) {
        int hypercubeIndex = threadId / SPLIT;
        int curThreadDensity = 0;

        // Find starting and ending indexes for each thread
        int startIndex = (threadId % SPLIT) * (hypercubeCount / SPLIT);
        int endIndex = startIndex + (hypercubeCount / SPLIT);
        if ((threadId % SPLIT) == SPLIT - 1) {
            endIndex += (hypercubeCount % SPLIT);
        }

        // Do a linear scan and calculate neighborhood density
        for (int i = startIndex; i < endIndex; i++) {
            if (checkImmediateNeighbor(hypercube + hypercubeIndex * DIM,
                                       hypercube + i * DIM, hypercubeCount, DIM)) {
                curThreadDensity += instancesCount[i];
            }
        }

        // use atomic addition to update neighborhood density of one hypercube
        atomicAdd(density + hypercubeIndex, curThreadDensity);
    }
    return;
}

// Calculate neighborhood density using fast tree
__global__ void optimNeighborhoodDensity(int *density, int *instancesCount,
                                         optimTreeNode *linearTree,
                                         int *hypercubeArray, int *childCount,
                                         int DIM, int hypercubeCount,
                                         int *dimNodes) {

    // Set thread indexes
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int initialDimNodeCount = childCount[0];
    int totalNodes = hypercubeCount * initialDimNodeCount;

    // If  root node has no children
    if (initialDimNodeCount == 0) {
        // Assign one thread per distinct hypercube
        if (threadId < hypercubeCount) {
            int hypercubeIndex = threadId;
            int parentIndex = 0;
            // Calculate neighborhood density
            int curThreadDensity = optimNeighborDensitySubTree(
                    hypercubeArray, linearTree, hypercubeIndex, instancesCount,
                    parentIndex, hypercubeCount, DIM);

            atomicAdd(density + hypercubeIndex, curThreadDensity);
        }

    } else {
        // Parallelize tree traversal further if root node has children
        if (threadId < totalNodes) {
            int hypercubeIndex = threadId / initialDimNodeCount;
            // Map multiple threads to children nodes of root
            int parentIndex = dimNodes[threadId % initialDimNodeCount];

            // Calculate neighborhood density by traversing a sub section of tree
            int curThreadDensity = optimNeighborDensitySubTree(
                    hypercubeArray, linearTree, hypercubeIndex, instancesCount,
                    parentIndex, hypercubeCount, DIM);

            // use atomic addition to update neighborhood density of one hypercube
            atomicAdd(density + hypercubeIndex, curThreadDensity);
        }
    }

    return;
}

// Supporting function to copy node data from locality optim tree to fast tree
__host__ void copyNode(optimTreeNode *root, treeNode *linearCurNode,
                       optimTreeNode *optimCurNode, int curDim) {
    // Copy node contents
    (*optimCurNode).startIndex = (*linearCurNode).startIndex;
    (*optimCurNode).endIndex = (*linearCurNode).endIndex;
    (*optimCurNode).coordinate = (*linearCurNode).coordinate;
    (*optimCurNode).nextChildIndex = (*linearCurNode).nextChildIndex;
    (*optimCurNode).curDim = curDim;

    // If current node has sibling, then set break to sibling
    if ((*linearCurNode).nextSiblingIndex != NONE) {
        (*optimCurNode).nextBreakIndex = (*linearCurNode).nextSiblingIndex;
    }
        // If not, then set break to parent node's break index
    else {
        (*optimCurNode).nextBreakIndex =
                root[(*linearCurNode).parentIndex].nextBreakIndex;
    }

    return;
}

// Build tree with optimized locality and traversal
__host__ void buildSuperOptimTree(treeNode *linearTree,
                                  optimTreeNode *superOptimTree) {
    // Set initial node
    (*superOptimTree).coordinate = (*linearTree).coordinate;
    (*superOptimTree).startIndex = (*linearTree).startIndex;
    (*superOptimTree).endIndex = (*linearTree).endIndex;
    (*superOptimTree).nextChildIndex = (*linearTree).nextChildIndex;
    (*superOptimTree).nextBreakIndex = NONE;
    (*superOptimTree).curDim = 0;

    if (linearTree[0].nextChildIndex == NONE) {
        return;
    }

    int curDim = 1;
    int curHypercubeIndex = 1;

    // iterate untill current index is none
    while (curHypercubeIndex != NONE) {

        copyNode(superOptimTree, linearTree + curHypercubeIndex,
                 superOptimTree + curHypercubeIndex, curDim);

        // Move to child node if present
        if (linearTree[curHypercubeIndex].nextChildIndex != NONE) {
            curHypercubeIndex = linearTree[curHypercubeIndex].nextChildIndex;
            curDim++;
        }
            // Move to sibling node is child node is not present
        else if (linearTree[curHypercubeIndex].nextSiblingIndex != NONE) {
            curHypercubeIndex = linearTree[curHypercubeIndex].nextSiblingIndex;
        }
            // Backtrack until current node has a sibling node
        else {
            while (curHypercubeIndex != NONE) {
                curHypercubeIndex = linearTree[curHypercubeIndex].parentIndex;
                curDim--;

                if (curHypercubeIndex != NONE &&
                    linearTree[curHypercubeIndex].nextSiblingIndex != NONE) {
                    curHypercubeIndex = linearTree[curHypercubeIndex].nextSiblingIndex;
                    break;
                }
            }
        }
    }

    return;
}

// Function to calculate outlier score
__host__ void calculateOutlierScore(float *outlierScore, int *neighborhoodDensity,
                                    map <vector<MY_DATATYPE>, vector<int>> hypercubeMap,
                                    int N, int maxNeighborhoodDensity) {

    int index = 0;

    // Calculate neighborhodd density
    for (auto itr = hypercubeMap.begin(); itr != hypercubeMap.end(); itr++) {
        for (int i: itr->second) {
            outlierScore[i] =
                    (float) (maxNeighborhoodDensity - neighborhoodDensity[index]) /
                    (float) (maxNeighborhoodDensity);
        }
        index++;
    }

    return;
}

__host__ void calculateOutlierScore(float *outlierScore, int *neighborhoodDensity,
                                    map <vector<int>, vector<int>> hypercubeMap,
                                    int N, int maxNeighborhoodDensity) {

    int index = 0;

    // Calculate neighborhodd density
    for (auto itr = hypercubeMap.begin(); itr != hypercubeMap.end(); itr++) {
        for (int i: itr->second) {
            outlierScore[i] =
                    (float) (maxNeighborhoodDensity - neighborhoodDensity[index]) /
                    (float) (maxNeighborhoodDensity);
        }
        index++;
    }

    return;
}

// Function to find min bits required to store hypercube dim
__host__ int findK(int BIN) {
    int k = 1;
    while (pow(2, k) < BIN) {
        k++;
    }

    return k;
}

// Naive approach
float naiveStrategy(int *d_hypercubeArray, int *h_neighborhoodDensity,
                    int *h_instancesCount, int distinctHypercubeCount, int BIN,
                    int DIM) {

    puts("using naive approach");

    // Variable declaration
    cudaEvent_t neighborhoodDensityStart, neighborhoodDensityStop;

    int *d_neighborhoodDensity = NULL;
    int *d_instancesCount = NULL;

    // Setting split to 8 - optimized threads per linear scan
    int SPLIT = 8;
    float neighborhoodDensityTime;

    dim3 dimBlock, dimGrid;

    // Memory
    size_t neighborhoodDensityMemory = sizeof(int) * distinctHypercubeCount;
    size_t instancesCountMemory = sizeof(int) * distinctHypercubeCount;

    // Allocate memory in device for neighborhood density and copy from host to
    // device
    cudaMalloc((void **) &d_neighborhoodDensity, neighborhoodDensityMemory);

    cudaMemcpy(d_neighborhoodDensity, h_neighborhoodDensity,
               neighborhoodDensityMemory, cudaMemcpyHostToDevice);

    // Allocate memory in device for instances count and copy from host to
    // device
    cudaMalloc((void **) &d_instancesCount, instancesCountMemory);

    cudaMemcpy(d_instancesCount, h_instancesCount, instancesCountMemory,
               cudaMemcpyHostToDevice);

    // Set threads per block for kernel function
    dimBlock.x = 256;

    dimGrid.x =
            ceil((float) (distinctHypercubeCount * SPLIT) / (float) (dimBlock.x));

    // Get timings for neighborhood density
    cudaEventCreate(&neighborhoodDensityStart);
    cudaEventCreate(&neighborhoodDensityStop);
    cudaEventRecord(neighborhoodDensityStart);

    // Calculate neighborhood density using naive approach
    naiveNeighborhoodDensity<<<dimGrid, dimBlock>>>(
            d_neighborhoodDensity, d_instancesCount, DIM, distinctHypercubeCount,
            d_hypercubeArray, SPLIT);

    cudaDeviceSynchronize();
    cudaEventRecord(neighborhoodDensityStop);
    cudaEventSynchronize(neighborhoodDensityStop);

    cudaEventElapsedTime(&neighborhoodDensityTime, neighborhoodDensityStart,
                         neighborhoodDensityStop);

    // Copy calculated neighborhood density from device to host
    cudaMemcpy(h_neighborhoodDensity, d_neighborhoodDensity,
               sizeof(int) * distinctHypercubeCount, cudaMemcpyDeviceToHost);

    // Return the time for calculating neighborhood density
    return neighborhoodDensityTime;
}

// Locality and traverasl optimized strategy
__host__ float
finalOptimTreeStrategy(int *h_hypercubeArray, int *d_hypercubeArray,
                       int *h_neighborhoodDensity, int *h_instancesCount,
                       int distinctHypercubeCount, int DIM, int MINSPLIT) {

    puts("Using locality and traversal optimized tree");

    // Variable declaration
    cudaEvent_t neighborhoodDensityStart, neighborhoodDensityStop;

    int *d_neighborhoodDensity = NULL;
    int *d_instancesCount = NULL;
    int *d_childCount = NULL;
    int *d_dimNodes = NULL;
    treeNode *h_linearTree = NULL;
    optimTreeNode *d_linearTree = NULL;

    int linearTreeCount = 0;
    int curDim = 0;
    float neighborhoodDensityTime;

    dim3 dimBlock, dimGrid;

    // Memory
    size_t neighborhoodDensityMemory = sizeof(int) * distinctHypercubeCount;
    size_t instancesCountMemory = sizeof(int) * distinctHypercubeCount;
    size_t childCountMemory = sizeof(int) * (DIM + 1);

    // Copy hypercube array from device to host
    cudaMemcpy(h_hypercubeArray, d_hypercubeArray,
               sizeof(int) * distinctHypercubeCount * DIM,
               cudaMemcpyDeviceToHost);

    // Allocate memory in device for neighborhood density and copy from host to
    // device
    cudaMalloc((void **) &d_neighborhoodDensity, neighborhoodDensityMemory);

    cudaMemcpy(d_neighborhoodDensity, h_neighborhoodDensity,
               neighborhoodDensityMemory, cudaMemcpyHostToDevice);

    // Allocate memory in device for instances count and copy from host to device
    cudaMalloc((void **) &d_instancesCount, instancesCountMemory);

    cudaMemcpy(d_instancesCount, h_instancesCount, instancesCountMemory,
               cudaMemcpyHostToDevice);

    // Build a simple tree
    // Array for counting nodes in each dim
    int *h_childCount = (int *) calloc((DIM + 1), sizeof(int));
    int *h_dimStart = (int *) calloc((DIM + 1), sizeof(int));

    // Append initial node
    appendNode(&h_linearTree, 0, distinctHypercubeCount - 1, NONE, NONE,
               &linearTreeCount);

    buildLinearTree(h_hypercubeArray, &h_linearTree, h_childCount, h_dimStart,
                    &linearTreeCount, curDim, distinctHypercubeCount, MINSPLIT,
                    DIM);

    // Memory
    size_t dimIndexMemory = sizeof(int) * h_childCount[0];
    size_t linearTreeMemory = sizeof(optimTreeNode) * linearTreeCount;

    // Build locality optimized tree
    treeNode *h_optimizedLinearTree =
            (treeNode *) malloc(sizeof(treeNode) * linearTreeCount);

    // Array to hold the indexes of nodes in first dimension
    // This is necessary for assigning multiple threads per tree
    int *h_dimNodes = (int *) malloc(dimIndexMemory);

    buildOptimizedLinearTree(h_linearTree, h_optimizedLinearTree, h_dimNodes);

    // Build locality and traversal optimized tree
    optimTreeNode *h_superOptimTree = (optimTreeNode *) malloc(linearTreeMemory);

    buildSuperOptimTree(h_optimizedLinearTree, h_superOptimTree);

    // Allocate memory in device and copy from host to device
    cudaMalloc((void **) &d_dimNodes, dimIndexMemory);

    cudaMemcpy(d_dimNodes, h_dimNodes, dimIndexMemory, cudaMemcpyHostToDevice);

    // Allocate memory in device for fast tree and copy from host to
    // device
    cudaMalloc((void **) &d_linearTree, linearTreeMemory);

    cudaMemcpy(d_linearTree, h_superOptimTree, linearTreeMemory,
               cudaMemcpyHostToDevice);

    // Allocate memory in device and copy from host to device
    cudaMalloc((void **) &d_childCount, childCountMemory);

    cudaMemcpy(d_childCount, h_childCount, childCountMemory,
               cudaMemcpyHostToDevice);

    // Set threads per block for kernel function
    dimBlock.x = 256;

    if (h_childCount[0] == 0) {
        dimGrid.x = ceil((float) (distinctHypercubeCount) / (float) (dimBlock.x));
    } else {
        dimGrid.x = ceil((float) (distinctHypercubeCount * h_childCount[0]) /
                         (float) (dimBlock.x));
    }

    // Get timings for neighborhood density
    cudaEventCreate(&neighborhoodDensityStart);
    cudaEventCreate(&neighborhoodDensityStop);
    cudaEventRecord(neighborhoodDensityStart);

    // Calculate neighborhood density using fast tree
    optimNeighborhoodDensity<<<dimGrid, dimBlock>>>(
            d_neighborhoodDensity, d_instancesCount, d_linearTree, d_hypercubeArray,
            d_childCount, DIM, distinctHypercubeCount, d_dimNodes);

    cudaDeviceSynchronize();
    cudaEventRecord(neighborhoodDensityStop);
    cudaEventSynchronize(neighborhoodDensityStop);

    cudaEventElapsedTime(&neighborhoodDensityTime, neighborhoodDensityStart,
                         neighborhoodDensityStop);

    // Copy calculated neighborhood density from device to host
    cudaMemcpy(h_neighborhoodDensity, d_neighborhoodDensity,
               neighborhoodDensityMemory, cudaMemcpyDeviceToHost);

    // Return the time for calculating neighborhood density
    return neighborhoodDensityTime;
}

// Locality optimized tree traversal strategy
__host__ float
localityOptimTreeStrategy(int *h_hypercubeArray, int *d_hypercubeArray,
                          int *h_neighborhoodDensity, int *h_instancesCount,
                          int distinctHypercubeCount, int DIM, int MINSPLIT) {
    puts("Using locality optimized tree");

    // Variable declaration
    cudaEvent_t neighborhoodDensityStart, neighborhoodDensityStop;

    int *d_neighborhoodDensity = NULL;
    int *d_instancesCount = NULL;
    int *d_childCount = NULL;
    int *d_dimNodes = NULL;
    treeNode *h_linearTree = NULL;
    treeNode *d_linearTree = NULL;

    int linearTreeCount = 0;
    int curDim = 0;
    float neighborhoodDensityTime;

    dim3 dimBlock, dimGrid;

    // Memory
    size_t neighborhoodDensityMemory = sizeof(int) * distinctHypercubeCount;
    size_t instancesCountMemory = sizeof(int) * distinctHypercubeCount;
    size_t childCountMemory = sizeof(int) * (DIM + 1);

    // Copy hypercube array from device to host
    cudaMemcpy(h_hypercubeArray, d_hypercubeArray,
               sizeof(int) * distinctHypercubeCount * DIM,
               cudaMemcpyDeviceToHost);

    // Allocate memory in device for neighborhood density and copy from host to
    // device
    cudaMalloc((void **) &d_neighborhoodDensity, neighborhoodDensityMemory);

    cudaMemcpy(d_neighborhoodDensity, h_neighborhoodDensity,
               neighborhoodDensityMemory, cudaMemcpyHostToDevice);

    // Allocate memory in device for instances Count and copy from host to device
    cudaMalloc((void **) &d_instancesCount, instancesCountMemory);

    cudaMemcpy(d_instancesCount, h_instancesCount, instancesCountMemory,
               cudaMemcpyHostToDevice);

    // Build a simple tree
    // Array for counting nodes in each dim
    int *h_childCount = (int *) calloc((DIM + 1), sizeof(int));
    // Array to hold the starting index of each dimension
    int *h_dimStart = (int *) calloc((DIM + 1), sizeof(int));

    // Append initial node
    appendNode(&h_linearTree, 0, distinctHypercubeCount - 1, NONE, NONE,
               &linearTreeCount);

    buildLinearTree(h_hypercubeArray, &h_linearTree, h_childCount, h_dimStart,
                    &linearTreeCount, curDim, distinctHypercubeCount, MINSPLIT,
                    DIM);

    // Memory
    size_t dimIndexMemory = sizeof(int) * h_childCount[0];
    size_t linearTreeMemory = sizeof(treeNode) * linearTreeCount;

    // Build locality optimized tree
    treeNode *h_optimizedLinearTree = (treeNode *) malloc(linearTreeMemory);

    // Array to hold the indexes of nodes in first dimension
    // This is necessary for assigning multiple threads per tree
    int *h_dimNodes = (int *) malloc(dimIndexMemory);

    buildOptimizedLinearTree(h_linearTree, h_optimizedLinearTree, h_dimNodes);

    // Allocate memory in device and copy from host to device
    cudaMalloc((void **) &d_dimNodes, dimIndexMemory);

    cudaMemcpy(d_dimNodes, h_dimNodes, dimIndexMemory, cudaMemcpyHostToDevice);

    // Allocate memory in device for locality optim tree and copy from host to
    // device
    cudaMalloc((void **) &d_linearTree, linearTreeMemory);

    cudaMemcpy(d_linearTree, h_optimizedLinearTree, linearTreeMemory,
               cudaMemcpyHostToDevice);

    // Allocate memory in device and copy from host to device
    cudaMalloc((void **) &d_childCount, childCountMemory);

    cudaMemcpy(d_childCount, h_childCount, childCountMemory,
               cudaMemcpyHostToDevice);

    // Set threads per block for kernel function
    dimBlock.x = 256;

    if (h_childCount[0] == 0) {
        dimGrid.x = ceil((float) (distinctHypercubeCount) / (float) (dimBlock.x));
    } else {
        dimGrid.x = ceil((float) (distinctHypercubeCount * h_childCount[0]) /
                         (float) (dimBlock.x));
    }

    // Get timings for neighborhood density
    cudaEventCreate(&neighborhoodDensityStart);
    cudaEventCreate(&neighborhoodDensityStop);
    cudaEventRecord(neighborhoodDensityStart);

    // Calculate neighborhood density using locality optimized tree
    neighborhoodDensity<<<dimGrid, dimBlock>>>(
            d_neighborhoodDensity, d_instancesCount, d_linearTree, d_hypercubeArray,
            d_childCount, DIM, distinctHypercubeCount, d_dimNodes);

    cudaDeviceSynchronize();
    cudaEventRecord(neighborhoodDensityStop);
    cudaEventSynchronize(neighborhoodDensityStop);

    cudaEventElapsedTime(&neighborhoodDensityTime, neighborhoodDensityStart,
                         neighborhoodDensityStop);

    // Copy calculated neighborhood density from device to host
    cudaMemcpy(h_neighborhoodDensity, d_neighborhoodDensity,
               neighborhoodDensityMemory, cudaMemcpyDeviceToHost);

    // Return the time for calculating neighborhood density
    return neighborhoodDensityTime;
}

// Simple tree traversal strategy
__host__ float simpleTreeStrategy(int *h_hypercubeArray, int *d_hypercubeArray,
                                  int *h_neighborhoodDensity,
                                  int *h_instancesCount,
                                  int distinctHypercubeCount, int DIM,
                                  int MINSPLIT) {
    puts("Using simple tree");

    // Variable declaration
    cudaEvent_t neighborhoodDensityStart, neighborhoodDensityStop;

    int *d_neighborhoodDensity = NULL;
    int *d_instancesCount = NULL;
    int *d_childCount = NULL;

    treeNode *h_linearTree = NULL;
    treeNode *d_linearTree = NULL;

    float neighborhoodDensityTime;
    int linearTreeCount = 0;
    int curDim = 0;

    // Memory
    size_t neighborhoodDensityMemory = sizeof(int) * distinctHypercubeCount;
    size_t instancesCountMemory = sizeof(int) * distinctHypercubeCount;
    size_t childCountMemory = sizeof(int) * (DIM + 1);

    dim3 dimBlock, dimGrid;

    // Copy hypercube array from device to host
    cudaMemcpy(h_hypercubeArray, d_hypercubeArray,
               sizeof(int) * distinctHypercubeCount * DIM,
               cudaMemcpyDeviceToHost);

    // Allocate memory in device for neighborhood density and copy from host to
    // device
    cudaMalloc((void **) &d_neighborhoodDensity, neighborhoodDensityMemory);

    cudaMemcpy(d_neighborhoodDensity, h_neighborhoodDensity,
               neighborhoodDensityMemory, cudaMemcpyHostToDevice);

    // Allocate memory in device for instances Count and copy from host to device
    cudaMalloc((void **) &d_instancesCount, instancesCountMemory);

    cudaMemcpy(d_instancesCount, h_instancesCount, instancesCountMemory,
               cudaMemcpyHostToDevice);

    // Build a linear tree
    // Array for counting nodes in each dim
    int *h_childCount = (int *) calloc((DIM + 1), sizeof(int));
    // Array to hold the starting index of each dimension
    int *h_dimStart = (int *) calloc((DIM + 1), sizeof(int));

    // Append initial node
    appendNode(&h_linearTree, 0, distinctHypercubeCount - 1, NONE, NONE,
               &linearTreeCount);

    buildLinearTree(h_hypercubeArray, &h_linearTree, h_childCount, h_dimStart,
                    &linearTreeCount, curDim, distinctHypercubeCount, MINSPLIT,
                    DIM);

    size_t linearTreeMemory = sizeof(treeNode) * linearTreeCount;

    // Allocate memory in device for simple tree and copy from host to device
    cudaMalloc((void **) &d_linearTree, linearTreeMemory);

    cudaMemcpy(d_linearTree, h_linearTree, linearTreeMemory,
               cudaMemcpyHostToDevice);

    // Allocate memory in device for nodes in each dim and copy from host to
    // device
    cudaMalloc((void **) &d_childCount, childCountMemory);

    cudaMemcpy(d_childCount, h_childCount, childCountMemory,
               cudaMemcpyHostToDevice);

    // Set threads per block for kernel function
    dimBlock.x = 256;

    if (h_childCount[0] == 0) {
        dimGrid.x = ceil((float) (distinctHypercubeCount) / (float) (dimBlock.x));
    } else {
        dimGrid.x = ceil((float) (distinctHypercubeCount * h_childCount[0]) /
                         (float) (dimBlock.x));
    }

    // Get timings for neighborhood density
    cudaEventCreate(&neighborhoodDensityStart);
    cudaEventCreate(&neighborhoodDensityStop);
    cudaEventRecord(neighborhoodDensityStart);

    // Calculate neighborhood density using simple tree
    simpleNeighborhoodDensity<<<dimGrid, dimBlock>>>(
            d_neighborhoodDensity, d_instancesCount, d_linearTree, d_hypercubeArray,
            d_childCount, DIM, distinctHypercubeCount);

    cudaDeviceSynchronize();
    cudaEventRecord(neighborhoodDensityStop);
    cudaEventSynchronize(neighborhoodDensityStop);

    cudaEventElapsedTime(&neighborhoodDensityTime, neighborhoodDensityStart,
                         neighborhoodDensityStop);

    // Copy calculated neighborhood density from device to host
    cudaMemcpy(h_neighborhoodDensity, d_neighborhoodDensity,
               neighborhoodDensityMemory, cudaMemcpyDeviceToHost);

    // Return the time for calculating neighborhood density
    return neighborhoodDensityTime;
}

// Build hypercube array - Non encoding
__global__ void buildNonEncodedHypercubeArray(int *hypercube, double *dataset,
                                              int N, int BIN, int DIM) {

    //int totalElementsPerBlock = N / blockDim.x;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    double length = (double) (1) / (double) BIN;

    if (threadId < N) {
        for (int i = threadId * DIM; i < threadId * DIM + DIM; i++) {
            hypercube[i] = (int) floor(dataset[i] / length);
        }
    }

    return;
}

// Reorder dimensions based on variance
__host__ double *reorderByDimensions(double *dataset, int N, int DIM) {
    double mean, devmean;
    vector <pair<double, int>> devDimPair;
    double *tempDataset = (double *) malloc(sizeof(double) * N * DIM);
    int sampleSize, sampledPoints;

    // Set default sampling size as 100
    // Ensure at least 1000 points are sampled in the dataset
    // If N < 1000, then sample all points in the dataset
    if (N <= 1000) {
        sampleSize = 1;
    } else if (N < 100000) {
        sampleSize = N / 1000;
    } else {
        sampleSize = 100;
    }

    sampledPoints = N / sampleSize;

    for (int i = 0; i < DIM; i++) {
        mean = 0.0;
        for (int j = 0; j < N; j += sampleSize) {
            mean += dataset[DIM * j + i];
        }

        mean /= sampledPoints;
        devmean = 0;

        for (int j = 0; j < N; j += sampleSize) {
            devmean += pow(dataset[DIM * j + i] - mean, 2);
        }

        devmean /= sampledPoints;
        devDimPair.emplace_back(sqrt(devmean), i);
    }

    sort(devDimPair.begin(), devDimPair.end(), greater<>());

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < DIM; j++) {
            tempDataset[i * DIM + j] = dataset[i * DIM + devDimPair[j].second];
        }
    }

    free(dataset);
    return tempDataset;
}