#include "hySortOD_lib.h"
#include <omp.h>

// Import dataset
int importDataset(char *fname, int N, double *dataset, int DIM) {

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
void normalizeDataset(double *dataset, int N, int DIM) {
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
void encodeHypercube(MY_DATATYPE *curHypercube, int *hypercube, int DIM,
                     int index, int encodeBlockSize, int k) {
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
void createHypercube(MY_DATATYPE *hypercube, double *dataset, int N, int DIM,
                     int BIN, int encodeBlockSize, int k) {

    double length = (double) (1) / (double) BIN;

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        int tempHypercube[DIM];
        for (int j = 0; j < DIM; j++) {
            tempHypercube[j] = (int) floor(dataset[DIM * i + j] / length);
        }
        encodeHypercube(hypercube, tempHypercube, DIM, i, encodeBlockSize, k);
    }
}

// Return number with k bits set to 1
int setBitsTo1(int k) { return pow(2, k) - 1; }

// Decode encoded hypercube into hypercube array
void decodeHypercube(MY_DATATYPE *hypercubeEncodedArray, int *hypercube,
                     int DIM, int hypercubeCount, int threadId,
                     int encodeBlockSize, int k) {

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
void buildHypercubeArray(MY_DATATYPE *hypercube, int *hypercubeArray,
                         int hypercubeCount, int DIM, int encodeBlockSize,
                         int k) {

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < hypercubeCount; i++) {

        decodeHypercube(hypercube + i * encodeBlockSize, hypercubeArray, DIM,
                        hypercubeCount, i, encodeBlockSize, k);
    }
    return;
}

// Supporting function to add node to a tree
void appendNode(treeNode **rootNode, int startIndex, int endIndex,
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
void buildLinearTree(int *hypercube, treeNode **linearTree, int *childCount,
                     int *dimStart, int *curCount, int curDim, int N,
                     int MINSPLIT, int DIM) {
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
int checkNeighbor(int index, int *hypercube, treeNode *linearTree,
                  int curIndex, int curDim) {
    if (curDim == 0) { return 1; }
    int curDif = abs(hypercube[index] - linearTree[curIndex].coordinate);
    if (curDif <= 1) {
        return 1;
    }
    return 0;
}

// Supporting function to check if current hypercube is an immediate neighbor
int checkImmediateNeighbor(int *hypercubeA, int *hypercubeB, int hypercubeCount,
                           int DIM) {

    for (int i = 0; i < DIM; i++) {
        if (abs(hypercubeA[i] - hypercubeB[i]) > 1) {
            return 0;
        }
    }
    return 1;
}

// Neighborhood density of subtree - locality optimized
int neighborDensitySubTree(int *hypercube, treeNode *linearTree,
                           int hypercubeIndex, int *instancesCount,
                           int parentIndex, int curDim, int N, int DIM) {

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
                                           hypercube + i * DIM, N, DIM)) {
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
void neighborhoodDensity(int *density, int *instancesCount,
                         treeNode *linearTree, int *hypercubeArray, int DIM,
                         int hypercubeCount) {

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < hypercubeCount; i++) {
        int hypercubeIndex = i;
        int parentIndex = 0;
        // Calculate neighborhood density
        int curThreadDensity = neighborDensitySubTree(
                hypercubeArray, linearTree, hypercubeIndex, instancesCount, parentIndex,
                0, hypercubeCount, DIM);
        // Update neighborhood density
        density[hypercubeIndex] += curThreadDensity;
    }

    return;
}

// Calculate neighborhood density using simple tree
void simpleNeighborhoodDensity(int *density, int *instancesCount,
                               treeNode *linearTree, int *hypercubeArray,
                               int DIM, int hypercubeCount) {

// Assign one thread per distinct hypercube
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < hypercubeCount; i++) {
        int parentIndex = 0;
        // Calculate neighborhood density
        int curDensity =
                neighborDensitySubTree(hypercubeArray, linearTree, i, instancesCount,
                                       parentIndex, 0, hypercubeCount, DIM);
        // Update neighborhood density
        density[i] += curDensity;
    }

    return;
}

// Supporting function to copy node contents
void copyContents(treeNode *linearCurNode, treeNode *optimCurNode,
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
int optimCheckNeighbor(int index, int *hypercube, optimTreeNode *linearTree,
                       int curIndex, int curDim) {
    if (curDim == 0) { return 1; }
    int curDif = abs(hypercube[index] - linearTree[curIndex].coordinate);
    if (curDif <= 1) {
        return 1;
    }
    return 0;
}

// Calculate neighborhood density of sub tree
int optimNeighborDensitySubTree(int *hypercube, optimTreeNode *linearTree,
                                int hypercubeIndex, int *instancesCount,
                                int parentIndex, int N, int DIM) {
    int curDensity = 0;
    int index;
    int curHypercubeIndex = parentIndex;
    int endIndex = linearTree[parentIndex].nextBreakIndex;
    int curDim;

    // traverse from starting index till ending index
    while (curHypercubeIndex != NONE && curHypercubeIndex != endIndex) {
        curDim = linearTree[curHypercubeIndex].curDim;
        index = hypercubeIndex * DIM + curDim - 1;

        // Check if current node is a leaf node
        if (linearTree[curHypercubeIndex].nextChildIndex == NONE) {
            for (int i = linearTree[curHypercubeIndex].startIndex;
                 i <= linearTree[curHypercubeIndex].endIndex; i++) {

                // Check if current node is an immediate neighbor
                if (checkImmediateNeighbor(hypercube + hypercubeIndex * DIM,
                                           hypercube + i * DIM, N, DIM)) {
                    curDensity += instancesCount[i];
                }
            }
        }

        // Move to chid node if present
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
void naiveNeighborhoodDensity(int *density, int *instancesCount, int DIM,
                              int hypercubeCount, int *hypercube) {

#pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < hypercubeCount; j++) {
        int curDensity = 0;
        // Do a linear scan and calculate neighborhood density
        for (int i = 0; i < hypercubeCount; i++) {
            if (checkImmediateNeighbor(hypercube + j * DIM, hypercube + i * DIM,
                                       hypercubeCount, DIM)) {
                curDensity += instancesCount[i];
            }
        }
        // Update neighborhood density of one hypercube
        density[j] += curDensity;
    }
    return;
}

// Calculate neighborhood density using fast tree
void optimNeighborhoodDensity(int *density, int *instancesCount,
                              optimTreeNode *linearTree, int *hypercubeArray,
                              int DIM, int hypercubeCount) {

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < hypercubeCount; i++) {
        int hypercubeIndex = i;
        int parentIndex = 0;
        // Calculate neighborhood density
        int curDensity = optimNeighborDensitySubTree(
                hypercubeArray, linearTree, hypercubeIndex, instancesCount, parentIndex,
                hypercubeCount, DIM);

        density[hypercubeIndex] += curDensity;
    }

    return;
}

// Supporting function to copy node data from locality optim tree to fast tree
void copyNode(optimTreeNode *root, treeNode *linearCurNode,
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
void buildSuperOptimTree(treeNode *linearTree, optimTreeNode *superOptimTree) {
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

    // iterate until current index is none
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
void calculateOutlierScore(float *outlierScore, int *neighborhoodDensity,
                           map <vector<MY_DATATYPE>, vector<int>> hypercubeMap,
                           int N, int maxNeighborhoodDensity) {

    int index = 0;

    // Calculate neighborhood density
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

void calculateOutlierScore(float *outlierScore, int *neighborhoodDensity,
                           map <vector<int>, vector<int>> hypercubeMap, int N,
                           int maxNeighborhoodDensity) {

    int index = 0;

    // Calculate neighborhood density
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
int findK(int BIN) {
    int k = 1;
    while (pow(2, k) < BIN) {
        k++;
    }

    return k;
}

// Naive approach
float naiveStrategy(int *h_hypercubeArray, int *h_neighborhoodDensity,
                    int *h_instancesCount, int distinctHypercubeCount, int BIN,
                    int DIM) {

    puts("using naive approach");

    // Start clock
    double neighborhoodDensityStart = omp_get_wtime();

    // Calculate neighborhood density using naive approach
    naiveNeighborhoodDensity(h_neighborhoodDensity, h_instancesCount, DIM,
                             distinctHypercubeCount, h_hypercubeArray);

    double neighborhoodDensityStop = omp_get_wtime();

    // Return the time for calculating neighborhood density
    return (float) (neighborhoodDensityStop - neighborhoodDensityStart);
}

// Locality and traversal optimized strategy
float finalOptimTreeStrategy(int *h_hypercubeArray, int *h_neighborhoodDensity,
                             int *h_instancesCount, int distinctHypercubeCount,
                             int DIM, int MINSPLIT) {

    puts("Using locality and traversal optimized tree");

    // Variable declaration
    treeNode *h_linearTree = NULL;

    int linearTreeCount = 0;
    int curDim = 0;

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

    // Build locality optimized tree
    treeNode *h_optimizedLinearTree =
            (treeNode *) malloc(sizeof(treeNode) * linearTreeCount);

    // Array to hold the indexes of nodes in first dimension
    // This is necessary for assigning multiple threads per tree
    int *h_dimNodes = (int *) malloc(sizeof(int) * h_childCount[0]);

    buildOptimizedLinearTree(h_linearTree, h_optimizedLinearTree, h_dimNodes);

    // Build locality and traversal optimized tree
    optimTreeNode *h_superOptimTree =
            (optimTreeNode *) malloc(sizeof(optimTreeNode) * linearTreeCount);

    buildSuperOptimTree(h_optimizedLinearTree, h_superOptimTree);

    // Get timings for neighborhood density
    double neighborhoodDensityStart = omp_get_wtime();
    // Calculate neighborhood density using fast tree
    optimNeighborhoodDensity(h_neighborhoodDensity, h_instancesCount,
                             h_superOptimTree, h_hypercubeArray, DIM,
                             distinctHypercubeCount);

    double neighborhoodDensityStop = omp_get_wtime();

    // Return the time for calculating neighborhood density
    return (float) (neighborhoodDensityStop - neighborhoodDensityStart);
}

// Locality optimized tree traversal strategy
float localityOptimTreeStrategy(int *h_hypercubeArray,
                                int *h_neighborhoodDensity,
                                int *h_instancesCount,
                                int distinctHypercubeCount, int DIM,
                                int MINSPLIT) {

    puts("Using locality optimized tree");

    // Variable declaration
    treeNode *h_linearTree = NULL;
    int linearTreeCount = 0;
    int curDim = 0;

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

    // Build locality optimized tree
    treeNode *h_optimizedLinearTree =
            (treeNode *) malloc(sizeof(treeNode) * linearTreeCount);

    // Array to hold the indexes of nodes in first dimension
    // This is necessary for assigning multiple threads per tree
    int *h_dimNodes = (int *) malloc(sizeof(int) * h_childCount[0]);

    buildOptimizedLinearTree(h_linearTree, h_optimizedLinearTree, h_dimNodes);

    double neighborhoodDensityStart = omp_get_wtime();
    // Calculate neighborhood density using locality optimized tree
    neighborhoodDensity(h_neighborhoodDensity, h_instancesCount,
                        h_optimizedLinearTree, h_hypercubeArray, DIM,
                        distinctHypercubeCount);

    double neighborhoodDensityStop = omp_get_wtime();

    // Return the time for calculating neighborhood density
    return (float) (neighborhoodDensityStop - neighborhoodDensityStart);
}

// Simple tree traversal strategy
float simpleTreeStrategy(int *h_hypercubeArray, int *h_neighborhoodDensity,
                         int *h_instancesCount, int distinctHypercubeCount,
                         int DIM, int MINSPLIT) {
    puts("Using simple tree");

    // Variable declaration
    treeNode *h_linearTree = NULL;
    int linearTreeCount = 0;
    int curDim = 0;

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

    double neighborhoodDensityStart = omp_get_wtime();

    // Calculate neighborhood density using simple tree
    simpleNeighborhoodDensity(h_neighborhoodDensity, h_instancesCount,
                              h_linearTree, h_hypercubeArray, DIM,
                              distinctHypercubeCount);

    double neighborhoodDensityStop = omp_get_wtime();

    // Return the time for calculating neighborhood density
    return (float) (neighborhoodDensityStop - neighborhoodDensityStart);
}

// Build hypercube array - Non encoding
void buildNonEncodedHypercubeArray(int *hypercube, double *dataset, int N,
                                   int BIN, int DIM) {

    double length = (double) (1) / (double) BIN;

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N * DIM; i++) {
        hypercube[i] = (int) floor(dataset[i] / length);
    }

    return;
}

// Reorder dimensions based on variance
double *reorderByDimensions(double *dataset, int N, int DIM) {
    double mean, devmean;
    vector<pair<double, int>> devDimPair;
    double *tempDataset = (double *)malloc(sizeof(double) * N * DIM);
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

    mean = 0;
    devmean = 0;

    for (int i = 0; i < DIM; i++) {
        mean += devDimPair[i].first;
    }
    mean /= DIM;

    for (int i = 0; i < DIM; i++) {
        devmean += pow(devDimPair[i].first - mean, 2);
    }
    devmean /= DIM;

    double coeffOfVariance = (sqrt(devmean)/mean)*100;

    if (coeffOfVariance > 30)
    {
        puts("Dimensions are reordered");
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

    puts("Dimensions are NOT reordered");
    free(tempDataset);
    return dataset;
}