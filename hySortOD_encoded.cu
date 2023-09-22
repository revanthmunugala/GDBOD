#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <map>
#include <vector>
using namespace std;

#define NONE -1
#define MY_DATATYPE unsigned int
const string approach[] = {"Naive", "Tree"};
const string treeSelect[] = {"Simple", "Locality optimized", "Locality and traversal optimized"};

// Global variable
//int DIM = 384;

typedef struct treeNode {
    int coordinate;
    int startIndex;
    int endIndex;
    int nextSiblingIndex;
    int nextChildIndex;
    int parentIndex;
} treeNode;

typedef struct optimTreeNode {
    int coordinate;
    int startIndex;
    int endIndex;
    int nextChildIndex;
    int nextBreakIndex;
    int curDim;
} optimTreeNode;

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


__device__ void encodeHypercube(MY_DATATYPE*curHypercube, int *hypercube, int DIM, int index,
                                int encodeBlockSize, int k)
                                {
    int idx = 0;
    int dimPerBlock = floor((double)(sizeof(MY_DATATYPE)*8)/(double)k);

    for (int i = 0; i < DIM; i++) {

        if (i % dimPerBlock == 0) {
            idx = i / dimPerBlock + index * encodeBlockSize;
        }
        curHypercube[idx] = curHypercube[idx] << k | hypercube[i];
    }
    return;
}

__global__ void createHypercube(MY_DATATYPE*hypercube, double *dataset, int N, int DIM,
                                int BIN, int encodeBlockSize, int k)
                                {
    int totalElements = N * DIM;
    int totalElementsPerBlock = blockDim.x / DIM;
    int threadId = blockIdx.x * totalElementsPerBlock * DIM + threadIdx.x;
    
    double length = (double) (1) / (double) BIN;

    extern __shared__ int tempHypercube[];

    if (threadId < totalElements && threadIdx.x < totalElementsPerBlock * DIM)
    {
        tempHypercube[threadIdx.x] = (int) floor(dataset[threadId] / length);
    }

    __syncthreads();

    int hypercubeIndex = (threadIdx.x / DIM) + blockIdx.x * totalElementsPerBlock;

    if ((threadIdx.x % DIM == 0) && hypercubeIndex < N && hypercubeIndex < (blockIdx.x + 1) * totalElementsPerBlock)
    {
        encodeHypercube(hypercube, tempHypercube + threadIdx.x, DIM, hypercubeIndex, encodeBlockSize, k);
    }

    return;
}


__host__ __device__ int setBitsTo1(int k)
{
    return pow(2,k)-1;
}

__device__ void decodeHypercube(MY_DATATYPE *hypercubeEncodedArray, int *hypercube, int DIM,
                                int hypercubeCount, int threadId, int encodeBlockSize, int k) {

    int lastIndex = encodeBlockSize - 1;
    int elementsPerBlock = floor((double)(sizeof(MY_DATATYPE)*8)/(double)k);
    int elementsPerLastBlock = DIM % (elementsPerBlock);
    int iterationCount = elementsPerLastBlock;
    int kBitsTo1 = setBitsTo1(k);

    if (elementsPerLastBlock == 0) {
        iterationCount = elementsPerBlock;
    }

    for (int i = (DIM - 1); i >= 0; i--) {

        if (hypercubeEncodedArray[lastIndex] == 0 && iterationCount == 0) {
            lastIndex--;
            iterationCount = elementsPerBlock;
        }

        iterationCount--;

        hypercube[threadId * DIM + i] = hypercubeEncodedArray[lastIndex] & kBitsTo1;
        hypercubeEncodedArray[lastIndex] = hypercubeEncodedArray[lastIndex] >> k;
    }

    return;
}

__global__ void buildHypercubeArray(MY_DATATYPE *hypercube, int *hypercubeArray,
                                    int hypercubeCount, int DIM, int encodeBlockSize, int k) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadId < hypercubeCount) {
        decodeHypercube(hypercube + threadId * encodeBlockSize, hypercubeArray, DIM, hypercubeCount,
                        threadId, encodeBlockSize, k);
    }
    return;
}

__host__ void appendNode(treeNode **rootNode, int startIndex, int endIndex,
                         int coordinate, int parentIndex, int *curCount) {
    treeNode *curNode = (treeNode *) malloc(sizeof(treeNode));
    curNode->startIndex = startIndex;
    curNode->endIndex = endIndex;
    curNode->coordinate = coordinate;
    curNode->parentIndex = parentIndex;
    curNode->nextChildIndex = NONE;
    curNode->nextSiblingIndex = NONE;

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

__host__ void buildLinearTree(int *hypercube, treeNode **linearTree,
                              int *childCount, int *dimStart, int *curCount,
                              int curDim, int N, int MINSPLIT, int DIM) {
    if (curDim == DIM) {
        return;
    }

    int curDimElementCount;

    if (curDim - 1 >= 0) {
        curDimElementCount = childCount[curDim - 1];
    } else {
        curDimElementCount = 1;
    }

    int dimensionStart = dimStart[curDim];
    int dimensionEnd = curDimElementCount + dimensionStart;

    for (int i = dimensionStart; i < dimensionEnd; i++) {
        if (curDim + 1 < DIM) {
            dimStart[curDim + 1] = i + 1;
        }

        int startIndex = (*linearTree)[i].startIndex;
        int endIndex = (*linearTree)[i].endIndex;

        if (endIndex - startIndex < MINSPLIT) {
            continue;
        }

        int curValue = hypercube[startIndex * DIM + curDim];
        int curChildCount = 0;
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

    buildLinearTree(hypercube, linearTree, childCount, dimStart, curCount,
                    curDim + 1, N, MINSPLIT, DIM);
    return;
}


__device__ int checkNeighbor(int index, int *hypercube, treeNode *linearTree,
                             int curIndex) {
    int curDif = abs(hypercube[index] - linearTree[curIndex].coordinate);
    if (curDif <= 1) {
        return 1;
    }
    return 0;
}

__device__ int checkImmediateNeighbor(int *hypercubeA, int *hypercubeB, int hypercubeCount, int DIM) {

    for (int i = 0; i < DIM; i++) {
        if (abs(hypercubeA[i] - hypercubeB[i]) > 1) {
            return 0;
        }
    }
    return 1;
}

__device__ int neighborDensitySubTree(int *hypercube, treeNode *linearTree, int hypercubeIndex,
                                      int *childCount, int *dimStart, int *instancesCount,
                                      int parentIndex, int curDim, int N, int DIM) {

    int curDensity = 0;
    int curHypercubeIndex = parentIndex;
    int index;
    int endIndex = linearTree[curHypercubeIndex].nextSiblingIndex;

    while (curHypercubeIndex != NONE && curHypercubeIndex != endIndex) {

        index = hypercubeIndex * DIM + (curDim - 1);

        if (linearTree[curHypercubeIndex].nextChildIndex == NONE) {
            for (int i = linearTree[curHypercubeIndex].startIndex;
                 i <= linearTree[curHypercubeIndex].endIndex; i++) {
            
                if (checkImmediateNeighbor(hypercube + hypercubeIndex * DIM, hypercube + i * DIM, N, DIM)) {
                    curDensity += instancesCount[i];
                }
            }
        }

        if ((linearTree[curHypercubeIndex].nextChildIndex != NONE &&
             checkNeighbor(index, hypercube, linearTree, curHypercubeIndex))) {

            int tempIndex = linearTree[curHypercubeIndex].nextChildIndex;

            while (tempIndex != NONE && (!checkNeighbor(index + 1, hypercube, linearTree, tempIndex))) {
                tempIndex = linearTree[tempIndex].nextSiblingIndex;
            }

            if (tempIndex != NONE) {
                curHypercubeIndex = tempIndex;
                curDim++;
            } else {
                while (curHypercubeIndex != NONE)
                {
                    if (curHypercubeIndex != NONE &&
                        linearTree[curHypercubeIndex].nextSiblingIndex != NONE) {
                        curHypercubeIndex = linearTree[curHypercubeIndex].nextSiblingIndex;
                        break;
                    }
                    curHypercubeIndex = linearTree[curHypercubeIndex].parentIndex;
                    curDim--;
                }
            }

        } else if (linearTree[curHypercubeIndex].nextSiblingIndex != NONE) {
            curHypercubeIndex = linearTree[curHypercubeIndex].nextSiblingIndex;
        } else {
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

__global__ void neighborhoodDensity(int *density, int *instancesCount,
                                    treeNode *linearTree, int *dimStart,
                                    int *hypercubeArray, int *childCount, int DIM,
                                    int hypercubeCount, int *dimNodes) {

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int initialDimNodeCount = childCount[0];
    int totalNodes = hypercubeCount * initialDimNodeCount;

    if (initialDimNodeCount == 0) {
        if (threadId < hypercubeCount) {
            int hypercubeIndex = threadId;
            int parentIndex = 0;
            int curThreadDensity = neighborDensitySubTree(hypercubeArray, linearTree, hypercubeIndex, childCount,
                                                          dimStart,
                                                          instancesCount, parentIndex, 1, hypercubeCount, DIM);

            atomicAdd(density + hypercubeIndex, curThreadDensity);
        }

    } else {
        if (threadId < totalNodes) {
            int hypercubeIndex = threadId / initialDimNodeCount;
            int parentIndex = dimNodes[threadId % initialDimNodeCount];

            int curThreadDensity = neighborDensitySubTree(hypercubeArray, linearTree, hypercubeIndex, childCount,
                                                          dimStart,
                                                          instancesCount, parentIndex, 1, hypercubeCount, DIM);

            atomicAdd(density + hypercubeIndex, curThreadDensity);
        }

    }

    return;
}

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

void buildOptimizedLinearTree(treeNode *linearTree,
                              treeNode *optimizedLinearTree, int *dimNodes) {
    int curHypercubeIndex = 1;
    int optimTreeCount = 0;

    if (linearTree[0].nextChildIndex == NONE) {
        copyContents(linearTree, optimizedLinearTree, NONE, NONE, NONE);
        optimTreeCount++;
        return;
    }

    copyContents(linearTree, optimizedLinearTree, NONE, NONE, 1);
    optimTreeCount++;
    copyContents(linearTree + 1, optimizedLinearTree + 1, 0, NONE, NONE);
    optimTreeCount++;
    int backTrackIndex = 1;
    int flag = 0;
    int curDim = 1;
    int count = 0;

    while (curHypercubeIndex != NONE) {

        if (curDim == 1) {
            dimNodes[count] = optimTreeCount - 1;
            count++;
        }

        if (linearTree[curHypercubeIndex].nextChildIndex != NONE) {

            curHypercubeIndex = linearTree[curHypercubeIndex].nextChildIndex;
            curDim++;

            if (flag == 1) {
                optimizedLinearTree[backTrackIndex].nextChildIndex = optimTreeCount;
                copyContents(linearTree + curHypercubeIndex,
                             optimizedLinearTree + optimTreeCount, backTrackIndex, NONE, NONE);
            } else {

                optimizedLinearTree[optimTreeCount - 1].nextChildIndex = optimTreeCount;
                copyContents(linearTree + curHypercubeIndex,
                             optimizedLinearTree + optimTreeCount, optimTreeCount - 1,
                             NONE, NONE);
            }

            backTrackIndex = optimTreeCount;
            optimTreeCount++;
            flag = 0;

        } else if (linearTree[curHypercubeIndex].nextSiblingIndex != NONE) {

            optimizedLinearTree[backTrackIndex].nextSiblingIndex = optimTreeCount;
            curHypercubeIndex = linearTree[curHypercubeIndex].nextSiblingIndex;

            int temp = optimizedLinearTree[backTrackIndex].parentIndex;
            backTrackIndex = optimTreeCount;

            copyContents(linearTree + curHypercubeIndex,
                         optimizedLinearTree + optimTreeCount, temp,
                         NONE, NONE);
            optimTreeCount++;
        } else {

            while (curHypercubeIndex != NONE) {
                curHypercubeIndex = linearTree[curHypercubeIndex].parentIndex;
                backTrackIndex = optimizedLinearTree[backTrackIndex].parentIndex;
                curDim--;

                if (curHypercubeIndex != NONE &&
                    linearTree[curHypercubeIndex].nextSiblingIndex != NONE) {
                    flag = 1;
                    curHypercubeIndex = linearTree[curHypercubeIndex].nextSiblingIndex;
                    int temp = optimizedLinearTree[backTrackIndex].parentIndex;

                    optimizedLinearTree[backTrackIndex].nextSiblingIndex = optimTreeCount;

                    backTrackIndex = optimTreeCount;

                    copyContents(linearTree + curHypercubeIndex,
                                 optimizedLinearTree + optimTreeCount, temp,
                                 NONE, NONE);
                    optimTreeCount++;
                    break;
                }
            }
        }
    }

    return;
}

__device__ int optimCheckNeighbor(int index, int *hypercube, optimTreeNode *linearTree,
                                  int curIndex) {
    int curDif = abs(hypercube[index] - linearTree[curIndex].coordinate);
    if (curDif <= 1) {
        return 1;
    }
    return 0;
}

__device__ int optimNeighborDensitySubTree(int *hypercube, optimTreeNode *linearTree, int hypercubeIndex,
                                           int *instancesCount, int parentIndex, int N, int DIM) {
    int curDensity = 0;
    int index;
    int curHypercubeIndex = parentIndex;
    int endIndex = linearTree[parentIndex].nextBreakIndex;

    while (curHypercubeIndex != NONE && curHypercubeIndex != endIndex)
    {
        index = hypercubeIndex * DIM + (linearTree[curHypercubeIndex].curDim - 1);

        if (linearTree[curHypercubeIndex].nextChildIndex == NONE) {
            for (int i = linearTree[curHypercubeIndex].startIndex;
                 i <= linearTree[curHypercubeIndex].endIndex; i++) 
                 {
                
                if (checkImmediateNeighbor(hypercube + hypercubeIndex * DIM, hypercube + i * DIM, N, DIM)) {
                    curDensity += instancesCount[i];
                }
            }
        }

        if (linearTree[curHypercubeIndex].nextChildIndex != NONE &&
            optimCheckNeighbor(index, hypercube, linearTree, curHypercubeIndex)) {
            curHypercubeIndex = linearTree[curHypercubeIndex].nextChildIndex;
        } else {
            curHypercubeIndex = linearTree[curHypercubeIndex].nextBreakIndex;
        }
    }

    return curDensity;
}

__global__ void naiveNeighborhoodDensity(int *density, int *instancesCount, int DIM,
                                         int hypercubeCount, int *hypercube, int SPLIT) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int totalThreads = hypercubeCount * SPLIT;

    if (threadId < totalThreads) {
        int hypercubeIndex = threadId / SPLIT;
        int curThreadDensity = 0;
        int startIndex = (threadId % SPLIT) * (hypercubeCount / SPLIT);
        int endIndex = startIndex + (hypercubeCount / SPLIT);
        if ((threadId % SPLIT) == SPLIT - 1) {
            endIndex += (hypercubeCount % SPLIT);
        }

        for (int i = startIndex; i < endIndex; i++) {
            if (checkImmediateNeighbor(hypercube + hypercubeIndex * DIM, hypercube + i * DIM, hypercubeCount, DIM)) {
                curThreadDensity += instancesCount[i];
            }
        }
        atomicAdd(density + hypercubeIndex, curThreadDensity);
    }
    return;
}

__global__ void optimNeighborhoodDensity(int *density, int *instancesCount,
                                         optimTreeNode *linearTree, int *hypercubeArray,
                                         int *childCount, int DIM,
                                         int hypercubeCount, int *dimNodes) {

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int initialDimNodeCount = childCount[0];
    int totalNodes = hypercubeCount * initialDimNodeCount;

    if (initialDimNodeCount == 0) {
        if (threadId < hypercubeCount) {
            int hypercubeIndex = threadId;
            int parentIndex = 0;
            int curThreadDensity = optimNeighborDensitySubTree(hypercubeArray, linearTree, hypercubeIndex,
                                                               instancesCount, parentIndex, hypercubeCount, DIM);

            atomicAdd(density + hypercubeIndex, curThreadDensity);
        }

    } else {
        if (threadId < totalNodes) {
            int hypercubeIndex = threadId / initialDimNodeCount;
            int parentIndex = dimNodes[threadId % initialDimNodeCount];

            int curThreadDensity = optimNeighborDensitySubTree(hypercubeArray, linearTree, hypercubeIndex,
                                                               instancesCount, parentIndex, hypercubeCount, DIM);

            atomicAdd(density + hypercubeIndex, curThreadDensity);
        }
    }

    return;
}

__host__ void copyNode(optimTreeNode *root, treeNode *linearCurNode, optimTreeNode *optimCurNode, int curDim) {
    (*optimCurNode).startIndex = (*linearCurNode).startIndex;
    (*optimCurNode).endIndex = (*linearCurNode).endIndex;
    (*optimCurNode).coordinate = (*linearCurNode).coordinate;
    (*optimCurNode).nextChildIndex = (*linearCurNode).nextChildIndex;
    (*optimCurNode).curDim = curDim;

    if ((*linearCurNode).nextSiblingIndex != NONE) {
        (*optimCurNode).nextBreakIndex = (*linearCurNode).nextSiblingIndex;
    } else {
        (*optimCurNode).nextBreakIndex = root[(*linearCurNode).parentIndex].nextBreakIndex;
    }
    return;
}

__host__ void buildSuperOptimTree(treeNode *linearTree, optimTreeNode *superOptimTree) 
{
    
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

    while (curHypercubeIndex != NONE) {

        copyNode(superOptimTree, linearTree + curHypercubeIndex, superOptimTree + curHypercubeIndex, curDim);

        if (linearTree[curHypercubeIndex].nextChildIndex != NONE) {
            curHypercubeIndex = linearTree[curHypercubeIndex].nextChildIndex;
            curDim++;
        } else if (linearTree[curHypercubeIndex].nextSiblingIndex != NONE) {
            curHypercubeIndex = linearTree[curHypercubeIndex].nextSiblingIndex;
        } else {
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


void calculateOutlierScore(float *outlierScore, int *neighborhoodDensity, map <vector<MY_DATATYPE>, vector<int>> hypercubeMap,
                           int N, int maxNeighborhoodDensity) {
    
    map < vector < MY_DATATYPE > , vector < int > > ::iterator itr;
   
    int index = 0;
    for (itr = hypercubeMap.begin(); itr != hypercubeMap.end(); itr++) {
        for (int i: itr->second) {
            outlierScore[i] =
                    (float) (maxNeighborhoodDensity - neighborhoodDensity[index]) / (float) (maxNeighborhoodDensity);
        }
        index++;
    }

    return;
}

int findK(int BIN)
{
    int k =1;

    while(pow(2,k) < BIN)
    {
        k++;
    }

    return k;
}

int main(int argc, char **argv) 
{
    
    // Process command-line arguments
    int N;
    int DIM;
    int BIN;
    int MINSPLIT; // MINSPLIT = 0 defaults to naive strategy
    int NORMALIZE = 1; // Set to 1 to normalize datasets - Does not affect timeTrails
    char inputFname[500] = ""; // Dataset
    int APPROACH = 1; // 0 for Naive strategy and 1 for Tree strategy
    int TREE_SELECT = 3; // Optional parameter (default -3) 1 for simple tree, 2 for tree with optimized locality, 3 for tree with optimized locality and traversal, Use 0 for naive strategy
    
    if (argc != 9)
    {
        fprintf(stderr, "Please provide the following on the command line: N (number of lines in the file), dimensionality (number of coordinates per point/feature vector), BIN (Bin parameter), Min Split(Threshold), Normalize (0 or 1), dataset filename. ");
         return 0;
    }

    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &DIM);
    sscanf(argv[3], "%d", &BIN);
    sscanf(argv[4], "%d", &MINSPLIT);
    sscanf(argv[5], "%d", &NORMALIZE);
    strcpy(inputFname, argv[6]);
    sscanf(argv[7], "%d", &APPROACH);
    sscanf(argv[8], "%d", &TREE_SELECT);
    
    if (N < 1 || DIM < 1 || BIN < 1 || MINSPLIT<0 || NORMALIZE > 1 || NORMALIZE < 0 || APPROACH < 0 || APPROACH > 1 || TREE_SELECT < 0 || TREE_SELECT > 3) {
        cout << "\nOne of the following are invalid: N, DIM, BIN , NORMALIZE, APPROACH, TREE_SELECT " << endl;
        return 0;
    } else {

        string currentTreeSelect = "NONE";

        if(APPROACH == 1)
        {
            currentTreeSelect = treeSelect[TREE_SELECT-1];
        }

        cout << "\nNumber of lines (N): " << N << " Dimensionality: " << DIM << " BIN Size: "<< BIN
               << " MinSplit: " << MINSPLIT << " Normalize: " <<NORMALIZE << " Filename: " << inputFname << " Approach: " << approach[APPROACH]
               << " Selected tree: " << currentTreeSelect << endl;

    }


     // allocate memory for dataset
        // pointer to entire dataset
    double *h_dataset = (double *) malloc(sizeof(double) * N * DIM);

    int ret = importDataset(inputFname, N, h_dataset, DIM);

    if (ret == 1) {
        return 0;
    }
    
    if (NORMALIZE == 1) {
        normalizeDataset(h_dataset, N, DIM);
    }

    cudaEvent_t totalTimeStart, totalTimeStop;
    cudaEventCreate(&totalTimeStart);
    cudaEventCreate(&totalTimeStop);

    cudaEventRecord(totalTimeStart);

    int blockDim = 32;

    while (blockDim <= DIM) {
        blockDim += 32;
    }

    if (blockDim > 1024) {
        printf("\nMax allowed dimensions is 1024");
        return 0;
    }

    int totalElements = N * DIM;

    int totalElementsPerBlock = blockDim / DIM;

    dim3 dimGrid(ceil((float) N / (float) totalElementsPerBlock), 1, 1);
    dim3 dimBlock(blockDim, 1, 1);

    int k = findK(BIN);

    int dimPerBlock = floor((double)(sizeof(MY_DATATYPE)*8)/(double)k);
  
    int encodeBlockSize = ceil((double) DIM / (double) dimPerBlock);

    MY_DATATYPE *d_hypercube = nullptr;

    double *d_dataset = nullptr;

    printf("Grid - %.0f , Block - %d\n", ceil((float) N / (float) totalElementsPerBlock), blockDim);

    MY_DATATYPE*h_hypercube = (MY_DATATYPE *) calloc(encodeBlockSize * N, sizeof(MY_DATATYPE));

    cudaMalloc((void **) &d_hypercube, (sizeof(MY_DATATYPE) * N * encodeBlockSize));

    cudaMalloc((void **) &d_dataset, sizeof(double) * totalElements);

    cudaMemcpy(d_dataset, h_dataset, sizeof(double) * totalElements,
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_hypercube, h_hypercube, (sizeof(MY_DATATYPE) * N * encodeBlockSize), cudaMemcpyHostToDevice);

    // Record time
    cudaEvent_t createHypercubeStart, createHypercubeStop;
    cudaEventCreate(&createHypercubeStart);
    cudaEventCreate(&createHypercubeStop);

    cudaEventRecord(createHypercubeStart);

    createHypercube<<<dimGrid, dimBlock,
    sizeof(int) * totalElementsPerBlock * DIM>>>(
            d_hypercube, d_dataset, N, DIM, BIN, encodeBlockSize, k);

    cudaDeviceSynchronize();

    cudaEventRecord(createHypercubeStop);
    cudaEventSynchronize(createHypercubeStop);

    // Calculate elapsed time
    float createHypercubeTime = 0;
    cudaEventElapsedTime(&createHypercubeTime, createHypercubeStart, createHypercubeStop);

    cudaMemcpy(h_hypercube, d_hypercube, (sizeof(MY_DATATYPE) * N * encodeBlockSize), cudaMemcpyDeviceToHost);

    // Improve memory util
    cudaFree(d_dataset);
    cudaFree(d_hypercube);

    map <vector<MY_DATATYPE>, vector<int>> h_hypercube_mapper;

    for (int i = 0; i + encodeBlockSize <= N * encodeBlockSize; i = i + encodeBlockSize) {

        vector<MY_DATATYPE> h_hypercube_key(h_hypercube + i, h_hypercube + i + encodeBlockSize);

        if (h_hypercube_mapper.find(h_hypercube_key) == h_hypercube_mapper.end()) {
            vector<int> h_hypercube_instance;
            h_hypercube_mapper[h_hypercube_key] = h_hypercube_instance;
        }

        h_hypercube_mapper[h_hypercube_key].push_back(i / encodeBlockSize);

    }

    int distinctHypercubeCount = 0;
    MY_DATATYPE* h_hypercubeDistinct = nullptr;

    int *h_instancesCount = nullptr;


    distinctHypercubeCount = h_hypercube_mapper.size();

    h_hypercubeDistinct = (MY_DATATYPE *) malloc(sizeof(MY_DATATYPE) * distinctHypercubeCount * encodeBlockSize);

    h_instancesCount = (int *) malloc(sizeof(int) * distinctHypercubeCount);
    
    map < vector <MY_DATATYPE> , vector < int > > ::iterator itr;

    int hypercubePos = 0;
    for (itr = h_hypercube_mapper.begin(); itr != h_hypercube_mapper.end(); itr++) {
        memcpy(h_hypercubeDistinct + hypercubePos, itr->first.data(), sizeof(MY_DATATYPE) * itr->first.size());
        h_instancesCount[hypercubePos / itr->first.size()] = itr->second.size();
        hypercubePos += itr->first.size();
    }

    // Building Array of Hypercubes

    printf("Distinct Hypercube Count: %d\n", distinctHypercubeCount);

    totalElements = distinctHypercubeCount * DIM;

    MY_DATATYPE *d_hypercubeDistinct;

    int *d_hypercubeArray;

    int *h_hypercubeArray = (int *) malloc(sizeof(int) * totalElements);

    cudaMalloc((void **) &d_hypercubeDistinct,
               sizeof(MY_DATATYPE) * distinctHypercubeCount * encodeBlockSize);

    cudaMalloc((void **) &d_hypercubeArray, sizeof(int) * totalElements);

    cudaMemcpy(d_hypercubeDistinct, h_hypercubeDistinct, sizeof(MY_DATATYPE) * distinctHypercubeCount * encodeBlockSize,
               cudaMemcpyHostToDevice);

    dimGrid.x = ceil((float) totalElements / (float) blockDim);

    printf("Grid - %.0f , Block - %d\n", ceil((float) totalElements / (float) blockDim), blockDim);

    // Record time here
    cudaEvent_t buildHypercubeArrayStart, buildHypercubeArrayStop;
    cudaEventCreate(&buildHypercubeArrayStart);
    cudaEventCreate(&buildHypercubeArrayStop);

    cudaEventRecord(buildHypercubeArrayStart);

    buildHypercubeArray<<<dimGrid, dimBlock>>>(
            d_hypercubeDistinct, d_hypercubeArray, distinctHypercubeCount, DIM, encodeBlockSize, k);

    cudaDeviceSynchronize();

    cudaEventRecord(buildHypercubeArrayStop);
    cudaEventSynchronize(buildHypercubeArrayStop);

    float buildHypercubeArrayTime = 0;
    cudaEventElapsedTime(&buildHypercubeArrayTime, buildHypercubeArrayStart, buildHypercubeArrayStop);

    int *h_neighborhoodDensity = (int *) calloc(distinctHypercubeCount, sizeof(int));
    int *d_neighborhoodDensity = NULL;
    int *d_instancesCount = NULL;

    cudaMalloc((void **) &d_neighborhoodDensity,
               sizeof(int) * distinctHypercubeCount);

    cudaMalloc((void **) &d_instancesCount, sizeof(int) * distinctHypercubeCount);

    cudaMemcpy(d_neighborhoodDensity, h_neighborhoodDensity,
               sizeof(int) * distinctHypercubeCount, cudaMemcpyHostToDevice);

    cudaMemcpy(d_instancesCount, h_instancesCount,
               sizeof(int) * distinctHypercubeCount, cudaMemcpyHostToDevice);

    cudaEvent_t neighborhoodDensityStart, neighborhoodDensityStop, buildTreeStart, buildTreeStop;
    cudaEvent_t buildSuperOptimTreeStart, buildSuperOptimTreeStop;
    cudaEvent_t buildOptimTreeStart, buildOptimTreeStop;
    float neighborhoodDensityTime = 0, treeTime = 0, superOptimTreeTime = 0, optimTreeTime = 0;


    if (MINSPLIT == 0) {
        dimBlock.x = 256;
        int SPLIT = BIN;
        dimGrid.x = ceil((float) (distinctHypercubeCount * SPLIT) / (float) (dimBlock.x));

        cudaEventCreate(&neighborhoodDensityStart);
        cudaEventCreate(&neighborhoodDensityStop);
        cudaEventRecord(neighborhoodDensityStart);

        naiveNeighborhoodDensity<<<dimGrid, dimBlock>>>(d_neighborhoodDensity, d_instancesCount, DIM,
                                                        distinctHypercubeCount, d_hypercubeArray, SPLIT);

        cudaDeviceSynchronize();
        cudaEventRecord(neighborhoodDensityStop);
        cudaEventSynchronize(neighborhoodDensityStop);
        cudaEventElapsedTime(&neighborhoodDensityTime, neighborhoodDensityStart, neighborhoodDensityStop);

    } else {

        cudaMemcpy(h_hypercubeArray, d_hypercubeArray, sizeof(int) * totalElements,
                   cudaMemcpyDeviceToHost);

        // Build a linear tree

        treeNode *h_linearTree = NULL;
        int linearTreeCount = 0;
        int curDim = 0;

        int *h_childCount = (int *) calloc((DIM + 1), sizeof(int));
        int *h_dimStart = (int *) calloc((DIM + 1), sizeof(int));

        appendNode(&h_linearTree, 0, distinctHypercubeCount - 1, NONE, NONE,
                   &linearTreeCount);

        cudaEventCreate(&buildTreeStart);
        cudaEventCreate(&buildTreeStop);

        cudaEventRecord(buildTreeStart);

        buildLinearTree(h_hypercubeArray, &h_linearTree, h_childCount, h_dimStart,
                        &linearTreeCount, curDim, distinctHypercubeCount, MINSPLIT, DIM);

        cudaDeviceSynchronize();

        cudaEventRecord(buildTreeStop);
        cudaEventSynchronize(buildTreeStop);

        cudaEventElapsedTime(&treeTime, buildTreeStart, buildTreeStop);

        printf("Tree Node Count: %d\n", linearTreeCount);

        // Build optimized linear tree with better cache hits
        // Build optimized linear tree here
        // Allocate memory for optim tree

        printf("Optim tree build start\n");

        treeNode *h_optimizedLinearTree =
                (treeNode *) malloc(sizeof(treeNode) * linearTreeCount);


        int *h_dimNodes = (int *) malloc(sizeof(int) * h_childCount[0]);

        cudaEventCreate(&buildOptimTreeStart);
        cudaEventCreate(&buildOptimTreeStop);

        cudaEventRecord(buildOptimTreeStart);

        buildOptimizedLinearTree(h_linearTree, h_optimizedLinearTree, h_dimNodes);

        cudaDeviceSynchronize();

        cudaEventRecord(buildOptimTreeStop);
        cudaEventSynchronize(buildOptimTreeStop);

        cudaEventElapsedTime(&optimTreeTime, buildOptimTreeStart, buildOptimTreeStop);

        puts("Optimized tree build done");

        optimTreeNode *h_superOptimTree = (optimTreeNode *) malloc(sizeof(optimTreeNode) * linearTreeCount);

        cudaEventCreate(&buildSuperOptimTreeStart);
        cudaEventCreate(&buildSuperOptimTreeStop);

        cudaEventRecord(buildSuperOptimTreeStart);

        buildSuperOptimTree(h_optimizedLinearTree, h_superOptimTree);

        cudaDeviceSynchronize();

        cudaEventRecord(buildSuperOptimTreeStop);
        cudaEventSynchronize(buildSuperOptimTreeStop);

        cudaEventElapsedTime(&superOptimTreeTime, buildSuperOptimTreeStart, buildSuperOptimTreeStop);

        puts("Super Optimized Tree build done!!");

        // Count neighborhood density of hypercubes

        h_dimStart[DIM] = h_childCount[DIM - 2] + h_dimStart[DIM - 1];

        optimTreeNode *d_linearTree = NULL;
        int *d_dimStart = NULL;
        int *d_childCount = NULL;

        //Adding new code below

        int *d_dimNodes = NULL;

        cudaMalloc((void **) &d_dimNodes, sizeof(int) * h_childCount[0]);

        cudaMemcpy(d_dimNodes, h_dimNodes,
                   sizeof(int) * h_childCount[0], cudaMemcpyHostToDevice);


        // Check below line again

        cudaMalloc((void **) &d_linearTree, sizeof(optimTreeNode) * linearTreeCount);

        cudaMalloc((void **) &d_dimStart, sizeof(int) * (DIM + 1));

        cudaMalloc((void **) &d_childCount, sizeof(int) * (DIM + 1));


        cudaMemcpy(d_linearTree, h_superOptimTree, sizeof(optimTreeNode) * linearTreeCount,
                   cudaMemcpyHostToDevice);


        cudaMemcpy(d_dimStart, h_dimStart, sizeof(int) * (DIM + 1),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_childCount, h_childCount, sizeof(int) * (DIM + 1),
                   cudaMemcpyHostToDevice);


        dimBlock.x = 256;
        if (h_childCount[0] == 0) {
            dimGrid.x = ceil((float) (distinctHypercubeCount) / (float) (dimBlock.x));
        } else {
            dimGrid.x = ceil((float) (distinctHypercubeCount * h_childCount[0]) / (float) (dimBlock.x));
        }
        printf("Grid - %d  , Block - %d\n", dimGrid.x, dimBlock.x);
        printf("Starting to process %d hypercubes\n", distinctHypercubeCount);

        // Record time here

        cudaEventCreate(&neighborhoodDensityStart);
        cudaEventCreate(&neighborhoodDensityStop);

        cudaEventRecord(neighborhoodDensityStart);

        optimNeighborhoodDensity<<<dimGrid, dimBlock>>>(
                d_neighborhoodDensity, d_instancesCount, d_linearTree, d_hypercubeArray,
                d_childCount, DIM, distinctHypercubeCount, d_dimNodes);

        cudaDeviceSynchronize();

        cudaEventRecord(neighborhoodDensityStop);
        cudaEventSynchronize(neighborhoodDensityStop);

        cudaEventElapsedTime(&neighborhoodDensityTime, neighborhoodDensityStart, neighborhoodDensityStop);

    }

    cudaMemcpy(h_neighborhoodDensity, d_neighborhoodDensity,
               sizeof(int) * distinctHypercubeCount, cudaMemcpyDeviceToHost);

    int maxNeighborhoodDensity = INT_MIN;

    for (int i = 0; i < distinctHypercubeCount; i++) {
        if (i < 50)
        printf("Index - %d - %d\n",i,h_neighborhoodDensity[i]);
        if (h_neighborhoodDensity[i] > maxNeighborhoodDensity) {
            maxNeighborhoodDensity = h_neighborhoodDensity[i];
        }
    }

    
    // Calculate Outlier Score

    float *h_outlierScore =
            (float *) malloc(sizeof(float) * N);

    calculateOutlierScore(
            h_outlierScore, h_neighborhoodDensity, h_hypercube_mapper, N,
            maxNeighborhoodDensity);
    clock_t end = clock();
    
   
    
    cudaDeviceSynchronize();
    cudaEventRecord(totalTimeStop);
    cudaEventSynchronize(totalTimeStop);

    float totalTime = 0;
    cudaEventElapsedTime(&totalTime, totalTimeStart, totalTimeStop);

    printf("============TIME RESULTS================\n");

    printf("Total time for execution is %f sec \n", (totalTime / 1000));

    printf("Total time for building hypercube is %f sec \n", (buildHypercubeArrayTime / 1000));

    //printf("Total time to sort is %f sec \n",(sortTime/1000));

    printf("Time for building linear Tree is %f sec \n", (treeTime / 1000));

    printf("Time for building optimized linear Tree is %f sec \n", (optimTreeTime / 1000));

    printf("Time for building super optimized linear Tree is %f sec \n", (superOptimTreeTime / 1000));

    printf("Time for neighborhood density is %f sec \n", (neighborhoodDensityTime / (1000)));

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        printf("%s \n", cudaGetErrorString(error));
    }

    // free dataset
    free(h_dataset);

    return 0;
}
