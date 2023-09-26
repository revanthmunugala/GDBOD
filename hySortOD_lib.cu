#include "hySortOD_lib.h"

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
          (double)((dataset[i] - minValue[j]) / (maxValue[j] - minValue[j]));
    } else {
      dataset[i] = 0;
    }
  }

  return;
}

__device__ void encodeHypercube(MY_DATATYPE *curHypercube, int *hypercube,
                                int DIM, int index, int encodeBlockSize,
                                int k) {
  int idx = 0;
  int dimPerBlock = floor((double)(sizeof(MY_DATATYPE) * 8) / (double)k);

  for (int i = 0; i < DIM; i++) {

    if (i % dimPerBlock == 0) {
      idx = i / dimPerBlock + index * encodeBlockSize;
    }
    curHypercube[idx] = curHypercube[idx] << k | hypercube[i];
  }
  return;
}

__global__ void createHypercube(MY_DATATYPE *hypercube, double *dataset, int N,
                                int DIM, int BIN, int encodeBlockSize, int k) {
  int totalElements = N * DIM;
  int totalElementsPerBlock = blockDim.x / DIM;
  int threadId = blockIdx.x * totalElementsPerBlock * DIM + threadIdx.x;

  double length = (double)(1) / (double)BIN;

  extern __shared__ int tempHypercube[];

  if (threadId < totalElements && threadIdx.x < totalElementsPerBlock * DIM) {
    tempHypercube[threadIdx.x] = (int)floor(dataset[threadId] / length);
  }

  __syncthreads();

  int hypercubeIndex = (threadIdx.x / DIM) + blockIdx.x * totalElementsPerBlock;

  if ((threadIdx.x % DIM == 0) && hypercubeIndex < N &&
      hypercubeIndex < (blockIdx.x + 1) * totalElementsPerBlock) {
    encodeHypercube(hypercube, tempHypercube + threadIdx.x, DIM, hypercubeIndex,
                    encodeBlockSize, k);
  }

  return;
}

__host__ __device__ int setBitsTo1(int k) { return pow(2, k) - 1; }

__device__ void decodeHypercube(MY_DATATYPE *hypercubeEncodedArray,
                                int *hypercube, int DIM, int hypercubeCount,
                                int threadId, int encodeBlockSize, int k) {

  int lastIndex = encodeBlockSize - 1;
  int elementsPerBlock = floor((double)(sizeof(MY_DATATYPE) * 8) / (double)k);
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
                                    int hypercubeCount, int DIM,
                                    int encodeBlockSize, int k) {
  int threadId = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadId < hypercubeCount) {
    decodeHypercube(hypercube + threadId * encodeBlockSize, hypercubeArray, DIM,
                    hypercubeCount, threadId, encodeBlockSize, k);
  }
  return;
}

__host__ void appendNode(treeNode **rootNode, int startIndex, int endIndex,
                         int coordinate, int parentIndex, int *curCount) {

  treeNode *curNode = (treeNode *)malloc(sizeof(treeNode));
  curNode->startIndex = startIndex;
  curNode->endIndex = endIndex;
  curNode->coordinate = coordinate;
  curNode->parentIndex = parentIndex;
  curNode->nextChildIndex = NONE;
  curNode->nextSiblingIndex = NONE;

  if ((*curCount) == 0) {
    (*rootNode) = (treeNode *)malloc(sizeof(treeNode) * ((*curCount) + 1));
  } else {
    (*rootNode) =
        (treeNode *)realloc((*rootNode), sizeof(treeNode) * ((*curCount) + 1));
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

__device__ int checkImmediateNeighbor(int *hypercubeA, int *hypercubeB,
                                      int hypercubeCount, int DIM) {

  for (int i = 0; i < DIM; i++) {
    if (abs(hypercubeA[i] - hypercubeB[i]) > 1) {
      return 0;
    }
  }
  return 1;
}

__device__ int neighborDensitySubTree(int *hypercube, treeNode *linearTree,
                                      int hypercubeIndex, int *childCount,
                                      int *instancesCount, int parentIndex,
                                      int curDim, int N, int DIM) {

  int curDensity = 0;
  int curHypercubeIndex = parentIndex;
  int index;
  int endIndex = linearTree[curHypercubeIndex].nextSiblingIndex;

  while (curHypercubeIndex != NONE && curHypercubeIndex != endIndex) {

    index = hypercubeIndex * DIM + (curDim - 1);

    if (linearTree[curHypercubeIndex].nextChildIndex == NONE) {
      for (int i = linearTree[curHypercubeIndex].startIndex;
           i <= linearTree[curHypercubeIndex].endIndex; i++) {

        if (checkImmediateNeighbor(hypercube + hypercubeIndex * DIM,
                                   hypercube + i * DIM, N, DIM)) {
          curDensity += instancesCount[i];
        }
      }
    }

    if ((linearTree[curHypercubeIndex].nextChildIndex != NONE &&
         checkNeighbor(index, hypercube, linearTree, curHypercubeIndex))) {

      int tempIndex = linearTree[curHypercubeIndex].nextChildIndex;

      while (tempIndex != NONE &&
             (!checkNeighbor(index + 1, hypercube, linearTree, tempIndex))) {
        tempIndex = linearTree[tempIndex].nextSiblingIndex;
      }

      if (tempIndex != NONE) {
        curHypercubeIndex = tempIndex;
        curDim++;
      } else {
        while (curHypercubeIndex != NONE) {
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
                                    treeNode *linearTree, int *hypercubeArray,
                                    int *childCount, int DIM,
                                    int hypercubeCount, int *dimNodes) {

  int threadId = threadIdx.x + blockIdx.x * blockDim.x;
  int initialDimNodeCount = childCount[0];
  int totalNodes = hypercubeCount * initialDimNodeCount;

  if (initialDimNodeCount == 0) {
    if (threadId < hypercubeCount) {
      int hypercubeIndex = threadId;
      int parentIndex = 0;
      int curThreadDensity = neighborDensitySubTree(
          hypercubeArray, linearTree, hypercubeIndex, childCount,
          instancesCount, parentIndex, 1, hypercubeCount, DIM);

      atomicAdd(density + hypercubeIndex, curThreadDensity);
    }

  } else {
    if (threadId < totalNodes) {
      int hypercubeIndex = threadId / initialDimNodeCount;
      int parentIndex = dimNodes[threadId % initialDimNodeCount];

      int curThreadDensity = neighborDensitySubTree(
          hypercubeArray, linearTree, hypercubeIndex, childCount,
          instancesCount, parentIndex, 1, hypercubeCount, DIM);

      atomicAdd(density + hypercubeIndex, curThreadDensity);
    }
  }

  return;
}

__global__ void simpleNeighborhoodDensity(int *density, int *instancesCount,
                                          treeNode *linearTree,
                                          int *hypercubeArray, int *childCount,
                                          int DIM, int hypercubeCount) {

  int threadId = threadIdx.x + blockIdx.x * blockDim.x;
  int initialDimNodeCount = childCount[0];
  int totalNodes = hypercubeCount * initialDimNodeCount;

  if (initialDimNodeCount == 0) {
    if (threadId < hypercubeCount) {
      int hypercubeIndex = threadId;
      int parentIndex = 0;
      int curThreadDensity = neighborDensitySubTree(
          hypercubeArray, linearTree, hypercubeIndex, childCount,
          instancesCount, parentIndex, 1, hypercubeCount, DIM);

      atomicAdd(density + hypercubeIndex, curThreadDensity);
    }

  } else {
    if (threadId < totalNodes) {
      int hypercubeIndex = threadId / initialDimNodeCount;
      int parentIndex = (threadId % initialDimNodeCount) + 1;

      int curThreadDensity = neighborDensitySubTree(
          hypercubeArray, linearTree, hypercubeIndex, childCount,
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

    } else if (linearTree[curHypercubeIndex].nextSiblingIndex != NONE) {

      optimizedLinearTree[backTrackIndex].nextSiblingIndex = optimTreeCount;
      curHypercubeIndex = linearTree[curHypercubeIndex].nextSiblingIndex;

      int temp = optimizedLinearTree[backTrackIndex].parentIndex;
      backTrackIndex = optimTreeCount;

      copyContents(linearTree + curHypercubeIndex,
                   optimizedLinearTree + optimTreeCount, temp, NONE, NONE);
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
                       optimizedLinearTree + optimTreeCount, temp, NONE, NONE);
          optimTreeCount++;
          break;
        }
      }
    }
  }

  return;
}

__device__ int optimCheckNeighbor(int index, int *hypercube,
                                  optimTreeNode *linearTree, int curIndex) {
  int curDif = abs(hypercube[index] - linearTree[curIndex].coordinate);
  if (curDif <= 1) {
    return 1;
  }
  return 0;
}

__device__ int optimNeighborDensitySubTree(int *hypercube,
                                           optimTreeNode *linearTree,
                                           int hypercubeIndex,
                                           int *instancesCount, int parentIndex,
                                           int N, int DIM) {
  int curDensity = 0;
  int index;
  int curHypercubeIndex = parentIndex;
  int endIndex = linearTree[parentIndex].nextBreakIndex;

  while (curHypercubeIndex != NONE && curHypercubeIndex != endIndex) {
    index = hypercubeIndex * DIM + (linearTree[curHypercubeIndex].curDim - 1);

    if (linearTree[curHypercubeIndex].nextChildIndex == NONE) {
      for (int i = linearTree[curHypercubeIndex].startIndex;
           i <= linearTree[curHypercubeIndex].endIndex; i++) {

        if (checkImmediateNeighbor(hypercube + hypercubeIndex * DIM,
                                   hypercube + i * DIM, N, DIM)) {
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

__global__ void naiveNeighborhoodDensity(int *density, int *instancesCount,
                                         int DIM, int hypercubeCount,
                                         int *hypercube, int SPLIT) {
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
      if (checkImmediateNeighbor(hypercube + hypercubeIndex * DIM,
                                 hypercube + i * DIM, hypercubeCount, DIM)) {
        curThreadDensity += instancesCount[i];
      }
    }
    atomicAdd(density + hypercubeIndex, curThreadDensity);
  }
  return;
}

__global__ void optimNeighborhoodDensity(int *density, int *instancesCount,
                                         optimTreeNode *linearTree,
                                         int *hypercubeArray, int *childCount,
                                         int DIM, int hypercubeCount,
                                         int *dimNodes) {

  int threadId = threadIdx.x + blockIdx.x * blockDim.x;
  int initialDimNodeCount = childCount[0];
  int totalNodes = hypercubeCount * initialDimNodeCount;

  if (initialDimNodeCount == 0) {
    if (threadId < hypercubeCount) {
      int hypercubeIndex = threadId;
      int parentIndex = 0;
      int curThreadDensity = optimNeighborDensitySubTree(
          hypercubeArray, linearTree, hypercubeIndex, instancesCount,
          parentIndex, hypercubeCount, DIM);

      atomicAdd(density + hypercubeIndex, curThreadDensity);
    }

  } else {
    if (threadId < totalNodes) {
      int hypercubeIndex = threadId / initialDimNodeCount;
      int parentIndex = dimNodes[threadId % initialDimNodeCount];

      int curThreadDensity = optimNeighborDensitySubTree(
          hypercubeArray, linearTree, hypercubeIndex, instancesCount,
          parentIndex, hypercubeCount, DIM);

      atomicAdd(density + hypercubeIndex, curThreadDensity);
    }
  }

  return;
}

__host__ void copyNode(optimTreeNode *root, treeNode *linearCurNode,
                       optimTreeNode *optimCurNode, int curDim) {
  (*optimCurNode).startIndex = (*linearCurNode).startIndex;
  (*optimCurNode).endIndex = (*linearCurNode).endIndex;
  (*optimCurNode).coordinate = (*linearCurNode).coordinate;
  (*optimCurNode).nextChildIndex = (*linearCurNode).nextChildIndex;
  (*optimCurNode).curDim = curDim;

  if ((*linearCurNode).nextSiblingIndex != NONE) {
    (*optimCurNode).nextBreakIndex = (*linearCurNode).nextSiblingIndex;
  } else {
    (*optimCurNode).nextBreakIndex =
        root[(*linearCurNode).parentIndex].nextBreakIndex;
  }
  return;
}

__host__ void buildSuperOptimTree(treeNode *linearTree,
                                  optimTreeNode *superOptimTree) {

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

    copyNode(superOptimTree, linearTree + curHypercubeIndex,
             superOptimTree + curHypercubeIndex, curDim);

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

void calculateOutlierScore(float *outlierScore, int *neighborhoodDensity,
                           map<vector<MY_DATATYPE>, vector<int>> hypercubeMap,
                           int N, int maxNeighborhoodDensity) {

  map<vector<MY_DATATYPE>, vector<int>>::iterator itr;

  int index = 0;
  for (itr = hypercubeMap.begin(); itr != hypercubeMap.end(); itr++) {
    for (int i : itr->second) {
      outlierScore[i] =
          (float)(maxNeighborhoodDensity - neighborhoodDensity[index]) /
          (float)(maxNeighborhoodDensity);
    }
    index++;
  }

  return;
}

int findK(int BIN) {
  int k = 1;

  while (pow(2, k) < BIN) {
    k++;
  }

  return k;
}

float naiveStrategy(int *d_hypercubeArray, int *h_neighborhoodDensity,
                    int *h_instancesCount, int distinctHypercubeCount, int BIN,
                    int DIM) {

  cudaEvent_t neighborhoodDensityStart, neighborhoodDensityStop;
  float neighborhoodDensityTime;

  int *d_neighborhoodDensity = NULL;

  cudaMalloc((void **)&d_neighborhoodDensity,
             sizeof(int) * distinctHypercubeCount);

  cudaMemcpy(d_neighborhoodDensity, h_neighborhoodDensity,
             sizeof(int) * distinctHypercubeCount, cudaMemcpyHostToDevice);

  int *d_instancesCount = NULL;

  cudaMalloc((void **)&d_instancesCount, sizeof(int) * distinctHypercubeCount);

  cudaMemcpy(d_instancesCount, h_instancesCount,
             sizeof(int) * distinctHypercubeCount, cudaMemcpyHostToDevice);

  // Use naive approach when approach is 0 or when MINSPLIT = 0
  dim3 dimBlock, dimGrid;

  dimBlock.x = 256;
  int SPLIT = BIN;
  dimGrid.x =
      ceil((float)(distinctHypercubeCount * SPLIT) / (float)(dimBlock.x));

  cudaEventCreate(&neighborhoodDensityStart);
  cudaEventCreate(&neighborhoodDensityStop);
  cudaEventRecord(neighborhoodDensityStart);

  naiveNeighborhoodDensity<<<dimGrid, dimBlock>>>(
      d_neighborhoodDensity, d_instancesCount, DIM, distinctHypercubeCount,
      d_hypercubeArray, SPLIT);

  cudaDeviceSynchronize();
  cudaEventRecord(neighborhoodDensityStop);
  cudaEventSynchronize(neighborhoodDensityStop);
  cudaEventElapsedTime(&neighborhoodDensityTime, neighborhoodDensityStart,
                       neighborhoodDensityStop);

  cudaMemcpy(h_neighborhoodDensity, d_neighborhoodDensity,
             sizeof(int) * distinctHypercubeCount, cudaMemcpyDeviceToHost);

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
  cudaMalloc((void **)&d_neighborhoodDensity, neighborhoodDensityMemory);

  cudaMemcpy(d_neighborhoodDensity, h_neighborhoodDensity,
             neighborhoodDensityMemory, cudaMemcpyHostToDevice);

  // Allocate memory in device for instances Count and copy from host to device
  cudaMalloc((void **)&d_instancesCount, instancesCountMemory);

  cudaMemcpy(d_instancesCount, h_instancesCount, instancesCountMemory,
             cudaMemcpyHostToDevice);

  // Build a simple tree
  // Array for counting nodes in each dim
  int *h_childCount = (int *)calloc((DIM + 1), sizeof(int));
  int *h_dimStart = (int *)calloc((DIM + 1), sizeof(int));

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
      (treeNode *)malloc(sizeof(treeNode) * linearTreeCount);

  // Array to hold the indexes of nodes in first dimension
  // This is necessary for assigning multiple threads per tree
  int *h_dimNodes = (int *)malloc(dimIndexMemory);

  buildOptimizedLinearTree(h_linearTree, h_optimizedLinearTree, h_dimNodes);

  // Build locality and traversal optimized tree
  optimTreeNode *h_superOptimTree = (optimTreeNode *)malloc(linearTreeMemory);

  buildSuperOptimTree(h_optimizedLinearTree, h_superOptimTree);

  // Allocate memory in device and copy from host to device
  cudaMalloc((void **)&d_dimNodes, dimIndexMemory);

  cudaMemcpy(d_dimNodes, h_dimNodes, dimIndexMemory, cudaMemcpyHostToDevice);

  // Allocate memory in device for fast tree and copy from host to
  // device
  cudaMalloc((void **)&d_linearTree, linearTreeMemory);

  cudaMemcpy(d_linearTree, h_superOptimTree, linearTreeMemory,
             cudaMemcpyHostToDevice);

  // Allocate memory in device and copy from host to device
  cudaMalloc((void **)&d_childCount, childCountMemory);

  cudaMemcpy(d_childCount, h_childCount, childCountMemory,
             cudaMemcpyHostToDevice);

  // Set threads per block for kernel function
  dimBlock.x = 256;

  if (h_childCount[0] == 0) {
    dimGrid.x = ceil((float)(distinctHypercubeCount) / (float)(dimBlock.x));
  } else {
    dimGrid.x = ceil((float)(distinctHypercubeCount * h_childCount[0]) /
                     (float)(dimBlock.x));
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
  cudaMalloc((void **)&d_neighborhoodDensity, neighborhoodDensityMemory);

  cudaMemcpy(d_neighborhoodDensity, h_neighborhoodDensity,
             neighborhoodDensityMemory, cudaMemcpyHostToDevice);

  // Allocate memory in device for instances Count and copy from host to device
  cudaMalloc((void **)&d_instancesCount, instancesCountMemory);

  cudaMemcpy(d_instancesCount, h_instancesCount, instancesCountMemory,
             cudaMemcpyHostToDevice);

  // Build a simple tree
  // Array for counting nodes in each dim
  int *h_childCount = (int *)calloc((DIM + 1), sizeof(int));
  // Array to hold the starting index of each dimension
  int *h_dimStart = (int *)calloc((DIM + 1), sizeof(int));

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
  treeNode *h_optimizedLinearTree = (treeNode *)malloc(linearTreeMemory);

  // Array to hold the indexes of nodes in first dimension
  // This is necessary for assigning multiple threads per tree
  int *h_dimNodes = (int *)malloc(dimIndexMemory);

  buildOptimizedLinearTree(h_linearTree, h_optimizedLinearTree, h_dimNodes);

  // Allocate memory in device and copy from host to device
  cudaMalloc((void **)&d_dimNodes, dimIndexMemory);

  cudaMemcpy(d_dimNodes, h_dimNodes, dimIndexMemory, cudaMemcpyHostToDevice);

  // Allocate memory in device for locality optim tree and copy from host to
  // device
  cudaMalloc((void **)&d_linearTree, linearTreeMemory);

  cudaMemcpy(d_linearTree, h_optimizedLinearTree, linearTreeMemory,
             cudaMemcpyHostToDevice);

  // Allocate memory in device and copy from host to device
  cudaMalloc((void **)&d_childCount, childCountMemory);

  cudaMemcpy(d_childCount, h_childCount, childCountMemory,
             cudaMemcpyHostToDevice);

  // Set threads per block for kernel function
  dimBlock.x = 256;

  if (h_childCount[0] == 0) {
    dimGrid.x = ceil((float)(distinctHypercubeCount) / (float)(dimBlock.x));
  } else {
    dimGrid.x = ceil((float)(distinctHypercubeCount * h_childCount[0]) /
                     (float)(dimBlock.x));
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
  cudaMalloc((void **)&d_neighborhoodDensity, neighborhoodDensityMemory);

  cudaMemcpy(d_neighborhoodDensity, h_neighborhoodDensity,
             neighborhoodDensityMemory, cudaMemcpyHostToDevice);

  // Allocate memory in device for instances Count and copy from host to device
  cudaMalloc((void **)&d_instancesCount, instancesCountMemory);

  cudaMemcpy(d_instancesCount, h_instancesCount, instancesCountMemory,
             cudaMemcpyHostToDevice);

  // Build a linear tree
  // Array for counting nodes in each dim
  int *h_childCount = (int *)calloc((DIM + 1), sizeof(int));
  // Array to hold the starting index of each dimension
  int *h_dimStart = (int *)calloc((DIM + 1), sizeof(int));

  // Append initial node
  appendNode(&h_linearTree, 0, distinctHypercubeCount - 1, NONE, NONE,
             &linearTreeCount);

  buildLinearTree(h_hypercubeArray, &h_linearTree, h_childCount, h_dimStart,
                  &linearTreeCount, curDim, distinctHypercubeCount, MINSPLIT,
                  DIM);

  size_t linearTreeMemory = sizeof(treeNode) * linearTreeCount;

  // Allocate memory in device for simple tree and copy from host to device
  cudaMalloc((void **)&d_linearTree, linearTreeMemory);

  cudaMemcpy(d_linearTree, h_linearTree, linearTreeMemory,
             cudaMemcpyHostToDevice);

  // Allocate memory in device for nodes in each dim and copy from host to
  // device
  cudaMalloc((void **)&d_childCount, childCountMemory);

  cudaMemcpy(d_childCount, h_childCount, childCountMemory,
             cudaMemcpyHostToDevice);

  // Set threads per block for kernel function
  dimBlock.x = 256;

  if (h_childCount[0] == 0) {
    dimGrid.x = ceil((float)(distinctHypercubeCount) / (float)(dimBlock.x));
  } else {
    dimGrid.x = ceil((float)(distinctHypercubeCount * h_childCount[0]) /
                     (float)(dimBlock.x));
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
