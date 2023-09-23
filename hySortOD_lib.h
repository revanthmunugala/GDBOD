#ifndef HYSORTOD_LIB_H
#define HYSORTOD_LIB_H

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

__host__ int importDataset(char *fname, int N, double *dataset, int DIM);

__host__ void normalizeDataset(double *dataset, int N, int DIM);

__device__ void encodeHypercube(MY_DATATYPE*curHypercube, int *hypercube, int DIM, int index,
                                int encodeBlockSize, int k);

__global__ void createHypercube(MY_DATATYPE*hypercube, double *dataset, int N, int DIM,
                                int BIN, int encodeBlockSize, int k);
                                

__device__ void decodeHypercube(MY_DATATYPE *hypercubeEncodedArray, int *hypercube, int DIM,
                                int hypercubeCount, int threadId, int encodeBlockSize, int k);

__global__ void buildHypercubeArray(MY_DATATYPE *hypercube, int *hypercubeArray,
                                    int hypercubeCount, int DIM, int encodeBlockSize, int k);

__host__ void appendNode(treeNode **rootNode, int startIndex, int endIndex,
                         int coordinate, int parentIndex, int *curCount);

__host__ void buildLinearTree(int *hypercube, treeNode **linearTree,
                              int *childCount, int *dimStart, int *curCount,
                              int curDim, int N, int MINSPLIT, int DIM);

__device__ int checkNeighbor(int index, int *hypercube, treeNode *linearTree,
                             int curIndex);

__device__ int checkImmediateNeighbor(int *hypercubeA, int *hypercubeB, int hypercubeCount, int DIM);

__device__ int neighborDensitySubTree(int *hypercube, treeNode *linearTree, int hypercubeIndex,
                                      int *childCount, int *dimStart, int *instancesCount,
                                      int parentIndex, int curDim, int N, int DIM);

__global__ void neighborhoodDensity(int *density, int *instancesCount,
                                    treeNode *linearTree, int *dimStart,
                                    int *hypercubeArray, int *childCount, int DIM,
                                    int hypercubeCount, int *dimNodes);

__host__ void copyContents(treeNode *linearCurNode, treeNode *optimCurNode,
                  int parentIndex, int siblingIndex, int childIndex);

__host__ void buildOptimizedLinearTree(treeNode *linearTree,
                              treeNode *optimizedLinearTree, int *dimNodes);

__device__ int optimCheckNeighbor(int index, int *hypercube, optimTreeNode *linearTree,
                                  int curIndex);

__device__ int optimNeighborDensitySubTree(int *hypercube, optimTreeNode *linearTree, int hypercubeIndex,
                                           int *instancesCount, int parentIndex, int N, int DIM);

__global__ void naiveNeighborhoodDensity(int *density, int *instancesCount, int DIM,
                                         int hypercubeCount, int *hypercube, int SPLIT);

__global__ void optimNeighborhoodDensity(int *density, int *instancesCount,
                                         optimTreeNode *linearTree, int *hypercubeArray,
                                         int *childCount, int DIM,
                                         int hypercubeCount, int *dimNodes) ;

__host__ void copyNode(optimTreeNode *root, treeNode *linearCurNode, optimTreeNode *optimCurNode, int curDim);

__host__ void buildSuperOptimTree(treeNode *linearTree, optimTreeNode *superOptimTree); 

__host__ void calculateOutlierScore(float *outlierScore, int *neighborhoodDensity, map <vector<MY_DATATYPE>, vector<int>> hypercubeMap,
                           int N, int maxNeighborhoodDensity);

__host__ int findK(int BIN);

__host__ __device__ int setBitsTo1(int k);

#endif