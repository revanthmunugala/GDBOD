#ifndef HYSORTOD_LIB_H
#define HYSORTOD_LIB_H

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>

using namespace std;

#define NONE -1
#define MY_DATATYPE uint64_t

const string approach[] = {"Naive", "Tree"};
const string treeSelect[] = {"Simple", "Locality optimized",
                             "Locality and traversal optimized"};

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

#include "hySortOD_lib.h"

// Import dataset
__host__ int importDataset(char *fname, int N, double *dataset, int DIM);

// Normalize dataset
__host__ void normalizeDataset(double *dataset, int N, int DIM);

// Encode hypercube coordinates
__device__ void encodeHypercube(MY_DATATYPE *curHypercube, int *hypercube,
                                int DIM, int index, int encodeBlockSize, int k);

// Map points to hypercube and encode
__global__ void createHypercube(MY_DATATYPE *hypercube, double *dataset, int N,
                                int DIM, int BIN, int encodeBlockSize, int k);

// Return number with k bits set to 1
__host__ __device__

int setBitsTo1(int k);

// Decode encoded hypercube into hypercube array
__device__ void decodeHypercube(MY_DATATYPE *hypercubeEncodedArray,
                                int *hypercube, int DIM, int hypercubeCount,
                                int threadId, int encodeBlockSize, int k);

// Decode encoded hypercube and build hypercube array
__global__ void buildHypercubeArray(MY_DATATYPE *hypercube, int *hypercubeArray,
                                    int hypercubeCount, int DIM,
                                    int encodeBlockSize, int k);

// Supporting function to add node to a tree
__host__ void appendNode(treeNode **rootNode, int startIndex, int endIndex,
                         int coordinate, int parentIndex, int *curCount);

// Build simple tree
__host__ void buildLinearTree(int *hypercube, treeNode **linearTree,
                              int *childCount, int *dimStart, int *curCount,
                              int curDim, int N, int MINSPLIT, int DIM);

// Supporting function to check if the difference between current coordinates is
// less than 1
__device__ int checkNeighbor(int index, int *hypercube, treeNode *linearTree,
                             int curIndex);

// Supporting function to check if current hypercube is an immediate neighbor
__device__ int checkImmediateNeighbor(int *hypercubeA, int *hypercubeB,
                                      int hypercubeCount, int DIM, int curDim=0);

// Neighborhood density of subtree - locality optimixzed
__device__ int neighborDensitySubTree(int *hypercube, treeNode *linearTree,
                                      int hypercubeIndex, int *childCount,
                                      int *instancesCount, int parentIndex,
                                      int curDim, int N, int DIM);

// Calculate neighborhood density using locality optimized tree
__global__ void neighborhoodDensity(int *density, int *instancesCount,
                                    treeNode *linearTree, int *hypercubeArray,
                                    int *childCount, int DIM,
                                    int hypercubeCount, int *dimNodes);

// Calculate neighborhood density using simple tree
__global__ void simpleNeighborhoodDensity(int *density, int *instancesCount,
                                          treeNode *linearTree,
                                          int *hypercubeArray, int *childCount,
                                          int DIM, int hypercubeCount);

// Supporting function to copy node contents
__host__ void copyContents(treeNode *linearCurNode, treeNode *optimCurNode,
                           int parentIndex, int siblingIndex, int childIndex);

// build locality optimized tree
void buildOptimizedLinearTree(treeNode *linearTree,
                              treeNode *optimizedLinearTree, int *dimNodes);

// Supporting function
__device__ int optimCheckNeighbor(int index, int *hypercube,
                                  optimTreeNode *linearTree, int curIndex);

// Calculate neighborhood density of sub tree
__device__ int optimNeighborDensitySubTree(int *hypercube,
                                           optimTreeNode *linearTree,
                                           int hypercubeIndex,
                                           int *instancesCount, int parentIndex,
                                           int N, int DIM);

// Calculate neighborhood density using naive apprach
__global__ void naiveNeighborhoodDensity(int *density, int *instancesCount,
                                         int DIM, int hypercubeCount,
                                         int *hypercube, int SPLIT);

// Calculate neighborhood density using fast tree
__global__ void optimNeighborhoodDensity(int *density, int *instancesCount,
                                         optimTreeNode *linearTree,
                                         int *hypercubeArray, int *childCount,
                                         int DIM, int hypercubeCount,
                                         int *dimNodes);

// Supporting function to copy node data from locality optim tree to fast tree
__host__ void copyNode(optimTreeNode *root, treeNode *linearCurNode,
                       optimTreeNode *optimCurNode, int curDim);

// Build tree with optimized locality and traversal
__host__ void buildSuperOptimTree(treeNode *linearTree,
                                  optimTreeNode *superOptimTree);

// Function to calculate outlier score
__host__ void calculateOutlierScore(float *outlierScore, int *neighborhoodDensity,
                                    map <vector<MY_DATATYPE>, vector<int>> hypercubeMap,
                                    int N, int maxNeighborhoodDensity);

__host__ void calculateOutlierScore(float *outlierScore, int *neighborhoodDensity,
                                    map <vector<int>, vector<int>> hypercubeMap,
                                    int N, int maxNeighborhoodDensity);

// Function to find min bits required to store hypercube dim
__host__ int findK(int BIN);

// Naive approach
float naiveStrategy(int *d_hypercubeArray, int *h_neighborhoodDensity,
                    int *h_instancesCount, int distinctHypercubeCount, int BIN,
                    int DIM);

// Locality and traverasl optimized strategy
__host__ float
finalOptimTreeStrategy(int *h_hypercubeArray, int *d_hypercubeArray,
                       int *h_neighborhoodDensity, int *h_instancesCount,
                       int distinctHypercubeCount, int DIM, int MINSPLIT);

// Locality optimized tree traversal strategy
__host__ float
localityOptimTreeStrategy(int *h_hypercubeArray, int *d_hypercubeArray,
                          int *h_neighborhoodDensity, int *h_instancesCount,
                          int distinctHypercubeCount, int DIM, int MINSPLIT);

// Simple tree traversal strategy
__host__ float simpleTreeStrategy(int *h_hypercubeArray, int *d_hypercubeArray,
                                  int *h_neighborhoodDensity,
                                  int *h_instancesCount,
                                  int distinctHypercubeCount, int DIM,
                                  int MINSPLIT);

// Build hypercube array - Non encoding
__global__ void buildNonEncodedHypercubeArray(int *hypercube, double *dataset,
                                              int N, int BIN, int DIM);

#endif