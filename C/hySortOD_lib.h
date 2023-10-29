#ifndef HYSORTOD_LIB_H
#define HYSORTOD_LIB_H

#include <chrono>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <map>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <algorithm>

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
int importDataset(char *fname, int N, double *dataset, int DIM);

// Normalize dataset
void normalizeDataset(double *dataset, int N, int DIM);

// Encode current hypercube coordinates
void encodeHypercube(MY_DATATYPE *curHypercube, int *hypercube, int DIM,
                     int index, int encodeBlockSize, int k);

// Map points to hypercube and encode
void createHypercube(MY_DATATYPE *hypercube, double *dataset, int N, int DIM,
                     int BIN, int encodeBlockSize, int k);

// Return number with k bits set to 1
int setBitsTo1(int k);

// Decode encoded hypercube into hypercube array
void decodeHypercube(MY_DATATYPE *hypercubeEncodedArray, int *hypercube,
                     int DIM, int hypercubeCount, int threadId,
                     int encodeBlockSize, int k);

// Decode encoded hypercube and build hypercube array
void buildHypercubeArray(MY_DATATYPE *hypercube, int *hypercubeArray,
                         int hypercubeCount, int DIM, int encodeBlockSize,
                         int k);

// Supporting function to add node to a tree
void appendNode(treeNode **rootNode, int startIndex, int endIndex,
                int coordinate, int parentIndex, int *curCount);

// Build simple tree
void buildLinearTree(int *hypercube, treeNode **linearTree, int *childCount,
                     int *dimStart, int *curCount, int curDim, int N,
                     int MINSPLIT, int DIM);

// Supporting function to check if the difference between current coordinates is
// less than 1
int checkNeighbor(int index, int *hypercube, treeNode *linearTree,
                  int curIndex, int curDim);

// Supporting function to check if current hypercube is an immediate neighbor
int checkImmediateNeighbor(int *hypercubeA, int *hypercubeB, int hypercubeCount,
                           int DIM);

// Neighborhood density of subtree - locality optimized
int neighborDensitySubTree(int *hypercube, treeNode *linearTree,
                           int hypercubeIndex, int *instancesCount,
                           int parentIndex, int curDim, int N, int DIM);

// Calculate neighborhood density using locality optimized tree
void neighborhoodDensity(int *density, int *instancesCount,
                         treeNode *linearTree, int *hypercubeArray, int DIM,
                         int hypercubeCount);

// Calculate neighborhood density using simple tree
void simpleNeighborhoodDensity(int *density, int *instancesCount,
                               treeNode *linearTree, int *hypercubeArray,
                               int DIM, int hypercubeCount);

// Supporting function to copy node contents
void copyContents(treeNode *linearCurNode, treeNode *optimCurNode,
                  int parentIndex, int siblingIndex, int childIndex);

// build locality optimized tree
void buildOptimizedLinearTree(treeNode *linearTree,
                              treeNode *optimizedLinearTree, int *dimNodes);

// Supporting function
int optimCheckNeighbor(int index, int *hypercube, optimTreeNode *linearTree,
                       int curIndex, int curDim);

// Calculate neighborhood density of subtree
int optimNeighborDensitySubTree(int *hypercube, optimTreeNode *linearTree,
                                int hypercubeIndex, int *instancesCount,
                                int parentIndex, int N, int DIM);

// Calculate neighborhood density using naive apprach
void naiveNeighborhoodDensity(int *density, int *instancesCount, int DIM,
                              int hypercubeCount, int *hypercube);

// Calculate neighborhood density using fast tree
void optimNeighborhoodDensity(int *density, int *instancesCount,
                              optimTreeNode *linearTree, int *hypercubeArray,
                              int DIM, int hypercubeCount);

// Supporting function to copy node data from locality optim tree to fast tree
void copyNode(optimTreeNode *root, treeNode *linearCurNode,
              optimTreeNode *optimCurNode, int curDim);

// Build tree with optimized locality and traversal
void buildSuperOptimTree(treeNode *linearTree, optimTreeNode *superOptimTree);

// Function to calculate outlier score
void calculateOutlierScore(float *outlierScore, int *neighborhoodDensity,
                           map <vector<MY_DATATYPE>, vector<int>> hypercubeMap,
                           int N, int maxNeighborhoodDensity);

void calculateOutlierScore(float *outlierScore, int *neighborhoodDensity,
                           map <vector<int>, vector<int>> hypercubeMap, int N,
                           int maxNeighborhoodDensity);

// Function to find min bits required to store hypercube dim
int findK(int BIN);

// Naive approach
float naiveStrategy(int *h_hypercubeArray, int *h_neighborhoodDensity,
                    int *h_instancesCount, int distinctHypercubeCount, int BIN,
                    int DIM);

// Locality and traversal optimized tree traversal strategy
float finalOptimTreeStrategy(int *h_hypercubeArray, int *h_neighborhoodDensity,
                             int *h_instancesCount, int distinctHypercubeCount,
                             int DIM, int MINSPLIT);

// Locality optimized tree traversal strategy
float localityOptimTreeStrategy(int *h_hypercubeArray,
                                int *h_neighborhoodDensity,
                                int *h_instancesCount,
                                int distinctHypercubeCount, int DIM,
                                int MINSPLIT);

// Simple tree traversal strategy
float simpleTreeStrategy(int *h_hypercubeArray, int *h_neighborhoodDensity,
                         int *h_instancesCount, int distinctHypercubeCount,
                         int DIM, int MINSPLIT);

// Build hypercube array - Non encoding
void buildNonEncodedHypercubeArray(int *hypercube, double *dataset, int N,
                                   int BIN, int DIM);

// Reorder dimensions based on variance
double*reorderByDimensions(double*dataset, int N, int DIM);

#endif