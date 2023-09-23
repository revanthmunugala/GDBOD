#include "hySortOD_lib.h"

using namespace std;

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
