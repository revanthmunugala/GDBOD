# GDBOD: Density-Based Outlier Detection Optimizing Locality and Fine-Grained Parallelism for Efficient Tree Traversals on the GPU

**Authors**: Revanth Munugala and Michael Gowanlock  
**Acknowledgements**: This material is based upon work supported by the National Science Foundation under Grant No. 2042155.

For more information, see the following paper:

> Munugala, Revanth Reddy, and Michael Gowanlock (2024). GDBOD: Density-Based Outlier Detection Exploiting Efficient Tree Traversals on the GPU. In *IEEE 31st International Conference on High Performance Computing, Data, and Analytics (HiPC)* (pp. 111–121).

---

## Overview

Outlier detection algorithms are employed across numerous application domains. In contrast to distance-based outlier detection algorithms that compute distances between points, hypercube-based algorithms reduce computational costs by evaluating the density of a point based on its enclosing hypercube.

A major limitation of state-of-the-art hypercube-based algorithms is that they do not scale to large datasets. This paper proposes **GPU Density-Based Outlier Detection (GDBOD)**, which is supported by efficient tree-based hypercube search methods.

We propose two GPU-friendly n-ary tree data structures for efficient hypercube searches that are optimized to obtain good locality and exploit the fine-grained parallelism afforded by the GPU.

We also propose:
- A data encoding method that compresses data to reduce the number of comparisons during distinct hypercube array construction.
- Coordinate reordering to enhance neighborhood search performance.
- Sequential and multi-core CPU algorithms for systems not equipped with GPUs.

### Performance

- Sequential CPU algorithm achieves a **mean speedup of 18.35×** over the state-of-the-art.
- Parallel GPU algorithm achieves a **mean speedup of 3.29×** over the multi-core CPU algorithm across 6 real-world datasets.
- With all optimizations, on A100, for SuSy dataset, we achieve:
  - **Peak compute throughput**: 89.52%
  - **L1 cache hits**: 91.90%
  - **L2 cache hits**: 99.78%

---

## Implementation Strategies

- `CUDA/` – GPU-accelerated implementation.
- `OpenMP/` – Multi-core CPU implementation.
- `C/` – Sequential CPU implementation.

---

## Parameters

Default parameters are specified in `hySortOD_encoded.cpp` (GPU version only).  
See the paper for CPU-specific default parameters.

| Parameter     | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `N`           | Number of points in the dataset                                             |
| `DIM`         | Dimensionality of the dataset                                               |
| `BIN`         | Number of bins (default = 7)                                                |
| `MINSPLIT`    | Threshold to limit the number of mapped hypercubes (default = 20)          |
| `NORMALIZE`   | Flag to normalize the dataset                                               |
| `APPROACH`    | Search strategy (default = tree strategy; other = naive strategy)          |
| `TREE_SELECT` | (default = 3) <br> 1: simple tree <br> 2: optimized locality <br> 3: optimized locality + traversal <br> 0: naive strategy |

---

## Compiling and Running the Code

The code is written in CUDA and C/C++.
Tested with CUDA v12.2 on NVIDIA A100 and Quadro RTX 5000 GPUs.

To **compile**, run:

```bash
make
```

To **run**, use:

```bash
./main <N> <DIM> <BIN> <MINSPLIT> <NORMALIZE> <DATASET_FILE> <APPROACH> <TREE_SELECT>
```

### Example:

```bash
./main 11620300 57 7 0 1 dataset/bigcross.txt 1 3
```

To **disable the encoding optimization**, switch from `hySortOD_encoded.cu` to `hySortOD_non_encoded.cu` in the Makefile.

---
