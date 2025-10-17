# Fundamentals
Revise and learn fundamentals with me. Specifically focussing on accelerator architectures (starting with GPUs).

Learning DL workloads from perspective of computer architecture.
Step 1 is to brush up and create a mental model of the GPU architecture. Create the model from two perspectives: 1. Memory subsystem 2. Compute units.
We then start by understanding the basic operations: GEMM and performance implications for this.

System details:
Environment used: Google collab
GPU used: Tesla T4
Available GPU RAM for the slice provided by collab: 15 GB vidmem
Available CPU RAM for the slice provided by collab: 12.7 GB
Disk: 112.67 GB

Module 1:
1. Naive matmul: The very basic GEMM operation. Non-optimized version.
