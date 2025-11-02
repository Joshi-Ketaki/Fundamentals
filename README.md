# Fundamentals
Revise and learn fundamentals with me. Specifically focussing on accelerator architectures (starting with GPUs).

Learning DL workloads from perspective of computer architecture.
Step 1 is to brush up and create a mental model of the GPU architecture. Create the model from two perspectives: 1. Memory subsystem 2. Compute units.
We then start by understanding the basic operations: GEMM and performance implications for this.

System details:
Environment used: LeetGPU cli (https://leetgpu.com/). I am using cycle-accurate mode.

Note: LeetGPU is a simulator and not actual hardware. So hardware perf counters etc. is not supported.
But I find this a good place to experiment as unlike collab and other platforms there is no limit to the usage hours.
Downside is that we do not have access to nsight etc as this is a simulator. Also the support is a bit basic, for instance, command line does not support command line arguments.

GPU used: RTX 3070 (Ampere)

Module 1: Understanding the fundamental DL operations from pov of GPU architecture.

1. Naive matmul: The very basic GEMM operation. Non-optimized version.
2. Tiled Matmul: Does matrix multiply using tiles.

Results: Note the execution times. This is the only measurable metric in leetGPUs as of now.

