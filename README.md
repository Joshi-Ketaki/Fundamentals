# Fundamentals
Note: This is a repository where I am using quick coding exercises as a playground. This is nowhere a production-level code. So please do not expect comments and documents yet. Following some initial playing around, I will be cleaning up and optimizing my code. It will also follow with some tutorials/documentation.

Revise and learn fundamentals with me. Specifically focussing on accelerator architectures (starting with GPUs).

Learning DL workloads from perspective of computer architecture. Step 1 is to brush up and create a mental model of the GPU architecture. Create the model from two perspectives: 1. Memory subsystem 2. Compute units. In Step 2, we then start by understanding the basic operations: GEMM and performance implications for this.

System details: Environment used: LeetGPU cli (https://leetgpu.com/). I am using cycle-accurate mode.

Note: LeetGPU is a simulator and not actual hardware. So hardware perf counters etc. is not supported. But I find this a good place to experiment as unlike collab and other platforms there is no limit to the usage hours. Downside is that we do not have access to nsight etc as this is a simulator. Also the support is a bit basic, for instance, command line does not support command line arguments.

GPU used: GV100

Module 1: Understanding the fundamental DL operations from pov of GPU architecture.

PART 1: Matmul operation

Naive matmul: The very basic GEMM operation. Non-optimized version. Tiled Matmul: Does matrix multiply using tiles. Naive and Tiled FP16: Use half precision in case of memory bound kernels. Tensor Cores: Use TC for optimal matmul (This is being debugged as of now.) PART 2: CNN operation

