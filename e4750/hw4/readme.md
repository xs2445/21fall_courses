# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2021)

## Assignment-4: Scan

Total points: 100

### References to help you with assignment
* [PyCuda Examples](https://github.com/inducer/pycuda/tree/main/examples)
* [PyOpenCL Examples](https://github.com/inducer/pyopencl/tree/main/examples)
* [NVidia Blog](https://developer.nvidia.com/blog/tag/cuda/)

### Primer
For this assignment you will be working on inclusive parallel scan on a 1D list (That is to implement parallel algorithm for all prefix sum). The scan operator will be the addition (plus) operator. There are many uses for all-prefix-sums, including, but not limited to sorting, lexical analysis, string comparison, polynomial evaluation, stream compaction, and building histograms and data structures (graphs, trees, etc.) in parallel. Your kernel should be able to handle input lists of arbitrary length. To simplify the lab, the input list will be at most of length 2048 × 65, 535 elements. This means that the computation can be performed using only one kernel launch. The boundary condition can be handled by filling “identity value (0 for sum)” into the shared memory of the last block when the length is not a multiple of the thread block size. Example: All-prefix-sums operation on the array [3 1 7 0 4 1 6 3], would return [3 4 11 11 14 16 22 25].

### Programming Part (70 points)

### Programming Task 1. 1-D Scan - Naive Python algorithm
(5 points) 1. Implement scan on a 1D list (prefix sum) in Python.   
(5 points) 2. Write test cases to confirm that your code is working correctly. Length of the input in the test case should be at max 5 elements.

### Programming Task 2. 1-D Scan - Programing in PyCuda and PyOpenCL 
(10 points) 1.  Implement a work inefficient scan algorithm using both PyOpenCL and PyCUDA. The input and output are the same as those of the serial one. ** Analyze the time complexity.** Hint: Check the course materials for naive scan algorithm.  
(20 points) 2. Implement a work efficient parallel scan algorithm using both PyOpenCL and PyCUDA. The input and output remain the same. Analyze the time complexity. Hint: Check the course materials for 'work efficient'.  
(10 points) 4. Write test cases to verify your output with naive python algorithm. Input cases of lengths 128, 2048, 128 X 2048, 2048 X 2048, 65535 X 2048.  
(20 points) 5. For the input cases of length 128, 2048, 128 X 2048, 2048 X 2048, 65535 X 2048 record the time taken(including memory transfer) for the three functions (naive python, work inefficient parallel, and work efficient parallel). Provide a graph of your time observations in the report and compare the performance of the algorithms, compare both space and time complexity.  

### Conceptual Problem - Stream Compaction (10 points) 
Consider the application of scan called Stream Compaction.
Sketch and describe the flow of stream compaction.
Write a pseudo code for an OpenCL kernel which implements stream compaction.

### Theory Problems (20 points) 
(5 points) 1. Consider that NVIDIA GPUs execute warps of 32 parallel threads using SIMT. What's the difference between SIMD and SIMT? What is the worst choice as the number of threads per block to chose in this case among the following and why?   
(A) 1  
(B) 16  
(C) 32  
(D) 64  

(5 points) 2. What is a bank conflict? Give an example for bank conflict.  
(5 points) 3. Conisder the following code for finding the sum of all elements in a vector. The following code doesn't always work correctly explain why? Also suggest how to fix this? (Hint: use atomics.)  
```
__global__ void vectSum(int* d_vect,size_t size, int* result){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < size){
        *result+=d_vect[tid];
        tid+=blockDim.x * gridDim.x;
    }
}
``` 
(5 points) 4. Is there a way to dynamically allocate shared memory? Explain how?  

### Templates for PyCuda and PyOpenCL
A common template for PyCuda and PyOpenCL is shown below.

```python
import relevant.libraries

class PrefixSum:
  def __init__(self):
	pass

  def prefix_sum_python(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def prefix_sum_gpu_naive(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def prefix_sum_gpu_work_efficient(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def test_prefix_sum_python(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def test_prefix_sum_gpu_naive(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def test_prefix_sum_gpu_work_efficient(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

```
Submit the assignment on courseworks. The submission should have with 3 files,     

Report: homwork_4_report_{uni}.pdf  
PyCuda code: homwork_4_pycuda_{uni}.py  
PyopenCL code: Homwork_4_pyopencl_{uni}.py
