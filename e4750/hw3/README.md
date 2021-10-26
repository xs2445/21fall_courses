# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2021)

## Assignment-3: 2D kernels, Matrices, Shared Memory, Constant Memory 

Total points: 100

### References to help you with assignment
* [PyCuda Examples](https://github.com/inducer/pycuda/tree/main/examples)
* [PyOpenCL Examples](https://github.com/inducer/pyopencl/tree/main/examples)
* [NVidia Blog](https://developer.nvidia.com/blog/tag/cuda/)

### Primer
For this assignment you will be working on 2D convolution. First you will pursue a simple(r) approach which doesn't use shared memory or constant memory. Later you will  incorporate the shared memory and/or constant memory, and compare the time taken for the operations.

Matrix convolution is primarily used in image processing for tasks such as image enhancing, encoding etc. A standard image convolution formula for a 5x5 convolution kernel A with matrix B is
```
C(i,j) = sum (m = 0 to 4) {
	 sum(n = 0 to 4) { 
	 	A[m][n] * B[i+m-2][j+n-2] 
	 } 
}
```
 where 0 <= i < B.height and 0 <= j < B.width. For this assignment you can assume that the elements that are "outside" the matrix B, are treated as if they had value zero. You can assume the kernel size is either 3 x 3 or 5 x 5 for this assignment but write your code to work for general odd dimnesion kernel sizes.

### Programming Part 

### Task 1 PyCuda(40 points)  
(10 points) 1. Write kernel function to perform the above convolution operation without using shared memory or constant memory. Name this kernel function conv_gpu_naive.   
(10 points) 2. Optimization one: Rewrite the kernel function using shared memory optimization. Name this kernel function conv_gpu_shared_mem.  
(5 points) 3. Optimization two: Rewrite the kernel function with matrix B initialized in constant memory. Name this kernel function conv_gpu_shared_and_constant_mem.  
(5 points) 4. Write test cases to verify your output with the scipy.signal.convolve2d function from scipy module in python. Name this test function test_conv_pycuda. Write at lease one working test case for each function.  
(10 points) 5. Record the time taken to execute convolution, including memory transfer operations for the following matrix A dimensions: 16 x 16, 64 x 64, 256 x 256, 1024 x 1024, 4096 x 4096. Run each case multiple times and record the average of the time.  
  
  
### Task 2 PyOpenCL(40 points) 
(10 points) 1. Write kernel function to perform the above convolution operation without shared memory or constant memory. Name this funcition conv_gpu_naive_openCL.
(10 points) 2. Optimization one: Rewrite the kernel function using shared memory optimization. Name this kernel function conv_gpu_shared_mem_openCL.  
(5 points) 3. Optimization two: Rewrite the kernel function with matrix B initialized in constant memory. Name this kernel function conv_gpu_shared_and_constant_mem_openCL.  
(5 points) 4. Write test cases to verify your output with the scipy.signal.convolve2d function from scipy module in python. Name this test function test_conv_pyopencl. Atleast write one working test case for each function.  
(10 points) 5. Record the time taken to execute convolution, including memory transfer operations for the following matrix A dimensions: 16 x 16, 64 x 64, 256 x 256, 1024 x 1024, 4096 x 4096. Run each case multiple times and record the average of the time.  

### Theory Problems(20 points) 
(5 points) 1. Compare the recorded times against the serial implementation for all the above cases. Which approach is faster in PyCuda? Which approach is faster in PyOpenCL? Why is that particular method better than the other.  
(5 points) 2. Can this approach be scaled for very big kernel functions? In both cases explain why?  
(5 points) 3. Explain what is row-major order and column-major order for storing multidimensional arrays? Does it matter how data is stored?  
(5 points) 4. Consider the following kernel for finding the sum of two matrics:  

```
__global__
void matrix_sum(float *A, float *B, float *C,
  int m, int n) {
  // assume 2D grid and block dimensions
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x < m && y < n) {
    int ij = x + y*m; // column-major order
    C[ij] = A[ij] + B[ij];
  } 
}

```

Now consider the call to the kernel:  
```
// optimization: copy data outside of the loop
cudaMemcpy(dA,...);
cudaMemcpy(dB,...);
for (int i=0; i<n; ++i)
  for (int j=0; j<n; ++j) {
    int ij = i + j*n; // column-major order
    matrix_sum<<<1,1>>>(dA+ij, dB+ij, dC+ij, 1,1);
  }
cudaMemcpy(hC,dC,...);

```  

Will this code work? If yes - is this efficient, why or why not. If not, what can you do to improve the efficiency? If not, what is the issue with this code and how would you fix the issue?  

### Templates

#### PyCuda

```python
import relevant.libraries

class Convolution:
  def __init__(self):
	"""
	Attributes for instance of EncoderDecoder module
	"""
	self.mod = self.getSourceModule()
	pass

  def getSourceModule(self):
	# kernel code wrapper
	kernelwrapper = """""" # you can either use a string or save the kernel in kernel.cu file and reference it here.
	# Compile the kernel code when an instance
	# of this class is made. 
	return SourceModule(kernelwrapper)

  def conv_gpu_naive(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def conv_gpu_shared_mem(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def conv_gpu_shared_and_constant_mem(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def test_conv_pycuda(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass
```

#### PyOpenCL
```python
import relevant.libraries

class Convolution:
  def __init__(self):
	"""
	Attributes for instance of clModule
	Includes OpenCL context, command queue, kernel code.
	"""

	# Get platform and device property
	NAME = 'NVIDIA CUDA'
	platforms = cl.get_platforms()
	devs = None
	for platform in platforms:
	    if platform.name == NAME:
	        devs = platform.get_devices()       

	# Create Context:
	self.ctx = cl.Context(devs)

	# Setup Command Queue:
	self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

	# kernel - Write your kernel code here or in a .cu file and link it here.
	kernel_code = """""" 

	# Build kernel code
	self.prg = cl.Program(self.ctx, kernel_code).build()

  def conv_gpu_naive(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def conv_gpu_shared_mem(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def conv_gpu_shared_and_constant_mem(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def test_conv_pyopencl(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass
```

### How to submit the solution to the assignment
Submit the assignment on courseworks. The submission should have with 3 files,   

Report: homwork_3_report_{uni}.pdf  
PyCuda code: homwork_3_pycuda_{uni}.pdf  
PyopenCL code: Homwork_3_pyopencl_{uni}.pdf
