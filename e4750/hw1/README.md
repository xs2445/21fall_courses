# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2021)

## Assignment-1: Introduction to Memory Access in PyCUDA & PyOpenCL

Due date: See in the courseworks.

Total points: 100

### Primer

The goal of the assignment is to discover the most efficient method(s) of host-to-device memory allocation. The assignment is divided into a programming section, and a theory section. The programming section contains two tasks - one for CUDA, the other for OpenCL. 

### Relevant Documentation

For PyOpenCL:
1. [OpenCL Runtime: Platforms, Devices & Contexts](https://documen.tician.de/pyopencl/runtime_platform.html)
2. [pyopencl.array](https://documen.tician.de/pyopencl/array.html#the-array-class)
3. [pyopencl.Buffer](https://documen.tician.de/pyopencl/runtime_memory.html#buffer)

For PyCUDA:
1. [Documentation Root](https://documen.tician.de/pycuda/index.html)
2. [Memory tools](https://documen.tician.de/pycuda/util.html#memory-pools)
3. [gpuarrays](https://documen.tician.de/pycuda/array.html)


## Programming Problem (80 points)

### Problem set up
Consider two 1D vectors *A* and *B* of length **N**. The task is to write code for PyOpenCL and for PyCUDA which performs vector addition on those two vectors in multiple ways, which are differentiated in how they interact with device memory. The programming problem is divided into two tasks, one each for OpenCL and CUDA. Make sure you complete both by following the instructions exactly. 

#### Task-1: PyOpenCL (30 points)

For PyOpenCL, you have been provided with the kernel code along with the assignment template. Your task is to build this kernel and use it as the basis for two methods of vector addition. The methods involve different ways of interacting with device and host memory. Read the instructions below and follow them exactly to complete task-1 of this assignment:

1. The kernel code for OpenCL is provided in the template at the end of this README file. 

2. *(10 points)* Write a function to perform vector addition on *A* and *B* by using `pyopencl.array` to load your vectors to device memory. Compare your results with the serial implementation on CPU. Record the time for the execution of the operation, including the memory transfer steps. 

3. *(10 points)* Write a function to perform vector addition on *A* and *B* by using `pyopencl.Buffer` to load your vectors to device memory. Compare your results with the serial implementation on CPU. Record the time for the execution of the operation, including the memory transfer steps. 

5. *(5 points)* Record your observations for 8 different array sizes(*L*) starting from array size as 10 and increasing the size after each observation by a factor of 10: [10, 100, 1000, ... ,100000000]. Call the functions which you wrote one-by-one to perform vector addition on the two arrays and record the **average** running time for each function call. Additionaly a function is provided (**numpyAdd**) to add two vectors on CPU host using standard numpy syntax, record the **average** running time for this third function for each of the array sizes.

6. *(5 points)* Plot the **average** execution times against the increasing array size (in orders of **L**)


#### Task-2: PyCUDA (50 points)

For PyCUDA, the coding problem will involve your first practical encounter with kernel codes, host-to-device memory transfers (and vice-versa), and certain key classes that PyCUDA provides for them. Read the instructions below carefully and complete your assignment as outlined:

1. *(10 points)* Write the kernel code for vector addition.

2. *(10 points)* Write a function (**explicitAdd**) to perform vector addition on *A* and *B* taking advantage of explicit memory allocation using `pycuda.driver.mem_alloc()`. Do not forget to retrieve the result from device memory using the appropriate PyCUDA function. Use `SourceModule` to compile the kernel which you defined earlier. Compare your results with the serial implementation on CPU. Record the following:
    1. Time taken to execute the operation including memory transfer.
    2. Time taken to execute the operation excluding memory transfer.

3. *(10 points)* Write a function (**implicitAdd**)  to perform vector addition on *A* and *B* **without** explicit memory allocation. Use `SourceModule` to compile the kernel which you defined earlier. Compare your results with the serial implementation on CPU. Time the execution. Record the time taken to complete the operation including memory transfer. 

4. *(5 points)* Write a function (**gpuarrayAdd_np**) to perform vector addition on *A* and *B* using the `gpuarray` class instead of allocating with `mem_alloc`. For this problem use numpy like syntax without actually calling the kernel for vector addition. Compare your results with the serial implementation on CPU.Use standard algebraic syntax to perform addition. Record the following:
    1. Time taken to execute the operation including memory transfer.
    2. Time taken to execute the operation excluding memory transfer.

5. *(5 points)* Write a function (**gpuarrayAdd**) to perform vector addition on *A* and *B* using the `gpuarray` class instead of allocating with `mem_alloc`. For this part of the problem call the kernel for vector addition. Compare your results with the serial implementation on CPU. Record the following:
    1. Time taken to execute the operation including memory transfer.
    2. Time taken to execute the operation excluding memory transfer.

6. Record your observations for 8 different array sizes(*L*) starting from array size as 10 and increasing the size after each observation by a factor of 10: [10, 100, 1000, ... ,100000000]. 

7. Call the functions which you wrote one-by-one to perform vector addition on the two arrays and record the **average** running time including memory transfer for each function call. 

8. Call the functions (**explicitAdd, gpuarrayAdd_np, gpuarrayAdd**) which you wrote one-by-one to perform vector addition on the two arrays and record the **average** running time excluding memory transfer for each function call. 

9. Additionaly a function is provided (**numpyAdd**) to add two vectors on CPU host using standard numpy syntax, record the **average** running time for this third function for each of the array sizes.

10. *(5 points)* Plot the **average** execution times (including memory transfer for GPU operations) against the increasing array size (in orders of **L**)

11. *(5 points)* Plot the **average** execution times (excluding memory transfer for GPU operations) against the increasing array size (in orders of **L**)

## Theory Problems (20 points)

1. *(4 points)* What is the difference between a thread, a task and a process? 

2. *(4 points)* Are all algorithms potentially scalable? (Expand question scope to cover Amdahl's Law?)

3. *(4 points)* Out of the two approaches explored in task-1 (PyOpenCL), which proved to be faster? Explore the PyOpenCL docs and source code to support your conclusions about the differences in execution time.

4. *(4 points)* Of the different approaches explored in task-2 (PyCUDA), which method(s) proved the fastest? Explore the PyCUDA docs and source code and explain how/why: (a) Normal python syntax can be used to perform operations on gpuarrays; (b) gpuarray execution (non-naive method) is comparable to using `mem_alloc`. 

5. *(4 points)* How does the parallel approaches in task-1(PyOpenCL) and task-2(PyCuda) compare to the serial execution on CPU? Which is faster and why? [Consider both cases, including memory transfer and excluding memory transfer]

## Code Templates

You **must** adhere to the template given in the starter code below - this is essential for all assignments to be graded fairly and equally. 

#### PyOpenCL Starter Code *(with kernel code)*

**Note:** The kernel code is only provided for OpenCL, and for the sole reason that this is the first assignment of the course.  

```python
"""
The code in this file is part of the instructor-provided template for Assignment-1, task-1, Fall 2021. 
"""


import relevant.libraries

class clModule:
    def __init__(self):
        """
        **Do not modify this code**
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

        # kernel - will not be provided for future assignments!
        kernel_code = """
            __kernel void sum(__global float* c, __global float* a, __global float* b, const unsigned int n)
            {
                unsigned int i = get_global_id(0);
                if (i < n) {
                    c[i] = a[i] + b[i];
                }
            }
        """ 

        # Build kernel code
        self.prg = cl.Program(self.ctx, kernel_code).build()

    def deviceAdd(self, a, b, length):
        """
        Function to perform vector addition using the cl.array class
        Arguments:
            a       :   1st Vector
            b       :   2nd Vector
            length  :   length of vectors.
        Returns:
            c       :   vector sum of arguments a and b
            time_   :   execution time for pocl function 
        """
        # device memory allocation

        # execute operation.

        # wait for execution to complete.

        # Record execution time 

        # Copy output from GPU to CPU [Use .get() method]

        # return a tuple of output of addition and time taken to execute the operation.

        pass

    def bufferAdd(self, a, b, length):
        """
        Function to perform vector addition using the cl.Buffer class
        Returns:
            c               :    vector sum of arguments a and b
            end - start     :    execution time for pocl function 
        """
        # Create three buffers (plans for areas of memory on the device)

        # execute operation with numpy syntax
        
        # Wait for execution to complete.

        # Record execution time 
        
        # Copy output from GPU to CPU [Use enqueue_copy]

        # return a tuple of output of addition and time taken to execute the operation.

        pass

    def numpyAdd(self, a, b, length):
        """
        Function to perform vector addition on host(CPU).
        Arguments:
            a       :   1st Vector
            b       :   2nd Vector
            length  :   length of vector a or b[since they are of same length] 
        """
        a = np.array(a)
        b = np.array(b)

        start = time.time()
        c = a + b
        end = time.time()

        return c, end - start


if __name__ == "__main__":
    # Define the number of iterations and starting lengths of vectors
    
    # Create an instance of the clModule class

    # Perform addition tests for increasing lengths of vectors
    # L = 10, 100, 1000 ..., (You can use np.random.randn to generate two vectors)

    # Compare outputs.

    # Plot the compute times
```

#### PyCUDA Starter Code

```python
"""
The code in this file is part of the instructor-provided template for Assignment-1, task-2, Fall 2021. 
"""

import relevant.libraries

class deviceAdd:
    def __init__(self):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """

        # Compile the kernel code when an instance
        # of this class is made. This way it only
        # needs to be done once for the 3 functions
        # you will call from this class.
        self.mod = self.getSourceModule()

    def getSourceModule(self):
        """
        Compiles Kernel in Source Module to be used by functions across the class.
        """
        # define your kernel below.
        kernelwrapper = """"""
        return SourceModule(kernelwrapper)

    
    def explicitAdd(self, a, b, length):
        """
        Function to perform on-device parallel vector addition
        by explicitly allocating device memory for host variables.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """

        # Event objects to mark the start and end points

        # Device memory allocation for input and output arrays

        # Copy data from host to device

        # Call the kernel function from the compiled module

        # Get grid and block dim
        
        # Record execution time and call the kernel loaded to the device

        # Wait for the event to complete

        # Copy result from device to the host

        # return a tuple of output of addition and time taken to execute the operation.
        pass

    
    def implicitAdd(self, a, b, length):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # Event objects to mark the start and end points

        # Get grid and block dim

        # Call the kernel function from the compiled module
        
        # Record execution time and call the kernel loaded to the device

        # Wait for the event to complete
        
        # return a tuple of output of addition and time taken to execute the operation.
        pass


    def gpuarrayAdd_np(self, a, b, length):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables and WITHOUT calling the kernel. The operation
        is defined using numpy-like syntax. 
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # Event objects to mark start and end points

        # Allocate device memory using gpuarray class        
        
        # Record execution time and execute operation with numpy syntax

        # Wait for the event to complete

        # Fetch result from device to host
        
        # return a tuple of output of addition and time taken to execute the operation.
        pass
        
    def gpuarrayAdd(self, a, b, length):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables and WITHOUT calling the kernel. The operation
        is defined using numpy-like syntax. 
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """

        # Create cuda events to mark the start and end of array.

        # Get function defined in class defination

        # Allocate device memory for a, b, output of addition using gpuarray class        
        
        # Get grid and block dim

        # Record execution time and execute operation with numpy syntax (for recording execution time use cuda event start)

        # Wait for the event to complete

        # Fetch result from device to host
        
        # return a tuple of output of addition and time taken to execute the operation.
        pass

    def numpyAdd(self, a, b):
        """
        Function to perform on-host vector addition. The operation
        is defined using numpy-like syntax. 
        Returns (addition result, execution time)
        """
        # Initialize empty array on host to store result
        start = time.time()
        c = np.add(a, b)
        end = time.time()
        
        return c, end - start

if __name__ == "__main__":

    # Define the number of iterations and starting lengths of vectors
    
    # Create an instance of the deviceAdd class

    # Perform addition tests for increasing lengths of vectors
    # L = 10, 100, 1000 ..., (You can use np.random.randn to generate two vectors)

    # Compare outputs.

    # Plot the compute times
```
