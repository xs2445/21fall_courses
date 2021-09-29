"""
The code in this file is part of the instructor-provided template for Assignment-1, task-2, Fall 2021. 
"""

import pycuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.driver as cuda
import math
import time


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
        kernelwrapper = """
        __global__
        void AddVector(float *a, float *b, float *c, length){
            int i = blockIdx.x*blockDim.x + threadIdx.x;
            if(i<length) c[i] = a[i]+b[i]
        }

        """
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
        start = cuda.Event()
        end = cuda.Event()

        # Device memory allocation for input and output arrays
            # a_d, b_d, c_d are arrays in the device
            # a, b, c are the arrays in the host
        a_d= cuda.mem_alloc(a.size*a.dtype.itemsize)
        b_d = cuda.mem_alloc(b.size*b.dtype.itemsize)

        # Copy data from host to device
        cuda.memcpy_htod(a_d, a)
        cuda.memcpy_htod(b_d,b)

        # Call the kernel function from the compiled module
        func = self.mod.get_function("AddVector")   # get function from Cuda C code


        # Get grid and block dim

        
        # Record execution time and call the kernel loaded to the device
        start.record()  # recording the time consumed
        c_d = func(a_d, b_d, length, block=(math.ceil(length/256)))
        end.record()
        time = start.time_till(end)


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
