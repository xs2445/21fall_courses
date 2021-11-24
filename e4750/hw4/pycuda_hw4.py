import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
import time
import matplotlib.pyplot as plt
import math



class PrefixSum:
    def __init__(self):
        self.mod = None
        self.getSourceModule()


    def getSourceModule(self):
        kernelwrapper = """
        //------------------------------------------------------------------------------
        // predefined parameters
        #define SECTION_SIZE 1024

        //------------------------------------------------------------------------------
        // Kogge-Stone scan kernel for arbitrary input length, naive-gpu
        // phase1&2 for inefficient scan

        __global__
        void KoggeStone_end(double *X, double *Y, double *end_ary, const int InputSize){
            
            // copy the input array to shared memory
            __shared__ double XY[SECTION_SIZE];

            const int tx = threadIdx.x;
            const unsigned long int i = blockDim.x*blockIdx.x + tx;

            if(i<InputSize){
                XY[tx] = X[i];
            }

            // iterative scan on XY
            float temp = 0.0f;
            for(unsigned int stride = 1; stride < blockDim.x; stride*=2){
                temp = 0;
                __syncthreads();
                if(tx >= stride) temp = XY[tx-stride];
                __syncthreads();
                XY[tx] += temp;
                //if(tx >= stride) XY[tx] += XY[tx-stride];
            }
            __syncthreads();
        
            if(i<InputSize) Y[i] = XY[tx];
            //Y[i] = XY[tx];

            // copy the last element of the block
            if(tx == SECTION_SIZE-1){
                end_ary[blockIdx.x] = XY[tx];
            }
            else if(i == InputSize-1){
                end_ary[blockIdx.x] = XY[tx];
            }

        }


        //------------------------------------------------------------------------------
        // phase 3 of inefficient scan

        __global__
        void phase3_koggestone(double *Y, double *S, const int InputSize){

            // current position in Y
            const unsigned long int y = blockDim.x*blockIdx.x + threadIdx.x;

            if(blockIdx.x > 0 && y<InputSize) Y[y] += S[blockIdx.x-1];
        }


        //------------------------------------------------------------------------------

        #define SECTION_SIZE_2 SECTION_SIZE*2

        //------------------------------------------------------------------------------
        // Brent-Kung scan kernel phase1&2 (efficient scan)
        // output two array Y: pre-fix sum, end_ary: last elements of each blocok.

        __global__ 
        void BrentKung_end(double *X, double *Y,  double *end_ary, const int InputSize){

            __shared__ double XY[SECTION_SIZE_2];

            // current position in block
            const unsigned int tx = threadIdx.x;
            // current position in X
            const unsigned long int i = 2*blockDim.x*blockIdx.x + tx;

            // each thread copy 2 element to shared memory
            if(i<InputSize) XY[tx] = X[i];
            if(i+blockDim.x < InputSize) XY[tx+blockDim.x] = X[i+blockDim.x];

            // reduction tree
            for(unsigned int stride=1; stride <= blockDim.x; stride*=2){
                __syncthreads();
                int index = (tx+1)*2*stride-1;
                if(index<SECTION_SIZE_2) XY[index] += XY[index - stride];                 
            }

            // distribution tree 
            for(unsigned int stride=SECTION_SIZE_2/4; stride > 0; stride /=2){
                int index = (tx+1)*2*stride-1;
                __syncthreads();
                if(index+stride < SECTION_SIZE_2) XY[index+stride] += XY[index];
            }

            __syncthreads();

            // output the last element of each block
            if(((tx+1) == SECTION_SIZE) && (i+blockDim.x < InputSize)){
                end_ary[blockIdx.x] = XY[tx+blockDim.x];
            }
            else if((i+1) == InputSize){
                end_ary[blockIdx.x] = XY[tx];
            }
            else if((i+blockDim.x+1) == InputSize){
                end_ary[blockIdx.x] = XY[tx+blockDim.x];
            }
            
            // output each block
            if(i<InputSize) Y[i] = XY[tx];
            if(i+blockDim.x < InputSize) Y[i+blockDim.x] = XY[tx+blockDim.x];
            
        }

        //------------------------------------------------------------------------------
        // phase 3 of efficient scan

        __global__
        void phase3_brentkung(double *Y, double *S, const int InputSize){

            // current position in Y
            const unsigned long int y = 2*(blockDim.x*blockIdx.x) + threadIdx.x;

            if(2*blockIdx.x > 0 && y<InputSize) Y[y] += S[blockIdx.x-1];
            
            if(2*blockIdx.x > 0 && y+blockDim.x < InputSize) Y[y+blockDim.x] += S[blockIdx.x-1];

        }

        """
		
        mod = SourceModule(kernelwrapper)

        self.mod = mod
    
    @staticmethod
    def prefix_sum_python(N, length):
        """
		Naive prefix sum serial implementation
        yi = x0 + ... + xi

		params:
		- N: input array

		return:
        - Y: output array
		"""

        start = time.time()

        Y = np.zeros_like(N)

        for i in range(length):
            sum_temp = 0
            for j in range(i+1):
                sum_temp += N[j]
            Y[i] = sum_temp
        
        return Y, (time.time()-start)*1e3

    @staticmethod
    def prefix_sum_python2(N, length):
        Y = np.zeros_like(N)

        Y[0] = N[0]
        for i in range(length-1):
            Y[i+1] = Y[i] + N[i+1]

        return Y


    def prefix_sum_gpu_naive(self, N, length):
        """
        Prefix_sum using Kogge-Stone scan kernel

		params:
		- N: input array

		return:
		- Y: result
		- time
		"""
        start = cuda.Event()
        end = cuda.Event()
        start.record()

        # times for iteration
        if length<=1024:
            n = 1
        else:
            n = math.ceil(math.log(length, 1024))

        Y_d_list = []
        E_d_list = [gpuarray.to_gpu(N)]

        blocksize = 1024
        BlockDim = (blocksize, 1, 1)
        gridsize_list = [length]

        func_red = self.mod.get_function("KoggeStone_end")
        func_dis = self.mod.get_function("phase3_koggestone")

        # reduction
        for i in range(n):

            # memory allocation for Y
            Y_d_list.append(gpuarray.zeros(gridsize_list[-1], dtype=np.float64))
            # gridsize for this step
            gridsize_list.append((gridsize_list[i]-1)//blocksize+1)
            # list of last elements for this step
            E_d_list.append(gpuarray.zeros(gridsize_list[-1], dtype=np.float64))

            func_red(E_d_list[i], Y_d_list[i], E_d_list[-1], np.int32(gridsize_list[i]), block=BlockDim, grid=(gridsize_list[-1],1,1))
            cuda.Context.synchronize()

        # distribution
        for i in range(n-1)[::-1]:
            func_dis(Y_d_list[i], Y_d_list[i+1], np.int32(gridsize_list[i]), block=BlockDim, grid=(gridsize_list[i+1],1,1))
            cuda.Context.synchronize()

        Y = Y_d_list[0].get().copy()
        
        end.record()
        end.synchronize()

        return Y, start.time_till(end)


    def prefix_sum_gpu_work_efficient(self, N, length):
        """
        Prefix_sum using Brent-Kung scan kernel

		params:
		- N: input array

		return:
		- Y: result
		- time
		"""
        start = cuda.Event()
        end = cuda.Event()
        start.record()

        # times for iteration
        if length<=2048:
            n = 1
        else:
            n = math.ceil(math.log(length, 2048))

        Y_d_list = []
        E_d_list = [gpuarray.to_gpu(N)]

        blocksize = 1024
        BlockDim = (blocksize, 1, 1)
        gridsize_list = [length]

        # function of reduction kernel
        func_red = self.mod.get_function("BrentKung_end")
        # function of distribution kernel
        func_dis = self.mod.get_function("phase3_brentkung")

        # reduction
        for i in range(n):

            # memory allocation for Y
            Y_d_list.append(gpuarray.zeros(gridsize_list[-1], dtype=np.float64))
            # gridsize for this step
            gridsize_list.append((gridsize_list[i]-1)//(blocksize*2)+1)
            # list of last elements for this step
            E_d_list.append(gpuarray.zeros(gridsize_list[-1], dtype=np.float64))

            func_red(E_d_list[i], Y_d_list[i], E_d_list[-1], np.int32(gridsize_list[i]), block=BlockDim, grid=(gridsize_list[-1],1,1))
            cuda.Context.synchronize()

        # distribution
        for i in range(n-1)[::-1]:
            func_dis(Y_d_list[i], Y_d_list[i+1], np.int32(gridsize_list[i]), block=BlockDim, grid=(gridsize_list[i+1],1,1))
            cuda.Context.synchronize()

        Y = Y_d_list[0].get().copy()
        end.record()
        end.synchronize()

        return Y, start.time_till(end)


    @staticmethod
    def test_prefix_sum_python(printing=True):
        """
		Test function prefix_sum_python
        yi = x0 + ... + xi

		params:
		- printing(boolean): print debug information

		return:
        - (bool): correct or not
		"""

        N = np.array([3,1,7,0,4,1,6,3])
        check_ary = np.array([3,4,11,11,14,16,22,25])
        Y = PrefixSum.prefix_sum_python(N, N.shape[0])

        if np.allclose(Y,check_ary):
            if printing:
                print('Input array: ', N)
                print('Output array: ', Y)
                print('Check array: ', check_ary)
                print('Correct!')
                return True
            else:
                return True
        else:
            if printing:
                print('Input array: ', N)
                print('Output array: ', Y)
                print('Check array: ', check_ary)
                print('Incorrect!')
                return False
            else:
                return False           


    def test_prefix_sum_gpu_naive(self):
        # implement this, note you can change the function signature (arguments and return type)
        pass

    def test_prefix_sum_gpu_work_efficient(self):
        # implement this, note you can change the function signature (arguments and return type)
        pass






if __name__ == '__main__':

    summer = PrefixSum()
    
    length_ary = [128, 2048, 128*2048, 2048*2048, 65535*2048]

    time_python_naive = []
    time_gpu_ineffcient = []
    time_gpu_efficient = []

    for length in length_ary:
        N = np.random.rand(length).astype(np.float64)
        _, time_p_n = summer.prefix_sum_python(N, length)
        _, time_g_i = summer.prefix_sum_gpu_naive(N, length)
        _, time_g_e = summer.prefix_sum_gpu_work_efficient(N, length)
        time_python_naive.append(time_p_n)
        time_gpu_ineffcient.append(time_g_i)
        time_gpu_efficient.append(time_g_e)
        print('Finished scan with length: %d' % (length))

    np.save('time_python_naive.npy', np.array(time_python_naive))
    np.save('time_gpu_ineffcient.npy', np.array(time_gpu_ineffcient))
    np.save('time_gpu_efficient.npy', np.array(time_gpu_efficient))

    plot_python_naive = np.load('time_python_naive.npy')
    plot_gpu_inefficient = np.load('time_gpu_ineffcient.npy')
    plot_gpu_efficient = np.load('time_gpu_efficient.npy')

    plt.figure()
    plt.plot(plot_python_naive, label='python_naive')
    plt.plot(plot_gpu_inefficient, label='gpu_inefficient')
    plt.plot(plot_gpu_efficient, label='gpu_efficient')
    plt.legend()
    plt.grid()
    plt.xlabel('Length of vectors')
    plt.ylabel('Processing time (ms)')
    plt.title('Comparison of different scan methods')
    plt.savefig('comparison.png')
    # plt.show()