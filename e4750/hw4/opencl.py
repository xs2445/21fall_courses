import pyopencl as cl
import numpy as np
import time
import pyopencl.array as array
import math
import matplotlib.pyplot as plt



class PrefixSum:
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

        kernel_code = """
         #define SECTION_SIZE 1024
         
        //-------------naive kernel phase1 start--------------------------------------
         // this kernel is used for calculating prefix of sections & prefix of S
        __kernel void KoggeStone_end(__global double *X, __global double *Y, __global double *end_ary, const int InputSize){
            
            const int tx = get_local_id(0);
            const int bx = get_group_id(0);
            const unsigned long int i = get_global_id(0);

            __local double XY[SECTION_SIZE];
            
            if(i<InputSize){
                XY[tx] = X[i];
            }

            // iterative scan on XY
            float temp = 0.0f;
            for(unsigned int stride = 1; stride < SECTION_SIZE; stride*=2){
                temp = 0;
                barrier(CLK_LOCAL_MEM_FENCE);
                if(tx >= stride) temp = XY[tx-stride];
                barrier(CLK_LOCAL_MEM_FENCE);
                XY[tx] += temp;
                //if(tx >= stride) XY[tx] += XY[tx-stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        
            if(i<InputSize) Y[i] = XY[tx];

            // copy the last element of the block
            if(tx == SECTION_SIZE-1){
                end_ary[bx] = XY[tx];
            }
            else if(i == InputSize-1){
                end_ary[bx] = XY[tx];
            }
        }
        
        //------------------------------------------------------------------------------
        // phase 3 of inefficient scan
        
        __kernel void phase3_koggestone(__global double *Y, __global double *S, const int InputSize)
        // double *in = out from last kernel; S = S;
        {
            // current position in Y
            const unsigned long int y = get_global_id(0);

            if(get_group_id(0) > 0 && y<InputSize) Y[y] += S[get_group_id(0)-1];
        }
        
        
        //------------------------------------------------------------------------------

        #define SECTION_SIZE_2 SECTION_SIZE*2

        //------------------------------------------------------------------------------
        // Brent-Kung scan kernel phase1&2 (efficient scan)
        // output two array Y: pre-fix sum, end_ary: last elements of each blocok.
        
        __kernel void BrentKung_end(__global double *X, __global double *Y, __global double *end_ary, const int InputSize)
        {
            __local double XY[SECTION_SIZE_2];

            // current position in block
            const unsigned int tx = get_local_id(0);
            // current position in X
            const unsigned long int i = 2*get_local_size(0)*get_group_id(0) + tx;

            // each thread copy 2 element to shared memory
            if(i<InputSize) XY[tx] = X[i];
            if(i+SECTION_SIZE < InputSize) XY[tx+SECTION_SIZE] = X[i+SECTION_SIZE];
            
            // reduction tree
            for(unsigned int stride=1; stride <= SECTION_SIZE; stride*=2){
                barrier(CLK_LOCAL_MEM_FENCE);
                int index = (tx+1)*2*stride-1;
                if(index<SECTION_SIZE_2) XY[index] += XY[index - stride];                 
            }

            // distribution tree 
            for(unsigned int stride=SECTION_SIZE_2/4; stride > 0; stride /=2){
                int index = (tx+1)*2*stride-1;
                barrier(CLK_LOCAL_MEM_FENCE);
                if(index+stride < SECTION_SIZE_2) XY[index+stride] += XY[index];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // output the last element of each block
            if(((tx+1) == SECTION_SIZE) && (i+SECTION_SIZE < InputSize)){
                end_ary[get_group_id(0)] = XY[tx+SECTION_SIZE];
            }
            else if((i+1) == InputSize){
                end_ary[get_group_id(0)] = XY[tx];
            }
            else if((i+SECTION_SIZE+1) == InputSize){
                end_ary[get_group_id(0)] = XY[tx+SECTION_SIZE];
            }
            
            // output each block
            if(i<InputSize) Y[i] = XY[tx];
            if(i+SECTION_SIZE < InputSize) Y[i+SECTION_SIZE] = XY[tx+SECTION_SIZE];
           
        }            

        //------------------------------------------------------------------------------
        // phase 3 of efficient scan


         __kernel void phase3_brentkung(__global double *Y, __global double *S, const int InputSize)
         {
            // current position in Y
            const unsigned long int y = 2*(get_local_size(0)*get_group_id(0)) + get_local_id(0);

            if(2*get_group_id(0) > 0 && y<InputSize) Y[y] += S[get_group_id(0)-1];
            
            if(2*get_group_id(0) > 0 && y+SECTION_SIZE < InputSize) Y[y+SECTION_SIZE] += S[get_group_id(0)-1];
         }
        """

        self.prg = cl.Program(self.ctx, kernel_code).build()

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
        """
		another naive prefix sum serial implementation
        yi = yi-1 + xi

		params:
		- N: input array

		return:
        - Y: output array
		"""

        start = time.time()

        Y = np.zeros_like(N)

        Y[0] = N[0]
        for i in range(length-1):
            Y[i+1] = Y[i] + N[i+1]

        return Y, (time.time()-start)*1e3

    def prefix_sum_gpu_naive(self, arrayin, length):
        """
        Prefix_sum using Kogge-Stone scan kernel

		params:
		- N: input array

		return:
		- Y: result
		- time
		"""
        t_start = time.time()

        if length<=1024:
            n = 1
        else:
            n = math.ceil(math.log(length, 1024))

        Y_d_list = []
        E_d_list = [array.to_device(self.queue, arrayin)]

        blocksize = 1024
        gridsize_list = [length]

        func_red = self.prg.KoggeStone_end
        func_dis = self.prg.phase3_koggestone

        for i in range(n):
            # memory allocation for Y
            Y_d_list.append(array.zeros(self.queue, gridsize_list[-1], dtype=np.float64))
            # gridsize for this step
            gridsize_list.append(((gridsize_list[i]-1)//blocksize+1)*blocksize)
            # list of last elements for this step
            E_d_list.append(array.zeros(self.queue, gridsize_list[-1], dtype=np.float64))

            event_red = func_red(
                self.queue, 
                (gridsize_list[-1],1,1), (blocksize,1,1),
                E_d_list[i].data, Y_d_list[i].data, E_d_list[-1].data, np.int32(gridsize_list[i])
            )
            event_red.wait()


        for i in range(n-1)[::-1]:
            event_dist = func_dis(
                self.queue, 
                (gridsize_list[i+1],1,1), (blocksize,1,1), 
                Y_d_list[i].data, Y_d_list[i+1].data, np.int32(gridsize_list[i])
            )
            event_dist.wait()

        Y = Y_d_list[0].get()

        t_end = time.time()

        return Y, (t_end-t_start)*1e3


    def prefix_sum_gpu_work_efficient(self, N, length):
        """
        Prefix_sum using Brent-Kung scan kernel

		params:
		- N: input array

		return:
		- Y: result
		- time
		"""
        t_start = time.time()

        if length<=2048:
            n = 1
        else:
            n = math.ceil(math.log(length, 2048))

        Y_d_list = []
        E_d_list = [array.to_device(self.queue, N)]

        blocksize = 1024
        gridsize_list = [length]

        func_red = self.prg.BrentKung_end
        func_dis = self.prg.phase3_brentkung

        for i in range(n):
            # memory allocation for Y
            Y_d_list.append(array.zeros(self.queue, gridsize_list[-1], dtype=np.float64))
            # gridsize for this step
            gridsize_list.append(((gridsize_list[i]-1)//blocksize+1)*blocksize)
            # list of last elements for this step
            E_d_list.append(array.zeros(self.queue, gridsize_list[-1], dtype=np.float64))

            event_red = func_red(
                self.queue, 
                (gridsize_list[-1],1,1), (blocksize,1,1),
                E_d_list[i].data, Y_d_list[i].data, E_d_list[-1].data, np.int32(gridsize_list[i])
            )
            event_red.wait()


        for i in range(n-1)[::-1]:
            event_dist = func_dis(
                self.queue, 
                (gridsize_list[i+1],1,1), (blocksize,1,1), 
                Y_d_list[i].data, Y_d_list[i+1].data, np.int32(gridsize_list[i])
            )
            event_dist.wait()

        Y = Y_d_list[0].get()

        t_end = time.time()

        return Y, (t_end-t_start)*1e3


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


if __name__ == '__main__':

    # summer = PrefixSum()
    
    length_ary = [128, 2048, 128*2048, 2048*2048, 65535*2048]

    # time_python_naive = []
    # time_gpu_ineffcient = []
    # time_gpu_efficient = []

    # for length in length_ary:
    #     N = np.random.rand(length).astype(np.float64)
    #     _, time_p_n = summer.prefix_sum_python2(N, length)
    #     _, time_g_i = summer.prefix_sum_gpu_naive(N, length)
    #     _, time_g_e = summer.prefix_sum_gpu_work_efficient(N, length)
    #     time_python_naive.append(time_p_n)
    #     time_gpu_ineffcient.append(time_g_i)
    #     time_gpu_efficient.append(time_g_e)
    #     print('Finished scan with length: %d' % (length))

    # np.save('time_python_naive_cl.npy', np.array(time_python_naive))
    # np.save('time_gpu_ineffcient_cl.npy', np.array(time_gpu_ineffcient))
    # np.save('time_gpu_efficient_cl.npy', np.array(time_gpu_efficient))

    plot_python_naive = np.load('time_python_naive_cl.npy')
    plot_gpu_inefficient = np.load('time_gpu_ineffcient_cl.npy')
    plot_gpu_efficient = np.load('time_gpu_efficient_cl.npy')

    plt.figure()
    plt.plot(plot_python_naive, label='python_naive')
    plt.plot(plot_gpu_inefficient, label='gpu_inefficient')
    plt.plot(plot_gpu_efficient, label='gpu_efficient')
    plt.legend()
    plt.grid()
    plt.xticks([0,1,2,3,4], length_ary)
    plt.xlabel('Length of vectors')
    plt.ylabel('Processing time (ms)')
    plt.title('Comparison of different scan methods (OpenCL)')
    plt.savefig('comparison_opencl.png')
    # plt.show()