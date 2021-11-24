import pyopencl as cl
import numpy as np
import pyopencl.array as array
import matplotlib.pyplot as plt
from pyopencl import Buffer, MemoryObject
from scipy.signal import convolve2d
import time


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
		kernel_code = """
		//****************************************************************************************
		// global memory function

		__kernel void conv_gpu_naive_openCL(__global float *N, __global float *M, __global float *P, 
											const int width, const int height, const int mask_width){
			
			// position of threads
			//const int row = get_local_size(0)*get_gruop_id(0) + get_local_id(0);
			//const int col = get_local_size(1)*get_gruop_id(1) + get_local_id(1);
			const int row = get_global_id(0);
			const int col = get_global_id(1);

			// copy to register
			const int mask_w = mask_width;
			const int n_w = width;
			const int n_h = height;

			// start point of the kernel
			const int col_start = col - mask_w/2;
			const int row_start = row - mask_w/2;

			float p_value = 0.0f;

			// see if the thread in the range of N
			if((row<n_h)&&(col<n_w)){

				for(int i=0; i<mask_w; i++){			// in y direction of mask

					int row_mask = mask_w - 1 - i;		// x coordinate in mask
					int row_n = row_start + i;			// x coordinate in N
					
					for(int j=0; j<mask_w; j++){		// in x direction of mask

						int col_mask = mask_w - 1 - j;	// y coordinate in mask
						int col_n = col_start + j;		// y coordinate in N

						// if in the range of N
						if ((col_n>=0) && (col_n<n_w) && (row_n>=0) && (row_n<n_h)){
							p_value += N[row_n*n_w+col_n] * M[row_mask*mask_w+col_mask];
						}
					}
				}
				P[row*n_w+col] = p_value;
			}
		}

		//****************************************************************************************

		// predefined parameters
		#define MASK_SIZE 5
		#define TILE_SIZE 12
		#define TILE_SIZE_PAD (TILE_SIZE + MASK_SIZE - 1)

		//****************************************************************************************
		// local memory function (which is shared memory in cuda)


		__kernel void conv_gpu_shared_mem_openCL(__global float *N, __global float *M, __global float *P, 
											const int width, const int height){


			__local float N_ds[TILE_SIZE_PAD][TILE_SIZE_PAD];		// copy input matrix to local memory (which is shared memory in cuda)
			__local float M_ds[MASK_SIZE][MASK_SIZE];				// copy mask to lcoal memory

			// current position of thread
			const int tx = get_local_id(0);
			const int ty = get_local_id(1);

			// current position of input matrix
			const int col = tx + get_group_id(0)*TILE_SIZE;
			const int row = ty + get_group_id(1)*TILE_SIZE;

			// start point of mask
			const int col_start = col - (MASK_SIZE-1)/2;
			const int row_start = row - (MASK_SIZE-1)/2;

			// copy mask to shared memory
			for(int i=0; i<MASK_SIZE; i++){				// in x direction of mask
				for(int j=0; j<MASK_SIZE; j++){			// in y direction of mask
					M_ds[i][j] = M[i*MASK_SIZE+j];
				}
			}

			// copy each tiled matrix into local memory
			if((row_start>=0) && (row_start<height) && (col_start>=0) && (col_start<width)){
				N_ds[ty][tx] = N[row_start*width+col_start];
			}
			else{
				N_ds[ty][tx] = 0.0f;
			}

			// need to wait all the thread have done copy
    		barrier(CLK_LOCAL_MEM_FENCE);
			
			float p_value = 0.0f;
			if((ty<TILE_SIZE) && (tx<TILE_SIZE)){			// in range of tile

				// in y direction of mask
				for(int i=0; i<MASK_SIZE; i++){
					int row_mask = MASK_SIZE - 1 - i;		// y coordinate in mask

					// in x direction of mask
					for(int j=0; j<MASK_SIZE; j++){
						int col_mask = MASK_SIZE - 1 - j;	// x coordinate in mask
						p_value += N_ds[ty+i][tx+j] * M_ds[row_mask][col_mask];
					}
				}
				if((row<height) && (col<width)){					// in range of input matrix
					P[row*width+col] = p_value;
				}
			}
		}

		//****************************************************************************************
		// shared memory + constant memory function
		
		__kernel void conv_gpu_shared_and_constant_mem_openCL(__global float *N, __constant float *M, __global float *P, 
											const int width, const int height){


			__local float N_ds[TILE_SIZE_PAD][TILE_SIZE_PAD];		// copy input matrix to local memory (which is shared memory in cuda)

			// current position of thread
			const int tx = get_local_id(0);
			const int ty = get_local_id(1);

			// current position of input matrix
			const int col = tx + get_group_id(0)*TILE_SIZE;
			const int row = ty + get_group_id(1)*TILE_SIZE;

			// start point of mask
			const int col_start = col - (MASK_SIZE-1)/2;
			const int row_start = row - (MASK_SIZE-1)/2;

			// copy each tiled matrix into local memory
			if((row_start>=0) && (row_start<height) && (col_start>=0) && (col_start<width)){
				N_ds[ty][tx] = N[row_start*width+col_start];
			}
			else{
				N_ds[ty][tx] = 0.0f;
			}

			// need to wait all the thread have done copy
    		barrier(CLK_LOCAL_MEM_FENCE);
			
			float p_value = 0.0f;
			if((ty<TILE_SIZE) && (tx<TILE_SIZE)){			// in range of tile

				// in y direction of mask
				for(int i=0; i<MASK_SIZE; i++){
					int row_mask = MASK_SIZE - 1 - i;		// y coordinate in mask

					// in x direction of mask
					for(int j=0; j<MASK_SIZE; j++){
						int col_mask = MASK_SIZE - 1 - j;	// x coordinate in mask
						p_value += N_ds[ty+i][tx+j] * M[row_mask*MASK_SIZE+col_mask];
					}
				}
				if((row<height) && (col<width)){			// in range of input matrix
					P[row*width+col] = p_value;
				}
			}
		}


		""" 
		
		# Build kernel code
		self.prg = cl.Program(self.ctx, kernel_code).build()
		self.mask_size = 5
		self.tile_size = 12
		self.tile_size_pad = self.tile_size + self.mask_size - 1
		

	def conv_gpu_naive(self, N, M):
		
		"""
		parallel convolution using global memory
		visit input matrix and mask from global memory in the kernel
		mask can be any size

		params:
		- N: input matrix
		- M: mask

		return:
		- P: result
		- time
		"""
		
		# import time
		# start to record
		t_start = time.time()

		height, width = N.shape
		mask_width = M.shape[0]

		# P = np.empty_like(N)

		# device memory allocation
		N_d = cl.array.to_device(self.queue, N)
		M_d = cl.array.to_device(self.queue, M)
		# P_d = cl.array.empty_like(N)
		P_d = cl.array.empty(self.queue, N_d.shape, N_d.dtype)
		
		GlobalSize = (height, width, 1)
		# workgroup size
		# Local_size = (height//workitem_size+1, width//workitem_size+1,1)
		self.prg.conv_gpu_naive_openCL(self.queue, GlobalSize, None, N_d.data, M_d.data, P_d.data, np.int32(width), 
										np.int32(height), np.int32(mask_width))

		# wait for execution to complete.
		self.queue.finish()

		# Copy output from GPU to CPU
		P = np.array(P_d.get())

		# Record execution time 
		time_ = time.time() - t_start

		return P, time_*1e3
		

	def conv_gpu_shared_mem(self, N, M):
		"""
		parallel convolution using local memory
		visit input matrix and mask from local memory in the kernel
		mask have to be 5*5

		params:
		- N: input matrix
		- M: mask

		return:
		- P: result
		- time
		"""
		# start to record
		t_start = time.time()

		height, width = N.shape

		# device memory allocation
		N_d = cl.array.to_device(self.queue, N)
		M_d = cl.array.to_device(self.queue, M)
		# P_d = cl.array.empty_like(N)
		P_d = cl.array.empty(self.queue, N_d.shape, N_d.dtype)
		
		# grid size   
		GlobalSize = ((height//self.tile_size+1)*self.tile_size_pad, (width//self.tile_size+1)*self.tile_size_pad)
		# workgroup size
		LocalSize = (self.tile_size_pad, self.tile_size_pad)

		func = self.prg.conv_gpu_shared_mem_openCL
		event = func(self.queue, GlobalSize, LocalSize, N_d.data, M_d.data, P_d.data, 
											np.int32(width), np.int32(height))

		event.wait()
		# time_event = event.profile.end - event.profile.start

		# Copy output from GPU to CPU
		P = P_d.get()
		
		# Record execution time 
		time_ = time.time() - t_start

		return np.array(P), time_*1e3

	def conv_gpu_shared_and_constant_mem(self, N, M):
		"""
		parallel convolution using local and constant memory
		visit input matrix from local memory and mask from constant memory in the kernel
		mask have to be 5*5

		params:
		- N: input matrix
		- M: mask

		return:
		- P: result
		- time
		"""
		# start to record
		t_start = time.time()

		height, width = N.shape

		# device memory allocation
		N_d = cl.array.to_device(self.queue, N)
		M_d = cl.array.to_device(self.queue, M)
		# P_d = cl.array.empty_like(N)
		P_d = cl.array.empty(self.queue, N_d.shape, N_d.dtype)
		
		# grid size   
		GlobalSize = ((height//self.tile_size+1)*self.tile_size_pad, (width//self.tile_size+1)*self.tile_size_pad)
		# workgroup size
		LocalSize = (self.tile_size_pad, self.tile_size_pad)

		func = self.prg.conv_gpu_shared_and_constant_mem_openCL
		event = func(self.queue, GlobalSize, LocalSize, N_d.data, M_d.data, P_d.data, 
											np.int32(width), np.int32(height))

		event.wait()
		# time_event = event.profile.end - event.profile.start

		# Copy output from GPU to CPU
		P = P_d.get()
		
		# Record execution time 
		time_ = time.time() - t_start

		return np.array(P), time_*1e3

	def conv_scipy(self, N, M):
		"""
		Serial convolution using scipy.signal.convolve2d

		params:
		- N: input matrix
		- M: mask

		return:
		- P: result
		- time
		"""
		start = time.time()

		P = convolve2d(N.astype(np.float32), M.astype(np.float32), mode='same')

		return P, (time.time()-start)*1000

	def test_conv_pyopencl(self, N_size):
		"""
		test the correctness and performance of the parallel function

		params:
		- N_size: size of the input matrix, N is quare

		return:
		- t_naive: time of naive function
		- t_sh:    time of shared memory function 
		- t_const: time of shared and constant function
		- t_scipy: tmie of scipy function
		"""

		N = np.random.rand(N_size,N_size).astype(np.float32)
		M = np.random.rand(self.mask_size,self.mask_size).astype(np.float32)

		P_cl_naive, t_naive = self.conv_gpu_naive(N,M)
		P_cl_sh, t_sh = self.conv_gpu_shared_mem(N,M)
		P_cl_const, t_const = self.conv_gpu_shared_and_constant_mem(N,M)
		P_sp, t_scipy = self.conv_scipy(N,M)
 
		if not np.allclose(P_cl_naive, P_sp):
			raise Exception('Result from method using global memory does not match the correct answer!',
			'The shape of input matrix is %d' % (N_size))
		if not np.allclose(P_cl_sh, P_sp):
			raise Exception('Result from method using shared memory does not match the correct answer!',
			'The shape of input matrix is %d' % (N_size))
		if not np.allclose(P_cl_const, P_sp):
			raise Exception('Result from method using shared and constant memory does not match the correct answer!',
			'The shape of input matrix is %d' % (N_size))

		return t_naive, t_sh, t_const, t_scipy



if __name__ == '__main__':
	conver = Convolution()

	time_hist = []

	test_num_list = [16, 64, 256, 1024, 4096]

	for n_size in test_num_list:
		time_hist_temp = []
		# each length test 10 times
		for _ in range(10):
			time_hist_temp.append(conver.test_conv_pyopencl(n_size))
		time_hist.append(np.mean(np.array(time_hist_temp), axis=0))
	time_hist = np.array(time_hist)

	plt.figure(dpi=100)
	plt.grid()
	plt.plot(test_num_list, time_hist[:,0], label='naive')
	plt.plot(test_num_list, time_hist[:,1], label='shared')
	plt.plot(test_num_list, time_hist[:,2], label='shared+const')
	plt.plot(test_num_list, time_hist[:,3], label='scipy')
	plt.ylabel('Processing Time (ms)')
	plt.xlabel('Size of input matrix')
	plt.title('Comparason between different convolution method')
	plt.legend()
	plt.savefig('./pyopencl_withscipy.png')

	plt.figure(dpi=100)
	plt.grid()
	plt.plot(test_num_list, time_hist[:,0], label='naive')
	plt.plot(test_num_list, time_hist[:,1], label='shared')
	plt.plot(test_num_list, time_hist[:,2], label='shared+const')
	plt.semilogx()
	plt.ylabel('Processing Time (ms)')
	plt.xlabel('Size of input matrix')
	plt.title('Comparason between different convolution method')
	plt.legend()
	plt.savefig('./pyopencl_withoutscipy.png')
