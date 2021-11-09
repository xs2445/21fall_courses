#!/usr/bin/env python


import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import convolve2d



class Convolution:
	def __init__(self):
		# """
		# Attributes for instance of EncoderDecoder module
		# """
		self.mod = None
		self.getSourceModule()
		self.tile_size = 12
		self.mask_size = 5
		self.block_size = self.tile_size + self.mask_size -1
		self.block_dim = (self.block_size, self.block_size, 1)
	
	def getSourceModule(self):
		# kernel code wrapper
		kernelwrapper = """
			#include <stdio.h>

			//****************************************************************************************
			// global memory function

			__global__ 
			void conv_gpu_naive(float *N, float *P, float *M, int height, int width, int mask_width){

				// the coordinate of thread (also coordinate in N or P)
				int col = blockDim.x * blockIdx.x + threadIdx.x;
				int row = blockDim.y * blockIdx.y + threadIdx.y;

				// copy to register
				int mask_w = mask_width;
				int n_w = width;
				int n_h = height;
				// start point of the kernel
				int col_start = col - mask_w/2;
				int row_start = row - mask_w/2;

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
			// shared memory function

			__global__
			void conv_gpu_shared_mem(float *N, float *P, float *M, int height, int width){

				// copy to register
				const int n_w = width;
				const int n_h = height;
				const int mask_size = MASK_SIZE;
				const int tile_size = TILE_SIZE;

				__shared__ float N_ds[TILE_SIZE_PAD][TILE_SIZE_PAD];	// copy part of input matrix into shared memory
				__shared__ float M_ds[MASK_SIZE][MASK_SIZE];			// copy mask into shared memory

				// current position of N_ds
				const int tx = threadIdx.x;
				const int ty = threadIdx.y;

				// current position of input matrix
				const int col = tx + blockIdx.x*tile_size;
				const int row = ty + blockIdx.y*tile_size;

				// start point of mask
				const int col_start = col - mask_size/2;
				const int row_start = row - mask_size/2;

				// copy mask to shared memory
				for(int i=0; i<mask_size; i++){				// in x direction of mask
					for(int j=0; j<mask_size; j++){			// in y direction of mask
						M_ds[i][j] = M[i*mask_size+j];
					}
				}

				// copy input matrix to shared memory by tile (each thread will copy 1 element to N_ds)
				// here each tile is enlarged to TILE_SIZE_PAD which is size of N_ds and block

				if((row_start>=0) && (row_start<n_h) && (col_start>=0) && (col_start<n_w)){
					N_ds[ty][tx] = N[row_start*n_w+col_start];
				}
				else{
					N_ds[ty][tx] = 0.0f;
				}

				// need to wait all the thread have done copy
				__syncthreads();

				float p_value = 0.0f;
				if((ty<tile_size) && (tx<tile_size)){			// in range of tile

					// in y direction of mask
					for(int i=0; i<mask_size; i++){
						int row_n = ty+i;						// y coordinate in N_ds
						int row_mask = mask_size - 1 - i;		// y coordinate in mask

						// in x direction of mask
						for(int j=0; j<mask_size; j++){
							int col_n = tx+j;					// x coordinate in N_ds
							int col_mask = mask_size - 1 - j;	// x coordinate in mask
							p_value += N_ds[row_n][col_n] * M_ds[row_mask][col_mask];
						}
					}
					if((row<n_h) && (col<n_w)){					// in range of input matrix
						P[row*n_w+col] = p_value;
					}
				}
			}

			//****************************************************************************************

			// allocate constant memory for mask
			__constant__ float M_c[MASK_SIZE*MASK_SIZE];

			//****************************************************************************************
			// shared + constant memory function

			__global__ 
			void conv_gpu_shared_and_constant_mem(float *N, float *P, float *M_c, int height, int width){
				
				// copy to register
				const int n_w = width;
				const int n_h = height;

				__shared__ float N_ds[TILE_SIZE_PAD][TILE_SIZE_PAD];	// copy part of input matrix into shared memory

				// current position of N_ds
				const int tx = threadIdx.x;
				const int ty = threadIdx.y;

				// current position of input matrix
				const int col = tx + blockIdx.x*TILE_SIZE;
				const int row = ty + blockIdx.y*TILE_SIZE;

				// start point of mask
				const int col_start = col - MASK_SIZE/2;
				const int row_start = row - MASK_SIZE/2;

				// mask is in the constant memory

				// copy input matrix to shared memory by tile (each thread will copy 1 element to N_ds)
				// here each tile is enlarged to TILE_SIZE_PAD which is size of N_ds and block

				if((row_start>=0) && (row_start<n_h) && (col_start>=0) && (col_start<n_w)){
					N_ds[ty][tx] = N[row_start*n_w+col_start];
				}
				else{
					N_ds[ty][tx] = 0.0f;
				}

				// need to wait all the thread have done copy
				__syncthreads();

				float p_value = 0.0f;
				if((ty<TILE_SIZE) && (tx<TILE_SIZE)){			// in range of tile

					// in y direction of mask
					for(int i=0; i<MASK_SIZE; i++){
						int row_n = ty+i;						// y coordinate in N_ds
						int row_mask = MASK_SIZE - 1 - i;		// y coordinate in mask

						// in x direction of mask
						for(int j=0; j<MASK_SIZE; j++){
							int col_n = tx+j;					// x coordinate in N_ds
							int col_mask = MASK_SIZE - 1 - j;	// x coordinate in mask
							p_value += N_ds[row_n][col_n] * M_c[row_mask*MASK_SIZE+col_mask];
						}
					}
					if((row<n_h) && (col<n_w)){					// in range of input matrix
						P[row*n_w+col] = p_value;
					}
				}
			}


		
		""" # you can either use a string or save the kernel in kernel.cu file and reference it here.
		# Compile the kernel code when an instance
		# of this class is made. 
		mod = SourceModule(kernelwrapper)
			
		self.mod = mod


	def getGridDim(self, height, width, dim_h, dim_w):
		GridDim = (height//dim_h+1, width//dim_w+1,1)
		return GridDim

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
		# implement this, note you can change the function signature (arguments and return type)
		start = cuda.Event()
		end = cuda.Event()
		start.record()

		# get shape of input matrix
		height, width = N.shape
		# get shape of mask
		mask_width = M.shape[0]
		# create the result matrix
		P = np.empty_like(N)

		# memory allocation on device
		# N_d = gpuarray.to_gpu(N)
		# M_d = gpuarray.to_gpu(M)
		# P_d = gpuarray.to_gpu(P)
		N_d = cuda.mem_alloc_like(N)
		M_d = cuda.mem_alloc_like(M)
		P_d = cuda.mem_alloc_like(P)

		# copy matrics to device
		cuda.memcpy_htod(M_d, M)
		cuda.memcpy_htod(N_d, N)

		# block and grid size
		blockdim = 32
		BlockDim = (blockdim, blockdim, 1)
		GridDim = self.getGridDim(height, width, dim_h=blockdim, dim_w=blockdim)

		func_conv = self.mod.get_function("conv_gpu_naive")
		func_conv(N_d, P_d, M_d, np.int32(height), np.int32(width), np.int32(mask_width), block=BlockDim, grid = GridDim)

		cuda.memcpy_dtoh(P, P_d)
		
		end.record()
		end.synchronize()
		time = start.time_till(end)

		return P, time

	def conv_gpu_shared_mem(self, N, M):
		"""
		parallel convolution using shared memory
		visit input matrix and mask from shared memory in the kernel
		mask have to be 5*5

		params:
		- N: input matrix
		- M: mask

		return:
		- P: result
		- time
		"""
		start = cuda.Event()
		end = cuda.Event()
		start.record()

		# get shape of input matrix
		height, width = N.shape

		P = np.empty_like(N)

		# memory allocation on device
		N_d = cuda.mem_alloc_like(N)
		M_d = cuda.mem_alloc_like(M)
		P_d = cuda.mem_alloc_like(P)

		# copy matrics to device
		cuda.memcpy_htod(M_d, M)
		cuda.memcpy_htod(N_d, N)

		# block and grid size
		BlockDim = self.block_dim
		GridDim = self.getGridDim(height, width, dim_h=self.tile_size, dim_w=self.tile_size)

		func_conv = self.mod.get_function("conv_gpu_shared_mem")
		func_conv(N_d, P_d, M_d, np.int32(height), np.int32(width), np.int32(self.tile_size), block=BlockDim, grid = GridDim)

		cuda.memcpy_dtoh(P, P_d)

		end.record()
		end.synchronize()
		time = start.time_till(end)

		return P, time

	def conv_gpu_shared_and_constant_mem(self, N, M):
		"""
		parallel convolution using shared memory and constant memory
		visit input matrix from shared memory, visit mask from constant memory in the kernel
		mask have to be 5*5

		params:
		- N: input matrix
		- M: mask

		return:
		- P: result
		- time
		"""

		start = cuda.Event()
		end = cuda.Event()
		start.record()

		# get shape of input matrix
		height, width = N.shape

		P = np.empty_like(N)

		# memory allocation on device
		N_d = cuda.mem_alloc_like(N)
		M_d = cuda.mem_alloc_like(M)
		P_d = cuda.mem_alloc_like(P)

		# copy matrics to device
		cuda.memcpy_htod(M_d, M)
		cuda.memcpy_htod(N_d, N)

		# block and grid size
		BlockDim = self.block_dim
		GridDim = self.getGridDim(height, width, dim_h=self.tile_size, dim_w=self.tile_size)

		func_conv = self.mod.get_function("conv_gpu_shared_and_constant_mem")
		func_conv(N_d, P_d, M_d, np.int32(height), np.int32(width), np.int32(self.tile_size), block=BlockDim, grid = GridDim)

		cuda.memcpy_dtoh(P, P_d)

		end.record()
		end.synchronize()
		time = start.time_till(end)

		return P, time

	@staticmethod
	def conv_scipy(N, M):
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


	def test_conv_pycuda(self, N_size):
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

		P_cu_naive, t_naive = self.conv_gpu_naive(N,M)
		P_cu_sh, t_sh = self.conv_gpu_shared_mem(N,M)
		P_cu_const, t_const = self.conv_gpu_shared_and_constant_mem(N,M)
		P_sp, t_scipy = self.conv_scipy(N,M)
 
		if not np.allclose(P_cu_naive, P_sp):
			raise Exception('Result from method using global memory does not match the correct answer!',
			'The shape of input matrix is %d' % (N_size))
		if not np.allclose(P_cu_sh, P_sp):
			raise Exception('Result from method using shared memory does not match the correct answer!',
			'The shape of input matrix is %d' % (N_size))
		if not np.allclose(P_cu_const, P_sp):
			raise Exception('Result from method using shared and constant memory does not match the correct answer!',
			'The shape of input matrix is %d' % (N_size))

		return t_naive, t_sh, t_const, t_scipy



if __name__ == "__main__":
	conver = Convolution()

	time_hist = []

	test_num_list = [16, 64, 256, 1024, 4096]

	for n_size in test_num_list:
		time_hist_temp = []
		for _ in range(10):
			time_hist_temp.append(conver.test_conv_pycuda(n_size))
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
	plt.savefig('./pycuda_withscipy.png')

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
	plt.savefig('./pycuda_withoutscipy.png')
