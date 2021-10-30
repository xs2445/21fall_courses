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
		self.mod = self.getSourceModule()
		pass
	
	def getSourceModule(self):
		# kernel code wrapper
		kernelwrapper = """
			#include <stdio.h>
			__global__ 
			void conv_gpu_naive(const float *N, float *P, const float *M, int height, int width, int mask_width){

				// the coordinate of thread (also coordinate in N or P)
				const int col = blockDim.x * blockIdx.x + threadIdx.x;
				const int row = blockDim.y * blockIdx.y + threadIdx.y;
				//printf("%d   %d\\n", row, col);

				// copy to register
				const int mask_w = mask_width;
				const int n_w = width;
				const int n_h = height;
				// start point of the kernel
				const int col_start = col - (mask_w-1)/2;
				const int row_start = row - (mask_w-1)/2;

				float p_value = 0;

				// in y direction of mask
				for(int i=0; i<mask_w; i++){
					// x coordinate in mask
					int row_mask = mask_w - 1 - i;
					// x coordinate in N
					int row_n = row_start + i;
					
					if((row_n>=0) && (row_n<n_h)){
					// in x direction of mask
					for(int j=0; j<mask_w; j++){
						// y coordinate in mask
						int col_mask = mask_w - 1 - j;
						// y coordinate in N
						int col_n = col_start + j;

						// if in the range of N
						//if((row_n>=0) && (row_n<n_h) && (col_n>=0) && (col_n<n_w)){
						if((col_n>=0) && (col_n<n_w)){
							p_value += N[row_n*n_w+col_n] * M[row_mask*mask_w+col_mask];
							//printf("%d  %d\\n", row, col);
						}
					}
					}
				}
				/*
				if((row>=0) && (row<height) && (col>=0) && (col<width)){
					//printf("%d  %d  %d\\n", row, col, float(N[row*width+col]));
					//printf("%d\\n", N[1]);
					printf("%d\\n", N[row*width+col]);
				}
				*/
				P[row*n_w+col] = p_value;
			}
		
		""" 
		# you can either use a string or save the kernel in kernel.cu file and reference it here.
		# Compile the kernel code when an instance
		# of this class is made. 
		return SourceModule(kernelwrapper)


	def getBlockGridDim(self, N, blocksize=16):
		BlockDim = (blocksize, blocksize,1)
		GridDim = (N.shape[0]//blocksize+1, N.shape[1]//blocksize+1,1)
		return BlockDim, GridDim

	def conv_gpu_naive(self, N, M):
		"""
		convolution with global memory
		:param N: input matrix
		:param M: mask
		:return:
		- out: a tensor with the same shape as x
		- cache: (train phase) cache a random dropout mask used in feedforward process
				(test phase) None
		"""
		# implement this, note you can change the function signature (arguments and return type)
		# convert the datatype
		N = N.astype(np.float32)
		M = M.astype(np.float32)

		func_conv = self.mod.get_function("conv_gpu_naive")
		height, width = N.shape
		# print(height,width)
		mask_width = M.shape[0]
		# print(mask_width)
		# the result matrix
		P = np.empty_like(N)
		# copy to device global memory
		N_d = gpuarray.to_gpu(N)
		M_d = gpuarray.to_gpu(M)
		P_d = gpuarray.to_gpu(P)

		# Allocate memory on device
		# N_d = cuda.mem_alloc(N.nbytes)
		# M_d = cuda.mem_alloc(M.nbytes)
		# P_d = cuda.mem_alloc(P.nbytes)
		# print(N.shape)
		# print(N.nbytes/N.shape[0]/N.shape[1])
		# Copy matrix to memory
		# cuda.memcpy_htod(N_d, N)
		# cuda.memcpy_htod(M_d, M)

		# block and grid size
		BlockDim, GridDim = self.getBlockGridDim(N)
		print(BlockDim, GridDim)

		func_conv(N_d, P_d, M_d, np.int32(height), np.int32(width), np.int32(mask_width), block=BlockDim, grid = GridDim)

		P = P_d.get()
		# cuda.memcpy_dtoh(P, P_d)

		return P


	def conv_gpu_shared_mem(self):
		# implement this, note you can change the function signature (arguments and return type)
		pass

	def conv_gpu_shared_and_constant_mem(self):
		# implement this, note you can change the function signature (arguments and return type)
		pass

	def test_conv_pycuda(self):
		# implement this, note you can change the function signature (arguments and return type)
		pass


if __name__ == "__main__":
	N = np.random.rand(5,5)
	M = np.random.rand(3,3)
	# M = np.array([[1,0,0],[0,0,0],[0,0,0]])
	# print(M)
	conver = Convolution()
	P_cu = conver.conv_gpu_naive(N,M)
	P_sp = convolve2d(N.astype(np.float32), M.astype(np.float32), mode='same')
	print(np.allclose(P_cu, P_sp))
	print(N,'\n')
	print(P_cu,'\n')
	print(P_sp)