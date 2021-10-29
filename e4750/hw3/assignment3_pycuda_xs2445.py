#!/usr/bin/env python

import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
import time
import matplotlib.pyplot as plt


class Convolution:
	def __init__(self):
		# """
		# Attributes for instance of EncoderDecoder module
		# """
		self.mod = self.getSourceModule()
		pass
	
	def getSourceModule(self):
		# kernel code wrapper
		kernelwrapper_naive = """
			#include <stdio.h>
			__global__ 
			void conv_gpu_naive(float *N, float *P, float *M, int height, int width, int mask_width){

				// the coordinate of thread (also coordinate in N or P)
				int col = blockDim.x * blockIdx.x + threadIdx.x;
				int row = blockDim.y * blockIdx.y + threadIdx.y;
				printf("%d  %d\\n", col,row);

				// copy to register
				int mask_w = mask_width;
				int n_w = width;
				int n_h = height;
				// start point of the kernel
				int col_start = col - mask_w/2;
				int row_start = row - mask_w/2;

				float p_value = 0.0f;

				// for every pixel in mask
				for(int i=0; i<mask_w; i++){
					// x coordinate in N
					int col_i = col_start + i;
					// if in the range of N
					if(col_i>=0 && col_i<n_w){
						for(int j=0; j<mask_w; j++){
							// y coordinate in N
							int row_i = row_start + j;
							//if in the range of N
							if(row_i>=0 && row_i<n_h){
								p_value += N[col_i*n_w+row_i] * M[i*mask_w+j];
								//int a = col_i*n_w+row_i;
								//printf("%d\\n", a);
							}
						}
					}
				}
				P[col*n_w+row] = p_value;
			}
		
		""" # you can either use a string or save the kernel in kernel.cu file and reference it here.
		# Compile the kernel code when an instance
		# of this class is made. 
		return SourceModule(kernelwrapper_naive)

	def getBlockGridDim(self, N, blocksize=32):
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
		func_conv = self.mod.get_function("conv_gpu_naive")
		height, width = N.shape
		print(height,width)
		mask_width = M.shape[0]
		print(mask_width)
		# the result matrix
		P = np.empty_like(N)
		# copy to device global memory
		N_d = gpuarray.to_gpu(N)
		M_d = gpuarray.to_gpu(M)
		P_d = gpuarray.to_gpu(P)
		# block and grid size
		BlockDim, GridDim = self.getBlockGridDim(N)

		func_conv(N_d, P_d, M_d, np.int32(height), np.int32(width), np.int32(mask_width), block=BlockDim, grid = GridDim)

		P = P_d.get()

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
	N = np.ones((32,32),dtype=np.float)


	M = np.array(([0,0,0],[0,1,0],[0,0,0]),dtype=np.float)
	conver = Convolution()
	P = conver.conv_gpu_naive(N,M)