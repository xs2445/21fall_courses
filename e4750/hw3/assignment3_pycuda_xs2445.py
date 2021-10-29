class Convolution:
    def __init__(self):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        # what's wrong here
		self.mod = self.getSourceModule()
		pass


	

	def getSourceModule(self):
		# kernel code wrapper
		kernelwrapper = """
			__global__ 
			void conv_gpu_naive(float *N, float *P, float *M, int height, int width, int mask_width){

				// the coordinate of thread (also coordinate in N or P)
				col = blockDim.x * blockIdx.x + threadIdx.x;
				row = blockDim.y * blockIdx.y + threadIdx.y;

				// copy to register
				int mask_w = mask_width
				int n_w = width
				int n_h = height
				// start point of the kernel
				int col_start = col - mask_w/2
				int row_start = row - mask_w/2

				float p_value = 0.0f

				// for every pixel in mask
				for(int i=0; i<mask_w; i++){
					// x coordinate in N
					col_i = col_start + i;
					// if in the range of N
					if(col_i>=0 && col_i<n_w){
						for(int j=0; j<mask_w; j++){
							// y coordinate in N
							row_i = row_start + j
							//if in the range of N
							if(row_i>=0 && row_i<n_h){
								p_value += N[col_i][row_i] * M[i][j]
							}
						}
					}
				}
				P[col][row] = p_value
			}
		
		""" # you can either use a string or save the kernel in kernel.cu file and reference it here.
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