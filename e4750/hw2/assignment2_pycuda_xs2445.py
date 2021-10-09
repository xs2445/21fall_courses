import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import sys

class cudaCipher:
    def __init__(self):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        # If you are using any helper function to make 
        # blocksize or gridsize calculations, you may define them
        # here as lambda functions. 
        # Quick lambda function to calculate grid dimensions

        # define block and grid dimensions
        self.blocksize = 256
        self.BlockDim = (self.blocksize,1,1)
        # self.GridDim = (math.ceil(self.length/256),1,1)
        
        
        # kernel code wrapper
        self.kernelwrapper = """
        __global__
        void Decrypt(char *sentence, char *decrypted, int length){
            int i = blockIdx.x*blockDim.x + threadIdx.x;
            if(i<length){
                int asc = int(sentence[i]);
                if((asc > 96) && (asc < 123)){
                    if(asc < 110) decrypted[i] = char(asc+13);
                    else decrypted[i] = char(asc-13);
                    }
                else decrypted[i] = char(asc);
            }
        }
        """

        # Compile the kernel code when an instance
        # of this class is made.
        self.mod = SourceModule(self.kernelwrapper)
    
    def devCipher(self, sentence):
        """
        Function to perform on-device parallel ROT-13 encrypt/decrypt
        by explicitly allocating device memory for host variables using
        gpuarray.
        Returns
            out                             :   encrypted/decrypted result
            time_ :   execution time in milliseconds
        """
        # create cuda events to record the execution time
        start_alc = cuda.Event()
        start_cpt = cuda.Event()
        end = cuda.Event()

        # Get kernel function
        func_dcp = self.mod.get_function("Decrypt")

        # culculate the gridDim
        # print(type(sentence))
        length = len(sentence)*sys.getsizeof(sentence)
        # print(length)
        GridDim = (math.ceil(length/self.blocksize),1,1)

        # Device memory allocation for input and output array(s)
            # convert string into np.chararaay
        sentence = np.char.asarray(sentence)
        # sentence = unicode(sentence, "utf-8") 
        decrypted = np.empty_like(sentence)
            # record execution time with memory allocation
        start_alc.record()
        sentence_d = gpuarray.to_gpu(sentence)
        decrypted_d = gpuarray.to_gpu(decrypted)
        # print(type(sentence))
        
        # Record execution time and execute operation.
        start_cpt.record()
        func_dcp(sentence_d, decrypted_d, np.int32(length), block=self.BlockDim, grid=GridDim)
        end.record()
        
        # Wait for the event to complete
        end.synchronize()
        time_alc = start_alc.time_till(end)
        time_cpt = start_cpt.time_till(end)

        # Fetch result from device to host
        decrypted = decrypted_d.get()
        # Convert output array back to string
        # decrypted = decrypted.decode('utf-8')
        # decrypted = literal_eval(sentence)
        # decrypted = str(decrypted)
        # print(type(decrypted))
        # decrypted = np.array2string(decrypted)
        decrypted = np.ndarray.tolist(decrypted)
        # print(type(decrypted))

        return decrypted, time_alc
    
    def pyCipher(self, sentence):
        """
        Function to perform parallel ROT-13 encrypt/decrypt using 
        vanilla python.

        Returns
            decrypted                       :   encrypted/decrypted result
            time_         :   execution time in milliseconds
        """
        start = time.time()
        # using ord() convert sting to ascii whose datatype is int
        text_asc = [ord(i) for i in sentence]
        decrypted = ''
        # decrypt
        for asc in text_asc:

            if asc > 96 and asc < 123:
                if asc < 110:
                    # using chr() to convert ascii int to string
                    decrypted += chr(asc+13)
                else:
                    decrypted += chr(asc-13)
            else:
                decrypted += chr(asc)
        # print(type(decrypted))
        return decrypted, time.time()-start



def main():
    # Main code
    # create an instance of cudaCipher
    cipher = cudaCipher()

    # Open text file to be deciphered.
    # Preprocess the file to separate sentences
    text_de = open('deciphertext.txt','r').read()

    # Split string into list populated with '.' as delimiter.
    sentences = text_de.split('. ')

    # Empty lists to hold deciphered sentences, execution times
    decrypted_c = []
    decrypted_p = []
    tc = []
    tp = []

    # Loop over each sentence in the list
    for sentence in sentences:
        temp_sc, temp_tc = cipher.devCipher(sentence)
        temp_sp, temp_tp = cipher.pyCipher(sentence)
        decrypted_c.append(temp_sc)
        decrypted_p.append(temp_sp)
        tc.append(temp_tc)
        tp.append(temp_tp)

    # post process the string(s) if required
    decrypted_cuda = []
    for sentence in decrypted_c:
        for i in sentence:
            a = i
        decrypted_cuda.append(a)

    tc = np.array(tc)
    tp = np.array(tp)

    # Execution time
    print("CUDA output cracked in ", tc.mean(), " milliseconds per sentence.")
    print("Python output cracked in ", tp.mean(), " milliseconds per sentence.")

    # check if the results match
    try:
        print("Checkpoint: Do python and kernel decryption match? Checking...")
        for i in range(len(decrypted_cuda)):
            # print(decrypted_cuda[i], decrypted_p[i])
            decrypted_cuda[i] += '. '
            decrypted_p[i] += '. '
            assert decrypted_cuda[i] == decrypted_p[i]
    # dump bad output to file for debugging
    except AssertionError:
        print('Checkpoint failed: Python and CUDA kernel decryption do not match. Try Again!')
    # If ciphers agree, proceed to write decrypted text to file and plot execution times
    else: 
        print("Checkpoint passed!")
        print("Writing decrypted text to file...")
        # Write cuda output to file
        decrypted_text = open('decrypted_text.txt', 'w')
        decrypted_text.write(''.join(decrypted_cuda))
        decrypted_text.close()
        # Dot plot the  per-sentence execution times
        plt.figure()
        plt.scatter(range(len(decrypted_cuda)), tc, label='pycuda')
        plt.scatter(range(len(decrypted_cuda)), tp, label='vanilla python')
        plt.legend()
        plt.grid()
        plt.title('PyCUDA, Comparison of processing time (including memory allocation)')
        plt.xlabel('Sentences')
        plt.ylabel('Processing Time (ms)')
        # plt.show()
        plt.savefig('comparison_cuda.jpg')


if __name__ == '__main__':
    main()