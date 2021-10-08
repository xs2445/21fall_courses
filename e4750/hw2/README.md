# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2021)

## Assignment-2: Deciphering Text and an Introduction to Profiling

Total points: 100

### References to help you with assignment
* [PyCuda Examples](https://github.com/inducer/pycuda/tree/main/examples)
* [PyOpenCL Examples](https://github.com/inducer/pyopencl/tree/main/examples)
* [NVidia Blog](https://developer.nvidia.com/blog/tag/cuda/)

### GUI Installation and Introduction to CUDA Profiling

Refer to Wiki section for CUDA Profiling and GUI installation. You can refer to recitation video on how to invoke Profiling using NSIGHT and what output is required to be submitted for the assignment.

### Outputs
Profiling output has to be submitted as a txt file. So your final submission will have 5 files:
1. 2 python files assignment_2_pycuda_<uni>.py, assignment_2_pyopencl_{uni}.py
2. 1 report assignment_2_report_{uni}.pdf
3. 1 profiling output file for pycuda: assignemnt_2_profile_{uni}.txt.
4. 1 file for deciphered text. (Note: You need to submit only one of the deciphered files considering both decryptions are equal, ofcourse marks will be deducted if the output does not match when I am testing it.)
    
All images related to GUI should be in report instead.

### Primer

The goal of this assignment is to introduce complexity to the 1D array manipulation kernel in OpenCL and CUDA. Additionally, you are expected to profile your PyCUDA code and form some conclusions from the visual profiling output you get. The assignment is divided into a programming section, and a theory section. The former contains two tasks - one for CUDA, the other for OpenCL. Information on CUDA Profiling will be introduced this week.

### Relevant Documentation

You may refer to the following links for this assignment to get started:

* [ROT-13 Cipher](https://en.wikipedia.org/wiki/ROT13)
* [ROT-13 converter](https://rot13.com/)


## Programming Problem (75 points)

For this assignment, you have been given an encoded text file. Your task is to correctly decode it using appropriate CUDA and OpenCL kernels, along with a naive function using basic string manipulation in Python. While you will still be dealing with 1D arrays, the kernels you write will not be as primitive as they were in the first assignment. 

Your task is simplified several ways:
1. You will only deal with lower case characters of the English alphabet. Numbers, punctuation are left untouched by the cipher. 
2. The cipher scheme is known to you. 

### ROT-N Ciphers

Rotation ciphers are some of the most popular ciphers. In this assignment we will deal with the most well known of them all - ROT-13, popularly known as the Caesar Cipher. 

If you are intercepting a coded message, it is unlikely you immediately know the cipher used. The larger the possible number of ciphers, the more computational power you need. (See: Enigma). If you had to write a code to iteratively try out every possible ROT-N cipher to decode some text, parallel compuation would be a great boon to have. This assignment is set in a similar vein. 

### Problem set up

Consider the file `deciphertext.txt`. It contains text coded in ROT-13. All characters are lower case, and none of the numbers or punctuation are encoded. You must write code to read this file, and decipher it on a per-sentence basis. The PyOpenCL and PyCUDA kernels must only convert lowercase characters, and ignore all other ASCII values. 

#### Task-1: PyOpenCL (30 points)

Read the given text file and preprocess it according to the requirements in the steps below:

1. *(7 points)* Write the kernel code for per-character conversion.

2. *(8 points)* Write a function to decipher an input string using the kernel code. Time the execition of the entire operation (including memory transfers).

3. *(5 points)* Write a function to decipher an input string using native python string manipulation. Time the execution of the entire operation. 

4. *(5 points )* Call the kernel function and python function iteratively for every sentence in the input text. You can stitch the decrypted sentences back together into a unified output for each method. Finally, use Python's exception handling (`try` and `except`) to compare the two decryption results. If they are equal, only then write the decryption to a file in the assignment directory. (You can save it to any file path you need to submit this file.)

5. *(5 points)* Save a dot-plot the of per-sentence execution time for both methods. Include this in report.

#### Task-2: PyCUDA & Profiling (40 points)

Read the given text file and preprocess it according to the requirements in the steps below:

1. *(5 points)* Write the kernel code for per-character conversion.

2. *(5 points)* Write a function to decipher an input string using the kernel code. Time the execition of the entire operation (including memory transfers).

3. Use the same naive python function to decipher sentence strings that you wrote for task-1. Time the execution of the entire operation. 

4. *(5 points )* Call the kernel function and python function iteratively for every sentence in the input text. You can stitch the decrypted sentences back together into a unified output for each method. Finally, use Python's exception handling (`try` and `except`) to compare the two decryption results. If they are equal, only then write the decryption to a file in the assignment directory. 

5. *(5 points)* Save a dot-plot the of per-sentence execution time for both methods. Include this in report.

6. *(10 points)* Profile your CUDA kernel using NSight. Take screenshots of what you see, and note your observations in the report.

7. *(10 points)* Based on the dot-plot, are all sentences deciphered in (roughly) equal time? If not, reason out why. Use the profiling output to explain the anomaly in execution time. 

### Deciphered Text (5 points)
The conditions for a full score are:
* The deciphered text matches the source text exactly, down to the spaces and punctuation. (i.e. total character count needs to be the same)
* All words in every sentence are correctly decoded. 

No points for identifying where the coded text is from; but if you happen to know - go ahead and write that in your report!

## Theory Problems (25 points)

1. *(5 points)* What is code profiling? How might profiling prove useful for CUDA development?

2. *(5 points)* Can two kernels be executed in parallel? If yes explain how, if no then explain why? You can consider multiple scenarios to answer this question.

3. *(5 points)* Cuda provides a "syncthreads" method, explain what is it and where is it used? Give an example for its application? Consider the following kernel code for doubling each vector, will syncthreads be helpful here?

```
__global__ void doublify(float *c_d, const float *a_d, const int len) {
        int t_id =  blockIdx.x * blockDim.x + threadIdx.x;
        c_d[t_id] = a_d[t_id]*2;
        __syncthreads();
}

```

4. *(5 points)* What's the difference between using "time.time()" and cuda events "event.record()" method to record execution time? Comment if the following pseudo codes describe correct usage of the methods for measuring time of doublify kernel introduced in previous question.

Cuda Events(event.record()):
```
event_start = cuda.Event()
event_start.record()
doublify(...) # assume this is a correct call to doublify kernel.
event_end.record()
event_end.synchronize()
time_taken = event_start.time_till(event_end) # comment on this.
```

Time (time.time()):
```
start = time.time()
doublify(...) # assume this is a correct call to doublify kernel.
end = time.time()
pycuda.driver.Context.synchronize() # assume this is the correct usage of synchronization all threads.
time_taken = end - start # comment on this.
```

5. *(5 points)* For a vector addition, assume that the vector length is 7500, each thread calculates one output element, and the thread block size is 512 threads. The programmer configures the kernel launch to have a minimal number of thread blocks to cover all output elements. How many threads will be in the grid? 

## Code Template

### PyOpenCL
```python
import relevant.libraries


class clCipher:
    def __init__(self):
        """
        Attributes for instance of clModule
        Includes OpenCL context, command queue, kernel code
        and input variables.
        """
        
        # Get platform and device property
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
        	if platform.name == NAME:
        		devs = platform.get_devices()       
        
        # Set up a command queue:
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        # kernel
        
        # Build kernel code
        

    def devCipher(self, sentence):
        """
        Function to perform on-device parallel ROT-13 encrypt/decrypt
        by explicitly allocating device memory for host variables.
        Returns
            decrypted :   decrypted/encrypted result
            time_     :   execution time in milliseconds
        """

        # Text pre-processing/list comprehension (if required)
        # Depends on how you approach the problem
        
        # device memory allocation
        
        # Call the kernel function and time event execution
        
        # OpenCL event profiling returns times in nanoseconds. 
        # Hence, 1e-6 will provide the time in milliseconds, 
        # making your plots easier to read.

        # Copy result to host memory
        
        return decrypted, time_

    
    def pyCipher(self, sentence):
        """
        Function to perform parallel ROT-13 encrypt/decrypt using 
        vanilla python. (String manipulation and list comprehension
        will prove useful.)

        Returns
            decrypted                  :   decrypted/encrypted result
            time_    :   execution time in milliseconds
        """
        
        return decrypted, time_


if __name__ == "__main__":
    # Main code

    # Open text file to be deciphered.
    # Preprocess the file to separate sentences


    # Loop over each sentence in the list
    for _ in _______:
        
    # Stitch decrypted sentences together
    
    print("OpenCL output cracked in ", tc, " milliseconds per sentence.")
    print("Python output cracked in ", tp, " milliseconds per sentence.")

    # Error check
    try:
        print("Checkpoint: Do python and kernel decryption match? Checking...")
        # compare outputs
    except ______:
        print("Checkpoint failed: Python and OpenCL kernel decryption do not match. Try Again!")
        # dump bad output to file for debugging
        

    # If ciphers agree, proceed to write decrypted text to file
    # and plot execution times

    if #conditions met: 

        # Write cuda output to file
        
        # Scatter plot the  per-sentence execution times    
```

### PyCUDA

```python
import relevant.libraries


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
        #
        
        
        # kernel code wrapper
        #
        

        # Compile the kernel code when an instance
        # of this class is made.

    
    def devCipher(self, sentence):
        """
        Function to perform on-device parallel ROT-13 encrypt/decrypt
        by explicitly allocating device memory for host variables using
        gpuarray.
        Returns
            out                             :   encrypted/decrypted result
            time_ :   execution time in milliseconds
        """
        # Get kernel function

        # Device memory allocation for input and output array(s)
        
        # Record execution time and execute operation.
        
        # Wait for the event to complete

        # Fetch result from device to host

        # Convert output array back to string

        return decrypted, time_

    
    def pyCipher(self, sentence):
        """
        Function to perform parallel ROT-13 encrypt/decrypt using 
        vanilla python.

        Returns
            decrypted                       :   encrypted/decrypted result
            time_         :   execution time in milliseconds
        """

        return decrypted, time_


if __name__ == "__main__":
    # Main code

    # Open text file to be deciphered.
    # Preprocess the file to separate sentences
    
    # Split string into list populated with '.' as delimiter.

    # Empty lists to hold deciphered sentences, execution times


    # Loop over each sentence in the list
    for _ in _______:
        
    # post process the string(s) if required
        
    # Execution time
    print("CUDA output cracked in ", tc, " milliseconds per sentence.")
    print("Python output cracked in ", tp, " milliseconds per sentence.")

    # Error check
    try:
        print("Checkpoint: Do python and kernel decryption match? Checking...")
        assert # something
        
    except _________:
        print("Checkpoint failed: Python and CUDA kernel decryption do not match. Try Again!")
        # dump bad output to file for debugging
        

    # If ciphers agree, proceed to write decrypted text to file
    # and plot execution times
    
    if #conditions met:
        print("Checkpoint passed!")
        print("Writing decrypted text to file...")

        # Write cuda output to file
        
        # Dot plot the  per-sentence execution times
```
