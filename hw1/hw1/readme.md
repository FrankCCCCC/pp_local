# Odd-Even Sort for MPI in C

# Implementation

## The Sorting Algorithm
I use the  Spread Sort of Boost library for local sorting. The Spread Sort is a novel hybrid radix sort algorithm. The time complexity given by the Boost official documentation is min(N logN, N key_length)(Note: N is the length of the sequence to be sorted). 

As for Odd-Even Sort, I use Baudet-Stevenson Odd-Even sort. The Baudet-Stevenson sort transfers the whole segment that the processes have to another one each time which reduces the transfer time extremely. It guarantees that it can finish within n phase for n parallel processes. 

## Other Features

#### Parallel Read/Write: 
For each process, just read/write the segment that it needs to handle.

#### Non-Blocking Message Passing: 
I use Non-Blocking message pass operation to pass the segment to another. It means that for each process it  would send the segment it has to the target process without waiting.

Note: The SOTA version is under last_ver/no_stop. 