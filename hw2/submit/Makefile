CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp -pthread
CXXFLAGS = $(CFLAGS)
hw2a_t=hw2a
hw2b_t=hw2b
# DEPS = libs/thread_pool/thread_pool.h libs/thread_pool/thread_pool.c libs/queue/queue.h libs/queue/queue.c

main: hw2a hw2b
	
hw2a:
	$(CXX) $(hw2a_t).cc $(CXXFLAGS) $(LDLIBS) -o $(hw2a_t)
hw2b:
	$(CXX) $(hw2b_t).cc $(CXXFLAGS) $(LDLIBS) -o $(hw2b_t)

.PHONY: clean
clean:
	rm -f $(hw2a_t) $(hw2b_t) $(hw2a_t:=.o) $(hw2b_t:=.o)
