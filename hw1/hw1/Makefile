# CC = mpicc
# CXX = mpicxx
# CXXFLAGS = -O3 -lm
# CFLAGS = -O3 -lm
# TARGETS = hw1

# .PHONY: all
# all: $(TARGETS)

# .PHONY: clean
# clean:
# 	rm -f $(TARGETS)


CC = mpicc
CXX = mpicxx
CXXFLAGS = -O3 -lm
CFLAGS = -O3 -lm
TARGETS = hw1
DEPS = timer/timer.h timer/timer.c

.PHONY: all
all: 
	$(CXX) $(TARGETS).cc $(DEPS) $(CXXFLAGS) -o $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)