NVCCFLAGS  = -O3 -std=c++11 -Xptxas=-v -arch=sm_61 
# NVCCFLAGS  = -O3 -std=c++11 -Xptxas=-v -arch=sm_61 -ftz=true -Xcompiler -rdynamic -lineinfo
NVCC     = nvcc 
LDFLAGS  := -lm

TARGETS = hw4-1

.PHONY: all
all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)
