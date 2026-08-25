NVCC := nvcc
ARCH := -arch=sm_75

# Debug symbols on by default so `make` / `make run` can be stepped through
# with cuda-gdb. Pass DEBUG=0 for an optimized, symbol-free build.
DEBUG ?= 1
ifeq ($(DEBUG),1)
CFLAGS := -O0 -g -G -std=c++17 -I../../common
else
CFLAGS := -O2 -std=c++17 -I../../common
endif

all: bin/01_stencil_singlegpu_baseline

bin/01_stencil_singlegpu_baseline: 01_stencil_singlegpu_baseline.cu | bin
	$(NVCC) $(ARCH) $(CFLAGS) -o $@ $<

bin:
	mkdir -p bin

run: all
	@echo "== bin/01_stencil_singlegpu_baseline =="
	@./bin/01_stencil_singlegpu_baseline
	@echo "NOTE: 02-05 require MPI/NCCL/NVSHMEM, not installed on this machine; not built. See README."

clean:
	rm -rf bin

.PHONY: all run clean
