NVCC := nvcc
ARCH := -arch=sm_75
CFLAGS := -O2 -std=c++17 -I../../common

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
