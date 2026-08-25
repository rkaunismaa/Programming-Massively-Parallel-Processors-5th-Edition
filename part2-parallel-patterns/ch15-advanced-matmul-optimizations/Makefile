NVCC := nvcc
# Debug symbols on by default so `make` / `make run` can be stepped through
# with cuda-gdb. Pass DEBUG=0 for an optimized, symbol-free build.
DEBUG ?= 1
ifeq ($(DEBUG),1)
CFLAGS := -O0 -g -G -std=c++17 -I../../common
else
CFLAGS := -O2 -std=c++17 -I../../common
endif
SM75_SRCS := 01_matmul_coarsened_larger_tiles.cu 02_matmul_register_tiled.cu 03_matmul_coalesced_output_store.cu 04_matmul_bank_conflict_free.cu
SM80_SRCS := 05_matmul_software_pipelined.cu
SM75_BINS := $(patsubst %.cu,bin/%,$(SM75_SRCS))
SM80_BINS := $(patsubst %.cu,bin/%,$(SM80_SRCS))
BINS := $(SM75_BINS) $(SM80_BINS)

all: $(BINS)

$(SM75_BINS): bin/%: %.cu | bin
	$(NVCC) -arch=sm_75 $(CFLAGS) -o $@ $<

$(SM80_BINS): bin/%: %.cu | bin
	$(NVCC) -arch=sm_80 $(CFLAGS) -o $@ $<

bin:
	mkdir -p bin

run: all
	@for b in $(BINS); do echo "== $$b =="; ./$$b || exit 1; done

clean:
	rm -rf bin

.PHONY: all run clean
