NVCC := nvcc
# Debug symbols on by default so `make` / `make run` can be stepped through
# with cuda-gdb. Pass DEBUG=0 for an optimized, symbol-free build.
DEBUG ?= 1
ifeq ($(DEBUG),1)
CFLAGS := -O0 -g -G -std=c++17 -I../../common
else
CFLAGS := -O2 -std=c++17 -I../../common
endif
SM75_SRCS := 01_coalesced_vs_uncoalesced_access.cu 02_vectorized_loads_float4.cu 03_shared_memory_bank_conflicts.cu 04_thread_coarsening.cu 05_loop_unrolling.cu
SM80_SRCS := 06_double_buffering_async_copy.cu
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
