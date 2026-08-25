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
SRCS := $(wildcard *.cu)
BINS := $(patsubst %.cu,bin/%,$(SRCS))

all: $(BINS)

bin/%: %.cu | bin
	$(NVCC) $(ARCH) $(CFLAGS) -o $@ $<

bin:
	mkdir -p bin

# §18.7's cooperative-groups grid-wide barrier (grid.sync() inside
# 05_bfs_cooperative_groups.cu) hangs in cudaDeviceSynchronize() when built
# with -G: forcing full device debug info collapses the simultaneous-SM-
# residency guarantee a cooperative-launch grid-wide sync depends on, so
# cudaDeviceGetAttribute(cudaDevAttrCooperativeLaunch, ...) still reports
# support but the kernel then deadlocks (confirmed: -O0 -g alone runs fine;
# adding -G reproduces the hang; -O2 runs fine). Build this one file without
# -G even in debug mode -- still with -g/-O0 for host-side debugging.
ifeq ($(DEBUG),1)
bin/05_bfs_cooperative_groups: 05_bfs_cooperative_groups.cu | bin
	$(NVCC) $(ARCH) -O0 -g -std=c++17 -I../../common -o $@ $<
endif

run: all
	@for b in $(BINS); do echo "== $$b =="; ./$$b || exit 1; done

clean:
	rm -rf bin

.PHONY: all run clean
