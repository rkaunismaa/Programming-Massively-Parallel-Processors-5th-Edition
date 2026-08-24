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

run: all
	@for b in $(BINS); do echo "== $$b =="; ./$$b || exit 1; done

clean:
	rm -rf bin

.PHONY: all run clean
