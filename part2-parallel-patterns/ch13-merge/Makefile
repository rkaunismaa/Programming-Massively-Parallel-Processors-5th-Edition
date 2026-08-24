NVCC := nvcc
ARCH := -arch=sm_75
CFLAGS := -O2 -std=c++17 -I../../common
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
