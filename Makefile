# Default BUILD, user can override: make BUILD=out
BUILD ?= build

# Default INSTALL, user can override: make INSTALL=ins
INSTALL ?= install

ifneq ($(MAKECMDGOALS),clean)
# CUDA: user can override, otherwise auto-detect with whereis
CUDA ?= $(shell whereis -b libcuda.so | awk '{print $$2}')

# Error if we still don't have anything
ifeq ($(strip $(CUDA)),)
$(error "libcuda.so not found; please run: make CUDA=/full/path/to/libcuda.so")
endif

# Always resolve CUDA through symlinks (even if user specified it)
# On Linux, readlink -f is easiest:
CUDA_RESOLVED := $(shell readlink -f "$(CUDA)")

# CUDAFILE = filename part of the resolved path
CUDAFILE := $(notdir $(CUDA_RESOLVED))
endif

.PHONY: all build install clean

all: build

build:
	mkdir -p $(BUILD)
	nvcc -g -arch sm_80 -c gvm.c -o $(BUILD)/gvm.o -Xcompiler "-Wno-implicit-function-declaration -fPIC"
	python3 tools/mapgen.py -obj $(BUILD)/gvm.o -map $(BUILD)/mapping.json
	python3 tools/symredef.py -in $(BUILD)/gvm.o -map $(BUILD)/mapping.json -t wrapper -out $(BUILD)/gvm.o.renamed
	python3 tools/symredef.py -in $(CUDA_RESOLVED) -map $(BUILD)/mapping.json -t cuda -out $(BUILD)/$(CUDAFILE).renamed
	patchelf --set-soname $(CUDAFILE).renamed $(BUILD)/$(CUDAFILE).renamed
	gcc -shared -o $(BUILD)/$(CUDAFILE) $(BUILD)/gvm.o.renamed $(BUILD)/$(CUDAFILE).renamed

install:
	mkdir -p $(INSTALL)
	mv $(INSTALL)/libcuda.so $(INSTALL)/libcuda.so.bak 2>/dev/null || true
	mv $(INSTALL)/libcuda.so.1 $(INSTALL)/libcuda.so.1.bak 2>/dev/null || true
	mv $(INSTALL)/$(CUDAFILE) $(INSTALL)/$(CUDAFILE).bak 2>/dev/null || true
	cp $(BUILD)/$(CUDAFILE).renamed $(INSTALL)/$(CUDAFILE).renamed
	cp $(BUILD)/$(CUDAFILE) $(INSTALL)/$(CUDAFILE)
	ln -s $(INSTALL)/$(CUDAFILE) $(INSTALL)/libcuda.so.1
	ln -s $(INSTALL)/libcuda.so.1 $(INSTALL)/libcuda.so

clean:
	rm -rf $(BUILD)
