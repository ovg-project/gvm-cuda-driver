# Intercept Layer for NVIDIA CUDA Driver
This is the source release of the Intercept Layer for NVIDIA CUDA Driver, tested with CUDA Driver 12.9 and GPU Driver 575.57.08.

## How to build
Easiest way to build:
```
make
```

To specify output dir of build:
```
make BUILD=<path to dir>
```

To specify CUDA driver the Intercept Layer is attaching to:
```
make CUDA=<path to cuda driver>
```

## How to install
Easiest way to install:
```
make install
```

To specify output dir of install:
```
make install INSTALL=<path to dir>
```
Note that is will first backup libcuda.so if exists in specified install dir, then remove all symlinks to the existing cuda driver in that driver if exists.

To specify CUDA driver the Intercept Layer is attaching to:
```
make install CUDA=<path to cuda driver>
```

## How to use
If you choose to replace existing cuda driver using:
```
sudo make install INSTALL=$(dirname $(whereis libcuda.so | awk '{print $2}'))
```
You should be able to work with any CUDA programs.
We've tested with `vllm`, `sglang`, `diffuser`, `llama-factory`, `llama.cpp`.

If you choose to put Intercept Layer somewhere else to keep your system clean:
```
make install INSTALL=<path to dir>
```
You can work with any CUDA programs using:
```
LD_LIBRARY_PATH=<path to dir>:$LD_LIBRARY_PATH <cuda programms>
```

For example:
```
LD_LIBRARY_PATH=<path to dir>:$LD_LIBRARY_PATH vllm serve meta-llama/Llama-3.2-3B
```
