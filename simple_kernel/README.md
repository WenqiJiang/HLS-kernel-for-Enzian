# simple_kernel

This folder contains the HLS kernel and the generated Verilog by Vitis HLS.

## Interface

HLS interface: https://github.com/WenqiJiang/HLS-kernel-for-Enzian/blob/main/simple_kernel/hls/src/vadd.cpp#L8-L28

It contains three arrays and three variables as arguments.

It has a configuration file which will be passed into Vivado, indicating the array-to-channel topology: https://github.com/WenqiJiang/HLS-kernel-for-Enzian/blob/main/simple_kernel/hls/connectivity.cfg

The generated Verilog interface: https://github.com/WenqiJiang/HLS-kernel-for-Enzian/blob/main/simple_kernel/verilog/vadd.v

The arrays are passed into the kernel as an argument representing their starting address, e.g., in the verilog file we have `wire   [63:0] in_pages_A;` for array `in_pages_A`.

