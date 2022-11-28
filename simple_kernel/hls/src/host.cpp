#include <stdint.h>
#include <chrono>

#include "host.hpp"

#include "constants.hpp"
// Wenqi: seems 2022.1 somehow does not support linking ap_uint.h to host?
// #include "ap_uint.h"


int main(int argc, char** argv)
{
    cl_int err;
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr 
    // is used if it is properly aligned. when not aligned, runtime had no choice but to create
    // its own host side buffer. So it is recommended to use this allocator if user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will 
    // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR 

    std::cout << "Allocating memory...\n";

    // in init
    size_t page_entry_num = 1000;
    size_t page_num_A = 10; 
    size_t page_num_B = 10; 

    // number of 512-bit entries that a page consumes 
    size_t bytes_per_page = page_entry_num % N_OBJ_PER_AXI == 0?
        64 * page_entry_num / N_OBJ_PER_AXI : 64 * (page_entry_num / N_OBJ_PER_AXI + 1);
    size_t bytes_page_A = page_num_A * bytes_per_page;
    size_t bytes_page_B = page_num_B * bytes_per_page;

    std::cout << "bytes_page_A: " << bytes_page_A << std::endl;
    std::cout << "bytes_page_B: " << bytes_page_B << std::endl;
    std::vector<int ,aligned_allocator<int>> in_pages_A(bytes_page_A / sizeof(int));
    std::vector<int ,aligned_allocator<int>> in_pages_B(bytes_page_B / sizeof(int));

    // size_t out_bytes = 10 * 1024 * 1024;
    size_t out_bytes = 4 * size_t(1000) * size_t(1000) * size_t(1000); // no more than 16 GB
    std::cout << "out_bytes: " << out_bytes << std::endl;
    std::vector<int64_t ,aligned_allocator<int64_t>> out(out_bytes / sizeof(int64_t));

// OPENCL HOST CODE AREA START

    std::vector<cl::Device> devices = get_devices();
    cl::Device device = devices[0];
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    //Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue q(context, device);

    // Import XCLBIN
    xclbin_file_name = argv[1];
    cl::Program::Binaries vadd_bins = import_binary_file();

    // Program and Kernel
    devices.resize(1);
    cl::Program program(context, devices, vadd_bins);
    cl::Kernel krnl_vector_add(program, "vadd");

    std::cout << "Finish loading bitstream...\n";
    // in init 
    OCL_CHECK(err, cl::Buffer buffer_in_pages_A   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            bytes_page_A, in_pages_A.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in_pages_B   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            bytes_page_B, in_pages_B.data(), &err));
	// out
    OCL_CHECK(err, cl::Buffer buffer_out(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            out_bytes, out.data(), &err));

    std::cout << "Finish allocate buffer...\n";

    int arg_counter = 0;    
    // in 
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(page_entry_num)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(page_num_A)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(page_num_B)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_in_pages_A));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_in_pages_B));

    // out
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_out));


    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
        // in
        buffer_in_pages_A,
        buffer_in_pages_B,
        buffer_out
        },0/* 0 means from host*/));

    std::cout << "Launching kernel...\n";
    // Launch the Kernel
    auto start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out},CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

    auto end = std::chrono::high_resolution_clock::now();
    double duration = (std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000.0);

    std::cout << "Duration (including memcpy out): " << duration << " sec" << std::endl; 

    std::cout << "Intersect pair number: " << out[0] << std::endl;
    std::cout << "Overall page per second = " << (page_num_A * page_num_B) / duration << std::endl;
    std::cout << "Number of comparison (and potentially insertion) per second = " << 
        (page_num_A * page_num_B) * (page_entry_num * page_entry_num) / duration << std::endl;

    std::cout << "TEST FINISHED (NO RESULT CHECK)" << std::endl; 

    return  0;
}
