#include <iostream>
#include <string>
#include <malloc.h>
#include <time.h> 
#include <sys/time.h>  
#include <chrono>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#ifdef EN_AVX
#include <x86intrin.h>
#endif
#include <boost/program_options.hpp>

#include "cBench.hpp"
#include "cProcess.hpp"

using namespace std;
using namespace fpga;

/* Def params */
constexpr auto const targetRegion = 0;
constexpr auto const defBenchRuns = 1;
constexpr auto const defAlign = pageSize;

/* Runtime */
constexpr auto const pageEntryNum = 1000;
constexpr auto const pageNumA = 10;
constexpr auto const pageNumB = 10;

constexpr auto const objBytes = 20;
constexpr auto const objBits = objBytes * 8;
constexpr auto const nObjPerAxi = 3;
constexpr auto const maxPageEntryNum = 10000;
constexpr auto const outBytes = 4UL * 1024UL * 1024UL * 1024UL;

constexpr auto const defBankA = 0;
constexpr auto const defBankB = 1;
constexpr auto const defBankOut = 1;

/**
 * @brief Reg map kernel controller
 * 
 */

constexpr auto const ADDR_AP_CTRL               = 0x00 >> 2;
constexpr auto const ADDR_GIE                   = 0x04 >> 2;
constexpr auto const ADDR_IER                   = 0x08 >> 2;
constexpr auto const ADDR_ISR                   = 0x0c >> 2;
constexpr auto const ADDR_PAGE_ENTRY_NUM_DATA_0 = 0x10 >> 2;
constexpr auto const ADDR_PAGE_ENTRY_NUM_CTRL   = 0x14 >> 2;
constexpr auto const ADDR_PAGE_NUM_A_DATA_0     = 0x18 >> 2;
constexpr auto const ADDR_PAGE_NUM_A_CTRL       = 0x1c >> 2;
constexpr auto const ADDR_PAGE_NUM_B_DATA_0     = 0x20 >> 2;
constexpr auto const ADDR_PAGE_NUM_B_CTRL       = 0x24 >> 2;
constexpr auto const ADDR_IN_PAGES_A_DATA_0     = 0x28 >> 2;
constexpr auto const ADDR_IN_PAGES_A_DATA_1     = 0x2c >> 2;
constexpr auto const ADDR_IN_PAGES_A_CTRL       = 0x30 >> 2;
constexpr auto const ADDR_IN_PAGES_B_DATA_0     = 0x34 >> 2;
constexpr auto const ADDR_IN_PAGES_B_DATA_1     = 0x38 >> 2;
constexpr auto const ADDR_IN_PAGES_B_CTRL       = 0x3c >> 2;
constexpr auto const ADDR_OUT_INTERSECT_DATA_0  = 0x40 >> 2;
constexpr auto const ADDR_OUT_INTERSECT_DATA_1  = 0x44 >> 2;
constexpr auto const ADDR_OUT_INTERSECT_CTRL    = 0x48 >> 2;

constexpr auto const krnlStart = 0x1;
constexpr auto const krnlStatus = 0x2;

/**
 * @brief Aligned allocation
 * 
 */
template <typename T>
struct aligned_allocator {
  using value_type = T;
  T* allocate(std::size_t num) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num) {
    free(p);
  }
};

/**
 * @brief Vector simple kernel
 * 
 */
int main(int argc, char *argv[])  
{
    // ---------------------------------------------------------------
    // Args 
    // ---------------------------------------------------------------

    // Read arguments
    boost::program_options::options_description programDescription("Options:");
    programDescription.add_options()
        ("runs,r", boost::program_options::value<uint32_t>(), "Number of  bench runs");
    
    boost::program_options::variables_map commandLineArgs;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, programDescription), commandLineArgs);
    boost::program_options::notify(commandLineArgs);

    uint32_t n_bench_runs = defBenchRuns;

    if(commandLineArgs.count("runs") > 0) n_bench_runs = commandLineArgs["runs"].as<uint32_t>();

    uint64_t bytes_per_page = pageEntryNum % nObjPerAxi == 0 ?
        64 * pageEntryNum / nObjPerAxi : 
        64 * (pageEntryNum / nObjPerAxi + 1); 
    uint64_t bytes_page_A = pageNumA * bytes_per_page;
    uint64_t bytes_page_B = pageNumA * bytes_per_page;

    uint64_t bytes_out = outBytes;

    PR_HEADER("PARAMS");
    std::cout << "Number of bench runs: " << n_bench_runs << std::endl;
    std::cout << "Bytes page A: " << bytes_page_A << std::endl;
    std::cout << "Bytes page B: " << bytes_page_B << std::endl;
    std::cout << "Bytes output: " << bytes_out << std::endl << std::endl;

    // ---------------------------------------------------------------
    // Init 
    // ---------------------------------------------------------------

    // Handles
    cProcess cproc(targetRegion, getpid());

    // Vectors
    std::vector<int ,aligned_allocator<int>> in_pages_A(bytes_page_A / sizeof(int));
    std::vector<int ,aligned_allocator<int>> in_pages_B(bytes_page_B / sizeof(int));
    std::vector<int64_t ,aligned_allocator<int64_t>> out_pages(bytes_out / sizeof(int64_t));

    // Offload (just empty?)
    uint64_t in_pages_A_pc = cproc.offload(in_pages_A.data(), bytes_page_A, defBankA);
    uint64_t in_pages_B_pc = cproc.offload(in_pages_B.data(), bytes_page_B, defBankB);
    uint64_t out_pages_pc = cproc.offload(out_pages.data(), bytes_out, defBankOut);
    std::cout << "Offload completed, pc_A: " << std::hex << in_pages_A_pc <<
        ", pc_B: " << in_pages_B_pc << ", pc_out: " << out_pages_pc << std::dec << std::endl;

    // Args (the way these registers are ordered is something else, Xilinx is great ...)
    cproc.setCSR(pageEntryNum, ADDR_PAGE_ENTRY_NUM_DATA_0);
    cproc.setCSR(pageNumA, ADDR_PAGE_NUM_A_DATA_0);
    cproc.setCSR(pageNumB, ADDR_PAGE_NUM_B_DATA_0);
    cproc.setCSR(LOW_32(in_pages_A_pc), ADDR_IN_PAGES_A_DATA_0);
    cproc.setCSR(HIGH_32(in_pages_A_pc), ADDR_IN_PAGES_A_DATA_1);
    cproc.setCSR(LOW_32(in_pages_B_pc), ADDR_IN_PAGES_B_DATA_0);
    cproc.setCSR(HIGH_32(in_pages_B_pc), ADDR_IN_PAGES_B_DATA_1);
    cproc.setCSR(LOW_32(out_pages_pc), ADDR_OUT_INTERSECT_DATA_0);
    cproc.setCSR(HIGH_32(out_pages_pc), ADDR_OUT_INTERSECT_DATA_1);

    // ---------------------------------------------------------------
    // Runs 
    // ---------------------------------------------------------------
    
    // Launch
    std::cout << "Launching kernel ..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    cproc.setCSR(krnlStart, ADDR_AP_CTRL);

    // Poll
    while((cproc.getCSR(ADDR_AP_CTRL) & krnlStatus) != krnlStatus)
        nanosleep((const struct timespec[]){{0, 100L}}, NULL);

    auto end = std::chrono::high_resolution_clock::now();
    double duration = (std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000.0);
    std::cout << "Duration: " << duration << " sec" << std::endl; 

    // Sync
    cproc.sync(out_pages.data());

    std::cout << "Intersect pair number: " << out_pages[0] << std::endl;
    std::cout << "Overall page per second = " << (pageNumA * pageNumB) / duration << std::endl;
    std::cout << "Number of comparison (and potentially insertion) per second = " << 
        (pageNumA * pageNumB) * (pageEntryNum * pageEntryNum) / duration << std::endl;

    std::cout << "TEST FINISHED (NO RESULT CHECK)" << std::endl << std::endl; 
    
    // ---------------------------------------------------------------
    // Release 
    // ---------------------------------------------------------------

    cproc.printDebug();

    return EXIT_SUCCESS;
}
