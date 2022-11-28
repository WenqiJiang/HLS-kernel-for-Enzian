#include "host.hpp"

#include <algorithm>
#include <chrono>
#include <unistd.h>
#include <limits>
#include <assert.h>
#include <string>
#include <stdint.h>

#include "constants.hpp"
// Wenqi: seems 2022.1 somehow does not support linking ap_uint.h to host?
// #include "ap_uint.h"

#include "./hnswlib/hnswlib.h"

// #define DEBUG

// boost::filesystem does not compile well, so implement this myself
std::string dir_concat(std::string dir1, std::string dir2) {
    if (dir1.back() != '/') {
        dir1 += '/';
    }
    return dir1 + dir2;
}

int main(int argc, char** argv)
{
    cl_int err;
    std::string db_name = "SBERT3000M";
    // std::string db_name = "SIFT100M";
    std::cout << "DB name: " << db_name << std::endl;
    
    std::string index_scan = "hnsw"; // hnsw or brute-force
    // std::string index_scan = "brute-force"; // hnsw or brute-force
    std::cout << "Index scan: " << index_scan << std::endl;

    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr 
    // is used if it is properly aligned. when not aligned, runtime had no choice but to create
    // its own host side buffer. So it is recommended to use this allocator if user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will 
    // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR 

    size_t query_num;
    size_t nlist;
    std::string data_dir_prefix;
    std::string raw_gt_vec_ID_suffix_dir;
    std::string raw_gt_dist_suffix_dir;
    std::string gnd_dir;
    std::string product_quantizer_dir_suffix;
    std::string query_vectors_dir_suffix;
    std::string vector_quantizer_dir_suffix;
    size_t raw_gt_vec_ID_size;
    size_t raw_gt_dist_size;
    size_t len_per_result; 
    size_t result_start_bias;
    if (strncmp(db_name.c_str(), "SBERT", 5) == 0) {
        if (db_name == "SBERT1000M") {
            nlist = 32768;
            data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SBERT1000M_IVF32768,PQ64_2shards/shard_0";
            raw_gt_vec_ID_suffix_dir = "gt_idx_1000M.ibin";
            raw_gt_dist_suffix_dir = "gt_dis_1000M.fbin";
            vector_quantizer_dir_suffix = "vector_quantizer_float32_32768_384_raw";
        }
        else if (db_name == "SBERT3000M") {
            nlist = 65536;
            data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SBERT3000M_IVF65536,PQ64_4shards/shard_0";
            raw_gt_vec_ID_suffix_dir = "gt_idx_3000M.ibin";
            raw_gt_dist_suffix_dir = "gt_dis_3000M.fbin";
            vector_quantizer_dir_suffix = "vector_quantizer_float32_65536_384_raw";
        }
        query_num = 10000;
        gnd_dir = "/mnt/scratch/wenqi/Faiss_experiments/sbert/";
        product_quantizer_dir_suffix = "product_quantizer_float32_64_256_6_raw";
        query_vectors_dir_suffix = "query_vectors_float32_10000_384_raw";
        raw_gt_vec_ID_size = (10000 * 1000 + 2) * sizeof(int);
        raw_gt_dist_size = (10000 * 1000 + 2) * sizeof(float);
        len_per_result = 1000;
        result_start_bias = 2;
    }

    std::cout << "Loading the first shard of the dataset... " << std::endl;

    ///////////     get data size from disk     //////////

    // PQ codes
    std::string PQ_codes_DRAM_0_dir_suffix("DDR_bank_0_PQ_raw");
    std::string PQ_codes_DRAM_0_dir = dir_concat(data_dir_prefix, PQ_codes_DRAM_0_dir_suffix);
    std::ifstream PQ_codes_DRAM_0_fstream(
        PQ_codes_DRAM_0_dir, 
        std::ios::in | std::ios::binary);
    PQ_codes_DRAM_0_fstream.seekg(0, PQ_codes_DRAM_0_fstream.end);
    size_t PQ_codes_DRAM_0_size =  PQ_codes_DRAM_0_fstream.tellg();
    if (!PQ_codes_DRAM_0_size) std::cout << "PQ_codes_DRAM_0_size is 0!";
    PQ_codes_DRAM_0_fstream.seekg(0, PQ_codes_DRAM_0_fstream.beg);

    std::string PQ_codes_DRAM_1_dir_suffix("DDR_bank_1_PQ_raw");
    std::string PQ_codes_DRAM_1_dir = dir_concat(data_dir_prefix, PQ_codes_DRAM_1_dir_suffix);
    std::ifstream PQ_codes_DRAM_1_fstream(
        PQ_codes_DRAM_1_dir, 
        std::ios::in | std::ios::binary);
    PQ_codes_DRAM_1_fstream.seekg(0, PQ_codes_DRAM_1_fstream.end);
    size_t PQ_codes_DRAM_1_size =  PQ_codes_DRAM_1_fstream.tellg();
    if (!PQ_codes_DRAM_1_size) std::cout << "PQ_codes_DRAM_1_size is 0!";
    PQ_codes_DRAM_1_fstream.seekg(0, PQ_codes_DRAM_1_fstream.beg);
    
    std::string PQ_codes_DRAM_2_dir_suffix("DDR_bank_2_PQ_raw");
    std::string PQ_codes_DRAM_2_dir = dir_concat(data_dir_prefix, PQ_codes_DRAM_2_dir_suffix);
    std::ifstream PQ_codes_DRAM_2_fstream(
        PQ_codes_DRAM_2_dir, 
        std::ios::in | std::ios::binary);
    PQ_codes_DRAM_2_fstream.seekg(0, PQ_codes_DRAM_2_fstream.end);
    size_t PQ_codes_DRAM_2_size =  PQ_codes_DRAM_2_fstream.tellg();
    if (!PQ_codes_DRAM_2_size) std::cout << "PQ_codes_DRAM_2_size is 0!";
    PQ_codes_DRAM_2_fstream.seekg(0, PQ_codes_DRAM_2_fstream.beg);
    
    std::string PQ_codes_DRAM_3_dir_suffix("DDR_bank_3_PQ_raw");
    std::string PQ_codes_DRAM_3_dir = dir_concat(data_dir_prefix, PQ_codes_DRAM_3_dir_suffix);
    std::ifstream PQ_codes_DRAM_3_fstream(
        PQ_codes_DRAM_3_dir, 
        std::ios::in | std::ios::binary);
    PQ_codes_DRAM_3_fstream.seekg(0, PQ_codes_DRAM_3_fstream.end);
    size_t PQ_codes_DRAM_3_size =  PQ_codes_DRAM_3_fstream.tellg();
    if (!PQ_codes_DRAM_3_size) std::cout << "PQ_codes_DRAM_3_size is 0!";
    PQ_codes_DRAM_3_fstream.seekg(0, PQ_codes_DRAM_3_fstream.beg);

    // vec IDs
    std::string vec_ID_DRAM_0_dir_suffix("DDR_bank_0_vec_ID_raw");
    std::string vec_ID_DRAM_0_dir = dir_concat(data_dir_prefix, vec_ID_DRAM_0_dir_suffix);
    std::ifstream vec_ID_DRAM_0_fstream(
        vec_ID_DRAM_0_dir, 
        std::ios::in | std::ios::binary);
    vec_ID_DRAM_0_fstream.seekg(0, vec_ID_DRAM_0_fstream.end);
    size_t vec_ID_DRAM_0_size =  vec_ID_DRAM_0_fstream.tellg();
    if (!vec_ID_DRAM_0_size) std::cout << "vec_ID_DRAM_0_size is 0!";
    vec_ID_DRAM_0_fstream.seekg(0, vec_ID_DRAM_0_fstream.beg);

    std::string vec_ID_DRAM_1_dir_suffix("DDR_bank_1_vec_ID_raw");
    std::string vec_ID_DRAM_1_dir = dir_concat(data_dir_prefix, vec_ID_DRAM_1_dir_suffix);
    std::ifstream vec_ID_DRAM_1_fstream(
        vec_ID_DRAM_1_dir, 
        std::ios::in | std::ios::binary);
    vec_ID_DRAM_1_fstream.seekg(0, vec_ID_DRAM_1_fstream.end);
    size_t vec_ID_DRAM_1_size =  vec_ID_DRAM_1_fstream.tellg();
    if (!vec_ID_DRAM_1_size) std::cout << "vec_ID_DRAM_1_size is 0!";
    vec_ID_DRAM_1_fstream.seekg(0, vec_ID_DRAM_1_fstream.beg);

    std::string vec_ID_DRAM_2_dir_suffix("DDR_bank_2_vec_ID_raw");
    std::string vec_ID_DRAM_2_dir = dir_concat(data_dir_prefix, vec_ID_DRAM_2_dir_suffix);
    std::ifstream vec_ID_DRAM_2_fstream(
        vec_ID_DRAM_2_dir, 
        std::ios::in | std::ios::binary);
    vec_ID_DRAM_2_fstream.seekg(0, vec_ID_DRAM_2_fstream.end);
    size_t vec_ID_DRAM_2_size =  vec_ID_DRAM_2_fstream.tellg();
    if (!vec_ID_DRAM_2_size) std::cout << "vec_ID_DRAM_2_size is 0!";
    vec_ID_DRAM_2_fstream.seekg(0, vec_ID_DRAM_2_fstream.beg);

    std::string vec_ID_DRAM_3_dir_suffix("DDR_bank_3_vec_ID_raw");
    std::string vec_ID_DRAM_3_dir = dir_concat(data_dir_prefix, vec_ID_DRAM_3_dir_suffix);
    std::ifstream vec_ID_DRAM_3_fstream(
        vec_ID_DRAM_3_dir, 
        std::ios::in | std::ios::binary);
    vec_ID_DRAM_3_fstream.seekg(0, vec_ID_DRAM_3_fstream.end);
    size_t vec_ID_DRAM_3_size =  vec_ID_DRAM_3_fstream.tellg();
    if (!vec_ID_DRAM_3_size) std::cout << "vec_ID_DRAM_3_size is 0!";
    vec_ID_DRAM_3_fstream.seekg(0, vec_ID_DRAM_3_fstream.beg);

    // control signals
    std::string nlist_PQ_codes_start_addr_dir_suffix("nlist_PQ_codes_start_addr");
    std::string nlist_PQ_codes_start_addr_dir = dir_concat(data_dir_prefix, nlist_PQ_codes_start_addr_dir_suffix);
    std::ifstream nlist_PQ_codes_start_addr_fstream(
        nlist_PQ_codes_start_addr_dir, 
        std::ios::in | std::ios::binary);
    nlist_PQ_codes_start_addr_fstream.seekg(0, nlist_PQ_codes_start_addr_fstream.end);
    size_t nlist_PQ_codes_start_addr_size =  nlist_PQ_codes_start_addr_fstream.tellg();
    if (!nlist_PQ_codes_start_addr_size) std::cout << "nlist_PQ_codes_start_addr_size is 0!";
    nlist_PQ_codes_start_addr_fstream.seekg(0, nlist_PQ_codes_start_addr_fstream.beg);

    std::string nlist_vec_ID_start_addr_dir_suffix("nlist_vec_ID_start_addr");
    std::string nlist_vec_ID_start_addr_dir = dir_concat(data_dir_prefix, nlist_vec_ID_start_addr_dir_suffix);
    std::ifstream nlist_vec_ID_start_addr_fstream(
        nlist_vec_ID_start_addr_dir, 
        std::ios::in | std::ios::binary);
    nlist_vec_ID_start_addr_fstream.seekg(0, nlist_vec_ID_start_addr_fstream.end);
    size_t nlist_vec_ID_start_addr_size =  nlist_vec_ID_start_addr_fstream.tellg();
    if (!nlist_vec_ID_start_addr_size) std::cout << "nlist_vec_ID_start_addr_size is 0!";
    nlist_vec_ID_start_addr_fstream.seekg(0, nlist_vec_ID_start_addr_fstream.beg);

    std::string nlist_num_vecs_dir_suffix("nlist_num_vecs");
    std::string nlist_num_vecs_dir = dir_concat(data_dir_prefix, nlist_num_vecs_dir_suffix);
    std::ifstream nlist_num_vecs_fstream(
        nlist_num_vecs_dir, 
        std::ios::in | std::ios::binary);
    nlist_num_vecs_fstream.seekg(0, nlist_num_vecs_fstream.end);
    size_t nlist_num_vecs_size =  nlist_num_vecs_fstream.tellg();
    if (!nlist_num_vecs_size) std::cout << "nlist_num_vecs_size is 0!";
    nlist_num_vecs_fstream.seekg(0, nlist_num_vecs_fstream.beg);

    std::string product_quantizer_dir = dir_concat(data_dir_prefix, product_quantizer_dir_suffix);
    std::ifstream product_quantizer_fstream(
        product_quantizer_dir, 
        std::ios::in | std::ios::binary);
    product_quantizer_fstream.seekg(0, product_quantizer_fstream.end);
    size_t product_quantizer_size =  product_quantizer_fstream.tellg();
    if (!product_quantizer_size) std::cout << "product_quantizer_size is 0!";
    product_quantizer_fstream.seekg(0, product_quantizer_fstream.beg);

    // ground truth 
    std::string raw_gt_vec_ID_dir = dir_concat(gnd_dir, raw_gt_vec_ID_suffix_dir);
    std::ifstream raw_gt_vec_ID_fstream(
        raw_gt_vec_ID_dir,
        std::ios::in | std::ios::binary);

    std::string raw_gt_dist_dir = dir_concat(gnd_dir, raw_gt_dist_suffix_dir);
    std::ifstream raw_gt_dist_fstream(
        raw_gt_dist_dir,
        std::ios::in | std::ios::binary);

    // info used to Select Cells to Scan

    std::string query_vectors_dir = dir_concat(data_dir_prefix, query_vectors_dir_suffix);
    std::ifstream query_vectors_fstream(
        query_vectors_dir, 
        std::ios::in | std::ios::binary);
    query_vectors_fstream.seekg(0, query_vectors_fstream.end);
    size_t query_vectors_size =  query_vectors_fstream.tellg();
    if (!query_vectors_size) std::cout << "query_vectors_size is 0!";
    query_vectors_fstream.seekg(0, query_vectors_fstream.beg);
    
    std::string vector_quantizer_dir = dir_concat(data_dir_prefix, vector_quantizer_dir_suffix);
    std::ifstream vector_quantizer_fstream(
        vector_quantizer_dir, 
        std::ios::in | std::ios::binary);
    vector_quantizer_fstream.seekg(0, vector_quantizer_fstream.end);
    size_t vector_quantizer_size =  vector_quantizer_fstream.tellg();
    if (!vector_quantizer_size) std::cout << "vector_quantizer_size is 0!";
    vector_quantizer_fstream.seekg(0, vector_quantizer_fstream.beg);

    //////////     Allocate Memory     //////////

    std::cout << "Allocating memory...\n";
    auto start_load = std::chrono::high_resolution_clock::now();

    // in init
    size_t nprobe = 32;

    assert (nprobe <= nlist);

    size_t meta_data_init_bytes = 3 * nlist * sizeof(int) + D * LUT_ENTRY_NUM * sizeof(float);

    assert(nlist * 4 ==  nlist_PQ_codes_start_addr_size);
    assert(nlist * 4 ==  nlist_vec_ID_start_addr_size);
    assert(nlist * 4 ==  nlist_num_vecs_size);
    assert(D * LUT_ENTRY_NUM * 4 == product_quantizer_size);

    std::vector<int ,aligned_allocator<int >> meta_data_init(meta_data_init_bytes / sizeof(int));

    // in runtime (should from network) in 512-bit packet
    size_t size_header = 1;
    size_t size_cell_IDs = nprobe * 4 % 64 == 0? nprobe * 4 / 64: nprobe * 4 / 64 + 1;
    size_t size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    size_t size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 

    size_t in_DRAM_bytes = query_num * 64 * (size_header + size_cell_IDs + size_query_vector + nprobe * size_center_vector);
    std::vector<int ,aligned_allocator<int >> in_DRAM(in_DRAM_bytes / sizeof(int));


    // in runtime (should from DRAM)
    std::vector<int ,aligned_allocator<int >> PQ_codes_DRAM_0(PQ_codes_DRAM_0_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> PQ_codes_DRAM_1(PQ_codes_DRAM_1_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> PQ_codes_DRAM_2(PQ_codes_DRAM_2_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> PQ_codes_DRAM_3(PQ_codes_DRAM_3_size / sizeof(int));

    std::vector<int ,aligned_allocator<int >> vec_ID_DRAM_0(vec_ID_DRAM_0_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> vec_ID_DRAM_1(vec_ID_DRAM_1_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> vec_ID_DRAM_2(vec_ID_DRAM_2_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> vec_ID_DRAM_3(vec_ID_DRAM_3_size / sizeof(int));
    
    // out
    // 128 is a random padding for network headers
    // header = 1 pkt
    size_t size_results_vec_ID = PRIORITY_QUEUE_LEN_L2 * 64 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 64 / 512 : PRIORITY_QUEUE_LEN_L2 * 64 / 512 + 1;
    size_t size_results_dist = PRIORITY_QUEUE_LEN_L2 * 32 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 32 / 512 : PRIORITY_QUEUE_LEN_L2 * 32 / 512 + 1;
    size_t size_results = 1 + size_results_vec_ID + size_results_dist; // in 512-bit packet
    size_t out_bytes = query_num * 64 * size_results;
    std::vector<int ,aligned_allocator<int>> out(out_bytes / sizeof(int));

    // the raw ground truth size is the same for idx_1M.ivecs, idx_10M.ivecs, idx_100M.ivecs
    // recall counts the very first nearest neighbor only
    size_t gt_vec_ID_size = 10000 * sizeof(int);
    std::vector<uint32_t, aligned_allocator<uint32_t>> raw_gt_vec_ID(raw_gt_vec_ID_size / sizeof(int), 0);
    std::vector<uint32_t, aligned_allocator<uint32_t>> gt_vec_ID(gt_vec_ID_size / sizeof(int), 0);
    
    size_t gt_dist_size = 10000 * sizeof(float);
    std::vector<float, aligned_allocator<float>> raw_gt_dist(raw_gt_dist_size / sizeof(float), 0);
    std::vector<float, aligned_allocator<float>> gt_dist(gt_dist_size / sizeof(float), 0);


    // on host side, used to Select Cells to Scan 
    std::vector<float ,aligned_allocator<float >> query_vectors(query_vectors_size / sizeof(float));
    std::vector<float ,aligned_allocator<float >> vector_quantizer(vector_quantizer_size / sizeof(float));


    //////////     load data from disk     //////////
    std::cout << "Loading data from disk...\n";

    // PQ codes
    char* PQ_codes_DRAM_0_char = (char*) malloc(PQ_codes_DRAM_0_size);
    PQ_codes_DRAM_0_fstream.read(PQ_codes_DRAM_0_char, PQ_codes_DRAM_0_size);
    if (!PQ_codes_DRAM_0_fstream) {
            std::cout << "error: only " << PQ_codes_DRAM_0_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&PQ_codes_DRAM_0[0], PQ_codes_DRAM_0_char, PQ_codes_DRAM_0_size);
    free(PQ_codes_DRAM_0_char);

    char* PQ_codes_DRAM_1_char = (char*) malloc(PQ_codes_DRAM_1_size);
    PQ_codes_DRAM_1_fstream.read(PQ_codes_DRAM_1_char, PQ_codes_DRAM_1_size);
    if (!PQ_codes_DRAM_1_fstream) {
            std::cout << "error: only " << PQ_codes_DRAM_1_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&PQ_codes_DRAM_1[0], PQ_codes_DRAM_1_char, PQ_codes_DRAM_1_size);
    free(PQ_codes_DRAM_1_char);

    char* PQ_codes_DRAM_2_char = (char*) malloc(PQ_codes_DRAM_2_size);
    PQ_codes_DRAM_2_fstream.read(PQ_codes_DRAM_2_char, PQ_codes_DRAM_2_size);
    if (!PQ_codes_DRAM_2_fstream) {
            std::cout << "error: only " << PQ_codes_DRAM_2_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&PQ_codes_DRAM_2[0], PQ_codes_DRAM_2_char, PQ_codes_DRAM_2_size);
    free(PQ_codes_DRAM_2_char);
    
    char* PQ_codes_DRAM_3_char = (char*) malloc(PQ_codes_DRAM_3_size);
    PQ_codes_DRAM_3_fstream.read(PQ_codes_DRAM_3_char, PQ_codes_DRAM_3_size);
    if (!PQ_codes_DRAM_3_fstream) {
            std::cout << "error: only " << PQ_codes_DRAM_3_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&PQ_codes_DRAM_3[0], PQ_codes_DRAM_3_char, PQ_codes_DRAM_3_size);
    free(PQ_codes_DRAM_3_char);

    // vec ID
    char* vec_ID_DRAM_0_char = (char*) malloc(vec_ID_DRAM_0_size);
    vec_ID_DRAM_0_fstream.read(vec_ID_DRAM_0_char, vec_ID_DRAM_0_size);
    if (!vec_ID_DRAM_0_fstream) {
            std::cout << "error: only " << vec_ID_DRAM_0_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&vec_ID_DRAM_0[0], vec_ID_DRAM_0_char, vec_ID_DRAM_0_size);
    free(vec_ID_DRAM_0_char);

    char* vec_ID_DRAM_1_char = (char*) malloc(vec_ID_DRAM_1_size);
    vec_ID_DRAM_1_fstream.read(vec_ID_DRAM_1_char, vec_ID_DRAM_1_size);
    if (!vec_ID_DRAM_1_fstream) {
            std::cout << "error: only " << vec_ID_DRAM_1_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&vec_ID_DRAM_1[0], vec_ID_DRAM_1_char, vec_ID_DRAM_1_size);
    free(vec_ID_DRAM_1_char);

    char* vec_ID_DRAM_2_char = (char*) malloc(vec_ID_DRAM_2_size);
    vec_ID_DRAM_2_fstream.read(vec_ID_DRAM_2_char, vec_ID_DRAM_2_size);
    if (!vec_ID_DRAM_2_fstream) {
            std::cout << "error: only " << vec_ID_DRAM_2_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&vec_ID_DRAM_2[0], vec_ID_DRAM_2_char, vec_ID_DRAM_2_size);
    free(vec_ID_DRAM_2_char);
    
    char* vec_ID_DRAM_3_char = (char*) malloc(vec_ID_DRAM_3_size);
    vec_ID_DRAM_3_fstream.read(vec_ID_DRAM_3_char, vec_ID_DRAM_3_size);
    if (!vec_ID_DRAM_3_fstream) {
            std::cout << "error: only " << vec_ID_DRAM_3_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&vec_ID_DRAM_3[0], vec_ID_DRAM_3_char, vec_ID_DRAM_3_size);
    free(vec_ID_DRAM_3_char);

    // control signals
    // meta_data_init = nlist_PQ_codes_start_addr, nlist_vec_ID_start_addr, nlist_num_vecs,
    char* nlist_PQ_codes_start_addr_char = (char*) malloc(nlist_PQ_codes_start_addr_size);
    nlist_PQ_codes_start_addr_fstream.read(nlist_PQ_codes_start_addr_char, nlist_PQ_codes_start_addr_size);
    if (!nlist_PQ_codes_start_addr_fstream) {
            std::cout << "error: only " << nlist_PQ_codes_start_addr_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&meta_data_init[0], nlist_PQ_codes_start_addr_char, nlist_PQ_codes_start_addr_size);
    free(nlist_PQ_codes_start_addr_char);

    char* nlist_vec_ID_start_addr_char = (char*) malloc(nlist_vec_ID_start_addr_size);
    nlist_vec_ID_start_addr_fstream.read(nlist_vec_ID_start_addr_char, nlist_vec_ID_start_addr_size);
    if (!nlist_vec_ID_start_addr_fstream) {
            std::cout << "error: only " << nlist_vec_ID_start_addr_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&meta_data_init[nlist], nlist_vec_ID_start_addr_char, nlist_vec_ID_start_addr_size);
    free(nlist_vec_ID_start_addr_char);
    
    char* nlist_num_vecs_char = (char*) malloc(nlist_num_vecs_size);
    nlist_num_vecs_fstream.read(nlist_num_vecs_char, nlist_num_vecs_size);
    if (!nlist_num_vecs_fstream) {
            std::cout << "error: only " << nlist_num_vecs_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&meta_data_init[2 * nlist], nlist_num_vecs_char, nlist_num_vecs_size);
    free(nlist_num_vecs_char);

    char* product_quantizer_char = (char*) malloc(product_quantizer_size);
    product_quantizer_fstream.read(product_quantizer_char, product_quantizer_size);
    if (!product_quantizer_fstream) {
            std::cout << "error: only " << product_quantizer_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&meta_data_init[3 * nlist], product_quantizer_char, product_quantizer_size);
    free(product_quantizer_char);

    // std::cout << "\n nlist_PQ_codes_start_addr (last cell): " << meta_data_init[1 * nlist - 1];
    // std::cout << "\n nlist_vec_ID_start_addr (last cell): " << meta_data_init[2 * nlist - 1];
    // std::cout << "\n nlist_num_vecs (last cell): " << meta_data_init[3 * nlist - 1];

    // // DEBUG: check whether PQ codes are correct
    // int cell = 5766;
    // int addr = meta_data_init[cell];
    // int addr_int_array = addr * 64 / 4;
    
    // std::cout << "cell ID for debug: " << cell << "\n";
    // std::cout << "PQ codes addr (AXI entry): " << addr << " ; in int array: " << addr_int_array << "\n";

    // unsigned char* PQ_code_tmp_0 = (unsigned char*) malloc(M);
    // unsigned char* PQ_code_tmp_1 = (unsigned char*) malloc(M);
    // unsigned char* PQ_code_tmp_2 = (unsigned char*) malloc(M);
    // unsigned char* PQ_code_tmp_3 = (unsigned char*) malloc(M);
    // memcpy(PQ_code_tmp_0, &PQ_codes_DRAM_0[addr_int_array], M);
    // memcpy(PQ_code_tmp_1, &PQ_codes_DRAM_1[addr_int_array], M);
    // memcpy(PQ_code_tmp_2, &PQ_codes_DRAM_2[addr_int_array], M);
    // memcpy(PQ_code_tmp_3, &PQ_codes_DRAM_3[addr_int_array], M);

    // std::cout << "\nCode channel 0:";
    // for (int i = 0; i < M; i ++) {
    //     unsigned int tmp = PQ_code_tmp_0[i];
    //     std::cout << " " << tmp;
    // }
    // std::cout << "\nCode channel 1:";
    // for (int i = 0; i < M; i ++) {
    //     unsigned int tmp = PQ_code_tmp_1[i];
    //     std::cout << " " << tmp;
    // }
    // std::cout << "\nCode channel 2:";
    // for (int i = 0; i < M; i ++) {
    //     unsigned int tmp = PQ_code_tmp_2[i];
    //     std::cout << " " << tmp;
    // }
    // std::cout << "\nCode channel 3:";
    // for (int i = 0; i < M; i ++) {
    //     unsigned int tmp = PQ_code_tmp_3[i];
    //     std::cout << " " << tmp;
    // }

    // ground truth
    char* raw_gt_vec_ID_char = (char*) malloc(raw_gt_vec_ID_size);
    raw_gt_vec_ID_fstream.read(raw_gt_vec_ID_char, raw_gt_vec_ID_size);
    if (!raw_gt_vec_ID_fstream) {
        std::cout << "error: only " << raw_gt_vec_ID_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&raw_gt_vec_ID[0], raw_gt_vec_ID_char, raw_gt_vec_ID_size);
    free(raw_gt_vec_ID_char);

    for (int i = 0; i < 10000; i++) {
        gt_vec_ID[i] = raw_gt_vec_ID[i * len_per_result + result_start_bias];
    }

    char* raw_gt_dist_char = (char*) malloc(raw_gt_dist_size);
    raw_gt_dist_fstream.read(raw_gt_dist_char, raw_gt_dist_size);
    if (!raw_gt_dist_fstream) {
        std::cout << "error: only " << raw_gt_dist_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&raw_gt_dist[0], raw_gt_dist_char, raw_gt_dist_size);
    free(raw_gt_dist_char);

    for (int i = 0; i < 10000; i++) {
        gt_dist[i] = raw_gt_dist[i * len_per_result + result_start_bias];
    }

    // on host, used to Select Cells to Scan
    char* query_vectors_char = (char*) malloc(query_vectors_size);
    query_vectors_fstream.read(query_vectors_char, query_vectors_size);
    if (!query_vectors_fstream) {
            std::cout << "error: only " << query_vectors_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&query_vectors[0], query_vectors_char, query_vectors_size);
    free(query_vectors_char);
    
    char* vector_quantizer_char = (char*) malloc(vector_quantizer_size);
    vector_quantizer_fstream.read(vector_quantizer_char, vector_quantizer_size);
    if (!vector_quantizer_fstream) {
            std::cout << "error: only " << vector_quantizer_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&vector_quantizer[0], vector_quantizer_char, vector_quantizer_size);
    free(vector_quantizer_char);

    assert(D * nlist * sizeof(float) == vector_quantizer_size);
    assert(D * 256 * sizeof(float) == product_quantizer_size);

    auto end_load = std::chrono::high_resolution_clock::now();
    double duration_load = (std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load).count());

    std::cout << "Duration memory allocation & disk load: " << duration_load << " ms" << std::endl; 


    //////////     Select Cells to Scan     //////////
   
    // HNSWlib index
    hnswlib::AlgorithmInterface<float>* alg_hnswlib;
    hnswlib::L2Space space(D);
    if (index_scan == "brute_force") {
        std::string brute_force_index_dir = dir_concat(data_dir_prefix, "hnswlib_brute_force_index.bin");
        std::ifstream f_hnswlib(brute_force_index_dir);
        bool hnswlib_index_exists = f_hnswlib.good();
        if (hnswlib_index_exists) {
            std::cout << "brute_force Index exists, loading index..." << std::endl;
            alg_hnswlib = new hnswlib::BruteforceSearch<float>(&space, brute_force_index_dir);
        }
        else {
            std::cout << "brute_force Index does not exist, creating new index..." << std::endl;
            alg_hnswlib = new hnswlib::BruteforceSearch<float>(&space, nlist);
            std::cout << "Adding data..." << std::endl;
            for (size_t i = 0; i < nlist; ++i) {
                alg_hnswlib->addPoint(vector_quantizer.data() + D * i, i);
            }
            alg_hnswlib->saveIndex(brute_force_index_dir);
        }
    } else if (index_scan == "hnsw") {
        std::string hnsw_index_dir = dir_concat(data_dir_prefix, "hnswlib_hnsw_index.bin");
        std::ifstream f_hnswlib(hnsw_index_dir);
        bool hnswlib_index_exists = f_hnswlib.good();
        if (hnswlib_index_exists) {
            std::cout << "HNSW Index exists, loading index..." << std::endl;
            alg_hnswlib = new hnswlib::HierarchicalNSW<float>(&space, hnsw_index_dir);
        }
        else {
            std::cout << "HNSW Index does not exist, creating new index..." << std::endl;
            size_t ef_construction = 128;
            alg_hnswlib = new hnswlib::HierarchicalNSW<float>(&space, nlist, ef_construction);
            std::cout << "Adding data..." << std::endl;
            for (size_t i = 0; i < nlist; ++i) {
                alg_hnswlib->addPoint(vector_quantizer.data() + D * i, i);
            }
            alg_hnswlib->saveIndex(hnsw_index_dir);
        }
    } else {
        std::cout << "index option does not exists, either brute_force or hnsw" << std::endl;
        exit(1);
    }

    std::cout << "selecting cells to scan..." << std::endl;

    auto start_LUT = std::chrono::high_resolution_clock::now();

    std::vector<std::pair <float, int>> dist_array(nlist); // (dist, cell_ID)

    // select cells to scan
    for (size_t query_id = 0; query_id < query_num; query_id++) {

        size_t start_addr_in_DRAM_per_query = query_id * (size_header + size_cell_IDs + size_query_vector + nprobe * size_center_vector) * 64 / sizeof(int);
        size_t start_addr_in_DRAM_cell_ID = start_addr_in_DRAM_per_query + size_header * 64 / sizeof(int);
        size_t start_addr_in_DRAM_query_vector = start_addr_in_DRAM_per_query + (size_header + size_cell_IDs) * 64 / sizeof(int);
        size_t start_addr_in_DRAM_center_vector = start_addr_in_DRAM_per_query + (size_header + size_cell_IDs + size_query_vector) * 64 / sizeof(int);

        const void* p = query_vectors.data() + query_id * D;
        // searchKNN return type: std::priority_queue<std::pair<dist_t, labeltype >>
        auto gd = alg_hnswlib->searchKnn(p, nprobe);
        assert(gd.size() == nprobe);
        int cnt = 0;
        while (!gd.empty()) {
            dist_array[cnt] = std::make_pair(gd.top().first, int(gd.top().second));
            gd.pop();
            cnt++;
        }

        // for (size_t nlist_id = 0; nlist_id < nlist; nlist_id ++) {

        //     float dist = 0;
        //     for (size_t d = 0; d < D; d++) {
        //         dist += 
        //             (query_vectors[query_id * D + d] - vector_quantizer[nlist_id * D + d]) *
        //             (query_vectors[query_id * D + d] - vector_quantizer[nlist_id * D + d]);
        //     }
        //     dist_array[nlist_id] = std::make_pair(dist, nlist_id);
        //     // std::cout << "dist: " << dist << " cell ID: " << nlist_id << "\n";
        // }
//         // select cells
//         std::nth_element(dist_array.begin(), dist_array.begin() + nprobe, dist_array.end()); // get nprobe smallest
//         std::sort(dist_array.begin(), dist_array.begin() + nprobe); // sort first nprobe
            
        // write cell ID
        for (size_t n = 0; n < nprobe; n++) {
            in_DRAM[start_addr_in_DRAM_cell_ID + n] = dist_array[n].second;
#ifdef DEBUG
            std::cout << "dist: " << dist_array[n].first << " cell ID: " << dist_array[n].second << "\n";
#endif
        } 

        // write query vector
	    memcpy(&in_DRAM[start_addr_in_DRAM_query_vector], &query_vectors[query_id * D], D * sizeof(float));

        // write center vectors      
        for (size_t nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            size_t cell_ID = dist_array[nprobe_id].second;
	        memcpy(&in_DRAM[start_addr_in_DRAM_center_vector + nprobe_id * D], 
                   &vector_quantizer[cell_ID * D], D * sizeof(float));
        }        
    }
    auto end_LUT = std::chrono::high_resolution_clock::now();
    double duration_LUT = (std::chrono::duration_cast<std::chrono::milliseconds>(end_LUT - start_LUT).count());

    std::cout << "Duration Select Cells to Scan: " << duration_LUT << " ms" << std::endl; 

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

	// ------------------------------------------------------------------
	// Create Buffers in Global Memory to store data
	//             o) buffer_in1 - stores source_in1
	//             o) buffer_in2 - stores source_in2
	//             o) buffer_ouput - stores Results
	// ------------------------------------------------------------------	
	

    // in init 
    OCL_CHECK(err, cl::Buffer buffer_meta_data_init   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            meta_data_init_bytes, meta_data_init.data(), &err));

	// in runtime (should from network)
    OCL_CHECK(err, cl::Buffer buffer_in_DRAM   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            in_DRAM_bytes, in_DRAM.data(), &err));

    // in runtime (should from DRAM)
    OCL_CHECK(err, cl::Buffer buffer_PQ_codes_DRAM_0   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            PQ_codes_DRAM_0_size, PQ_codes_DRAM_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_PQ_codes_DRAM_1   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            PQ_codes_DRAM_1_size, PQ_codes_DRAM_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_PQ_codes_DRAM_2   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            PQ_codes_DRAM_2_size, PQ_codes_DRAM_2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_PQ_codes_DRAM_3   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            PQ_codes_DRAM_3_size, PQ_codes_DRAM_3.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in_vec_ID_DRAM_0   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vec_ID_DRAM_0_size, vec_ID_DRAM_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in_vec_ID_DRAM_1   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vec_ID_DRAM_1_size, vec_ID_DRAM_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in_vec_ID_DRAM_2   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vec_ID_DRAM_2_size, vec_ID_DRAM_2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in_vec_ID_DRAM_3   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vec_ID_DRAM_3_size, vec_ID_DRAM_3.data(), &err));

	// out
    OCL_CHECK(err, cl::Buffer buffer_out(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            out_bytes, out.data(), &err));


    // in init
    OCL_CHECK(err, err = krnl_vector_add.setArg(0, int(query_num)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(1, int(nlist)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(2, int(nprobe)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(3, buffer_meta_data_init));

    // in runtime (should from network)
    OCL_CHECK(err, err = krnl_vector_add.setArg(4, buffer_in_DRAM));

    // in runtime (should from DRAM)
    OCL_CHECK(err, err = krnl_vector_add.setArg(5, buffer_PQ_codes_DRAM_0));
    OCL_CHECK(err, err = krnl_vector_add.setArg(6, buffer_PQ_codes_DRAM_1));
    OCL_CHECK(err, err = krnl_vector_add.setArg(7, buffer_PQ_codes_DRAM_2));
    OCL_CHECK(err, err = krnl_vector_add.setArg(8, buffer_PQ_codes_DRAM_3));
    OCL_CHECK(err, err = krnl_vector_add.setArg(9, buffer_in_vec_ID_DRAM_0));
    OCL_CHECK(err, err = krnl_vector_add.setArg(10, buffer_in_vec_ID_DRAM_1));
    OCL_CHECK(err, err = krnl_vector_add.setArg(11, buffer_in_vec_ID_DRAM_2));
    OCL_CHECK(err, err = krnl_vector_add.setArg(12, buffer_in_vec_ID_DRAM_3));

    // out
    OCL_CHECK(err, err = krnl_vector_add.setArg(13, buffer_out));



    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
        // in init
        buffer_meta_data_init,
        // in runtime (should from network)
        buffer_in_DRAM, 
        // in runtime (should from DRAM)
        buffer_PQ_codes_DRAM_0,
        buffer_PQ_codes_DRAM_1,
        buffer_PQ_codes_DRAM_2,
        buffer_PQ_codes_DRAM_3,
        buffer_in_vec_ID_DRAM_0, 
        buffer_in_vec_ID_DRAM_1, 
        buffer_in_vec_ID_DRAM_2, 
        buffer_in_vec_ID_DRAM_3},0/* 0 means from host*/));

    std::cout << "Launching FPGA task...\n";
    auto start_kernel = std::chrono::high_resolution_clock::now();
    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out},CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    
    auto end_kernel = std::chrono::high_resolution_clock::now();
    double duration_kernel = (std::chrono::duration_cast<std::chrono::milliseconds>(end_kernel - start_kernel).count());

    std::cout << "Duration FPGA kernel (include memcpy back): " << duration_kernel << " millisec" << std::endl; 

// OPENCL HOST CODE AREA END

    // std::cout << "all outs (in int): \n";
    // for (int i = 0; i < out_bytes / sizeof(int); i++) {
    //     std::cout << " " << out[i] << " ";
    // }
    // std::cout << "end all outs (in int): \n";

    // Compare the results of the Device to the ground truth
    std::cout << "Comparing Results..." << std::endl;
    
    int match_count_R1_at_1 = 0;
    int match_count_R1_at_10 = 0;
    int match_count_R1_at_100 = 0;
    int match_count_R_at_1 = 0;
    int match_count_R_at_10 = 0;
    int match_count_R_at_100 = 0;

    for (int query_id = 0; query_id < query_num; query_id++) {

#ifdef DEBUG
        std::cout << "query ID: " << query_id << std::endl;
#endif


        std::vector<long> hw_result_vec_ID_partial(TOPK, 0);
        std::vector<float> hw_result_dist_partial(TOPK, 0);
        std::vector<std::pair<float, int>> hw_result_pair(TOPK);

        int start_result_vec_ID_addr = (query_id * size_results + 1) * 64 / sizeof(int);
        int start_result_dist_addr = (query_id * size_results + 1 + size_results_vec_ID) * 64 / sizeof(int);

        // Load data
        memcpy(&hw_result_vec_ID_partial[0], &out[start_result_vec_ID_addr], 8 * TOPK);
        memcpy(&hw_result_dist_partial[0], &out[start_result_dist_addr], 4 * TOPK);
        for (int k = 0; k < TOPK; k++) {
            hw_result_pair[k] = std::make_pair(hw_result_dist_partial[k], hw_result_vec_ID_partial[k]);
        }
        std::sort(hw_result_pair.begin(), hw_result_pair.end());
        for (int k = 0; k < TOPK; k++) {
            hw_result_dist_partial[k] = hw_result_pair[k].first;
            hw_result_vec_ID_partial[k] = hw_result_pair[k].second;
        }
        

        int start_addr_gt = query_id * len_per_result + result_start_bias;

        // R1@K
        for (int k = 0; k < 1; k++) {
            if (hw_result_vec_ID_partial[k] == raw_gt_vec_ID[start_addr_gt]) {
                match_count_R1_at_1++;
                break;
            }
        } 
        for (int k = 0; k < 10; k++) {
            if (hw_result_vec_ID_partial[k] == raw_gt_vec_ID[start_addr_gt]) {
                match_count_R1_at_10++;
                break;
            }
        } 
        for (int k = 0; k < TOPK; k++) {
#ifdef DEBUG
            // std::cout << "hw: " << hw_result_vec_ID_partial[k] << " gt: " << gt_vec_ID[query_id] << 
            //     "hw dist: " << hw_result_dist_partial[k] << " gt dist: " << gt_dist[query_id] << std::endl;
#endif 
            if (hw_result_vec_ID_partial[k] == raw_gt_vec_ID[start_addr_gt]) {
                match_count_R1_at_100++;
                break;
            }
        } 

        // R@K
        std::unordered_set<size_t> gt_set;
        for (int k = 0; k < 1; k++) {
            gt_set.insert(raw_gt_vec_ID[start_addr_gt + k]);
        }
        for (int k = 0; k < 1; k++) {
            // count actually means contain here...
            // https://stackoverflow.com/questions/42532550/why-does-stdset-not-have-a-contains-member-function
            if (gt_set.count(hw_result_vec_ID_partial[k])) { 
                match_count_R_at_1++;
            }
        }
        for (int k = 0; k < 10; k++) {
            gt_set.insert(raw_gt_vec_ID[start_addr_gt + k]);
        }
        for (int k = 0; k < 10; k++) {
            // count actually means contain here...
            // https://stackoverflow.com/questions/42532550/why-does-stdset-not-have-a-contains-member-function
            if (gt_set.count(hw_result_vec_ID_partial[k])) { 
                match_count_R_at_10++;
            }
        }
        for (int k = 0; k < 100; k++) {
            gt_set.insert(raw_gt_vec_ID[start_addr_gt + k]);
        }
        for (int k = 0; k < 100; k++) {
            // count actually means contain here...
            // https://stackoverflow.com/questions/42532550/why-does-stdset-not-have-a-contains-member-function
            if (gt_set.count(hw_result_vec_ID_partial[k])) { 
                match_count_R_at_100++;
            }
        }
    }

    std::cout << "R1@1: " << float(match_count_R1_at_1) / query_num << std::endl;
    std::cout << "R1@10: " << float(match_count_R1_at_10) / query_num << std::endl;
    std::cout << "R1@100: " << float(match_count_R1_at_100) / query_num << std::endl;
    std::cout << "R@1: " << float(match_count_R_at_1) / (query_num * 1) << std::endl;
    std::cout << "R@10: " << float(match_count_R_at_10) / (query_num * 10) << std::endl;
    std::cout << "R@100: " << float(match_count_R_at_100) / (query_num * 100) << std::endl;


#ifdef DEBUG
    for (int query_id = 0; query_id < query_num; query_id++) {

        std::vector<long> hw_result_vec_ID_partial(TOPK, 0);
        std::vector<float> hw_result_dist_partial(TOPK, 0);

        int start_result_vec_ID_addr = (query_id * size_results + 1) * 64 / sizeof(int);
        int start_result_dist_addr = (query_id * size_results + 1 + size_results_vec_ID) * 64 / sizeof(int);

        // Load data
        memcpy(&hw_result_vec_ID_partial[0], &out[start_result_vec_ID_addr], 8 * TOPK);
        memcpy(&hw_result_dist_partial[0], &out[start_result_dist_addr], 4 * TOPK);
        
        // Check correctness
        for (int k = 0; k < TOPK; k++) {
            assert(hw_result_vec_ID_partial[k] != LARGE_NUM);
            assert(hw_result_vec_ID_partial[k] != LARGE_NUM + 1);
        } 
    }
#endif

    std::cout << "TEST FINISHED" << std::endl; 

    return  0;
}
