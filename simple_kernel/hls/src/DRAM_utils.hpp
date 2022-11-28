#pragma once

#include "types.hpp"

// send pairs of pages to the compute PE(s)
//   page_num_A * page_num_B pairs in total
// Note: the current design underutilizes the bandwidth for convenience
void input_loader(
    // input
    int page_entry_num, // number of entries per page
    int page_num_A, // number of pages for input A
    int page_num_B, // number of pages for input B
    const ap_uint<512>* in_pages_A,
    const ap_uint<512>* in_pages_B,
    // output
    hls::stream<obj_t>& s_page_A,
    hls::stream<obj_t>& s_page_B
    ) {
    
    // number of 512-bit entries that a page consumes 
    const int addr_per_page = page_entry_num % N_OBJ_PER_AXI == 0?
        page_entry_num / N_OBJ_PER_AXI : page_entry_num / N_OBJ_PER_AXI + 1;

    for (int a = 0; a < page_num_A; a++) {
        for (int b = 0; b < page_num_B; b++) {
            int start_addr_A = addr_per_page * a;
            int start_addr_B = addr_per_page * b;

            for (int i = 0; i < addr_per_page; i++) {
#pragma HLS pipeline // II=3 // needs N_OBJ_PER_AXI cycles
            // parse the input to three outputs
                ap_uint<512> reg_A = in_pages_A[start_addr_A + i];
                ap_uint<512> reg_B = in_pages_B[start_addr_B + i];

                for (int j = 0; j < N_OBJ_PER_AXI; j++) {

                    // page A
                    ap_uint<32> obj_id_A_ap_uint_32 = reg_A.range(
                      j * OBJ_BITS + 32 * 1 - 1, j * OBJ_BITS + 32 * 0);
                    ap_uint<32> left_A_ap_uint_32 = reg_A.range(
                      j * OBJ_BITS + 32 * 2 - 1, j * OBJ_BITS + 32 * 1);
                    ap_uint<32> right_A_ap_uint_32 = reg_A.range(
                      j * OBJ_BITS + 32 * 3 - 1, j * OBJ_BITS + 32 * 2);
                    ap_uint<32> top_A_ap_uint_32 = reg_A.range(
                      j * OBJ_BITS + 32 * 4 - 1, j * OBJ_BITS + 32 * 3);
                    ap_uint<32> bottom_A_ap_uint_32 = reg_A.range(
                      j * OBJ_BITS + 32 * 5 - 1, j * OBJ_BITS + 32 * 4);

                    obj_t obj_A;
                    obj_A.obj_id = *((int*) (&obj_id_A_ap_uint_32));
                    obj_A.left = *((float*) (&left_A_ap_uint_32)); 
                    obj_A.right = *((float*) (&right_A_ap_uint_32)); 
                    obj_A.top = *((float*) (&top_A_ap_uint_32)); 
                    obj_A.bottom = *((float*) (&bottom_A_ap_uint_32)); 
                    if (i * N_OBJ_PER_AXI + j < page_entry_num) {
                      s_page_A.write(obj_A);
                    }

                    // Page B
                    ap_uint<32> obj_id_B_ap_uint_32 = reg_B.range(
                      j * OBJ_BITS + 32 * 1 - 1, j * OBJ_BITS + 32 * 0);
                    ap_uint<32> left_B_ap_uint_32 = reg_B.range(
                      j * OBJ_BITS + 32 * 2 - 1, j * OBJ_BITS + 32 * 1);
                    ap_uint<32> right_B_ap_uint_32 = reg_B.range(
                      j * OBJ_BITS + 32 * 3 - 1, j * OBJ_BITS + 32 * 2);
                    ap_uint<32> top_B_ap_uint_32 = reg_B.range(
                      j * OBJ_BITS + 32 * 4 - 1, j * OBJ_BITS + 32 * 3);
                    ap_uint<32> bottom_B_ap_uint_32 = reg_B.range(
                      j * OBJ_BITS + 32 * 5 - 1, j * OBJ_BITS + 32 * 4);

                    obj_t obj_B;
                    obj_B.obj_id = *((int*) (&obj_id_B_ap_uint_32));
                    obj_B.left = *((float*) (&left_B_ap_uint_32)); 
                    obj_B.right = *((float*) (&right_B_ap_uint_32)); 
                    obj_B.top = *((float*) (&top_B_ap_uint_32)); 
                    obj_B.bottom = *((float*) (&bottom_B_ap_uint_32)); 
                    if (i * N_OBJ_PER_AXI + j < page_entry_num) {
                      s_page_B.write(obj_B);
                    }
                }
            }
        }
    }
}


void write_results(
    // input
    int page_entry_num,
    int page_pair_num, // number of page pairs to join, e.g., page_num_A * page_num_B
    hls::stream<int>& s_join_finish,  // per page pair
    hls::stream<result_t>& s_result_pair, 
    // out format: the first number writes total intersection count, 
    //   while the rest are intersect ID pairs
    ap_uint<64>* out_intersect) {

    ap_uint<64> total_intersect_count = 0;
    ap_uint<64> max_write_num = ap_uint<64>(page_entry_num) * ap_uint<64>(page_entry_num) * ap_uint<64>(page_pair_num);

    // wait until the first data has come
    while (s_join_finish.empty() && s_result_pair.empty()) {}

    // addr_bias = 1;
    for (ap_uint<64> i = 0; i < max_write_num; i++) {
#pragma HLS pipeline II=1
        if (!s_join_finish.empty() && s_result_pair.empty()) {
            int break_signal = s_join_finish.read(); // must read to make dataflow work
            total_intersect_count = i; 
            break;
        }
        result_t result = s_result_pair.read();
        ap_uint<64> result_ap_uint_64;
        result_ap_uint_64.range(31, 0) = result.obj_id_A;
        result_ap_uint_64.range(63, 32) = result.obj_id_B;
        out_intersect[i + 1] = result_ap_uint_64;
    }

    // write the number of intersection in the first address
    out_intersect[0] = total_intersect_count;
}

// // slow in writing, no burst inferred
// void write_results(
//     // input
//     int page_pair_num, // number of page pairs to join, e.g., page_num_A * page_num_B
//     hls::stream<int>& s_join_finish,  // per page pair
//     hls::stream<result_t>& s_result_pair, 
//     // out format: the first number writes total intersection count, 
//     //   while the rest are intersect ID pairs
//     ap_uint<64>* out_intersect) {

//     ap_uint<64> total_intersect_count = 0;
//     while (true) {
// #pragma HLS pipeline II=1

//         if (!s_result_pair.empty()) {
//             total_intersect_count += 1; 
//             result_t result = s_result_pair.read();
//             ap_uint<64> result_ap_uint_64;
//             result_ap_uint_64.range(31, 0) = result.obj_id_A;
//             result_ap_uint_64.range(63, 32) = result.obj_id_B;
//             ap_uint<64> out_addr = total_intersect_count + 1;
//             out_intersect[out_addr] = result_ap_uint_64;
//         } else if (!s_join_finish.empty() && s_result_pair.empty()) {
//             int break_signal = s_join_finish.read(); // must read to make dataflow work
//             break;
//         }
//     }

//     // write the number of intersection in the first address
//     out_intersect[0] = total_intersect_count;
// }