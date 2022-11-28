#include "ADC.hpp"
#include "constants.hpp"
#include "DRAM_utils.hpp"
#include "helpers.hpp"
#include "hierarchical_priority_queue.hpp"
#include "LUT_construction.hpp"
#include "types.hpp"

void DRAM_input_stream(
    int query_num,
    int nprobe,
    ap_uint<512>* in_DRAM,
    hls::stream<ap_uint<512>>& s_kernel_network_in) {

    int start_addr = 0;

    for (int query_id = 0; query_id < query_num; query_id++) {

        ap_uint<512> header = in_DRAM[start_addr];
        ap_uint<32> query_id_unused_uint = header.range(31, 0);
        ap_uint<32> nprobe_unused_uint = header.range(63, 32);
        int query_id_unused = *((int*) (&query_id_unused_uint));
        int nprobe_unused = *((int*) (&nprobe_unused_uint));

        int size_header = 1;
        int size_cell_IDs = nprobe * 4 % 64 == 0? nprobe * 4 / 64: nprobe * 4 / 64 + 1;
        const int size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
        const int size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
        int size_query = size_header + size_cell_IDs + size_query_vector + nprobe * size_center_vector;

        for (int i = 0; i < size_query; i++) {
            s_kernel_network_in.write(in_DRAM[start_addr + i]);
        }
        start_addr += size_query;
    }
}

void network_input_processing(
    int query_num,
    int nprobe,
    // in runtime
    hls::stream<ap_uint<512>>& s_kernel_network_in,
    // out
    hls::stream<int>& s_cell_ID,
    hls::stream<ap_uint<512> >& s_query_vectors,
    hls::stream<ap_uint<512> >& s_center_vectors
    ) {

    // Format: foe each query
    // packet 0: header (query_id, nprobe), in the future, nprobe is streamed from network
    // packet 1~k: cell_IDs to scan -> size = ceil(nprobe * 4 / 64) 
    // packet k~n: query_vectors
    // packet n~m: center_vectors

    // query format: store in 512-bit packets, pad 0 for the last packet if needed
    const int size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    const int size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 

    for (int query_id = 0; query_id < query_num; query_id++) {

        // header meta
        ap_uint<512> header = s_kernel_network_in.read();
        ap_uint<32> query_id_unused_uint = header.range(31, 0);
        ap_uint<32> nprobe_unused_uint = header.range(63, 32);
        int query_id_unused = *((int*) (&query_id_unused_uint));
        int nprobe_unused = *((int*) (&nprobe_unused_uint));

        int size_cell_IDs = nprobe * 4 % 64 == 0? nprobe * 4 / 64: nprobe * 4 / 64 + 1;

        // cell_IDs
        for (int i = 0; i < size_cell_IDs; i++) {

            ap_uint<512> pkt = s_kernel_network_in.read();

            for (int j = 0; j < 16; j++) {

                ap_uint<32> cell_ID_uint = pkt.range(32 * j + 31, 32 * j);
                int cell_ID = *((int*) (&cell_ID_uint));

                int cell_count = i * 16 + j;
                if (cell_count < nprobe) {
                    s_cell_ID.write(cell_ID);
                }
            }
        }

        // query vec
        for (int i = 0; i < size_query_vector; i++) {
            s_query_vectors.write(s_kernel_network_in.read());
        }

        // center vec
        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {
            for (int i = 0; i < size_center_vector; i++) {
                s_center_vectors.write(s_kernel_network_in.read());
            }
        }
    }
}

void network_output_processing(
    int query_num,
    hls::stream<result_t> &s_output, 
    hls::stream<ap_uint<512>>& s_kernel_network_out) {

    // Format: foe each query
    // packet 0: header (query_id, topK)
    // packet 1~k: topK results_pair (vec_ID, dist) -> size = ceil(topK * 8 / 64) 

    // in 512-bit packets
    const int size_results_vec_ID = PRIORITY_QUEUE_LEN_L2 * 64 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 64 / 512 : PRIORITY_QUEUE_LEN_L2 * 64 / 512 + 1;
    const int size_results_dist = PRIORITY_QUEUE_LEN_L2 * 32 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 32 / 512 : PRIORITY_QUEUE_LEN_L2 * 32 / 512 + 1;

    ap_uint<64> vec_ID_buffer [size_results_vec_ID * (512 / 64)] = { 0 };
    float dist_buffer[size_results_dist * (512 / 32)] = { 0 };

    // only write the last iteration
    for (int i = 0; i < query_num; i++) {
#pragma HLS pipeline II=1

        ap_uint<512> header = 0;
        ap_uint<32> query_id_header = i;
        ap_uint<32> topK_header = PRIORITY_QUEUE_LEN_L2;
        header.range(31, 0) = query_id_header;
        header.range(63, 32) = topK_header;
        s_kernel_network_out.write(header);

        for (int k = 0; k < TOPK; k++) {
            result_t raw_output = s_output.read();
            vec_ID_buffer[k] = raw_output.vec_ID;
            dist_buffer[k] = raw_output.dist;
        }

        // send vec IDs first
        for (int j = 0; j < size_results_vec_ID; j++) {
            ap_uint<512> pkt = 0;

            for (int k = 0; k < 512 / 64; k++) {
                
                    ap_uint<64> vec_ID = vec_ID_buffer[j * 512 / 64 + k];
                    pkt.range(64 * k + 63, 64 * k) = vec_ID;
                    
            }
            s_kernel_network_out.write(pkt);
        }

        // then send disk
        for (int j = 0; j < size_results_dist; j++) {
            ap_uint<512> pkt = 0;

            for (int k = 0; k < 512 / 32; k++) {
                
                    float dist = dist_buffer[j * 512 / 32 + k];
                    ap_uint<32> dist_uint = *((ap_uint<32>*) (&dist));
                    pkt.range(32 * k + 31, 32 * k) = dist_uint;
            }
            s_kernel_network_out.write(pkt);
        }
    } 
}

void DRAM_output_stream(
    int query_num,
    int nprobe,
    hls::stream<ap_uint<512>>& s_kernel_network_out,
    ap_uint<512>* out_DRAM) {

    // header = 1 pkt
    const int size_results_vec_ID = PRIORITY_QUEUE_LEN_L2 * 64 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 64 / 512 : PRIORITY_QUEUE_LEN_L2 * 64 / 512 + 1;
    const int size_results_dist = PRIORITY_QUEUE_LEN_L2 * 32 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 32 / 512 : PRIORITY_QUEUE_LEN_L2 * 32 / 512 + 1;
    int size_results = 1 + size_results_vec_ID + size_results_dist;
    
    for (int query_id = 0; query_id < query_num; query_id++) {
        for (int i = 0; i < size_results; i++) {
            out_DRAM[query_id * size_results + i] = s_kernel_network_out.read();
        }
    }
}

extern "C" {

void vadd(  
    // in init
    int query_num, 
    int nlist,
    int nprobe,
    int* meta_data_init, // which consists of the following three stuffs:
                    // int* nlist_PQ_codes_start_addr,
                    // int* nlist_vec_ID_start_addr,
                    // int* nlist_num_vecs,
                    // float* product_quantizer

    // in runtime (should from network)
    ap_uint<512>* in_DRAM, // simulate network 
    // int* cell_ID_DRAM, // query_num * nprobe
    // ap_uint<512>* LUT_DRAM, // query_num * nprobe * 256 * M

    // in runtime (should from DRAM)
    const ap_uint<512>* PQ_codes_DRAM_0,
    const ap_uint<512>* PQ_codes_DRAM_1,
    const ap_uint<512>* PQ_codes_DRAM_2,
    const ap_uint<512>* PQ_codes_DRAM_3,
    ap_uint<64>* vec_ID_DRAM_0,
    ap_uint<64>* vec_ID_DRAM_1,
    ap_uint<64>* vec_ID_DRAM_2,
    ap_uint<64>* vec_ID_DRAM_3,

    // out
    ap_uint<512>* out_DRAM)
{
// Share the same AXI interface with several control signals (but they are not allowed in same dataflow)
//    https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Controlling-AXI4-Burst-Behavior

// in init
#pragma HLS INTERFACE m_axi port=meta_data_init  offset=slave bundle=gmem_control

// in runtime (should from network)
// #pragma HLS INTERFACE m_axi port=cell_ID_DRAM offset=slave bundle=gmem3
// #pragma HLS INTERFACE m_axi port=LUT_DRAM offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=in_DRAM offset=slave bundle=gmem3

// in runtime (should from DRAM)
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_0 offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_1 offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_2 offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_3 offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_0  offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_1  offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_2  offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_3  offset=slave bundle=gmem12

// out
#pragma HLS INTERFACE m_axi port=out_DRAM  offset=slave bundle=gmem13

#pragma HLS dataflow

////////////////////     Network Input     ////////////////////

    hls::stream<ap_uint<512>> s_kernel_network_in;
#pragma HLS stream variable=s_kernel_network_in depth=512
    
    DRAM_input_stream(
        query_num,
        nprobe,
        in_DRAM,
        s_kernel_network_in);

    hls::stream<int> s_cell_ID;
#pragma HLS stream variable=s_cell_ID depth=256
    
    hls::stream<ap_uint<512> > s_query_vectors;
#pragma HLS stream variable=s_query_vectors depth=256
    
    hls::stream<ap_uint<512> > s_center_vectors;
#pragma HLS stream variable=s_center_vectors depth=256

    network_input_processing(
        query_num,
        nprobe,
        // in runtime
        s_kernel_network_in,
        // out
        s_cell_ID,
        s_query_vectors, 
        s_center_vectors);


////////////////////     0. Initialization     ////////////////////

    hls::stream<int> s_nlist_PQ_codes_start_addr;
#pragma HLS stream variable=s_nlist_PQ_codes_start_addr depth=256

    hls::stream<int> s_nlist_vec_ID_start_addr; // the top 10 numbers
#pragma HLS stream variable=s_nlist_vec_ID_start_addr depth=256
    
    hls::stream<int> s_nlist_num_vecs;
#pragma HLS stream variable=s_nlist_num_vecs depth=256

    hls::stream<float> s_product_quantizer_init;
#pragma HLS stream variable=s_product_quantizer_init depth=256

    load_meta_data(
        nlist,
        meta_data_init,
        s_nlist_PQ_codes_start_addr,
        s_nlist_vec_ID_start_addr,
        s_nlist_num_vecs,
        s_product_quantizer_init);

////////////////////     1. Construct LUT     ////////////////////

    hls::stream<distance_LUT_parallel_t> s_distance_LUT_first;
#pragma HLS stream variable=s_distance_LUT_first depth=512

    hls::stream<distance_LUT_parallel_t> s_distance_LUT[ADC_PE_NUM];
#pragma HLS stream variable=s_distance_LUT depth=8
#pragma HLS array_partition variable=s_distance_LUT complete
// #pragma HLS resource variable=s_distance_LUT core=FIFO_SRL

    LUT_construction_wrapper(
    // init
    query_num,
    nprobe, 
    s_product_quantizer_init,
    // runtime input from network
    s_query_vectors,
    s_center_vectors,
    // output
    s_distance_LUT_first);

////////////////////     2. ADC     ////////////////////

    hls::stream<int> s_cell_ID_get_cell_addr_and_size;
#pragma HLS stream variable=s_cell_ID_get_cell_addr_and_size depth=256
    
    hls::stream<int> s_cell_ID_load_PQ_codes;
#pragma HLS stream variable=s_cell_ID_load_PQ_codes depth=256

    replicate_s_cell_ID(
        query_num,
        nprobe,
        s_cell_ID,
        s_cell_ID_get_cell_addr_and_size,
        s_cell_ID_load_PQ_codes);

    hls::stream<int> s_scanned_entries_every_cell;
#pragma HLS stream variable=s_scanned_entries_every_cell depth=256
// #pragma HLS resource variable=s_scanned_entries_every_cell core=FIFO_SRL
    
    hls::stream<int> s_last_valid_PE_ID;
#pragma HLS stream variable=s_last_valid_PE_ID depth=256
// #pragma HLS resource variable=s_last_valid_PE_ID core=FIFO_SRL
    
    hls::stream<int> s_start_addr_every_cell;
#pragma HLS stream variable=s_start_addr_every_cell depth=256
// #pragma HLS resource variable=s_start_addr_every_cell core=FIFO_SRL
    
    hls::stream<int> s_control_iter_num_per_query;
#pragma HLS stream variable=s_control_iter_num_per_query depth=256
// #pragma HLS resource variable=s_control_iter_num_per_query core=FIFO_SRL
    
    get_cell_addr_and_size(
        // in init
        query_num, 
	    nlist,
        nprobe,
        s_nlist_PQ_codes_start_addr,
        s_nlist_num_vecs,
        // in runtime
        s_cell_ID_get_cell_addr_and_size,
        // out
        s_scanned_entries_every_cell,
        s_last_valid_PE_ID,
        s_start_addr_every_cell,
        s_control_iter_num_per_query);

    hls::stream<int> s_scanned_entries_every_cell_ADC[ADC_PE_NUM];
#pragma HLS stream variable=s_scanned_entries_every_cell_ADC depth=256
#pragma HLS array_partition variable=s_scanned_entries_every_cell_ADC complete
// #pragma HLS resource variable=s_scanned_entries_every_cell_ADC core=FIFO_SRL

    hls::stream<int> s_scanned_entries_every_cell_load_PQ_codes;
#pragma HLS stream variable=s_scanned_entries_every_cell_load_PQ_codes depth=256
// #pragma HLS resource variable=s_scanned_entries_every_cell_load_PQ_codes core=FIFO_SRL

    replicate_s_scanned_entries_every_cell(
        // in
        query_num,
        nprobe,
        s_scanned_entries_every_cell,
        // out
        s_scanned_entries_every_cell_ADC,
        s_scanned_entries_every_cell_load_PQ_codes);

    hls::stream<PQ_in_t> s_PQ_codes[ADC_PE_NUM];
#pragma HLS stream variable=s_PQ_codes depth=8
#pragma HLS array_partition variable=s_PQ_codes complete
// #pragma HLS resource variable=s_PQ_codes core=FIFO_SRL

    load_PQ_codes(
        // in init
        query_num, 
        nprobe,
        // in runtime
        s_cell_ID_load_PQ_codes,
        s_scanned_entries_every_cell_load_PQ_codes,
        s_last_valid_PE_ID,
        s_start_addr_every_cell,
        PQ_codes_DRAM_0,
        PQ_codes_DRAM_1,
        PQ_codes_DRAM_2,
        PQ_codes_DRAM_3,
        // out
        s_PQ_codes);

    hls::stream<PQ_out_t> s_PQ_result[ADC_PE_NUM];
#pragma HLS stream variable=s_PQ_result depth=8
#pragma HLS array_partition variable=s_PQ_result complete
// #pragma HLS resource variable=s_PQ_result core=FIFO_SRL



    // PE 0, with longer input FIFO
#if ADC_DOUBLE_BUF_ENABLE == 0
        PQ_lookup_computation(
            query_num, 
            nprobe,
            // input streams
            s_distance_LUT_first,
            s_PQ_codes[0],
            s_scanned_entries_every_cell_ADC[0],
            // output streams
            s_distance_LUT[0],
            s_PQ_result[0]);
#elif ADC_DOUBLE_BUF_ENABLE == 1
        PQ_lookup_computation_double_buffer(
            query_num, 
            nprobe,
            // input streams
            s_distance_LUT_first,
            s_PQ_codes[0],
            s_scanned_entries_every_cell_ADC[0],
            // output streams
            s_distance_LUT[0],
            s_PQ_result[0]);
#endif

    for (int s = 1; s < ADC_PE_NUM; s++) {
#pragma HLS unroll

#if ADC_DOUBLE_BUF_ENABLE == 0
        PQ_lookup_computation(
            query_num, 
            nprobe,
            // input streams
            s_distance_LUT[s - 1],
            s_PQ_codes[s],
            s_scanned_entries_every_cell_ADC[s],
            // output streams
            s_distance_LUT[s],
            s_PQ_result[s]);
#elif ADC_DOUBLE_BUF_ENABLE == 1
        PQ_lookup_computation_double_buffer(
            query_num, 
            nprobe,
            // input streams
            s_distance_LUT[s - 1],
            s_PQ_codes[s],
            s_scanned_entries_every_cell_ADC[s],
            // output streams
            s_distance_LUT[s],
            s_PQ_result[s]);
#endif
    }

    dummy_distance_LUT_consumer(
        query_num, 
        nprobe,
        s_distance_LUT[ADC_PE_NUM - 1]);

////////////////////     3. K-Selection     ////////////////////

    hls::stream<result_t> s_output; // the topK numbers
#pragma HLS stream variable=s_output depth=256

    hierarchical_priority_queue( 
        query_num, 
        nlist,
        s_nlist_vec_ID_start_addr,
        s_control_iter_num_per_query, 
        s_PQ_result,
        vec_ID_DRAM_0,
        vec_ID_DRAM_1,
        vec_ID_DRAM_2,
        vec_ID_DRAM_3,
        s_output);

////////////////////     Network Output     ////////////////////

    hls::stream<ap_uint<512>> s_kernel_network_out; 
#pragma HLS stream variable=s_kernel_network_out depth=512

    network_output_processing(
        query_num,
        s_output, 
        s_kernel_network_out);

    DRAM_output_stream(
        query_num,
        nprobe,
        s_kernel_network_out,
        out_DRAM);
}

}
