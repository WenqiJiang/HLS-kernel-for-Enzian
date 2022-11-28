#pragma once

// join between page pairs
void join_page(
    // input
    int page_entry_num, // number of entries per page
    int page_pair_num, // number of page pairs to join, e.g., page_num_A * page_num_B
    hls::stream<obj_t>& s_page_A,
    hls::stream<obj_t>& s_page_B,
    // output
    // hls::stream<ap_uint<64> >& s_intersect_count, // per page pair
    hls::stream<int>& s_join_finish, // one single signal after everything is written
    hls::stream<result_t>& s_result_pair) {

    obj_t page_A[MAX_PAGE_ENTRY_NUM];
    obj_t page_B[MAX_PAGE_ENTRY_NUM];

    for (int i = 0; i < page_pair_num; i++) {

        // load the two pages 
        for (int j = 0; j < page_entry_num; j++) {
#pragma HLS pipeline II=1
            page_A[j] = s_page_A.read();
            page_B[j] = s_page_B.read();
        }

        // perform the join 
        // ap_uint<64> intersect_count = 0;
        for (int m = 0; m < page_entry_num; m++) {

            obj_t obj_A = page_A[m];

            for (int n = 0; n < page_entry_num; n++) {
#pragma HLS pipeline II=1

                obj_t obj_B = page_B[n];

                // // point overlap is NOT regarded as overlap, only region overlap counts
                // bool overlap = 
                //     // horizontal overlap (no horizontal overlap within parentheses)
                //     !((obj_A.right <= obj_B.left) || (obj_B.right <= obj_A.left)) && 
                //     // vertical no overlap (no vertical overlap within parentheses)
                //     !((obj_A.bottom >= obj_B.top) || (obj_B.bottom >= obj_A.top));

                // point overlap is regarded as overlap, only region overlap counts
                bool overlap = 
                    // horizontal overlap (no horizontal overlap within parentheses)
                    !((obj_A.right < obj_B.left) || (obj_B.right < obj_A.left)) && 
                    // vertical no overlap (no vertical overlap within parentheses)
                    !((obj_A.bottom > obj_B.top) || (obj_B.bottom > obj_A.top));

                if (overlap) {
                    // intersect_count++;
                    result_t result;
                    result.obj_id_A = obj_A.obj_id;
                    result.obj_id_B = obj_B.obj_id;
                    s_result_pair.write(result);
                }
            }
        }
        // s_intersect_count.write(intersect_count);
    }
    s_join_finish.write(1);
}