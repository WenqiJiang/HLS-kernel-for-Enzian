// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Version: 2022.1
// Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module vadd_split_single_stream (
        ap_clk,
        ap_rst,
        ap_start,
        start_full_n,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        start_out,
        start_write,
        query_num_dout,
        query_num_num_data_valid,
        query_num_fifo_cap,
        query_num_empty_n,
        query_num_read,
        s_PQ_result_0_dout,
        s_PQ_result_0_num_data_valid,
        s_PQ_result_0_fifo_cap,
        s_PQ_result_0_empty_n,
        s_PQ_result_0_read,
        s_scanned_entries_every_cell_Replicated_i_i_0_dout,
        s_scanned_entries_every_cell_Replicated_i_i_0_num_data_valid,
        s_scanned_entries_every_cell_Replicated_i_i_0_fifo_cap,
        s_scanned_entries_every_cell_Replicated_i_i_0_empty_n,
        s_scanned_entries_every_cell_Replicated_i_i_0_read,
        s_insertion_per_queue_L1_i_0_din,
        s_insertion_per_queue_L1_i_0_num_data_valid,
        s_insertion_per_queue_L1_i_0_fifo_cap,
        s_insertion_per_queue_L1_i_0_full_n,
        s_insertion_per_queue_L1_i_0_write,
        s_insertion_per_queue_L1_i_1_din,
        s_insertion_per_queue_L1_i_1_num_data_valid,
        s_insertion_per_queue_L1_i_1_fifo_cap,
        s_insertion_per_queue_L1_i_1_full_n,
        s_insertion_per_queue_L1_i_1_write,
        s_input_splitted_i_0_din,
        s_input_splitted_i_0_num_data_valid,
        s_input_splitted_i_0_fifo_cap,
        s_input_splitted_i_0_full_n,
        s_input_splitted_i_0_write,
        s_input_splitted_i_1_din,
        s_input_splitted_i_1_num_data_valid,
        s_input_splitted_i_1_fifo_cap,
        s_input_splitted_i_1_full_n,
        s_input_splitted_i_1_write,
        query_num_c125_din,
        query_num_c125_num_data_valid,
        query_num_c125_fifo_cap,
        query_num_c125_full_n,
        query_num_c125_write,
        query_num_c126_din,
        query_num_c126_num_data_valid,
        query_num_c126_fifo_cap,
        query_num_c126_full_n,
        query_num_c126_write,
        ap_ext_blocking_n,
        ap_str_blocking_n,
        ap_int_blocking_n
);

parameter    ap_ST_fsm_state1 = 5'd1;
parameter    ap_ST_fsm_state2 = 5'd2;
parameter    ap_ST_fsm_state3 = 5'd4;
parameter    ap_ST_fsm_state4 = 5'd8;
parameter    ap_ST_fsm_state5 = 5'd16;

input   ap_clk;
input   ap_rst;
input   ap_start;
input   start_full_n;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
output   start_out;
output   start_write;
input  [31:0] query_num_dout;
input  [1:0] query_num_num_data_valid;
input  [1:0] query_num_fifo_cap;
input   query_num_empty_n;
output   query_num_read;
input  [95:0] s_PQ_result_0_dout;
input  [3:0] s_PQ_result_0_num_data_valid;
input  [3:0] s_PQ_result_0_fifo_cap;
input   s_PQ_result_0_empty_n;
output   s_PQ_result_0_read;
input  [31:0] s_scanned_entries_every_cell_Replicated_i_i_0_dout;
input  [3:0] s_scanned_entries_every_cell_Replicated_i_i_0_num_data_valid;
input  [3:0] s_scanned_entries_every_cell_Replicated_i_i_0_fifo_cap;
input   s_scanned_entries_every_cell_Replicated_i_i_0_empty_n;
output   s_scanned_entries_every_cell_Replicated_i_i_0_read;
output  [31:0] s_insertion_per_queue_L1_i_0_din;
input  [3:0] s_insertion_per_queue_L1_i_0_num_data_valid;
input  [3:0] s_insertion_per_queue_L1_i_0_fifo_cap;
input   s_insertion_per_queue_L1_i_0_full_n;
output   s_insertion_per_queue_L1_i_0_write;
output  [31:0] s_insertion_per_queue_L1_i_1_din;
input  [3:0] s_insertion_per_queue_L1_i_1_num_data_valid;
input  [3:0] s_insertion_per_queue_L1_i_1_fifo_cap;
input   s_insertion_per_queue_L1_i_1_full_n;
output   s_insertion_per_queue_L1_i_1_write;
output  [95:0] s_input_splitted_i_0_din;
input  [3:0] s_input_splitted_i_0_num_data_valid;
input  [3:0] s_input_splitted_i_0_fifo_cap;
input   s_input_splitted_i_0_full_n;
output   s_input_splitted_i_0_write;
output  [95:0] s_input_splitted_i_1_din;
input  [3:0] s_input_splitted_i_1_num_data_valid;
input  [3:0] s_input_splitted_i_1_fifo_cap;
input   s_input_splitted_i_1_full_n;
output   s_input_splitted_i_1_write;
output  [31:0] query_num_c125_din;
input  [1:0] query_num_c125_num_data_valid;
input  [1:0] query_num_c125_fifo_cap;
input   query_num_c125_full_n;
output   query_num_c125_write;
output  [31:0] query_num_c126_din;
input  [1:0] query_num_c126_num_data_valid;
input  [1:0] query_num_c126_fifo_cap;
input   query_num_c126_full_n;
output   query_num_c126_write;
output   ap_ext_blocking_n;
output   ap_str_blocking_n;
output   ap_int_blocking_n;

reg ap_done;
reg ap_idle;
reg start_write;
reg query_num_read;
reg s_PQ_result_0_read;
reg s_scanned_entries_every_cell_Replicated_i_i_0_read;
reg[31:0] s_insertion_per_queue_L1_i_0_din;
reg s_insertion_per_queue_L1_i_0_write;
reg s_insertion_per_queue_L1_i_1_write;
reg[95:0] s_input_splitted_i_0_din;
reg s_input_splitted_i_0_write;
reg s_input_splitted_i_1_write;
reg query_num_c125_write;
reg query_num_c126_write;

reg    real_start;
reg    start_once_reg;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [4:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    internal_ap_ready;
reg    query_num_blk_n;
reg    s_PQ_result_0_blk_n;
wire    ap_CS_fsm_state3;
wire   [0:0] icmp_ln105_fu_230_p2;
reg    s_scanned_entries_every_cell_Replicated_i_i_0_blk_n;
wire    ap_CS_fsm_state2;
wire   [0:0] icmp_ln100_fu_153_p2;
reg    s_insertion_per_queue_L1_i_0_blk_n;
reg    s_insertion_per_queue_L1_i_1_blk_n;
reg    s_input_splitted_i_0_blk_n;
reg    query_num_c125_blk_n;
reg    query_num_c126_blk_n;
reg   [31:0] query_num_read_reg_253;
wire   [30:0] add_ln100_fu_158_p2;
reg   [30:0] add_ln100_reg_261;
reg   [31:0] tmp_4_reg_266;
wire   [31:0] half_scanned_entries_every_cell_fu_212_p3;
reg   [31:0] half_scanned_entries_every_cell_reg_271;
wire    grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_start;
wire    grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_done;
wire    grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_idle;
wire    grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_ready;
wire   [95:0] grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_input_splitted_i_0_din;
wire    grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_input_splitted_i_0_write;
wire    grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_PQ_result_0_read;
wire   [95:0] grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_input_splitted_i_1_din;
wire    grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_input_splitted_i_1_write;
wire    grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_ext_blocking_n;
wire    grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_str_blocking_n;
wire    grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_int_blocking_n;
reg    grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_start_reg;
wire    ap_CS_fsm_state4;
wire    ap_CS_fsm_state5;
reg   [30:0] query_id_fu_70;
reg    ap_block_state3;
reg    ap_block_state1;
reg    ap_block_state2;
wire   [31:0] add_ln106_fu_236_p2;
wire   [31:0] zext_ln92_fu_149_p1;
wire   [31:0] sub_ln103_fu_172_p2;
wire   [30:0] lshr_ln103_7_fu_178_p4;
wire   [31:0] zext_ln103_fu_188_p1;
wire   [30:0] lshr_ln103_8_fu_198_p4;
wire   [0:0] tmp_fu_164_p3;
wire   [31:0] sub_ln103_4_fu_192_p2;
wire   [31:0] zext_ln103_4_fu_208_p1;
wire   [31:0] shl_ln105_fu_220_p2;
wire   [31:0] sub_ln105_fu_225_p2;
reg   [4:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
reg    ap_ST_fsm_state2_blk;
reg    ap_ST_fsm_state3_blk;
wire    ap_ST_fsm_state4_blk;
reg    ap_ST_fsm_state5_blk;
wire    ap_int_blocking_cur_n;
reg    ap_ext_blocking_sub_n;
reg    ap_wait_0;
reg    ap_sub_ext_blocking_0;
reg    ap_str_blocking_sub_n;
reg    ap_sub_str_blocking_0;
reg    ap_int_blocking_sub_n;
reg    ap_sub_int_blocking_0;
reg    ap_condition_189;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 start_once_reg = 1'b0;
#0 ap_done_reg = 1'b0;
#0 ap_CS_fsm = 5'd1;
#0 grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_start_reg = 1'b0;
end

vadd_split_single_stream_Pipeline_VITIS_LOOP_114_2 grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_start),
    .ap_done(grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_done),
    .ap_idle(grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_idle),
    .ap_ready(grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_ready),
    .s_input_splitted_i_0_din(grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_input_splitted_i_0_din),
    .s_input_splitted_i_0_num_data_valid(4'd0),
    .s_input_splitted_i_0_fifo_cap(4'd0),
    .s_input_splitted_i_0_full_n(s_input_splitted_i_0_full_n),
    .s_input_splitted_i_0_write(grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_input_splitted_i_0_write),
    .s_PQ_result_0_dout(s_PQ_result_0_dout),
    .s_PQ_result_0_num_data_valid(4'd0),
    .s_PQ_result_0_fifo_cap(4'd0),
    .s_PQ_result_0_empty_n(s_PQ_result_0_empty_n),
    .s_PQ_result_0_read(grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_PQ_result_0_read),
    .s_input_splitted_i_1_din(grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_input_splitted_i_1_din),
    .s_input_splitted_i_1_num_data_valid(4'd0),
    .s_input_splitted_i_1_fifo_cap(4'd0),
    .s_input_splitted_i_1_full_n(s_input_splitted_i_1_full_n),
    .s_input_splitted_i_1_write(grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_input_splitted_i_1_write),
    .half_scanned_entries_every_cell(half_scanned_entries_every_cell_reg_271),
    .ap_ext_blocking_n(grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_ext_blocking_n),
    .ap_str_blocking_n(grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_str_blocking_n),
    .ap_int_blocking_n(grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_int_blocking_n)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if ((~((s_scanned_entries_every_cell_Replicated_i_i_0_empty_n == 1'b0) & (icmp_ln100_fu_153_p2 == 1'd1)) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln100_fu_153_p2 == 1'd0))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state4)) begin
            grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_start_reg <= 1'b1;
        end else if ((grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_ready == 1'b1)) begin
            grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        start_once_reg <= 1'b0;
    end else begin
        if (((internal_ap_ready == 1'b0) & (real_start == 1'b1))) begin
            start_once_reg <= 1'b1;
        end else if ((internal_ap_ready == 1'b1)) begin
            start_once_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((query_num_c126_full_n == 1'b0) | (query_num_c125_full_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_id_fu_70 <= 31'd0;
    end else if ((~((s_insertion_per_queue_L1_i_1_full_n == 1'b0) | ((s_input_splitted_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_PQ_result_0_empty_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd0)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1))) & (1'b1 == ap_CS_fsm_state3))) begin
        query_id_fu_70 <= add_ln100_reg_261;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        add_ln100_reg_261 <= add_ln100_fu_158_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln100_fu_153_p2 == 1'd1))) begin
        half_scanned_entries_every_cell_reg_271 <= half_scanned_entries_every_cell_fu_212_p3;
        tmp_4_reg_266 <= s_scanned_entries_every_cell_Replicated_i_i_0_dout;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state1)) begin
        query_num_read_reg_253 <= query_num_dout;
    end
end

always @ (*) begin
    if (((query_num_c126_full_n == 1'b0) | (query_num_c125_full_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0))) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

always @ (*) begin
    if (((s_scanned_entries_every_cell_Replicated_i_i_0_empty_n == 1'b0) & (icmp_ln100_fu_153_p2 == 1'd1))) begin
        ap_ST_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_fsm_state2_blk = 1'b0;
    end
end

always @ (*) begin
    if (((s_insertion_per_queue_L1_i_1_full_n == 1'b0) | ((s_input_splitted_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_PQ_result_0_empty_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd0)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)))) begin
        ap_ST_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_fsm_state3_blk = 1'b0;
    end
end

assign ap_ST_fsm_state4_blk = 1'b0;

always @ (*) begin
    if ((grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_done == 1'b0)) begin
        ap_ST_fsm_state5_blk = 1'b1;
    end else begin
        ap_ST_fsm_state5_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((s_scanned_entries_every_cell_Replicated_i_i_0_empty_n == 1'b0) & (icmp_ln100_fu_153_p2 == 1'd1)) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln100_fu_153_p2 == 1'd0))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_wait_0 & ap_sub_ext_blocking_0) == 1'b1)) begin
        ap_ext_blocking_sub_n = 1'b0;
    end else begin
        ap_ext_blocking_sub_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (real_start == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((ap_wait_0 & ap_sub_int_blocking_0) == 1'b1)) begin
        ap_int_blocking_sub_n = 1'b0;
    end else begin
        ap_int_blocking_sub_n = 1'b1;
    end
end

always @ (*) begin
    if (((ap_wait_0 & ap_sub_str_blocking_0) == 1'b1)) begin
        ap_str_blocking_sub_n = 1'b0;
    end else begin
        ap_str_blocking_sub_n = 1'b1;
    end
end

always @ (*) begin
    if ((grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_ext_blocking_n == 1'b0)) begin
        ap_sub_ext_blocking_0 = 1'b1;
    end else begin
        ap_sub_ext_blocking_0 = 1'b0;
    end
end

always @ (*) begin
    if ((grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_int_blocking_n == 1'b0)) begin
        ap_sub_int_blocking_0 = 1'b1;
    end else begin
        ap_sub_int_blocking_0 = 1'b0;
    end
end

always @ (*) begin
    if ((grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_str_blocking_n == 1'b0)) begin
        ap_sub_str_blocking_0 = 1'b1;
    end else begin
        ap_sub_str_blocking_0 = 1'b0;
    end
end

always @ (*) begin
    if ((ap_ST_fsm_state5 == ap_CS_fsm)) begin
        ap_wait_0 = 1'b1;
    end else begin
        ap_wait_0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((s_scanned_entries_every_cell_Replicated_i_i_0_empty_n == 1'b0) & (icmp_ln100_fu_153_p2 == 1'd1)) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln100_fu_153_p2 == 1'd0))) begin
        internal_ap_ready = 1'b1;
    end else begin
        internal_ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_blk_n = query_num_empty_n;
    end else begin
        query_num_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_c125_blk_n = query_num_c125_full_n;
    end else begin
        query_num_c125_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((query_num_c126_full_n == 1'b0) | (query_num_c125_full_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_c125_write = 1'b1;
    end else begin
        query_num_c125_write = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_c126_blk_n = query_num_c126_full_n;
    end else begin
        query_num_c126_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((query_num_c126_full_n == 1'b0) | (query_num_c125_full_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_c126_write = 1'b1;
    end else begin
        query_num_c126_write = 1'b0;
    end
end

always @ (*) begin
    if ((~((query_num_c126_full_n == 1'b0) | (query_num_c125_full_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_read = 1'b1;
    end else begin
        query_num_read = 1'b0;
    end
end

always @ (*) begin
    if (((start_once_reg == 1'b0) & (start_full_n == 1'b0))) begin
        real_start = 1'b0;
    end else begin
        real_start = ap_start;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) & (icmp_ln105_fu_230_p2 == 1'd1))) begin
        s_PQ_result_0_blk_n = s_PQ_result_0_empty_n;
    end else begin
        s_PQ_result_0_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((s_insertion_per_queue_L1_i_1_full_n == 1'b0) | ((s_input_splitted_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_PQ_result_0_empty_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd0)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1))) & (1'b1 == ap_CS_fsm_state3) & (icmp_ln105_fu_230_p2 == 1'd1))) begin
        s_PQ_result_0_read = 1'b1;
    end else if ((1'b1 == ap_CS_fsm_state5)) begin
        s_PQ_result_0_read = grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_PQ_result_0_read;
    end else begin
        s_PQ_result_0_read = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) & (icmp_ln105_fu_230_p2 == 1'd1))) begin
        s_input_splitted_i_0_blk_n = s_input_splitted_i_0_full_n;
    end else begin
        s_input_splitted_i_0_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((s_insertion_per_queue_L1_i_1_full_n == 1'b0) | ((s_input_splitted_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_PQ_result_0_empty_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd0)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1))) & (1'b1 == ap_CS_fsm_state3) & (icmp_ln105_fu_230_p2 == 1'd1))) begin
        s_input_splitted_i_0_din = s_PQ_result_0_dout;
    end else if ((1'b1 == ap_CS_fsm_state5)) begin
        s_input_splitted_i_0_din = grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_input_splitted_i_0_din;
    end else begin
        s_input_splitted_i_0_din = grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_input_splitted_i_0_din;
    end
end

always @ (*) begin
    if ((~((s_insertion_per_queue_L1_i_1_full_n == 1'b0) | ((s_input_splitted_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_PQ_result_0_empty_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd0)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1))) & (1'b1 == ap_CS_fsm_state3) & (icmp_ln105_fu_230_p2 == 1'd1))) begin
        s_input_splitted_i_0_write = 1'b1;
    end else if ((1'b1 == ap_CS_fsm_state5)) begin
        s_input_splitted_i_0_write = grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_input_splitted_i_0_write;
    end else begin
        s_input_splitted_i_0_write = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        s_input_splitted_i_1_write = grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_input_splitted_i_1_write;
    end else begin
        s_input_splitted_i_1_write = 1'b0;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_fsm_state3) & (icmp_ln105_fu_230_p2 == 1'd0)) | ((1'b1 == ap_CS_fsm_state3) & (icmp_ln105_fu_230_p2 == 1'd1)))) begin
        s_insertion_per_queue_L1_i_0_blk_n = s_insertion_per_queue_L1_i_0_full_n;
    end else begin
        s_insertion_per_queue_L1_i_0_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_condition_189)) begin
        if ((icmp_ln105_fu_230_p2 == 1'd1)) begin
            s_insertion_per_queue_L1_i_0_din = add_ln106_fu_236_p2;
        end else if ((icmp_ln105_fu_230_p2 == 1'd0)) begin
            s_insertion_per_queue_L1_i_0_din = half_scanned_entries_every_cell_reg_271;
        end else begin
            s_insertion_per_queue_L1_i_0_din = 'bx;
        end
    end else begin
        s_insertion_per_queue_L1_i_0_din = 'bx;
    end
end

always @ (*) begin
    if (((~((s_insertion_per_queue_L1_i_1_full_n == 1'b0) | ((s_input_splitted_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_PQ_result_0_empty_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd0)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1))) & (1'b1 == ap_CS_fsm_state3) & (icmp_ln105_fu_230_p2 == 1'd0)) | (~((s_insertion_per_queue_L1_i_1_full_n == 1'b0) | ((s_input_splitted_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_PQ_result_0_empty_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd0)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1))) & (1'b1 == ap_CS_fsm_state3) & (icmp_ln105_fu_230_p2 == 1'd1)))) begin
        s_insertion_per_queue_L1_i_0_write = 1'b1;
    end else begin
        s_insertion_per_queue_L1_i_0_write = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_insertion_per_queue_L1_i_1_blk_n = s_insertion_per_queue_L1_i_1_full_n;
    end else begin
        s_insertion_per_queue_L1_i_1_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((s_insertion_per_queue_L1_i_1_full_n == 1'b0) | ((s_input_splitted_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_PQ_result_0_empty_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd0)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1))) & (1'b1 == ap_CS_fsm_state3))) begin
        s_insertion_per_queue_L1_i_1_write = 1'b1;
    end else begin
        s_insertion_per_queue_L1_i_1_write = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln100_fu_153_p2 == 1'd1))) begin
        s_scanned_entries_every_cell_Replicated_i_i_0_blk_n = s_scanned_entries_every_cell_Replicated_i_i_0_empty_n;
    end else begin
        s_scanned_entries_every_cell_Replicated_i_i_0_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((s_scanned_entries_every_cell_Replicated_i_i_0_empty_n == 1'b0) & (icmp_ln100_fu_153_p2 == 1'd1)) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln100_fu_153_p2 == 1'd1))) begin
        s_scanned_entries_every_cell_Replicated_i_i_0_read = 1'b1;
    end else begin
        s_scanned_entries_every_cell_Replicated_i_i_0_read = 1'b0;
    end
end

always @ (*) begin
    if (((start_once_reg == 1'b0) & (real_start == 1'b1))) begin
        start_write = 1'b1;
    end else begin
        start_write = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if ((~((query_num_c126_full_n == 1'b0) | (query_num_c125_full_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if ((~((s_scanned_entries_every_cell_Replicated_i_i_0_empty_n == 1'b0) & (icmp_ln100_fu_153_p2 == 1'd1)) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln100_fu_153_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else if ((~((s_scanned_entries_every_cell_Replicated_i_i_0_empty_n == 1'b0) & (icmp_ln100_fu_153_p2 == 1'd1)) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln100_fu_153_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end
        end
        ap_ST_fsm_state3 : begin
            if ((~((s_insertion_per_queue_L1_i_1_full_n == 1'b0) | ((s_input_splitted_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_PQ_result_0_empty_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd0)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1))) & (1'b1 == ap_CS_fsm_state3))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state5;
        end
        ap_ST_fsm_state5 : begin
            if (((1'b1 == ap_CS_fsm_state5) & (grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln100_fu_158_p2 = (query_id_fu_70 + 31'd1);

assign add_ln106_fu_236_p2 = (half_scanned_entries_every_cell_reg_271 + 32'd1);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

always @ (*) begin
    ap_block_state1 = ((query_num_c126_full_n == 1'b0) | (query_num_c125_full_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0));
end

always @ (*) begin
    ap_block_state2 = ((s_scanned_entries_every_cell_Replicated_i_i_0_empty_n == 1'b0) & (icmp_ln100_fu_153_p2 == 1'd1));
end

always @ (*) begin
    ap_block_state3 = ((s_insertion_per_queue_L1_i_1_full_n == 1'b0) | ((s_input_splitted_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_PQ_result_0_empty_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd0)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)));
end

always @ (*) begin
    ap_condition_189 = (~((s_insertion_per_queue_L1_i_1_full_n == 1'b0) | ((s_input_splitted_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_PQ_result_0_empty_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd0)) | ((s_insertion_per_queue_L1_i_0_full_n == 1'b0) & (icmp_ln105_fu_230_p2 == 1'd1))) & (1'b1 == ap_CS_fsm_state3));
end

assign ap_ext_blocking_n = (ap_ext_blocking_sub_n & 1'b1);

assign ap_int_blocking_cur_n = (s_scanned_entries_every_cell_Replicated_i_i_0_blk_n & s_insertion_per_queue_L1_i_1_blk_n & s_insertion_per_queue_L1_i_0_blk_n & s_input_splitted_i_0_blk_n & s_PQ_result_0_blk_n & query_num_c126_blk_n & query_num_c125_blk_n & query_num_blk_n);

assign ap_int_blocking_n = (ap_int_blocking_sub_n & ap_int_blocking_cur_n);

assign ap_ready = internal_ap_ready;

assign ap_str_blocking_n = (ap_str_blocking_sub_n & 1'b1);

assign grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_start = grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_ap_start_reg;

assign half_scanned_entries_every_cell_fu_212_p3 = ((tmp_fu_164_p3[0:0] == 1'b1) ? sub_ln103_4_fu_192_p2 : zext_ln103_4_fu_208_p1);

assign icmp_ln100_fu_153_p2 = (($signed(zext_ln92_fu_149_p1) < $signed(query_num_read_reg_253)) ? 1'b1 : 1'b0);

assign icmp_ln105_fu_230_p2 = ((sub_ln105_fu_225_p2 == 32'd1) ? 1'b1 : 1'b0);

assign lshr_ln103_7_fu_178_p4 = {{sub_ln103_fu_172_p2[31:1]}};

assign lshr_ln103_8_fu_198_p4 = {{s_scanned_entries_every_cell_Replicated_i_i_0_dout[31:1]}};

assign query_num_c125_din = query_num_dout;

assign query_num_c126_din = query_num_dout;

assign s_input_splitted_i_1_din = grp_split_single_stream_Pipeline_VITIS_LOOP_114_2_fu_130_s_input_splitted_i_1_din;

assign s_insertion_per_queue_L1_i_1_din = half_scanned_entries_every_cell_reg_271;

assign shl_ln105_fu_220_p2 = half_scanned_entries_every_cell_reg_271 << 32'd1;

assign start_out = real_start;

assign sub_ln103_4_fu_192_p2 = (32'd0 - zext_ln103_fu_188_p1);

assign sub_ln103_fu_172_p2 = (32'd0 - s_scanned_entries_every_cell_Replicated_i_i_0_dout);

assign sub_ln105_fu_225_p2 = (tmp_4_reg_266 - shl_ln105_fu_220_p2);

assign tmp_fu_164_p3 = s_scanned_entries_every_cell_Replicated_i_i_0_dout[32'd31];

assign zext_ln103_4_fu_208_p1 = lshr_ln103_8_fu_198_p4;

assign zext_ln103_fu_188_p1 = lshr_ln103_7_fu_178_p4;

assign zext_ln92_fu_149_p1 = query_id_fu_70;

endmodule //vadd_split_single_stream
