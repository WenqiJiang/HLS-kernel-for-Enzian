// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Version: 2022.1
// Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module vadd_network_input_processing (
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
        nprobe_dout,
        nprobe_num_data_valid,
        nprobe_fifo_cap,
        nprobe_empty_n,
        nprobe_read,
        s_kernel_network_in65_dout,
        s_kernel_network_in65_num_data_valid,
        s_kernel_network_in65_fifo_cap,
        s_kernel_network_in65_empty_n,
        s_kernel_network_in65_read,
        s_cell_ID66_din,
        s_cell_ID66_num_data_valid,
        s_cell_ID66_fifo_cap,
        s_cell_ID66_full_n,
        s_cell_ID66_write,
        s_query_vectors67_din,
        s_query_vectors67_num_data_valid,
        s_query_vectors67_fifo_cap,
        s_query_vectors67_full_n,
        s_query_vectors67_write,
        s_center_vectors68_din,
        s_center_vectors68_num_data_valid,
        s_center_vectors68_fifo_cap,
        s_center_vectors68_full_n,
        s_center_vectors68_write,
        query_num_c139_din,
        query_num_c139_num_data_valid,
        query_num_c139_fifo_cap,
        query_num_c139_full_n,
        query_num_c139_write,
        query_num_c157_din,
        query_num_c157_num_data_valid,
        query_num_c157_fifo_cap,
        query_num_c157_full_n,
        query_num_c157_write,
        query_num_c158_din,
        query_num_c158_num_data_valid,
        query_num_c158_fifo_cap,
        query_num_c158_full_n,
        query_num_c158_write,
        nprobe_c167_din,
        nprobe_c167_num_data_valid,
        nprobe_c167_fifo_cap,
        nprobe_c167_full_n,
        nprobe_c167_write,
        nprobe_c185_din,
        nprobe_c185_num_data_valid,
        nprobe_c185_fifo_cap,
        nprobe_c185_full_n,
        nprobe_c185_write,
        ap_ext_blocking_n,
        ap_str_blocking_n,
        ap_int_blocking_n
);

parameter    ap_ST_fsm_state1 = 10'd1;
parameter    ap_ST_fsm_state2 = 10'd2;
parameter    ap_ST_fsm_state3 = 10'd4;
parameter    ap_ST_fsm_state4 = 10'd8;
parameter    ap_ST_fsm_state5 = 10'd16;
parameter    ap_ST_fsm_state6 = 10'd32;
parameter    ap_ST_fsm_state7 = 10'd64;
parameter    ap_ST_fsm_state8 = 10'd128;
parameter    ap_ST_fsm_state9 = 10'd256;
parameter    ap_ST_fsm_state10 = 10'd512;

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
input  [31:0] nprobe_dout;
input  [1:0] nprobe_num_data_valid;
input  [1:0] nprobe_fifo_cap;
input   nprobe_empty_n;
output   nprobe_read;
input  [511:0] s_kernel_network_in65_dout;
input  [9:0] s_kernel_network_in65_num_data_valid;
input  [9:0] s_kernel_network_in65_fifo_cap;
input   s_kernel_network_in65_empty_n;
output   s_kernel_network_in65_read;
output  [31:0] s_cell_ID66_din;
input  [8:0] s_cell_ID66_num_data_valid;
input  [8:0] s_cell_ID66_fifo_cap;
input   s_cell_ID66_full_n;
output   s_cell_ID66_write;
output  [511:0] s_query_vectors67_din;
input  [8:0] s_query_vectors67_num_data_valid;
input  [8:0] s_query_vectors67_fifo_cap;
input   s_query_vectors67_full_n;
output   s_query_vectors67_write;
output  [511:0] s_center_vectors68_din;
input  [8:0] s_center_vectors68_num_data_valid;
input  [8:0] s_center_vectors68_fifo_cap;
input   s_center_vectors68_full_n;
output   s_center_vectors68_write;
output  [31:0] query_num_c139_din;
input  [1:0] query_num_c139_num_data_valid;
input  [1:0] query_num_c139_fifo_cap;
input   query_num_c139_full_n;
output   query_num_c139_write;
output  [31:0] query_num_c157_din;
input  [1:0] query_num_c157_num_data_valid;
input  [1:0] query_num_c157_fifo_cap;
input   query_num_c157_full_n;
output   query_num_c157_write;
output  [31:0] query_num_c158_din;
input  [1:0] query_num_c158_num_data_valid;
input  [1:0] query_num_c158_fifo_cap;
input   query_num_c158_full_n;
output   query_num_c158_write;
output  [31:0] nprobe_c167_din;
input  [1:0] nprobe_c167_num_data_valid;
input  [1:0] nprobe_c167_fifo_cap;
input   nprobe_c167_full_n;
output   nprobe_c167_write;
output  [31:0] nprobe_c185_din;
input  [1:0] nprobe_c185_num_data_valid;
input  [1:0] nprobe_c185_fifo_cap;
input   nprobe_c185_full_n;
output   nprobe_c185_write;
output   ap_ext_blocking_n;
output   ap_str_blocking_n;
output   ap_int_blocking_n;

reg ap_done;
reg ap_idle;
reg start_write;
reg query_num_read;
reg nprobe_read;
reg s_kernel_network_in65_read;
reg s_cell_ID66_write;
reg s_query_vectors67_write;
reg s_center_vectors68_write;
reg query_num_c139_write;
reg query_num_c157_write;
reg query_num_c158_write;
reg nprobe_c167_write;
reg nprobe_c185_write;

reg    real_start;
reg    start_once_reg;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [9:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    internal_ap_ready;
reg    query_num_blk_n;
reg    nprobe_blk_n;
reg    s_kernel_network_in65_blk_n;
wire    ap_CS_fsm_state2;
wire   [0:0] icmp_ln59_fu_321_p2;
reg    query_num_c139_blk_n;
reg    query_num_c157_blk_n;
reg    query_num_c158_blk_n;
reg    nprobe_c167_blk_n;
reg    nprobe_c185_blk_n;
reg   [31:0] nprobe_read_reg_344;
reg   [31:0] query_num_read_reg_349;
wire   [27:0] size_cell_IDs_fu_275_p3;
reg   [27:0] size_cell_IDs_reg_354;
wire   [36:0] bound_fu_303_p2;
reg   [36:0] bound_reg_359;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_start;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_done;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_idle;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_ready;
wire   [31:0] grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_s_cell_ID66_din;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_s_cell_ID66_write;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_s_kernel_network_in65_read;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_ext_blocking_n;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_str_blocking_n;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_int_blocking_n;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_start;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_done;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_idle;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_ready;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_s_kernel_network_in65_read;
wire   [511:0] grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_s_query_vectors67_din;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_s_query_vectors67_write;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_ext_blocking_n;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_str_blocking_n;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_int_blocking_n;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_start;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_done;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_idle;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_ready;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_s_kernel_network_in65_read;
wire   [511:0] grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_s_center_vectors68_din;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_s_center_vectors68_write;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_ext_blocking_n;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_str_blocking_n;
wire    grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_int_blocking_n;
reg    grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_start_reg;
wire    ap_CS_fsm_state3;
wire    ap_CS_fsm_state4;
reg    grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_start_reg;
wire    ap_CS_fsm_state6;
wire    ap_CS_fsm_state7;
reg    grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_start_reg;
wire    ap_CS_fsm_state8;
reg   [9:0] ap_NS_fsm;
wire    ap_NS_fsm_state9;
wire    ap_CS_fsm_state10;
reg   [30:0] query_id_fu_96;
wire   [30:0] add_ln59_fu_326_p2;
reg    ap_block_state2;
reg    ap_block_state1;
wire   [3:0] empty_331_fu_191_p1;
wire   [5:0] mul_cast_fu_195_p3;
wire   [31:0] empty_fu_185_p2;
wire   [31:0] p_neg_fu_217_p2;
wire   [25:0] p_lshr_cast_cast_fu_223_p4;
wire   [26:0] p_lshr_cast_cast_cast_fu_233_p1;
wire   [25:0] p_lshr_f_cast_fu_243_p4;
wire   [0:0] tmp_fu_209_p3;
wire   [26:0] p_neg_t_fu_237_p2;
wire   [26:0] p_lshr_f_cast_cast_fu_253_p1;
wire   [26:0] div_fu_257_p3;
wire  signed [27:0] div_cast_fu_265_p1;
wire   [0:0] cmp2_fu_203_p2;
wire   [27:0] add_fu_269_p2;
wire   [34:0] tmp_s_fu_291_p3;
wire   [36:0] p_shl_fu_283_p3;
wire   [36:0] p_shl2_fu_299_p1;
wire   [31:0] zext_ln59_fu_317_p1;
reg    ap_ST_fsm_state1_blk;
reg    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
reg    ap_ST_fsm_state4_blk;
wire    ap_ST_fsm_state5_blk;
wire    ap_ST_fsm_state6_blk;
reg    ap_ST_fsm_state7_blk;
wire    ap_ST_fsm_state8_blk;
wire    ap_ST_fsm_state9_blk;
reg    ap_ST_fsm_state10_blk;
wire    ap_int_blocking_cur_n;
reg    ap_ext_blocking_sub_n;
reg    ap_wait_0;
reg    ap_sub_ext_blocking_0;
reg    ap_wait_1;
reg    ap_sub_ext_blocking_1;
reg    ap_wait_2;
reg    ap_sub_ext_blocking_2;
reg    ap_str_blocking_sub_n;
reg    ap_sub_str_blocking_0;
reg    ap_sub_str_blocking_1;
reg    ap_sub_str_blocking_2;
reg    ap_int_blocking_sub_n;
reg    ap_sub_int_blocking_0;
reg    ap_sub_int_blocking_1;
reg    ap_sub_int_blocking_2;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 start_once_reg = 1'b0;
#0 ap_done_reg = 1'b0;
#0 ap_CS_fsm = 10'd1;
#0 grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_start_reg = 1'b0;
#0 grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_start_reg = 1'b0;
#0 grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_start_reg = 1'b0;
end

vadd_network_input_processing_Pipeline_VITIS_LOOP_71_2 grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_start),
    .ap_done(grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_done),
    .ap_idle(grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_idle),
    .ap_ready(grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_ready),
    .s_cell_ID66_din(grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_s_cell_ID66_din),
    .s_cell_ID66_num_data_valid(9'd0),
    .s_cell_ID66_fifo_cap(9'd0),
    .s_cell_ID66_full_n(s_cell_ID66_full_n),
    .s_cell_ID66_write(grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_s_cell_ID66_write),
    .size_cell_IDs(size_cell_IDs_reg_354),
    .nprobe_load(nprobe_read_reg_344),
    .s_kernel_network_in65_dout(s_kernel_network_in65_dout),
    .s_kernel_network_in65_num_data_valid(10'd0),
    .s_kernel_network_in65_fifo_cap(10'd0),
    .s_kernel_network_in65_empty_n(s_kernel_network_in65_empty_n),
    .s_kernel_network_in65_read(grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_s_kernel_network_in65_read),
    .ap_ext_blocking_n(grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_ext_blocking_n),
    .ap_str_blocking_n(grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_str_blocking_n),
    .ap_int_blocking_n(grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_int_blocking_n)
);

vadd_network_input_processing_Pipeline_VITIS_LOOP_88_4 grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_start),
    .ap_done(grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_done),
    .ap_idle(grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_idle),
    .ap_ready(grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_ready),
    .s_kernel_network_in65_dout(s_kernel_network_in65_dout),
    .s_kernel_network_in65_num_data_valid(10'd0),
    .s_kernel_network_in65_fifo_cap(10'd0),
    .s_kernel_network_in65_empty_n(s_kernel_network_in65_empty_n),
    .s_kernel_network_in65_read(grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_s_kernel_network_in65_read),
    .s_query_vectors67_din(grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_s_query_vectors67_din),
    .s_query_vectors67_num_data_valid(9'd0),
    .s_query_vectors67_fifo_cap(9'd0),
    .s_query_vectors67_full_n(s_query_vectors67_full_n),
    .s_query_vectors67_write(grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_s_query_vectors67_write),
    .ap_ext_blocking_n(grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_ext_blocking_n),
    .ap_str_blocking_n(grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_str_blocking_n),
    .ap_int_blocking_n(grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_int_blocking_n)
);

vadd_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6 grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_start),
    .ap_done(grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_done),
    .ap_idle(grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_idle),
    .ap_ready(grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_ready),
    .s_kernel_network_in65_dout(s_kernel_network_in65_dout),
    .s_kernel_network_in65_num_data_valid(10'd0),
    .s_kernel_network_in65_fifo_cap(10'd0),
    .s_kernel_network_in65_empty_n(s_kernel_network_in65_empty_n),
    .s_kernel_network_in65_read(grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_s_kernel_network_in65_read),
    .s_center_vectors68_din(grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_s_center_vectors68_din),
    .s_center_vectors68_num_data_valid(9'd0),
    .s_center_vectors68_fifo_cap(9'd0),
    .s_center_vectors68_full_n(s_center_vectors68_full_n),
    .s_center_vectors68_write(grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_s_center_vectors68_write),
    .bound(bound_reg_359),
    .ap_ext_blocking_n(grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_ext_blocking_n),
    .ap_str_blocking_n(grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_str_blocking_n),
    .ap_int_blocking_n(grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_int_blocking_n)
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
        end else if ((~((s_kernel_network_in65_empty_n == 1'b0) & (icmp_ln59_fu_321_p2 == 1'd1)) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln59_fu_321_p2 == 1'd0))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state3)) begin
            grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_start_reg <= 1'b1;
        end else if ((grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_ready == 1'b1)) begin
            grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state6)) begin
            grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_start_reg <= 1'b1;
        end else if ((grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_ready == 1'b1)) begin
            grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b1 == ap_NS_fsm_state9) & (1'b1 == ap_CS_fsm_state8))) begin
            grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_start_reg <= 1'b1;
        end else if ((grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_ready == 1'b1)) begin
            grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_start_reg <= 1'b0;
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
    if ((~((query_num_c158_full_n == 1'b0) | (query_num_c157_full_n == 1'b0) | (query_num_c139_full_n == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0) | (nprobe_c185_full_n == 1'b0) | (nprobe_c167_full_n == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_id_fu_96 <= 31'd0;
    end else if ((~((s_kernel_network_in65_empty_n == 1'b0) & (icmp_ln59_fu_321_p2 == 1'd1)) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln59_fu_321_p2 == 1'd1))) begin
        query_id_fu_96 <= add_ln59_fu_326_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state1)) begin
        bound_reg_359[36 : 3] <= bound_fu_303_p2[36 : 3];
        nprobe_read_reg_344 <= nprobe_dout;
        query_num_read_reg_349 <= query_num_dout;
        size_cell_IDs_reg_354 <= size_cell_IDs_fu_275_p3;
    end
end

always @ (*) begin
    if ((grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_done == 1'b0)) begin
        ap_ST_fsm_state10_blk = 1'b1;
    end else begin
        ap_ST_fsm_state10_blk = 1'b0;
    end
end

always @ (*) begin
    if (((query_num_c158_full_n == 1'b0) | (query_num_c157_full_n == 1'b0) | (query_num_c139_full_n == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0) | (nprobe_c185_full_n == 1'b0) | (nprobe_c167_full_n == 1'b0))) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

always @ (*) begin
    if (((s_kernel_network_in65_empty_n == 1'b0) & (icmp_ln59_fu_321_p2 == 1'd1))) begin
        ap_ST_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_fsm_state2_blk = 1'b0;
    end
end

assign ap_ST_fsm_state3_blk = 1'b0;

always @ (*) begin
    if ((grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_done == 1'b0)) begin
        ap_ST_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_fsm_state4_blk = 1'b0;
    end
end

assign ap_ST_fsm_state5_blk = 1'b0;

assign ap_ST_fsm_state6_blk = 1'b0;

always @ (*) begin
    if ((grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_done == 1'b0)) begin
        ap_ST_fsm_state7_blk = 1'b1;
    end else begin
        ap_ST_fsm_state7_blk = 1'b0;
    end
end

assign ap_ST_fsm_state8_blk = 1'b0;

assign ap_ST_fsm_state9_blk = 1'b0;

always @ (*) begin
    if ((~((s_kernel_network_in65_empty_n == 1'b0) & (icmp_ln59_fu_321_p2 == 1'd1)) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln59_fu_321_p2 == 1'd0))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = ap_done_reg;
    end
end

always @ (*) begin
    if ((((ap_wait_2 & ap_sub_ext_blocking_2) == 1'b1) & ((ap_wait_1 & ap_sub_ext_blocking_1) == 1'b1) & ((ap_wait_0 & ap_sub_ext_blocking_0) == 1'b1))) begin
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
    if ((((ap_wait_2 & ap_sub_int_blocking_2) == 1'b1) & ((ap_wait_1 & ap_sub_int_blocking_1) == 1'b1) & ((ap_wait_0 & ap_sub_int_blocking_0) == 1'b1))) begin
        ap_int_blocking_sub_n = 1'b0;
    end else begin
        ap_int_blocking_sub_n = 1'b1;
    end
end

always @ (*) begin
    if ((((ap_wait_2 & ap_sub_str_blocking_2) == 1'b1) & ((ap_wait_1 & ap_sub_str_blocking_1) == 1'b1) & ((ap_wait_0 & ap_sub_str_blocking_0) == 1'b1))) begin
        ap_str_blocking_sub_n = 1'b0;
    end else begin
        ap_str_blocking_sub_n = 1'b1;
    end
end

always @ (*) begin
    if ((grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_ext_blocking_n == 1'b0)) begin
        ap_sub_ext_blocking_0 = 1'b1;
    end else begin
        ap_sub_ext_blocking_0 = 1'b0;
    end
end

always @ (*) begin
    if ((grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_ext_blocking_n == 1'b0)) begin
        ap_sub_ext_blocking_1 = 1'b1;
    end else begin
        ap_sub_ext_blocking_1 = 1'b0;
    end
end

always @ (*) begin
    if ((grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_ext_blocking_n == 1'b0)) begin
        ap_sub_ext_blocking_2 = 1'b1;
    end else begin
        ap_sub_ext_blocking_2 = 1'b0;
    end
end

always @ (*) begin
    if ((grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_int_blocking_n == 1'b0)) begin
        ap_sub_int_blocking_0 = 1'b1;
    end else begin
        ap_sub_int_blocking_0 = 1'b0;
    end
end

always @ (*) begin
    if ((grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_int_blocking_n == 1'b0)) begin
        ap_sub_int_blocking_1 = 1'b1;
    end else begin
        ap_sub_int_blocking_1 = 1'b0;
    end
end

always @ (*) begin
    if ((grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_int_blocking_n == 1'b0)) begin
        ap_sub_int_blocking_2 = 1'b1;
    end else begin
        ap_sub_int_blocking_2 = 1'b0;
    end
end

always @ (*) begin
    if ((grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_str_blocking_n == 1'b0)) begin
        ap_sub_str_blocking_0 = 1'b1;
    end else begin
        ap_sub_str_blocking_0 = 1'b0;
    end
end

always @ (*) begin
    if ((grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_str_blocking_n == 1'b0)) begin
        ap_sub_str_blocking_1 = 1'b1;
    end else begin
        ap_sub_str_blocking_1 = 1'b0;
    end
end

always @ (*) begin
    if ((grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_str_blocking_n == 1'b0)) begin
        ap_sub_str_blocking_2 = 1'b1;
    end else begin
        ap_sub_str_blocking_2 = 1'b0;
    end
end

always @ (*) begin
    if ((ap_ST_fsm_state4 == ap_CS_fsm)) begin
        ap_wait_0 = 1'b1;
    end else begin
        ap_wait_0 = 1'b0;
    end
end

always @ (*) begin
    if ((ap_ST_fsm_state7 == ap_CS_fsm)) begin
        ap_wait_1 = 1'b1;
    end else begin
        ap_wait_1 = 1'b0;
    end
end

always @ (*) begin
    if ((ap_ST_fsm_state10 == ap_CS_fsm)) begin
        ap_wait_2 = 1'b1;
    end else begin
        ap_wait_2 = 1'b0;
    end
end

always @ (*) begin
    if ((~((s_kernel_network_in65_empty_n == 1'b0) & (icmp_ln59_fu_321_p2 == 1'd1)) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln59_fu_321_p2 == 1'd0))) begin
        internal_ap_ready = 1'b1;
    end else begin
        internal_ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        nprobe_blk_n = nprobe_empty_n;
    end else begin
        nprobe_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        nprobe_c167_blk_n = nprobe_c167_full_n;
    end else begin
        nprobe_c167_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((query_num_c158_full_n == 1'b0) | (query_num_c157_full_n == 1'b0) | (query_num_c139_full_n == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0) | (nprobe_c185_full_n == 1'b0) | (nprobe_c167_full_n == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        nprobe_c167_write = 1'b1;
    end else begin
        nprobe_c167_write = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        nprobe_c185_blk_n = nprobe_c185_full_n;
    end else begin
        nprobe_c185_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((query_num_c158_full_n == 1'b0) | (query_num_c157_full_n == 1'b0) | (query_num_c139_full_n == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0) | (nprobe_c185_full_n == 1'b0) | (nprobe_c167_full_n == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        nprobe_c185_write = 1'b1;
    end else begin
        nprobe_c185_write = 1'b0;
    end
end

always @ (*) begin
    if ((~((query_num_c158_full_n == 1'b0) | (query_num_c157_full_n == 1'b0) | (query_num_c139_full_n == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0) | (nprobe_c185_full_n == 1'b0) | (nprobe_c167_full_n == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        nprobe_read = 1'b1;
    end else begin
        nprobe_read = 1'b0;
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
        query_num_c139_blk_n = query_num_c139_full_n;
    end else begin
        query_num_c139_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((query_num_c158_full_n == 1'b0) | (query_num_c157_full_n == 1'b0) | (query_num_c139_full_n == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0) | (nprobe_c185_full_n == 1'b0) | (nprobe_c167_full_n == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_c139_write = 1'b1;
    end else begin
        query_num_c139_write = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_c157_blk_n = query_num_c157_full_n;
    end else begin
        query_num_c157_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((query_num_c158_full_n == 1'b0) | (query_num_c157_full_n == 1'b0) | (query_num_c139_full_n == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0) | (nprobe_c185_full_n == 1'b0) | (nprobe_c167_full_n == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_c157_write = 1'b1;
    end else begin
        query_num_c157_write = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (real_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_c158_blk_n = query_num_c158_full_n;
    end else begin
        query_num_c158_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((query_num_c158_full_n == 1'b0) | (query_num_c157_full_n == 1'b0) | (query_num_c139_full_n == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0) | (nprobe_c185_full_n == 1'b0) | (nprobe_c167_full_n == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_c158_write = 1'b1;
    end else begin
        query_num_c158_write = 1'b0;
    end
end

always @ (*) begin
    if ((~((query_num_c158_full_n == 1'b0) | (query_num_c157_full_n == 1'b0) | (query_num_c139_full_n == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0) | (nprobe_c185_full_n == 1'b0) | (nprobe_c167_full_n == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
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
    if ((1'b1 == ap_CS_fsm_state4)) begin
        s_cell_ID66_write = grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_s_cell_ID66_write;
    end else begin
        s_cell_ID66_write = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state10)) begin
        s_center_vectors68_write = grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_s_center_vectors68_write;
    end else begin
        s_center_vectors68_write = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln59_fu_321_p2 == 1'd1))) begin
        s_kernel_network_in65_blk_n = s_kernel_network_in65_empty_n;
    end else begin
        s_kernel_network_in65_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((s_kernel_network_in65_empty_n == 1'b0) & (icmp_ln59_fu_321_p2 == 1'd1)) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln59_fu_321_p2 == 1'd1))) begin
        s_kernel_network_in65_read = 1'b1;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        s_kernel_network_in65_read = grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_s_kernel_network_in65_read;
    end else if ((1'b1 == ap_CS_fsm_state7)) begin
        s_kernel_network_in65_read = grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_s_kernel_network_in65_read;
    end else if ((1'b1 == ap_CS_fsm_state4)) begin
        s_kernel_network_in65_read = grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_s_kernel_network_in65_read;
    end else begin
        s_kernel_network_in65_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        s_query_vectors67_write = grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_s_query_vectors67_write;
    end else begin
        s_query_vectors67_write = 1'b0;
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
            if ((~((query_num_c158_full_n == 1'b0) | (query_num_c157_full_n == 1'b0) | (query_num_c139_full_n == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0) | (nprobe_c185_full_n == 1'b0) | (nprobe_c167_full_n == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if ((~((s_kernel_network_in65_empty_n == 1'b0) & (icmp_ln59_fu_321_p2 == 1'd1)) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln59_fu_321_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else if ((~((s_kernel_network_in65_empty_n == 1'b0) & (icmp_ln59_fu_321_p2 == 1'd1)) & (1'b1 == ap_CS_fsm_state2) & (icmp_ln59_fu_321_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end
        end
        ap_ST_fsm_state3 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        ap_ST_fsm_state4 : begin
            if (((1'b1 == ap_CS_fsm_state4) & (grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state7;
        end
        ap_ST_fsm_state7 : begin
            if (((1'b1 == ap_CS_fsm_state7) & (grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end
        end
        ap_ST_fsm_state8 : begin
            ap_NS_fsm = ap_ST_fsm_state9;
        end
        ap_ST_fsm_state9 : begin
            ap_NS_fsm = ap_ST_fsm_state10;
        end
        ap_ST_fsm_state10 : begin
            if (((1'b1 == ap_CS_fsm_state10) & (grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state10;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_fu_269_p2 = ($signed(div_cast_fu_265_p1) + $signed(28'd1));

assign add_ln59_fu_326_p2 = (query_id_fu_96 + 31'd1);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state10 = ap_CS_fsm[32'd9];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state7 = ap_CS_fsm[32'd6];

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd7];

assign ap_NS_fsm_state9 = ap_NS_fsm[32'd8];

always @ (*) begin
    ap_block_state1 = ((query_num_c158_full_n == 1'b0) | (query_num_c157_full_n == 1'b0) | (query_num_c139_full_n == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1) | (real_start == 1'b0) | (nprobe_c185_full_n == 1'b0) | (nprobe_c167_full_n == 1'b0));
end

always @ (*) begin
    ap_block_state2 = ((s_kernel_network_in65_empty_n == 1'b0) & (icmp_ln59_fu_321_p2 == 1'd1));
end

assign ap_ext_blocking_n = (ap_ext_blocking_sub_n & 1'b1);

assign ap_int_blocking_cur_n = (s_kernel_network_in65_blk_n & query_num_c158_blk_n & query_num_c157_blk_n & query_num_c139_blk_n & query_num_blk_n & nprobe_c185_blk_n & nprobe_c167_blk_n & nprobe_blk_n);

assign ap_int_blocking_n = (ap_int_blocking_sub_n & ap_int_blocking_cur_n);

assign ap_ready = internal_ap_ready;

assign ap_str_blocking_n = (ap_str_blocking_sub_n & 1'b1);

assign bound_fu_303_p2 = (p_shl_fu_283_p3 - p_shl2_fu_299_p1);

assign cmp2_fu_203_p2 = ((mul_cast_fu_195_p3 == 6'd0) ? 1'b1 : 1'b0);

assign div_cast_fu_265_p1 = $signed(div_fu_257_p3);

assign div_fu_257_p3 = ((tmp_fu_209_p3[0:0] == 1'b1) ? p_neg_t_fu_237_p2 : p_lshr_f_cast_cast_fu_253_p1);

assign empty_331_fu_191_p1 = nprobe_dout[3:0];

assign empty_fu_185_p2 = nprobe_dout << 32'd2;

assign grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_start = grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_ap_start_reg;

assign grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_start = grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_ap_start_reg;

assign grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_start = grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_ap_start_reg;

assign icmp_ln59_fu_321_p2 = (($signed(zext_ln59_fu_317_p1) < $signed(query_num_read_reg_349)) ? 1'b1 : 1'b0);

assign mul_cast_fu_195_p3 = {{empty_331_fu_191_p1}, {2'd0}};

assign nprobe_c167_din = nprobe_dout;

assign nprobe_c185_din = nprobe_dout;

assign p_lshr_cast_cast_cast_fu_233_p1 = p_lshr_cast_cast_fu_223_p4;

assign p_lshr_cast_cast_fu_223_p4 = {{p_neg_fu_217_p2[31:6]}};

assign p_lshr_f_cast_cast_fu_253_p1 = p_lshr_f_cast_fu_243_p4;

assign p_lshr_f_cast_fu_243_p4 = {{nprobe_dout[29:4]}};

assign p_neg_fu_217_p2 = (32'd0 - empty_fu_185_p2);

assign p_neg_t_fu_237_p2 = (27'd0 - p_lshr_cast_cast_cast_fu_233_p1);

assign p_shl2_fu_299_p1 = tmp_s_fu_291_p3;

assign p_shl_fu_283_p3 = {{nprobe_dout}, {5'd0}};

assign query_num_c139_din = query_num_dout;

assign query_num_c157_din = query_num_dout;

assign query_num_c158_din = query_num_dout;

assign s_cell_ID66_din = grp_network_input_processing_Pipeline_VITIS_LOOP_71_2_fu_158_s_cell_ID66_din;

assign s_center_vectors68_din = grp_network_input_processing_Pipeline_VITIS_LOOP_93_5_VITIS_LOOP_94_6_fu_176_s_center_vectors68_din;

assign s_query_vectors67_din = grp_network_input_processing_Pipeline_VITIS_LOOP_88_4_fu_168_s_query_vectors67_din;

assign size_cell_IDs_fu_275_p3 = ((cmp2_fu_203_p2[0:0] == 1'b1) ? div_cast_fu_265_p1 : add_fu_269_p2);

assign start_out = real_start;

assign tmp_fu_209_p3 = nprobe_dout[32'd29];

assign tmp_s_fu_291_p3 = {{nprobe_dout}, {3'd0}};

assign zext_ln59_fu_317_p1 = query_id_fu_96;

always @ (posedge ap_clk) begin
    bound_reg_359[2:0] <= 3'b000;
end

endmodule //vadd_network_input_processing