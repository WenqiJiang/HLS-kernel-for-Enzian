// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Version: 2022.1
// Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module vadd_merge_streams_38_8_s (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        query_num_dout,
        query_num_num_data_valid,
        query_num_fifo_cap,
        query_num_empty_n,
        query_num_read,
        s_intermediate_result_with_vec_ID_i_0_dout,
        s_intermediate_result_with_vec_ID_i_0_num_data_valid,
        s_intermediate_result_with_vec_ID_i_0_fifo_cap,
        s_intermediate_result_with_vec_ID_i_0_empty_n,
        s_intermediate_result_with_vec_ID_i_0_read,
        s_intermediate_result_with_vec_ID_i_1_dout,
        s_intermediate_result_with_vec_ID_i_1_num_data_valid,
        s_intermediate_result_with_vec_ID_i_1_fifo_cap,
        s_intermediate_result_with_vec_ID_i_1_empty_n,
        s_intermediate_result_with_vec_ID_i_1_read,
        s_intermediate_result_with_vec_ID_i_2_dout,
        s_intermediate_result_with_vec_ID_i_2_num_data_valid,
        s_intermediate_result_with_vec_ID_i_2_fifo_cap,
        s_intermediate_result_with_vec_ID_i_2_empty_n,
        s_intermediate_result_with_vec_ID_i_2_read,
        s_intermediate_result_with_vec_ID_i_3_dout,
        s_intermediate_result_with_vec_ID_i_3_num_data_valid,
        s_intermediate_result_with_vec_ID_i_3_fifo_cap,
        s_intermediate_result_with_vec_ID_i_3_empty_n,
        s_intermediate_result_with_vec_ID_i_3_read,
        s_intermediate_result_with_vec_ID_i_4_dout,
        s_intermediate_result_with_vec_ID_i_4_num_data_valid,
        s_intermediate_result_with_vec_ID_i_4_fifo_cap,
        s_intermediate_result_with_vec_ID_i_4_empty_n,
        s_intermediate_result_with_vec_ID_i_4_read,
        s_intermediate_result_with_vec_ID_i_5_dout,
        s_intermediate_result_with_vec_ID_i_5_num_data_valid,
        s_intermediate_result_with_vec_ID_i_5_fifo_cap,
        s_intermediate_result_with_vec_ID_i_5_empty_n,
        s_intermediate_result_with_vec_ID_i_5_read,
        s_intermediate_result_with_vec_ID_i_6_dout,
        s_intermediate_result_with_vec_ID_i_6_num_data_valid,
        s_intermediate_result_with_vec_ID_i_6_fifo_cap,
        s_intermediate_result_with_vec_ID_i_6_empty_n,
        s_intermediate_result_with_vec_ID_i_6_read,
        s_intermediate_result_with_vec_ID_i_7_dout,
        s_intermediate_result_with_vec_ID_i_7_num_data_valid,
        s_intermediate_result_with_vec_ID_i_7_fifo_cap,
        s_intermediate_result_with_vec_ID_i_7_empty_n,
        s_intermediate_result_with_vec_ID_i_7_read,
        s_merged_intermediate_result_i_din,
        s_merged_intermediate_result_i_num_data_valid,
        s_merged_intermediate_result_i_fifo_cap,
        s_merged_intermediate_result_i_full_n,
        s_merged_intermediate_result_i_write,
        query_num_c116_din,
        query_num_c116_num_data_valid,
        query_num_c116_fifo_cap,
        query_num_c116_full_n,
        query_num_c116_write,
        ap_ext_blocking_n,
        ap_str_blocking_n,
        ap_int_blocking_n
);

parameter    ap_ST_fsm_state1 = 3'd1;
parameter    ap_ST_fsm_state2 = 3'd2;
parameter    ap_ST_fsm_state3 = 3'd4;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
input  [31:0] query_num_dout;
input  [1:0] query_num_num_data_valid;
input  [1:0] query_num_fifo_cap;
input   query_num_empty_n;
output   query_num_read;
input  [95:0] s_intermediate_result_with_vec_ID_i_0_dout;
input  [3:0] s_intermediate_result_with_vec_ID_i_0_num_data_valid;
input  [3:0] s_intermediate_result_with_vec_ID_i_0_fifo_cap;
input   s_intermediate_result_with_vec_ID_i_0_empty_n;
output   s_intermediate_result_with_vec_ID_i_0_read;
input  [95:0] s_intermediate_result_with_vec_ID_i_1_dout;
input  [3:0] s_intermediate_result_with_vec_ID_i_1_num_data_valid;
input  [3:0] s_intermediate_result_with_vec_ID_i_1_fifo_cap;
input   s_intermediate_result_with_vec_ID_i_1_empty_n;
output   s_intermediate_result_with_vec_ID_i_1_read;
input  [95:0] s_intermediate_result_with_vec_ID_i_2_dout;
input  [3:0] s_intermediate_result_with_vec_ID_i_2_num_data_valid;
input  [3:0] s_intermediate_result_with_vec_ID_i_2_fifo_cap;
input   s_intermediate_result_with_vec_ID_i_2_empty_n;
output   s_intermediate_result_with_vec_ID_i_2_read;
input  [95:0] s_intermediate_result_with_vec_ID_i_3_dout;
input  [3:0] s_intermediate_result_with_vec_ID_i_3_num_data_valid;
input  [3:0] s_intermediate_result_with_vec_ID_i_3_fifo_cap;
input   s_intermediate_result_with_vec_ID_i_3_empty_n;
output   s_intermediate_result_with_vec_ID_i_3_read;
input  [95:0] s_intermediate_result_with_vec_ID_i_4_dout;
input  [3:0] s_intermediate_result_with_vec_ID_i_4_num_data_valid;
input  [3:0] s_intermediate_result_with_vec_ID_i_4_fifo_cap;
input   s_intermediate_result_with_vec_ID_i_4_empty_n;
output   s_intermediate_result_with_vec_ID_i_4_read;
input  [95:0] s_intermediate_result_with_vec_ID_i_5_dout;
input  [3:0] s_intermediate_result_with_vec_ID_i_5_num_data_valid;
input  [3:0] s_intermediate_result_with_vec_ID_i_5_fifo_cap;
input   s_intermediate_result_with_vec_ID_i_5_empty_n;
output   s_intermediate_result_with_vec_ID_i_5_read;
input  [95:0] s_intermediate_result_with_vec_ID_i_6_dout;
input  [3:0] s_intermediate_result_with_vec_ID_i_6_num_data_valid;
input  [3:0] s_intermediate_result_with_vec_ID_i_6_fifo_cap;
input   s_intermediate_result_with_vec_ID_i_6_empty_n;
output   s_intermediate_result_with_vec_ID_i_6_read;
input  [95:0] s_intermediate_result_with_vec_ID_i_7_dout;
input  [3:0] s_intermediate_result_with_vec_ID_i_7_num_data_valid;
input  [3:0] s_intermediate_result_with_vec_ID_i_7_fifo_cap;
input   s_intermediate_result_with_vec_ID_i_7_empty_n;
output   s_intermediate_result_with_vec_ID_i_7_read;
output  [95:0] s_merged_intermediate_result_i_din;
input  [9:0] s_merged_intermediate_result_i_num_data_valid;
input  [9:0] s_merged_intermediate_result_i_fifo_cap;
input   s_merged_intermediate_result_i_full_n;
output   s_merged_intermediate_result_i_write;
output  [31:0] query_num_c116_din;
input  [1:0] query_num_c116_num_data_valid;
input  [1:0] query_num_c116_fifo_cap;
input   query_num_c116_full_n;
output   query_num_c116_write;
output   ap_ext_blocking_n;
output   ap_str_blocking_n;
output   ap_int_blocking_n;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg query_num_read;
reg s_intermediate_result_with_vec_ID_i_0_read;
reg s_intermediate_result_with_vec_ID_i_1_read;
reg s_intermediate_result_with_vec_ID_i_2_read;
reg s_intermediate_result_with_vec_ID_i_3_read;
reg s_intermediate_result_with_vec_ID_i_4_read;
reg s_intermediate_result_with_vec_ID_i_5_read;
reg s_intermediate_result_with_vec_ID_i_6_read;
reg s_intermediate_result_with_vec_ID_i_7_read;
reg s_merged_intermediate_result_i_write;
reg query_num_c116_write;

reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [2:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    query_num_blk_n;
reg    query_num_c116_blk_n;
reg   [31:0] query_num_read_reg_99;
wire   [40:0] bound4_fu_92_p2;
reg   [40:0] bound4_reg_104;
wire    ap_CS_fsm_state2;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_start;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_done;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_idle;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_ready;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_6_read;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_5_read;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_4_read;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_3_read;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_2_read;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_1_read;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_0_read;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_7_read;
wire   [95:0] grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_merged_intermediate_result_i_din;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_merged_intermediate_result_i_write;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_ext_blocking_n;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_str_blocking_n;
wire    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_int_blocking_n;
reg    grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_start_reg;
wire    ap_CS_fsm_state3;
reg    ap_block_state1;
wire   [31:0] bound4_fu_92_p0;
wire   [9:0] bound4_fu_92_p1;
reg   [2:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
reg    ap_ST_fsm_state3_blk;
wire    ap_int_blocking_cur_n;
reg    ap_ext_blocking_sub_n;
reg    ap_wait_0;
reg    ap_sub_ext_blocking_0;
reg    ap_str_blocking_sub_n;
reg    ap_sub_str_blocking_0;
reg    ap_int_blocking_sub_n;
reg    ap_sub_int_blocking_0;
wire   [40:0] bound4_fu_92_p00;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_done_reg = 1'b0;
#0 ap_CS_fsm = 3'd1;
#0 grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_start_reg = 1'b0;
end

vadd_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3 grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_start),
    .ap_done(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_done),
    .ap_idle(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_idle),
    .ap_ready(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_ready),
    .s_intermediate_result_with_vec_ID_i_6_dout(s_intermediate_result_with_vec_ID_i_6_dout),
    .s_intermediate_result_with_vec_ID_i_6_num_data_valid(4'd0),
    .s_intermediate_result_with_vec_ID_i_6_fifo_cap(4'd0),
    .s_intermediate_result_with_vec_ID_i_6_empty_n(s_intermediate_result_with_vec_ID_i_6_empty_n),
    .s_intermediate_result_with_vec_ID_i_6_read(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_6_read),
    .s_intermediate_result_with_vec_ID_i_5_dout(s_intermediate_result_with_vec_ID_i_5_dout),
    .s_intermediate_result_with_vec_ID_i_5_num_data_valid(4'd0),
    .s_intermediate_result_with_vec_ID_i_5_fifo_cap(4'd0),
    .s_intermediate_result_with_vec_ID_i_5_empty_n(s_intermediate_result_with_vec_ID_i_5_empty_n),
    .s_intermediate_result_with_vec_ID_i_5_read(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_5_read),
    .s_intermediate_result_with_vec_ID_i_4_dout(s_intermediate_result_with_vec_ID_i_4_dout),
    .s_intermediate_result_with_vec_ID_i_4_num_data_valid(4'd0),
    .s_intermediate_result_with_vec_ID_i_4_fifo_cap(4'd0),
    .s_intermediate_result_with_vec_ID_i_4_empty_n(s_intermediate_result_with_vec_ID_i_4_empty_n),
    .s_intermediate_result_with_vec_ID_i_4_read(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_4_read),
    .s_intermediate_result_with_vec_ID_i_3_dout(s_intermediate_result_with_vec_ID_i_3_dout),
    .s_intermediate_result_with_vec_ID_i_3_num_data_valid(4'd0),
    .s_intermediate_result_with_vec_ID_i_3_fifo_cap(4'd0),
    .s_intermediate_result_with_vec_ID_i_3_empty_n(s_intermediate_result_with_vec_ID_i_3_empty_n),
    .s_intermediate_result_with_vec_ID_i_3_read(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_3_read),
    .s_intermediate_result_with_vec_ID_i_2_dout(s_intermediate_result_with_vec_ID_i_2_dout),
    .s_intermediate_result_with_vec_ID_i_2_num_data_valid(4'd0),
    .s_intermediate_result_with_vec_ID_i_2_fifo_cap(4'd0),
    .s_intermediate_result_with_vec_ID_i_2_empty_n(s_intermediate_result_with_vec_ID_i_2_empty_n),
    .s_intermediate_result_with_vec_ID_i_2_read(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_2_read),
    .s_intermediate_result_with_vec_ID_i_1_dout(s_intermediate_result_with_vec_ID_i_1_dout),
    .s_intermediate_result_with_vec_ID_i_1_num_data_valid(4'd0),
    .s_intermediate_result_with_vec_ID_i_1_fifo_cap(4'd0),
    .s_intermediate_result_with_vec_ID_i_1_empty_n(s_intermediate_result_with_vec_ID_i_1_empty_n),
    .s_intermediate_result_with_vec_ID_i_1_read(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_1_read),
    .s_intermediate_result_with_vec_ID_i_0_dout(s_intermediate_result_with_vec_ID_i_0_dout),
    .s_intermediate_result_with_vec_ID_i_0_num_data_valid(4'd0),
    .s_intermediate_result_with_vec_ID_i_0_fifo_cap(4'd0),
    .s_intermediate_result_with_vec_ID_i_0_empty_n(s_intermediate_result_with_vec_ID_i_0_empty_n),
    .s_intermediate_result_with_vec_ID_i_0_read(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_0_read),
    .s_intermediate_result_with_vec_ID_i_7_dout(s_intermediate_result_with_vec_ID_i_7_dout),
    .s_intermediate_result_with_vec_ID_i_7_num_data_valid(4'd0),
    .s_intermediate_result_with_vec_ID_i_7_fifo_cap(4'd0),
    .s_intermediate_result_with_vec_ID_i_7_empty_n(s_intermediate_result_with_vec_ID_i_7_empty_n),
    .s_intermediate_result_with_vec_ID_i_7_read(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_7_read),
    .s_merged_intermediate_result_i_din(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_merged_intermediate_result_i_din),
    .s_merged_intermediate_result_i_num_data_valid(10'd0),
    .s_merged_intermediate_result_i_fifo_cap(10'd0),
    .s_merged_intermediate_result_i_full_n(s_merged_intermediate_result_i_full_n),
    .s_merged_intermediate_result_i_write(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_merged_intermediate_result_i_write),
    .bound4(bound4_reg_104),
    .ap_ext_blocking_n(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_ext_blocking_n),
    .ap_str_blocking_n(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_str_blocking_n),
    .ap_int_blocking_n(grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_int_blocking_n)
);

vadd_mul_32ns_10ns_41_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 10 ),
    .dout_WIDTH( 41 ))
mul_32ns_10ns_41_1_1_U7763(
    .din0(bound4_fu_92_p0),
    .din1(bound4_fu_92_p1),
    .dout(bound4_fu_92_p2)
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
        end else if (((1'b1 == ap_CS_fsm_state3) & (grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_done == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state2)) begin
            grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_start_reg <= 1'b1;
        end else if ((grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_ready == 1'b1)) begin
            grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        bound4_reg_104 <= bound4_fu_92_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state1)) begin
        query_num_read_reg_99 <= query_num_dout;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) | (query_num_c116_full_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1))) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

always @ (*) begin
    if ((grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_done == 1'b0)) begin
        ap_ST_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_fsm_state3_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) & (grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_done == 1'b1))) begin
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
    if (((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
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
    if (((1'b1 == ap_CS_fsm_state3) & (grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_done == 1'b1))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
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
    if ((grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_ext_blocking_n == 1'b0)) begin
        ap_sub_ext_blocking_0 = 1'b1;
    end else begin
        ap_sub_ext_blocking_0 = 1'b0;
    end
end

always @ (*) begin
    if ((grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_int_blocking_n == 1'b0)) begin
        ap_sub_int_blocking_0 = 1'b1;
    end else begin
        ap_sub_int_blocking_0 = 1'b0;
    end
end

always @ (*) begin
    if ((grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_str_blocking_n == 1'b0)) begin
        ap_sub_str_blocking_0 = 1'b1;
    end else begin
        ap_sub_str_blocking_0 = 1'b0;
    end
end

always @ (*) begin
    if ((ap_ST_fsm_state3 == ap_CS_fsm)) begin
        ap_wait_0 = 1'b1;
    end else begin
        ap_wait_0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_blk_n = query_num_empty_n;
    end else begin
        query_num_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_c116_blk_n = query_num_c116_full_n;
    end else begin
        query_num_c116_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_start == 1'b0) | (query_num_c116_full_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_c116_write = 1'b1;
    end else begin
        query_num_c116_write = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_start == 1'b0) | (query_num_c116_full_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_read = 1'b1;
    end else begin
        query_num_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_intermediate_result_with_vec_ID_i_0_read = grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_0_read;
    end else begin
        s_intermediate_result_with_vec_ID_i_0_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_intermediate_result_with_vec_ID_i_1_read = grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_1_read;
    end else begin
        s_intermediate_result_with_vec_ID_i_1_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_intermediate_result_with_vec_ID_i_2_read = grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_2_read;
    end else begin
        s_intermediate_result_with_vec_ID_i_2_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_intermediate_result_with_vec_ID_i_3_read = grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_3_read;
    end else begin
        s_intermediate_result_with_vec_ID_i_3_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_intermediate_result_with_vec_ID_i_4_read = grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_4_read;
    end else begin
        s_intermediate_result_with_vec_ID_i_4_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_intermediate_result_with_vec_ID_i_5_read = grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_5_read;
    end else begin
        s_intermediate_result_with_vec_ID_i_5_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_intermediate_result_with_vec_ID_i_6_read = grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_6_read;
    end else begin
        s_intermediate_result_with_vec_ID_i_6_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_intermediate_result_with_vec_ID_i_7_read = grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_intermediate_result_with_vec_ID_i_7_read;
    end else begin
        s_intermediate_result_with_vec_ID_i_7_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_merged_intermediate_result_i_write = grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_merged_intermediate_result_i_write;
    end else begin
        s_merged_intermediate_result_i_write = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if ((~((ap_start == 1'b0) | (query_num_c116_full_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            if (((1'b1 == ap_CS_fsm_state3) & (grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

always @ (*) begin
    ap_block_state1 = ((ap_start == 1'b0) | (query_num_c116_full_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1));
end

assign ap_ext_blocking_n = (ap_ext_blocking_sub_n & 1'b1);

assign ap_int_blocking_cur_n = (query_num_c116_blk_n & query_num_blk_n);

assign ap_int_blocking_n = (ap_int_blocking_sub_n & ap_int_blocking_cur_n);

assign ap_str_blocking_n = (ap_str_blocking_sub_n & 1'b1);

assign bound4_fu_92_p0 = bound4_fu_92_p00;

assign bound4_fu_92_p00 = query_num_read_reg_99;

assign bound4_fu_92_p1 = 41'd304;

assign grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_start = grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_ap_start_reg;

assign query_num_c116_din = query_num_dout;

assign s_merged_intermediate_result_i_din = grp_merge_streams_38_8_Pipeline_VITIS_LOOP_175_1_VITIS_LOOP_176_2_VITIS_LOOP_177_3_fu_66_s_merged_intermediate_result_i_din;

endmodule //vadd_merge_streams_38_8_s
