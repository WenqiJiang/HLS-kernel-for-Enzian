// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Version: 2022.1
// Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module vadd_replicate_s_scanned_entries_every_cell (
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
        nprobe_dout,
        nprobe_num_data_valid,
        nprobe_fifo_cap,
        nprobe_empty_n,
        nprobe_read,
        s_scanned_entries_every_cell75_dout,
        s_scanned_entries_every_cell75_num_data_valid,
        s_scanned_entries_every_cell75_fifo_cap,
        s_scanned_entries_every_cell75_empty_n,
        s_scanned_entries_every_cell75_read,
        s_scanned_entries_every_cell_ADC_0_din,
        s_scanned_entries_every_cell_ADC_0_num_data_valid,
        s_scanned_entries_every_cell_ADC_0_fifo_cap,
        s_scanned_entries_every_cell_ADC_0_full_n,
        s_scanned_entries_every_cell_ADC_0_write,
        s_scanned_entries_every_cell_ADC_1_din,
        s_scanned_entries_every_cell_ADC_1_num_data_valid,
        s_scanned_entries_every_cell_ADC_1_fifo_cap,
        s_scanned_entries_every_cell_ADC_1_full_n,
        s_scanned_entries_every_cell_ADC_1_write,
        s_scanned_entries_every_cell_ADC_2_din,
        s_scanned_entries_every_cell_ADC_2_num_data_valid,
        s_scanned_entries_every_cell_ADC_2_fifo_cap,
        s_scanned_entries_every_cell_ADC_2_full_n,
        s_scanned_entries_every_cell_ADC_2_write,
        s_scanned_entries_every_cell_ADC_3_din,
        s_scanned_entries_every_cell_ADC_3_num_data_valid,
        s_scanned_entries_every_cell_ADC_3_fifo_cap,
        s_scanned_entries_every_cell_ADC_3_full_n,
        s_scanned_entries_every_cell_ADC_3_write,
        s_scanned_entries_every_cell_load_PQ_codes79_din,
        s_scanned_entries_every_cell_load_PQ_codes79_num_data_valid,
        s_scanned_entries_every_cell_load_PQ_codes79_fifo_cap,
        s_scanned_entries_every_cell_load_PQ_codes79_full_n,
        s_scanned_entries_every_cell_load_PQ_codes79_write,
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
input  [31:0] nprobe_dout;
input  [1:0] nprobe_num_data_valid;
input  [1:0] nprobe_fifo_cap;
input   nprobe_empty_n;
output   nprobe_read;
input  [31:0] s_scanned_entries_every_cell75_dout;
input  [8:0] s_scanned_entries_every_cell75_num_data_valid;
input  [8:0] s_scanned_entries_every_cell75_fifo_cap;
input   s_scanned_entries_every_cell75_empty_n;
output   s_scanned_entries_every_cell75_read;
output  [31:0] s_scanned_entries_every_cell_ADC_0_din;
input  [8:0] s_scanned_entries_every_cell_ADC_0_num_data_valid;
input  [8:0] s_scanned_entries_every_cell_ADC_0_fifo_cap;
input   s_scanned_entries_every_cell_ADC_0_full_n;
output   s_scanned_entries_every_cell_ADC_0_write;
output  [31:0] s_scanned_entries_every_cell_ADC_1_din;
input  [8:0] s_scanned_entries_every_cell_ADC_1_num_data_valid;
input  [8:0] s_scanned_entries_every_cell_ADC_1_fifo_cap;
input   s_scanned_entries_every_cell_ADC_1_full_n;
output   s_scanned_entries_every_cell_ADC_1_write;
output  [31:0] s_scanned_entries_every_cell_ADC_2_din;
input  [8:0] s_scanned_entries_every_cell_ADC_2_num_data_valid;
input  [8:0] s_scanned_entries_every_cell_ADC_2_fifo_cap;
input   s_scanned_entries_every_cell_ADC_2_full_n;
output   s_scanned_entries_every_cell_ADC_2_write;
output  [31:0] s_scanned_entries_every_cell_ADC_3_din;
input  [8:0] s_scanned_entries_every_cell_ADC_3_num_data_valid;
input  [8:0] s_scanned_entries_every_cell_ADC_3_fifo_cap;
input   s_scanned_entries_every_cell_ADC_3_full_n;
output   s_scanned_entries_every_cell_ADC_3_write;
output  [31:0] s_scanned_entries_every_cell_load_PQ_codes79_din;
input  [8:0] s_scanned_entries_every_cell_load_PQ_codes79_num_data_valid;
input  [8:0] s_scanned_entries_every_cell_load_PQ_codes79_fifo_cap;
input   s_scanned_entries_every_cell_load_PQ_codes79_full_n;
output   s_scanned_entries_every_cell_load_PQ_codes79_write;
output   ap_ext_blocking_n;
output   ap_str_blocking_n;
output   ap_int_blocking_n;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg query_num_read;
reg nprobe_read;
reg s_scanned_entries_every_cell75_read;
reg s_scanned_entries_every_cell_ADC_0_write;
reg s_scanned_entries_every_cell_ADC_1_write;
reg s_scanned_entries_every_cell_ADC_2_write;
reg s_scanned_entries_every_cell_ADC_3_write;
reg s_scanned_entries_every_cell_load_PQ_codes79_write;

reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [2:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    query_num_blk_n;
reg    nprobe_blk_n;
reg   [31:0] nprobe_read_reg_84;
reg   [31:0] query_num_read_reg_89;
wire   [63:0] bound_fu_77_p2;
reg   [63:0] bound_reg_94;
wire    ap_CS_fsm_state2;
wire    grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_start;
wire    grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_done;
wire    grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_idle;
wire    grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_ready;
wire    grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell75_read;
wire   [31:0] grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_0_din;
wire    grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_0_write;
wire   [31:0] grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_1_din;
wire    grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_1_write;
wire   [31:0] grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_2_din;
wire    grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_2_write;
wire   [31:0] grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_3_din;
wire    grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_3_write;
wire   [31:0] grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_load_PQ_codes79_din;
wire    grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_load_PQ_codes79_write;
wire    grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_ext_blocking_n;
wire    grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_str_blocking_n;
wire    grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_int_blocking_n;
reg    grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_start_reg;
wire    ap_CS_fsm_state3;
reg    ap_block_state1;
wire   [31:0] bound_fu_77_p0;
wire   [31:0] bound_fu_77_p1;
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
wire   [63:0] bound_fu_77_p00;
wire   [63:0] bound_fu_77_p10;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_done_reg = 1'b0;
#0 ap_CS_fsm = 3'd1;
#0 grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_start_reg = 1'b0;
end

vadd_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2 grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_start),
    .ap_done(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_done),
    .ap_idle(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_idle),
    .ap_ready(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_ready),
    .s_scanned_entries_every_cell75_dout(s_scanned_entries_every_cell75_dout),
    .s_scanned_entries_every_cell75_num_data_valid(9'd0),
    .s_scanned_entries_every_cell75_fifo_cap(9'd0),
    .s_scanned_entries_every_cell75_empty_n(s_scanned_entries_every_cell75_empty_n),
    .s_scanned_entries_every_cell75_read(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell75_read),
    .s_scanned_entries_every_cell_ADC_0_din(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_0_din),
    .s_scanned_entries_every_cell_ADC_0_num_data_valid(9'd0),
    .s_scanned_entries_every_cell_ADC_0_fifo_cap(9'd0),
    .s_scanned_entries_every_cell_ADC_0_full_n(s_scanned_entries_every_cell_ADC_0_full_n),
    .s_scanned_entries_every_cell_ADC_0_write(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_0_write),
    .s_scanned_entries_every_cell_ADC_1_din(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_1_din),
    .s_scanned_entries_every_cell_ADC_1_num_data_valid(9'd0),
    .s_scanned_entries_every_cell_ADC_1_fifo_cap(9'd0),
    .s_scanned_entries_every_cell_ADC_1_full_n(s_scanned_entries_every_cell_ADC_1_full_n),
    .s_scanned_entries_every_cell_ADC_1_write(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_1_write),
    .s_scanned_entries_every_cell_ADC_2_din(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_2_din),
    .s_scanned_entries_every_cell_ADC_2_num_data_valid(9'd0),
    .s_scanned_entries_every_cell_ADC_2_fifo_cap(9'd0),
    .s_scanned_entries_every_cell_ADC_2_full_n(s_scanned_entries_every_cell_ADC_2_full_n),
    .s_scanned_entries_every_cell_ADC_2_write(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_2_write),
    .s_scanned_entries_every_cell_ADC_3_din(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_3_din),
    .s_scanned_entries_every_cell_ADC_3_num_data_valid(9'd0),
    .s_scanned_entries_every_cell_ADC_3_fifo_cap(9'd0),
    .s_scanned_entries_every_cell_ADC_3_full_n(s_scanned_entries_every_cell_ADC_3_full_n),
    .s_scanned_entries_every_cell_ADC_3_write(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_3_write),
    .s_scanned_entries_every_cell_load_PQ_codes79_din(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_load_PQ_codes79_din),
    .s_scanned_entries_every_cell_load_PQ_codes79_num_data_valid(9'd0),
    .s_scanned_entries_every_cell_load_PQ_codes79_fifo_cap(9'd0),
    .s_scanned_entries_every_cell_load_PQ_codes79_full_n(s_scanned_entries_every_cell_load_PQ_codes79_full_n),
    .s_scanned_entries_every_cell_load_PQ_codes79_write(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_load_PQ_codes79_write),
    .bound(bound_reg_94),
    .ap_ext_blocking_n(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_ext_blocking_n),
    .ap_str_blocking_n(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_str_blocking_n),
    .ap_int_blocking_n(grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_int_blocking_n)
);

vadd_mul_32ns_32ns_64_1_1 #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .dout_WIDTH( 64 ))
mul_32ns_32ns_64_1_1_U4039(
    .din0(bound_fu_77_p0),
    .din1(bound_fu_77_p1),
    .dout(bound_fu_77_p2)
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
        end else if (((grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state3))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state2)) begin
            grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_start_reg <= 1'b1;
        end else if ((grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_ready == 1'b1)) begin
            grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        bound_reg_94 <= bound_fu_77_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state1)) begin
        nprobe_read_reg_84 <= nprobe_dout;
        query_num_read_reg_89 <= query_num_dout;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1))) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

always @ (*) begin
    if ((grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_done == 1'b0)) begin
        ap_ST_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_fsm_state3_blk = 1'b0;
    end
end

always @ (*) begin
    if (((grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state3))) begin
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
    if (((grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state3))) begin
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
    if ((grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_ext_blocking_n == 1'b0)) begin
        ap_sub_ext_blocking_0 = 1'b1;
    end else begin
        ap_sub_ext_blocking_0 = 1'b0;
    end
end

always @ (*) begin
    if ((grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_int_blocking_n == 1'b0)) begin
        ap_sub_int_blocking_0 = 1'b1;
    end else begin
        ap_sub_int_blocking_0 = 1'b0;
    end
end

always @ (*) begin
    if ((grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_str_blocking_n == 1'b0)) begin
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
        nprobe_blk_n = nprobe_empty_n;
    end else begin
        nprobe_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_start == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        nprobe_read = 1'b1;
    end else begin
        nprobe_read = 1'b0;
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
    if ((~((ap_start == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        query_num_read = 1'b1;
    end else begin
        query_num_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_scanned_entries_every_cell75_read = grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell75_read;
    end else begin
        s_scanned_entries_every_cell75_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_scanned_entries_every_cell_ADC_0_write = grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_0_write;
    end else begin
        s_scanned_entries_every_cell_ADC_0_write = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_scanned_entries_every_cell_ADC_1_write = grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_1_write;
    end else begin
        s_scanned_entries_every_cell_ADC_1_write = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_scanned_entries_every_cell_ADC_2_write = grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_2_write;
    end else begin
        s_scanned_entries_every_cell_ADC_2_write = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_scanned_entries_every_cell_ADC_3_write = grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_3_write;
    end else begin
        s_scanned_entries_every_cell_ADC_3_write = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        s_scanned_entries_every_cell_load_PQ_codes79_write = grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_load_PQ_codes79_write;
    end else begin
        s_scanned_entries_every_cell_load_PQ_codes79_write = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if ((~((ap_start == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            if (((grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state3))) begin
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
    ap_block_state1 = ((ap_start == 1'b0) | (nprobe_empty_n == 1'b0) | (query_num_empty_n == 1'b0) | (ap_done_reg == 1'b1));
end

assign ap_ext_blocking_n = (ap_ext_blocking_sub_n & 1'b1);

assign ap_int_blocking_cur_n = (query_num_blk_n & nprobe_blk_n);

assign ap_int_blocking_n = (ap_int_blocking_sub_n & ap_int_blocking_cur_n);

assign ap_str_blocking_n = (ap_str_blocking_sub_n & 1'b1);

assign bound_fu_77_p0 = bound_fu_77_p00;

assign bound_fu_77_p00 = query_num_read_reg_89;

assign bound_fu_77_p1 = bound_fu_77_p10;

assign bound_fu_77_p10 = nprobe_read_reg_84;

assign grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_start = grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_ap_start_reg;

assign s_scanned_entries_every_cell_ADC_0_din = grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_0_din;

assign s_scanned_entries_every_cell_ADC_1_din = grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_1_din;

assign s_scanned_entries_every_cell_ADC_2_din = grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_2_din;

assign s_scanned_entries_every_cell_ADC_3_din = grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_ADC_3_din;

assign s_scanned_entries_every_cell_load_PQ_codes79_din = grp_replicate_s_scanned_entries_every_cell_Pipeline_VITIS_LOOP_31_1_VITIS_LOOP_33_2_fu_54_s_scanned_entries_every_cell_load_PQ_codes79_din;

endmodule //vadd_replicate_s_scanned_entries_every_cell
