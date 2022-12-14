// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Version: 2022.1
// Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module vadd_LUT_construction_sub_PE_1_Pipeline_VITIS_LOOP_193_5_VITIS_LOOP_194_6 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        s_sub_query_vectors_i_1_dout,
        s_sub_query_vectors_i_1_num_data_valid,
        s_sub_query_vectors_i_1_fifo_cap,
        s_sub_query_vectors_i_1_empty_n,
        s_sub_query_vectors_i_1_read,
        sub_query_vector_local_1031,
        sub_query_vector_local_1030,
        sub_query_vector_local_1029,
        sub_query_vector_local_1028,
        sub_query_vector_local_1027,
        sub_query_vector_local_1026,
        sub_query_vector_local_1025,
        sub_query_vector_local_1024,
        sub_query_vector_local_1023,
        sub_query_vector_local_1022,
        sub_query_vector_local_1021,
        sub_query_vector_local_1020,
        sub_query_vector_local_1019,
        sub_query_vector_local_1018,
        sub_query_vector_local_1017,
        sub_query_vector_local_1016,
        sub_query_vector_local_1015,
        sub_query_vector_local_1014,
        sub_query_vector_local_1013,
        sub_query_vector_local_1012,
        sub_query_vector_local_1011,
        sub_query_vector_local_1010,
        sub_query_vector_local_1009,
        sub_query_vector_local,
        sub_query_vector_local_1055_out,
        sub_query_vector_local_1055_out_ap_vld,
        sub_query_vector_local_1054_out,
        sub_query_vector_local_1054_out_ap_vld,
        sub_query_vector_local_1053_out,
        sub_query_vector_local_1053_out_ap_vld,
        sub_query_vector_local_1052_out,
        sub_query_vector_local_1052_out_ap_vld,
        sub_query_vector_local_1051_out,
        sub_query_vector_local_1051_out_ap_vld,
        sub_query_vector_local_1050_out,
        sub_query_vector_local_1050_out_ap_vld,
        sub_query_vector_local_1049_out,
        sub_query_vector_local_1049_out_ap_vld,
        sub_query_vector_local_1048_out,
        sub_query_vector_local_1048_out_ap_vld,
        sub_query_vector_local_1047_out,
        sub_query_vector_local_1047_out_ap_vld,
        sub_query_vector_local_1046_out,
        sub_query_vector_local_1046_out_ap_vld,
        sub_query_vector_local_1045_out,
        sub_query_vector_local_1045_out_ap_vld,
        sub_query_vector_local_1044_out,
        sub_query_vector_local_1044_out_ap_vld,
        sub_query_vector_local_1043_out,
        sub_query_vector_local_1043_out_ap_vld,
        sub_query_vector_local_1042_out,
        sub_query_vector_local_1042_out_ap_vld,
        sub_query_vector_local_1041_out,
        sub_query_vector_local_1041_out_ap_vld,
        sub_query_vector_local_1040_out,
        sub_query_vector_local_1040_out_ap_vld,
        sub_query_vector_local_1039_out,
        sub_query_vector_local_1039_out_ap_vld,
        sub_query_vector_local_1038_out,
        sub_query_vector_local_1038_out_ap_vld,
        sub_query_vector_local_1037_out,
        sub_query_vector_local_1037_out_ap_vld,
        sub_query_vector_local_1036_out,
        sub_query_vector_local_1036_out_ap_vld,
        sub_query_vector_local_1035_out,
        sub_query_vector_local_1035_out_ap_vld,
        sub_query_vector_local_1034_out,
        sub_query_vector_local_1034_out_ap_vld,
        sub_query_vector_local_1033_out,
        sub_query_vector_local_1033_out_ap_vld,
        sub_query_vector_local_1032_out,
        sub_query_vector_local_1032_out_ap_vld,
        ap_ext_blocking_n,
        ap_str_blocking_n,
        ap_int_blocking_n
);

parameter    ap_ST_fsm_pp0_stage0 = 1'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [31:0] s_sub_query_vectors_i_1_dout;
input  [1:0] s_sub_query_vectors_i_1_num_data_valid;
input  [1:0] s_sub_query_vectors_i_1_fifo_cap;
input   s_sub_query_vectors_i_1_empty_n;
output   s_sub_query_vectors_i_1_read;
input  [31:0] sub_query_vector_local_1031;
input  [31:0] sub_query_vector_local_1030;
input  [31:0] sub_query_vector_local_1029;
input  [31:0] sub_query_vector_local_1028;
input  [31:0] sub_query_vector_local_1027;
input  [31:0] sub_query_vector_local_1026;
input  [31:0] sub_query_vector_local_1025;
input  [31:0] sub_query_vector_local_1024;
input  [31:0] sub_query_vector_local_1023;
input  [31:0] sub_query_vector_local_1022;
input  [31:0] sub_query_vector_local_1021;
input  [31:0] sub_query_vector_local_1020;
input  [31:0] sub_query_vector_local_1019;
input  [31:0] sub_query_vector_local_1018;
input  [31:0] sub_query_vector_local_1017;
input  [31:0] sub_query_vector_local_1016;
input  [31:0] sub_query_vector_local_1015;
input  [31:0] sub_query_vector_local_1014;
input  [31:0] sub_query_vector_local_1013;
input  [31:0] sub_query_vector_local_1012;
input  [31:0] sub_query_vector_local_1011;
input  [31:0] sub_query_vector_local_1010;
input  [31:0] sub_query_vector_local_1009;
input  [31:0] sub_query_vector_local;
output  [31:0] sub_query_vector_local_1055_out;
output   sub_query_vector_local_1055_out_ap_vld;
output  [31:0] sub_query_vector_local_1054_out;
output   sub_query_vector_local_1054_out_ap_vld;
output  [31:0] sub_query_vector_local_1053_out;
output   sub_query_vector_local_1053_out_ap_vld;
output  [31:0] sub_query_vector_local_1052_out;
output   sub_query_vector_local_1052_out_ap_vld;
output  [31:0] sub_query_vector_local_1051_out;
output   sub_query_vector_local_1051_out_ap_vld;
output  [31:0] sub_query_vector_local_1050_out;
output   sub_query_vector_local_1050_out_ap_vld;
output  [31:0] sub_query_vector_local_1049_out;
output   sub_query_vector_local_1049_out_ap_vld;
output  [31:0] sub_query_vector_local_1048_out;
output   sub_query_vector_local_1048_out_ap_vld;
output  [31:0] sub_query_vector_local_1047_out;
output   sub_query_vector_local_1047_out_ap_vld;
output  [31:0] sub_query_vector_local_1046_out;
output   sub_query_vector_local_1046_out_ap_vld;
output  [31:0] sub_query_vector_local_1045_out;
output   sub_query_vector_local_1045_out_ap_vld;
output  [31:0] sub_query_vector_local_1044_out;
output   sub_query_vector_local_1044_out_ap_vld;
output  [31:0] sub_query_vector_local_1043_out;
output   sub_query_vector_local_1043_out_ap_vld;
output  [31:0] sub_query_vector_local_1042_out;
output   sub_query_vector_local_1042_out_ap_vld;
output  [31:0] sub_query_vector_local_1041_out;
output   sub_query_vector_local_1041_out_ap_vld;
output  [31:0] sub_query_vector_local_1040_out;
output   sub_query_vector_local_1040_out_ap_vld;
output  [31:0] sub_query_vector_local_1039_out;
output   sub_query_vector_local_1039_out_ap_vld;
output  [31:0] sub_query_vector_local_1038_out;
output   sub_query_vector_local_1038_out_ap_vld;
output  [31:0] sub_query_vector_local_1037_out;
output   sub_query_vector_local_1037_out_ap_vld;
output  [31:0] sub_query_vector_local_1036_out;
output   sub_query_vector_local_1036_out_ap_vld;
output  [31:0] sub_query_vector_local_1035_out;
output   sub_query_vector_local_1035_out_ap_vld;
output  [31:0] sub_query_vector_local_1034_out;
output   sub_query_vector_local_1034_out_ap_vld;
output  [31:0] sub_query_vector_local_1033_out;
output   sub_query_vector_local_1033_out_ap_vld;
output  [31:0] sub_query_vector_local_1032_out;
output   sub_query_vector_local_1032_out_ap_vld;
output   ap_ext_blocking_n;
output   ap_str_blocking_n;
output   ap_int_blocking_n;

reg ap_idle;
reg s_sub_query_vectors_i_1_read;
reg sub_query_vector_local_1055_out_ap_vld;
reg sub_query_vector_local_1054_out_ap_vld;
reg sub_query_vector_local_1053_out_ap_vld;
reg sub_query_vector_local_1052_out_ap_vld;
reg sub_query_vector_local_1051_out_ap_vld;
reg sub_query_vector_local_1050_out_ap_vld;
reg sub_query_vector_local_1049_out_ap_vld;
reg sub_query_vector_local_1048_out_ap_vld;
reg sub_query_vector_local_1047_out_ap_vld;
reg sub_query_vector_local_1046_out_ap_vld;
reg sub_query_vector_local_1045_out_ap_vld;
reg sub_query_vector_local_1044_out_ap_vld;
reg sub_query_vector_local_1043_out_ap_vld;
reg sub_query_vector_local_1042_out_ap_vld;
reg sub_query_vector_local_1041_out_ap_vld;
reg sub_query_vector_local_1040_out_ap_vld;
reg sub_query_vector_local_1039_out_ap_vld;
reg sub_query_vector_local_1038_out_ap_vld;
reg sub_query_vector_local_1037_out_ap_vld;
reg sub_query_vector_local_1036_out_ap_vld;
reg sub_query_vector_local_1035_out_ap_vld;
reg sub_query_vector_local_1034_out_ap_vld;
reg sub_query_vector_local_1033_out_ap_vld;
reg sub_query_vector_local_1032_out_ap_vld;

(* fsm_encoding = "none" *) reg   [0:0] ap_CS_fsm;
wire    ap_CS_fsm_pp0_stage0;
wire    ap_enable_reg_pp0_iter0;
reg    ap_enable_reg_pp0_iter1;
reg    ap_idle_pp0;
wire    ap_block_state1_pp0_stage0_iter0;
reg    ap_block_state2_pp0_stage0_iter1;
reg    ap_block_pp0_stage0_subdone;
wire   [0:0] icmp_ln193_fu_748_p2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    s_sub_query_vectors_i_1_blk_n;
wire    ap_block_pp0_stage0;
wire   [4:0] add_ln196_fu_826_p2;
reg   [4:0] add_ln196_reg_1265;
reg    ap_block_pp0_stage0_11001;
reg   [2:0] j_fu_184;
wire   [2:0] add_ln194_fu_832_p2;
wire    ap_loop_init;
reg   [2:0] ap_sig_allocacmp_j_load;
reg   [2:0] i_fu_188;
wire   [2:0] select_ln184_9_fu_786_p3;
reg   [2:0] ap_sig_allocacmp_i_load;
reg   [4:0] indvar_flatten44_fu_192;
wire   [4:0] add_ln193_fu_754_p2;
reg   [4:0] ap_sig_allocacmp_indvar_flatten44_load;
reg   [31:0] sub_query_vector_local_393_fu_196;
wire   [31:0] sub_query_vector_local_418_fu_853_p1;
reg   [31:0] sub_query_vector_local_394_fu_200;
reg   [31:0] sub_query_vector_local_395_fu_204;
reg   [31:0] sub_query_vector_local_396_fu_208;
reg   [31:0] sub_query_vector_local_397_fu_212;
reg   [31:0] sub_query_vector_local_398_fu_216;
reg   [31:0] sub_query_vector_local_399_fu_220;
reg   [31:0] sub_query_vector_local_400_fu_224;
reg   [31:0] sub_query_vector_local_401_fu_228;
reg   [31:0] sub_query_vector_local_402_fu_232;
reg   [31:0] sub_query_vector_local_403_fu_236;
reg   [31:0] sub_query_vector_local_404_fu_240;
reg   [31:0] sub_query_vector_local_405_fu_244;
reg   [31:0] sub_query_vector_local_406_fu_248;
reg   [31:0] sub_query_vector_local_407_fu_252;
reg   [31:0] sub_query_vector_local_408_fu_256;
reg   [31:0] sub_query_vector_local_409_fu_260;
reg   [31:0] sub_query_vector_local_410_fu_264;
reg   [31:0] sub_query_vector_local_411_fu_268;
reg   [31:0] sub_query_vector_local_412_fu_272;
reg   [31:0] sub_query_vector_local_413_fu_276;
reg   [31:0] sub_query_vector_local_414_fu_280;
reg   [31:0] sub_query_vector_local_415_fu_284;
reg   [31:0] sub_query_vector_local_416_fu_288;
reg    ap_block_pp0_stage0_01001;
wire   [0:0] icmp_ln194_fu_766_p2;
wire   [2:0] add_ln193_9_fu_780_p2;
wire   [2:0] select_ln184_fu_772_p3;
wire   [1:0] trunc_ln196_fu_798_p1;
wire   [2:0] shl_ln196_fu_810_p2;
wire   [4:0] shl_ln_fu_802_p3;
wire   [4:0] zext_ln196_fu_816_p1;
wire   [4:0] sub_ln196_fu_820_p2;
wire   [4:0] zext_ln184_fu_794_p1;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg   [0:0] ap_NS_fsm;
wire    ap_int_blocking_cur_n;
wire    ap_enable_pp0;
wire    ap_start_int;
reg    ap_condition_541;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 1'd1;
#0 ap_enable_reg_pp0_iter1 = 1'b0;
#0 ap_done_reg = 1'b0;
end

vadd_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage0),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_pp0_stage0;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((ap_loop_exit_ready == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if ((1'b1 == ap_condition_exit_pp0_iter0_stage0)) begin
            ap_enable_reg_pp0_iter1 <= 1'b0;
        end else if (((1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_enable_reg_pp0_iter1 <= ap_start_int;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if (((icmp_ln193_fu_748_p2 == 1'd0) & (ap_enable_reg_pp0_iter0 == 1'b1))) begin
            i_fu_188 <= select_ln184_9_fu_786_p3;
        end else if ((ap_loop_init == 1'b1)) begin
            i_fu_188 <= 3'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if (((icmp_ln193_fu_748_p2 == 1'd0) & (ap_enable_reg_pp0_iter0 == 1'b1))) begin
            indvar_flatten44_fu_192 <= add_ln193_fu_754_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten44_fu_192 <= 5'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if (((icmp_ln193_fu_748_p2 == 1'd0) & (ap_enable_reg_pp0_iter0 == 1'b1))) begin
            j_fu_184 <= add_ln194_fu_832_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            j_fu_184 <= 3'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_393_fu_196 <= sub_query_vector_local;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd0 == add_ln196_reg_1265))) begin
            sub_query_vector_local_393_fu_196 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_394_fu_200 <= sub_query_vector_local_1009;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd1 == add_ln196_reg_1265))) begin
            sub_query_vector_local_394_fu_200 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_395_fu_204 <= sub_query_vector_local_1010;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd2 == add_ln196_reg_1265))) begin
            sub_query_vector_local_395_fu_204 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_396_fu_208 <= sub_query_vector_local_1011;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd3 == add_ln196_reg_1265))) begin
            sub_query_vector_local_396_fu_208 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_397_fu_212 <= sub_query_vector_local_1012;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd4 == add_ln196_reg_1265))) begin
            sub_query_vector_local_397_fu_212 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_398_fu_216 <= sub_query_vector_local_1013;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd5 == add_ln196_reg_1265))) begin
            sub_query_vector_local_398_fu_216 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_399_fu_220 <= sub_query_vector_local_1014;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd6 == add_ln196_reg_1265))) begin
            sub_query_vector_local_399_fu_220 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_400_fu_224 <= sub_query_vector_local_1015;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd7 == add_ln196_reg_1265))) begin
            sub_query_vector_local_400_fu_224 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_401_fu_228 <= sub_query_vector_local_1016;
        end else if (((5'd8 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_401_fu_228 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_402_fu_232 <= sub_query_vector_local_1017;
        end else if (((5'd9 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_402_fu_232 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_403_fu_236 <= sub_query_vector_local_1018;
        end else if (((5'd10 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_403_fu_236 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_404_fu_240 <= sub_query_vector_local_1019;
        end else if (((5'd11 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_404_fu_240 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_405_fu_244 <= sub_query_vector_local_1020;
        end else if (((5'd12 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_405_fu_244 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_406_fu_248 <= sub_query_vector_local_1021;
        end else if (((5'd13 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_406_fu_248 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_407_fu_252 <= sub_query_vector_local_1022;
        end else if (((5'd14 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_407_fu_252 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_408_fu_256 <= sub_query_vector_local_1023;
        end else if (((5'd15 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_408_fu_256 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_409_fu_260 <= sub_query_vector_local_1024;
        end else if (((5'd16 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_409_fu_260 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_410_fu_264 <= sub_query_vector_local_1025;
        end else if (((5'd17 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_410_fu_264 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_411_fu_268 <= sub_query_vector_local_1026;
        end else if (((5'd18 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_411_fu_268 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_412_fu_272 <= sub_query_vector_local_1027;
        end else if (((5'd19 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_412_fu_272 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_413_fu_276 <= sub_query_vector_local_1028;
        end else if (((5'd20 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_413_fu_276 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_414_fu_280 <= sub_query_vector_local_1029;
        end else if (((5'd21 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_414_fu_280 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_415_fu_284 <= sub_query_vector_local_1030;
        end else if (((5'd22 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_415_fu_284 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_416_fu_288 <= sub_query_vector_local_1031;
        end else if ((1'b1 == ap_condition_541)) begin
            sub_query_vector_local_416_fu_288 <= sub_query_vector_local_418_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln193_fu_748_p2 == 1'd0) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        add_ln196_reg_1265 <= add_ln196_fu_826_p2;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_exit_ready == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b0) & (ap_idle_pp0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
        ap_idle_pp0 = 1'b1;
    end else begin
        ap_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (1'b1 == ap_CS_fsm_pp0_stage0) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_i_load = 3'd0;
    end else begin
        ap_sig_allocacmp_i_load = i_fu_188;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (1'b1 == ap_CS_fsm_pp0_stage0) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten44_load = 5'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten44_load = indvar_flatten44_fu_192;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (1'b1 == ap_CS_fsm_pp0_stage0) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_j_load = 3'd0;
    end else begin
        ap_sig_allocacmp_j_load = j_fu_184;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        s_sub_query_vectors_i_1_blk_n = s_sub_query_vectors_i_1_empty_n;
    end else begin
        s_sub_query_vectors_i_1_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        s_sub_query_vectors_i_1_read = 1'b1;
    end else begin
        s_sub_query_vectors_i_1_read = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1032_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1032_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1033_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1033_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1034_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1034_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1035_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1035_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1036_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1036_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1037_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1037_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1038_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1038_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1039_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1039_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1040_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1040_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1041_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1041_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1042_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1042_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1043_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1043_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1044_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1044_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1045_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1045_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1046_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1046_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1047_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1047_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1048_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1048_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1049_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1049_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1050_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1050_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1051_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1051_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1052_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1052_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1053_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1053_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1054_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1054_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_1055_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_1055_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_pp0_stage0 : begin
            ap_NS_fsm = ap_ST_fsm_pp0_stage0;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln193_9_fu_780_p2 = (ap_sig_allocacmp_i_load + 3'd1);

assign add_ln193_fu_754_p2 = (ap_sig_allocacmp_indvar_flatten44_load + 5'd1);

assign add_ln194_fu_832_p2 = (select_ln184_fu_772_p3 + 3'd1);

assign add_ln196_fu_826_p2 = (sub_ln196_fu_820_p2 + zext_ln184_fu_794_p1);

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd0];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_01001 = ((s_sub_query_vectors_i_1_empty_n == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1));
end

always @ (*) begin
    ap_block_pp0_stage0_11001 = ((s_sub_query_vectors_i_1_empty_n == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1));
end

always @ (*) begin
    ap_block_pp0_stage0_subdone = ((s_sub_query_vectors_i_1_empty_n == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1));
end

assign ap_block_state1_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state2_pp0_stage0_iter1 = (s_sub_query_vectors_i_1_empty_n == 1'b0);
end

always @ (*) begin
    ap_condition_541 = ((ap_enable_reg_pp0_iter1 == 1'b1) & ((5'd23 == add_ln196_reg_1265) | ((5'd24 == add_ln196_reg_1265) | ((5'd25 == add_ln196_reg_1265) | ((5'd26 == add_ln196_reg_1265) | ((5'd27 == add_ln196_reg_1265) | ((5'd28 == add_ln196_reg_1265) | ((5'd29 == add_ln196_reg_1265) | ((5'd30 == add_ln196_reg_1265) | (5'd31 == add_ln196_reg_1265))))))))));
end

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_enable_reg_pp0_iter0 = ap_start_int;

assign ap_ext_blocking_n = (1'b1 & 1'b1);

assign ap_int_blocking_cur_n = s_sub_query_vectors_i_1_blk_n;

assign ap_int_blocking_n = (ap_int_blocking_cur_n & 1'b1);

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign ap_str_blocking_n = (1'b1 & 1'b1);

assign icmp_ln193_fu_748_p2 = ((ap_sig_allocacmp_indvar_flatten44_load == 5'd24) ? 1'b1 : 1'b0);

assign icmp_ln194_fu_766_p2 = ((ap_sig_allocacmp_j_load == 3'd4) ? 1'b1 : 1'b0);

assign select_ln184_9_fu_786_p3 = ((icmp_ln194_fu_766_p2[0:0] == 1'b1) ? add_ln193_9_fu_780_p2 : ap_sig_allocacmp_i_load);

assign select_ln184_fu_772_p3 = ((icmp_ln194_fu_766_p2[0:0] == 1'b1) ? 3'd0 : ap_sig_allocacmp_j_load);

assign shl_ln196_fu_810_p2 = select_ln184_fu_772_p3 << 3'd1;

assign shl_ln_fu_802_p3 = {{trunc_ln196_fu_798_p1}, {3'd0}};

assign sub_ln196_fu_820_p2 = (shl_ln_fu_802_p3 - zext_ln196_fu_816_p1);

assign sub_query_vector_local_1032_out = sub_query_vector_local_393_fu_196;

assign sub_query_vector_local_1033_out = sub_query_vector_local_394_fu_200;

assign sub_query_vector_local_1034_out = sub_query_vector_local_395_fu_204;

assign sub_query_vector_local_1035_out = sub_query_vector_local_396_fu_208;

assign sub_query_vector_local_1036_out = sub_query_vector_local_397_fu_212;

assign sub_query_vector_local_1037_out = sub_query_vector_local_398_fu_216;

assign sub_query_vector_local_1038_out = sub_query_vector_local_399_fu_220;

assign sub_query_vector_local_1039_out = sub_query_vector_local_400_fu_224;

assign sub_query_vector_local_1040_out = sub_query_vector_local_401_fu_228;

assign sub_query_vector_local_1041_out = sub_query_vector_local_402_fu_232;

assign sub_query_vector_local_1042_out = sub_query_vector_local_403_fu_236;

assign sub_query_vector_local_1043_out = sub_query_vector_local_404_fu_240;

assign sub_query_vector_local_1044_out = sub_query_vector_local_405_fu_244;

assign sub_query_vector_local_1045_out = sub_query_vector_local_406_fu_248;

assign sub_query_vector_local_1046_out = sub_query_vector_local_407_fu_252;

assign sub_query_vector_local_1047_out = sub_query_vector_local_408_fu_256;

assign sub_query_vector_local_1048_out = sub_query_vector_local_409_fu_260;

assign sub_query_vector_local_1049_out = sub_query_vector_local_410_fu_264;

assign sub_query_vector_local_1050_out = sub_query_vector_local_411_fu_268;

assign sub_query_vector_local_1051_out = sub_query_vector_local_412_fu_272;

assign sub_query_vector_local_1052_out = sub_query_vector_local_413_fu_276;

assign sub_query_vector_local_1053_out = sub_query_vector_local_414_fu_280;

assign sub_query_vector_local_1054_out = sub_query_vector_local_415_fu_284;

assign sub_query_vector_local_1055_out = sub_query_vector_local_416_fu_288;

assign sub_query_vector_local_418_fu_853_p1 = s_sub_query_vectors_i_1_dout;

assign trunc_ln196_fu_798_p1 = select_ln184_fu_772_p3[1:0];

assign zext_ln184_fu_794_p1 = select_ln184_9_fu_786_p3;

assign zext_ln196_fu_816_p1 = shl_ln196_fu_810_p2;

endmodule //vadd_LUT_construction_sub_PE_1_Pipeline_VITIS_LOOP_193_5_VITIS_LOOP_194_6
