// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Version: 2022.1
// Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module vadd_LUT_construction_sub_PE_4_Pipeline_VITIS_LOOP_193_5_VITIS_LOOP_194_6 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        s_sub_query_vectors_i_4_dout,
        s_sub_query_vectors_i_4_num_data_valid,
        s_sub_query_vectors_i_4_fifo_cap,
        s_sub_query_vectors_i_4_empty_n,
        s_sub_query_vectors_i_4_read,
        sub_query_vector_local_383,
        sub_query_vector_local_382,
        sub_query_vector_local_381,
        sub_query_vector_local_380,
        sub_query_vector_local_379,
        sub_query_vector_local_378,
        sub_query_vector_local_377,
        sub_query_vector_local_376,
        sub_query_vector_local_375,
        sub_query_vector_local_374,
        sub_query_vector_local_373,
        sub_query_vector_local_372,
        sub_query_vector_local_371,
        sub_query_vector_local_370,
        sub_query_vector_local_369,
        sub_query_vector_local_368,
        sub_query_vector_local_367,
        sub_query_vector_local_366,
        sub_query_vector_local_365,
        sub_query_vector_local_364,
        sub_query_vector_local_363,
        sub_query_vector_local_362,
        sub_query_vector_local_361,
        sub_query_vector_local,
        sub_query_vector_local_407_out,
        sub_query_vector_local_407_out_ap_vld,
        sub_query_vector_local_406_out,
        sub_query_vector_local_406_out_ap_vld,
        sub_query_vector_local_405_out,
        sub_query_vector_local_405_out_ap_vld,
        sub_query_vector_local_404_out,
        sub_query_vector_local_404_out_ap_vld,
        sub_query_vector_local_403_out,
        sub_query_vector_local_403_out_ap_vld,
        sub_query_vector_local_402_out,
        sub_query_vector_local_402_out_ap_vld,
        sub_query_vector_local_401_out,
        sub_query_vector_local_401_out_ap_vld,
        sub_query_vector_local_400_out,
        sub_query_vector_local_400_out_ap_vld,
        sub_query_vector_local_399_out,
        sub_query_vector_local_399_out_ap_vld,
        sub_query_vector_local_398_out,
        sub_query_vector_local_398_out_ap_vld,
        sub_query_vector_local_397_out,
        sub_query_vector_local_397_out_ap_vld,
        sub_query_vector_local_396_out,
        sub_query_vector_local_396_out_ap_vld,
        sub_query_vector_local_395_out,
        sub_query_vector_local_395_out_ap_vld,
        sub_query_vector_local_394_out,
        sub_query_vector_local_394_out_ap_vld,
        sub_query_vector_local_393_out,
        sub_query_vector_local_393_out_ap_vld,
        sub_query_vector_local_392_out,
        sub_query_vector_local_392_out_ap_vld,
        sub_query_vector_local_391_out,
        sub_query_vector_local_391_out_ap_vld,
        sub_query_vector_local_390_out,
        sub_query_vector_local_390_out_ap_vld,
        sub_query_vector_local_389_out,
        sub_query_vector_local_389_out_ap_vld,
        sub_query_vector_local_388_out,
        sub_query_vector_local_388_out_ap_vld,
        sub_query_vector_local_387_out,
        sub_query_vector_local_387_out_ap_vld,
        sub_query_vector_local_386_out,
        sub_query_vector_local_386_out_ap_vld,
        sub_query_vector_local_385_out,
        sub_query_vector_local_385_out_ap_vld,
        sub_query_vector_local_384_out,
        sub_query_vector_local_384_out_ap_vld,
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
input  [31:0] s_sub_query_vectors_i_4_dout;
input  [1:0] s_sub_query_vectors_i_4_num_data_valid;
input  [1:0] s_sub_query_vectors_i_4_fifo_cap;
input   s_sub_query_vectors_i_4_empty_n;
output   s_sub_query_vectors_i_4_read;
input  [31:0] sub_query_vector_local_383;
input  [31:0] sub_query_vector_local_382;
input  [31:0] sub_query_vector_local_381;
input  [31:0] sub_query_vector_local_380;
input  [31:0] sub_query_vector_local_379;
input  [31:0] sub_query_vector_local_378;
input  [31:0] sub_query_vector_local_377;
input  [31:0] sub_query_vector_local_376;
input  [31:0] sub_query_vector_local_375;
input  [31:0] sub_query_vector_local_374;
input  [31:0] sub_query_vector_local_373;
input  [31:0] sub_query_vector_local_372;
input  [31:0] sub_query_vector_local_371;
input  [31:0] sub_query_vector_local_370;
input  [31:0] sub_query_vector_local_369;
input  [31:0] sub_query_vector_local_368;
input  [31:0] sub_query_vector_local_367;
input  [31:0] sub_query_vector_local_366;
input  [31:0] sub_query_vector_local_365;
input  [31:0] sub_query_vector_local_364;
input  [31:0] sub_query_vector_local_363;
input  [31:0] sub_query_vector_local_362;
input  [31:0] sub_query_vector_local_361;
input  [31:0] sub_query_vector_local;
output  [31:0] sub_query_vector_local_407_out;
output   sub_query_vector_local_407_out_ap_vld;
output  [31:0] sub_query_vector_local_406_out;
output   sub_query_vector_local_406_out_ap_vld;
output  [31:0] sub_query_vector_local_405_out;
output   sub_query_vector_local_405_out_ap_vld;
output  [31:0] sub_query_vector_local_404_out;
output   sub_query_vector_local_404_out_ap_vld;
output  [31:0] sub_query_vector_local_403_out;
output   sub_query_vector_local_403_out_ap_vld;
output  [31:0] sub_query_vector_local_402_out;
output   sub_query_vector_local_402_out_ap_vld;
output  [31:0] sub_query_vector_local_401_out;
output   sub_query_vector_local_401_out_ap_vld;
output  [31:0] sub_query_vector_local_400_out;
output   sub_query_vector_local_400_out_ap_vld;
output  [31:0] sub_query_vector_local_399_out;
output   sub_query_vector_local_399_out_ap_vld;
output  [31:0] sub_query_vector_local_398_out;
output   sub_query_vector_local_398_out_ap_vld;
output  [31:0] sub_query_vector_local_397_out;
output   sub_query_vector_local_397_out_ap_vld;
output  [31:0] sub_query_vector_local_396_out;
output   sub_query_vector_local_396_out_ap_vld;
output  [31:0] sub_query_vector_local_395_out;
output   sub_query_vector_local_395_out_ap_vld;
output  [31:0] sub_query_vector_local_394_out;
output   sub_query_vector_local_394_out_ap_vld;
output  [31:0] sub_query_vector_local_393_out;
output   sub_query_vector_local_393_out_ap_vld;
output  [31:0] sub_query_vector_local_392_out;
output   sub_query_vector_local_392_out_ap_vld;
output  [31:0] sub_query_vector_local_391_out;
output   sub_query_vector_local_391_out_ap_vld;
output  [31:0] sub_query_vector_local_390_out;
output   sub_query_vector_local_390_out_ap_vld;
output  [31:0] sub_query_vector_local_389_out;
output   sub_query_vector_local_389_out_ap_vld;
output  [31:0] sub_query_vector_local_388_out;
output   sub_query_vector_local_388_out_ap_vld;
output  [31:0] sub_query_vector_local_387_out;
output   sub_query_vector_local_387_out_ap_vld;
output  [31:0] sub_query_vector_local_386_out;
output   sub_query_vector_local_386_out_ap_vld;
output  [31:0] sub_query_vector_local_385_out;
output   sub_query_vector_local_385_out_ap_vld;
output  [31:0] sub_query_vector_local_384_out;
output   sub_query_vector_local_384_out_ap_vld;
output   ap_ext_blocking_n;
output   ap_str_blocking_n;
output   ap_int_blocking_n;

reg ap_idle;
reg s_sub_query_vectors_i_4_read;
reg sub_query_vector_local_407_out_ap_vld;
reg sub_query_vector_local_406_out_ap_vld;
reg sub_query_vector_local_405_out_ap_vld;
reg sub_query_vector_local_404_out_ap_vld;
reg sub_query_vector_local_403_out_ap_vld;
reg sub_query_vector_local_402_out_ap_vld;
reg sub_query_vector_local_401_out_ap_vld;
reg sub_query_vector_local_400_out_ap_vld;
reg sub_query_vector_local_399_out_ap_vld;
reg sub_query_vector_local_398_out_ap_vld;
reg sub_query_vector_local_397_out_ap_vld;
reg sub_query_vector_local_396_out_ap_vld;
reg sub_query_vector_local_395_out_ap_vld;
reg sub_query_vector_local_394_out_ap_vld;
reg sub_query_vector_local_393_out_ap_vld;
reg sub_query_vector_local_392_out_ap_vld;
reg sub_query_vector_local_391_out_ap_vld;
reg sub_query_vector_local_390_out_ap_vld;
reg sub_query_vector_local_389_out_ap_vld;
reg sub_query_vector_local_388_out_ap_vld;
reg sub_query_vector_local_387_out_ap_vld;
reg sub_query_vector_local_386_out_ap_vld;
reg sub_query_vector_local_385_out_ap_vld;
reg sub_query_vector_local_384_out_ap_vld;

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
reg    s_sub_query_vectors_i_4_blk_n;
wire    ap_block_pp0_stage0;
wire   [4:0] add_ln196_fu_826_p2;
reg   [4:0] add_ln196_reg_1265;
reg    ap_block_pp0_stage0_11001;
reg   [2:0] j_fu_184;
wire   [2:0] add_ln194_fu_832_p2;
wire    ap_loop_init;
reg   [2:0] ap_sig_allocacmp_j_load;
reg   [2:0] i_fu_188;
wire   [2:0] select_ln184_6_fu_786_p3;
reg   [2:0] ap_sig_allocacmp_i_load;
reg   [4:0] indvar_flatten44_fu_192;
wire   [4:0] add_ln193_fu_754_p2;
reg   [4:0] ap_sig_allocacmp_indvar_flatten44_load;
reg   [31:0] sub_query_vector_local_246_fu_196;
wire   [31:0] sub_query_vector_local_271_fu_853_p1;
reg   [31:0] sub_query_vector_local_247_fu_200;
reg   [31:0] sub_query_vector_local_248_fu_204;
reg   [31:0] sub_query_vector_local_249_fu_208;
reg   [31:0] sub_query_vector_local_250_fu_212;
reg   [31:0] sub_query_vector_local_251_fu_216;
reg   [31:0] sub_query_vector_local_252_fu_220;
reg   [31:0] sub_query_vector_local_253_fu_224;
reg   [31:0] sub_query_vector_local_254_fu_228;
reg   [31:0] sub_query_vector_local_255_fu_232;
reg   [31:0] sub_query_vector_local_256_fu_236;
reg   [31:0] sub_query_vector_local_257_fu_240;
reg   [31:0] sub_query_vector_local_258_fu_244;
reg   [31:0] sub_query_vector_local_259_fu_248;
reg   [31:0] sub_query_vector_local_260_fu_252;
reg   [31:0] sub_query_vector_local_261_fu_256;
reg   [31:0] sub_query_vector_local_262_fu_260;
reg   [31:0] sub_query_vector_local_263_fu_264;
reg   [31:0] sub_query_vector_local_264_fu_268;
reg   [31:0] sub_query_vector_local_265_fu_272;
reg   [31:0] sub_query_vector_local_266_fu_276;
reg   [31:0] sub_query_vector_local_267_fu_280;
reg   [31:0] sub_query_vector_local_268_fu_284;
reg   [31:0] sub_query_vector_local_269_fu_288;
reg    ap_block_pp0_stage0_01001;
wire   [0:0] icmp_ln194_fu_766_p2;
wire   [2:0] add_ln193_6_fu_780_p2;
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
            i_fu_188 <= select_ln184_6_fu_786_p3;
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
            sub_query_vector_local_246_fu_196 <= sub_query_vector_local;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd0 == add_ln196_reg_1265))) begin
            sub_query_vector_local_246_fu_196 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_247_fu_200 <= sub_query_vector_local_361;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd1 == add_ln196_reg_1265))) begin
            sub_query_vector_local_247_fu_200 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_248_fu_204 <= sub_query_vector_local_362;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd2 == add_ln196_reg_1265))) begin
            sub_query_vector_local_248_fu_204 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_249_fu_208 <= sub_query_vector_local_363;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd3 == add_ln196_reg_1265))) begin
            sub_query_vector_local_249_fu_208 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_250_fu_212 <= sub_query_vector_local_364;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd4 == add_ln196_reg_1265))) begin
            sub_query_vector_local_250_fu_212 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_251_fu_216 <= sub_query_vector_local_365;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd5 == add_ln196_reg_1265))) begin
            sub_query_vector_local_251_fu_216 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_252_fu_220 <= sub_query_vector_local_366;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd6 == add_ln196_reg_1265))) begin
            sub_query_vector_local_252_fu_220 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_253_fu_224 <= sub_query_vector_local_367;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (5'd7 == add_ln196_reg_1265))) begin
            sub_query_vector_local_253_fu_224 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_254_fu_228 <= sub_query_vector_local_368;
        end else if (((5'd8 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_254_fu_228 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_255_fu_232 <= sub_query_vector_local_369;
        end else if (((5'd9 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_255_fu_232 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_256_fu_236 <= sub_query_vector_local_370;
        end else if (((5'd10 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_256_fu_236 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_257_fu_240 <= sub_query_vector_local_371;
        end else if (((5'd11 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_257_fu_240 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_258_fu_244 <= sub_query_vector_local_372;
        end else if (((5'd12 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_258_fu_244 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_259_fu_248 <= sub_query_vector_local_373;
        end else if (((5'd13 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_259_fu_248 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_260_fu_252 <= sub_query_vector_local_374;
        end else if (((5'd14 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_260_fu_252 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_261_fu_256 <= sub_query_vector_local_375;
        end else if (((5'd15 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_261_fu_256 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_262_fu_260 <= sub_query_vector_local_376;
        end else if (((5'd16 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_262_fu_260 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_263_fu_264 <= sub_query_vector_local_377;
        end else if (((5'd17 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_263_fu_264 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_264_fu_268 <= sub_query_vector_local_378;
        end else if (((5'd18 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_264_fu_268 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_265_fu_272 <= sub_query_vector_local_379;
        end else if (((5'd19 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_265_fu_272 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_266_fu_276 <= sub_query_vector_local_380;
        end else if (((5'd20 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_266_fu_276 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_267_fu_280 <= sub_query_vector_local_381;
        end else if (((5'd21 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_267_fu_280 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_268_fu_284 <= sub_query_vector_local_382;
        end else if (((5'd22 == add_ln196_reg_1265) & (ap_enable_reg_pp0_iter1 == 1'b1))) begin
            sub_query_vector_local_268_fu_284 <= sub_query_vector_local_271_fu_853_p1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            sub_query_vector_local_269_fu_288 <= sub_query_vector_local_383;
        end else if ((1'b1 == ap_condition_541)) begin
            sub_query_vector_local_269_fu_288 <= sub_query_vector_local_271_fu_853_p1;
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
        s_sub_query_vectors_i_4_blk_n = s_sub_query_vectors_i_4_empty_n;
    end else begin
        s_sub_query_vectors_i_4_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        s_sub_query_vectors_i_4_read = 1'b1;
    end else begin
        s_sub_query_vectors_i_4_read = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_384_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_384_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_385_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_385_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_386_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_386_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_387_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_387_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_388_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_388_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_389_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_389_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_390_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_390_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_391_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_391_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_392_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_392_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_393_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_393_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_394_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_394_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_395_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_395_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_396_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_396_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_397_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_397_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_398_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_398_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_399_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_399_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_400_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_400_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_401_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_401_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_402_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_402_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_403_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_403_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_404_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_404_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_405_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_405_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_406_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_406_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln193_fu_748_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        sub_query_vector_local_407_out_ap_vld = 1'b1;
    end else begin
        sub_query_vector_local_407_out_ap_vld = 1'b0;
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

assign add_ln193_6_fu_780_p2 = (ap_sig_allocacmp_i_load + 3'd1);

assign add_ln193_fu_754_p2 = (ap_sig_allocacmp_indvar_flatten44_load + 5'd1);

assign add_ln194_fu_832_p2 = (select_ln184_fu_772_p3 + 3'd1);

assign add_ln196_fu_826_p2 = (sub_ln196_fu_820_p2 + zext_ln184_fu_794_p1);

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd0];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_01001 = ((s_sub_query_vectors_i_4_empty_n == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1));
end

always @ (*) begin
    ap_block_pp0_stage0_11001 = ((s_sub_query_vectors_i_4_empty_n == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1));
end

always @ (*) begin
    ap_block_pp0_stage0_subdone = ((s_sub_query_vectors_i_4_empty_n == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b1));
end

assign ap_block_state1_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state2_pp0_stage0_iter1 = (s_sub_query_vectors_i_4_empty_n == 1'b0);
end

always @ (*) begin
    ap_condition_541 = ((ap_enable_reg_pp0_iter1 == 1'b1) & ((5'd23 == add_ln196_reg_1265) | ((5'd24 == add_ln196_reg_1265) | ((5'd25 == add_ln196_reg_1265) | ((5'd26 == add_ln196_reg_1265) | ((5'd27 == add_ln196_reg_1265) | ((5'd28 == add_ln196_reg_1265) | ((5'd29 == add_ln196_reg_1265) | ((5'd30 == add_ln196_reg_1265) | (5'd31 == add_ln196_reg_1265))))))))));
end

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_enable_reg_pp0_iter0 = ap_start_int;

assign ap_ext_blocking_n = (1'b1 & 1'b1);

assign ap_int_blocking_cur_n = s_sub_query_vectors_i_4_blk_n;

assign ap_int_blocking_n = (ap_int_blocking_cur_n & 1'b1);

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign ap_str_blocking_n = (1'b1 & 1'b1);

assign icmp_ln193_fu_748_p2 = ((ap_sig_allocacmp_indvar_flatten44_load == 5'd24) ? 1'b1 : 1'b0);

assign icmp_ln194_fu_766_p2 = ((ap_sig_allocacmp_j_load == 3'd4) ? 1'b1 : 1'b0);

assign select_ln184_6_fu_786_p3 = ((icmp_ln194_fu_766_p2[0:0] == 1'b1) ? add_ln193_6_fu_780_p2 : ap_sig_allocacmp_i_load);

assign select_ln184_fu_772_p3 = ((icmp_ln194_fu_766_p2[0:0] == 1'b1) ? 3'd0 : ap_sig_allocacmp_j_load);

assign shl_ln196_fu_810_p2 = select_ln184_fu_772_p3 << 3'd1;

assign shl_ln_fu_802_p3 = {{trunc_ln196_fu_798_p1}, {3'd0}};

assign sub_ln196_fu_820_p2 = (shl_ln_fu_802_p3 - zext_ln196_fu_816_p1);

assign sub_query_vector_local_271_fu_853_p1 = s_sub_query_vectors_i_4_dout;

assign sub_query_vector_local_384_out = sub_query_vector_local_246_fu_196;

assign sub_query_vector_local_385_out = sub_query_vector_local_247_fu_200;

assign sub_query_vector_local_386_out = sub_query_vector_local_248_fu_204;

assign sub_query_vector_local_387_out = sub_query_vector_local_249_fu_208;

assign sub_query_vector_local_388_out = sub_query_vector_local_250_fu_212;

assign sub_query_vector_local_389_out = sub_query_vector_local_251_fu_216;

assign sub_query_vector_local_390_out = sub_query_vector_local_252_fu_220;

assign sub_query_vector_local_391_out = sub_query_vector_local_253_fu_224;

assign sub_query_vector_local_392_out = sub_query_vector_local_254_fu_228;

assign sub_query_vector_local_393_out = sub_query_vector_local_255_fu_232;

assign sub_query_vector_local_394_out = sub_query_vector_local_256_fu_236;

assign sub_query_vector_local_395_out = sub_query_vector_local_257_fu_240;

assign sub_query_vector_local_396_out = sub_query_vector_local_258_fu_244;

assign sub_query_vector_local_397_out = sub_query_vector_local_259_fu_248;

assign sub_query_vector_local_398_out = sub_query_vector_local_260_fu_252;

assign sub_query_vector_local_399_out = sub_query_vector_local_261_fu_256;

assign sub_query_vector_local_400_out = sub_query_vector_local_262_fu_260;

assign sub_query_vector_local_401_out = sub_query_vector_local_263_fu_264;

assign sub_query_vector_local_402_out = sub_query_vector_local_264_fu_268;

assign sub_query_vector_local_403_out = sub_query_vector_local_265_fu_272;

assign sub_query_vector_local_404_out = sub_query_vector_local_266_fu_276;

assign sub_query_vector_local_405_out = sub_query_vector_local_267_fu_280;

assign sub_query_vector_local_406_out = sub_query_vector_local_268_fu_284;

assign sub_query_vector_local_407_out = sub_query_vector_local_269_fu_288;

assign trunc_ln196_fu_798_p1 = select_ln184_fu_772_p3[1:0];

assign zext_ln184_fu_794_p1 = select_ln184_6_fu_786_p3;

assign zext_ln196_fu_816_p1 = shl_ln196_fu_810_p2;

endmodule //vadd_LUT_construction_sub_PE_4_Pipeline_VITIS_LOOP_193_5_VITIS_LOOP_194_6