// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Version: 2022.1
// Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module vadd_center_vector_dispatcher_Pipeline_VITIS_LOOP_127_4 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        reg_V,
        tmp,
        center_vector_buffer_address0,
        center_vector_buffer_ce0,
        center_vector_buffer_we0,
        center_vector_buffer_d0,
        ap_ext_blocking_n,
        ap_str_blocking_n,
        ap_int_blocking_n
);

parameter    ap_ST_fsm_state1 = 1'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [511:0] reg_V;
input  [8:0] tmp;
output  [8:0] center_vector_buffer_address0;
output   center_vector_buffer_ce0;
output   center_vector_buffer_we0;
output  [31:0] center_vector_buffer_d0;
output   ap_ext_blocking_n;
output   ap_str_blocking_n;
output   ap_int_blocking_n;

reg ap_idle;
reg center_vector_buffer_ce0;
reg center_vector_buffer_we0;

(* fsm_encoding = "none" *) reg   [0:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    ap_block_state1_pp0_stage0_iter0;
wire   [0:0] icmp_ln127_fu_73_p2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
wire   [63:0] zext_ln131_fu_126_p1;
reg   [4:0] j_fu_36;
wire   [4:0] add_ln127_fu_79_p2;
wire    ap_loop_init;
reg   [4:0] ap_sig_allocacmp_j_2;
wire   [3:0] trunc_ln674_fu_89_p1;
wire   [8:0] shl_ln_fu_93_p3;
wire   [511:0] zext_ln674_fu_101_p1;
wire   [511:0] lshr_ln674_fu_105_p2;
wire   [31:0] single_val_uint_fu_111_p1;
wire   [8:0] zext_ln127_fu_85_p1;
wire   [8:0] add_ln131_fu_120_p2;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg   [0:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_start_int;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 1'd1;
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
        ap_CS_fsm <= ap_ST_fsm_state1;
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
        end else if (((ap_loop_exit_ready == 1'b1) & (1'b1 == ap_CS_fsm_state1) & (ap_start_int == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start_int == 1'b1))) begin
        if ((icmp_ln127_fu_73_p2 == 1'd0)) begin
            j_fu_36 <= add_ln127_fu_79_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            j_fu_36 <= 5'd0;
        end
    end
end

always @ (*) begin
    if ((ap_start_int == 1'b0)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln127_fu_73_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (ap_start_int == 1'b1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_exit_ready == 1'b1) & (1'b1 == ap_CS_fsm_state1) & (ap_start_int == 1'b1))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start_int == 1'b1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_init == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_sig_allocacmp_j_2 = 5'd0;
    end else begin
        ap_sig_allocacmp_j_2 = j_fu_36;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start_int == 1'b1))) begin
        center_vector_buffer_ce0 = 1'b1;
    end else begin
        center_vector_buffer_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln127_fu_73_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state1) & (ap_start_int == 1'b1))) begin
        center_vector_buffer_we0 = 1'b1;
    end else begin
        center_vector_buffer_we0 = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln127_fu_79_p2 = (ap_sig_allocacmp_j_2 + 5'd1);

assign add_ln131_fu_120_p2 = (tmp + zext_ln127_fu_85_p1);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

assign ap_ext_blocking_n = (1'b1 & 1'b1);

assign ap_int_blocking_n = (1'b1 & 1'b1);

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign ap_str_blocking_n = (1'b1 & 1'b1);

assign center_vector_buffer_address0 = zext_ln131_fu_126_p1;

assign center_vector_buffer_d0 = single_val_uint_fu_111_p1;

assign icmp_ln127_fu_73_p2 = ((ap_sig_allocacmp_j_2 == 5'd16) ? 1'b1 : 1'b0);

assign lshr_ln674_fu_105_p2 = reg_V >> zext_ln674_fu_101_p1;

assign shl_ln_fu_93_p3 = {{trunc_ln674_fu_89_p1}, {5'd0}};

assign single_val_uint_fu_111_p1 = lshr_ln674_fu_105_p2[31:0];

assign trunc_ln674_fu_89_p1 = ap_sig_allocacmp_j_2[3:0];

assign zext_ln127_fu_85_p1 = ap_sig_allocacmp_j_2;

assign zext_ln131_fu_126_p1 = add_ln131_fu_120_p2;

assign zext_ln674_fu_101_p1 = shl_ln_fu_93_p3;

endmodule //vadd_center_vector_dispatcher_Pipeline_VITIS_LOOP_127_4
