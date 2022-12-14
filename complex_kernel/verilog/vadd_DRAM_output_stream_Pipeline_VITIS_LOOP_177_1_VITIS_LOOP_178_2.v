// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Version: 2022.1
// Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module vadd_DRAM_output_stream_Pipeline_VITIS_LOOP_177_1_VITIS_LOOP_178_2 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        s_kernel_network_out80_dout,
        s_kernel_network_out80_num_data_valid,
        s_kernel_network_out80_fifo_cap,
        s_kernel_network_out80_empty_n,
        s_kernel_network_out80_read,
        m_axi_gmem13_AWVALID,
        m_axi_gmem13_AWREADY,
        m_axi_gmem13_AWADDR,
        m_axi_gmem13_AWID,
        m_axi_gmem13_AWLEN,
        m_axi_gmem13_AWSIZE,
        m_axi_gmem13_AWBURST,
        m_axi_gmem13_AWLOCK,
        m_axi_gmem13_AWCACHE,
        m_axi_gmem13_AWPROT,
        m_axi_gmem13_AWQOS,
        m_axi_gmem13_AWREGION,
        m_axi_gmem13_AWUSER,
        m_axi_gmem13_WVALID,
        m_axi_gmem13_WREADY,
        m_axi_gmem13_WDATA,
        m_axi_gmem13_WSTRB,
        m_axi_gmem13_WLAST,
        m_axi_gmem13_WID,
        m_axi_gmem13_WUSER,
        m_axi_gmem13_ARVALID,
        m_axi_gmem13_ARREADY,
        m_axi_gmem13_ARADDR,
        m_axi_gmem13_ARID,
        m_axi_gmem13_ARLEN,
        m_axi_gmem13_ARSIZE,
        m_axi_gmem13_ARBURST,
        m_axi_gmem13_ARLOCK,
        m_axi_gmem13_ARCACHE,
        m_axi_gmem13_ARPROT,
        m_axi_gmem13_ARQOS,
        m_axi_gmem13_ARREGION,
        m_axi_gmem13_ARUSER,
        m_axi_gmem13_RVALID,
        m_axi_gmem13_RREADY,
        m_axi_gmem13_RDATA,
        m_axi_gmem13_RLAST,
        m_axi_gmem13_RID,
        m_axi_gmem13_RFIFONUM,
        m_axi_gmem13_RUSER,
        m_axi_gmem13_RRESP,
        m_axi_gmem13_BVALID,
        m_axi_gmem13_BREADY,
        m_axi_gmem13_BRESP,
        m_axi_gmem13_BID,
        m_axi_gmem13_BUSER,
        sext_ln177,
        mul_ln177_1,
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
input  [511:0] s_kernel_network_out80_dout;
input  [9:0] s_kernel_network_out80_num_data_valid;
input  [9:0] s_kernel_network_out80_fifo_cap;
input   s_kernel_network_out80_empty_n;
output   s_kernel_network_out80_read;
output   m_axi_gmem13_AWVALID;
input   m_axi_gmem13_AWREADY;
output  [63:0] m_axi_gmem13_AWADDR;
output  [0:0] m_axi_gmem13_AWID;
output  [31:0] m_axi_gmem13_AWLEN;
output  [2:0] m_axi_gmem13_AWSIZE;
output  [1:0] m_axi_gmem13_AWBURST;
output  [1:0] m_axi_gmem13_AWLOCK;
output  [3:0] m_axi_gmem13_AWCACHE;
output  [2:0] m_axi_gmem13_AWPROT;
output  [3:0] m_axi_gmem13_AWQOS;
output  [3:0] m_axi_gmem13_AWREGION;
output  [0:0] m_axi_gmem13_AWUSER;
output   m_axi_gmem13_WVALID;
input   m_axi_gmem13_WREADY;
output  [511:0] m_axi_gmem13_WDATA;
output  [63:0] m_axi_gmem13_WSTRB;
output   m_axi_gmem13_WLAST;
output  [0:0] m_axi_gmem13_WID;
output  [0:0] m_axi_gmem13_WUSER;
output   m_axi_gmem13_ARVALID;
input   m_axi_gmem13_ARREADY;
output  [63:0] m_axi_gmem13_ARADDR;
output  [0:0] m_axi_gmem13_ARID;
output  [31:0] m_axi_gmem13_ARLEN;
output  [2:0] m_axi_gmem13_ARSIZE;
output  [1:0] m_axi_gmem13_ARBURST;
output  [1:0] m_axi_gmem13_ARLOCK;
output  [3:0] m_axi_gmem13_ARCACHE;
output  [2:0] m_axi_gmem13_ARPROT;
output  [3:0] m_axi_gmem13_ARQOS;
output  [3:0] m_axi_gmem13_ARREGION;
output  [0:0] m_axi_gmem13_ARUSER;
input   m_axi_gmem13_RVALID;
output   m_axi_gmem13_RREADY;
input  [511:0] m_axi_gmem13_RDATA;
input   m_axi_gmem13_RLAST;
input  [0:0] m_axi_gmem13_RID;
input  [8:0] m_axi_gmem13_RFIFONUM;
input  [0:0] m_axi_gmem13_RUSER;
input  [1:0] m_axi_gmem13_RRESP;
input   m_axi_gmem13_BVALID;
output   m_axi_gmem13_BREADY;
input  [1:0] m_axi_gmem13_BRESP;
input  [0:0] m_axi_gmem13_BID;
input  [0:0] m_axi_gmem13_BUSER;
input  [57:0] sext_ln177;
input  [35:0] mul_ln177_1;
output   ap_ext_blocking_n;
output   ap_str_blocking_n;
output   ap_int_blocking_n;

reg ap_idle;
reg s_kernel_network_out80_read;
reg m_axi_gmem13_WVALID;

(* fsm_encoding = "none" *) reg   [0:0] ap_CS_fsm;
wire    ap_CS_fsm_pp0_stage0;
wire    ap_enable_reg_pp0_iter0;
reg    ap_enable_reg_pp0_iter1;
reg    ap_enable_reg_pp0_iter2;
reg    ap_enable_reg_pp0_iter3;
reg    ap_idle_pp0;
wire    ap_block_state1_pp0_stage0_iter0;
wire    ap_block_state2_pp0_stage0_iter1;
reg   [0:0] icmp_ln177_reg_134;
reg    ap_block_state3_pp0_stage0_iter2;
wire    ap_block_state4_pp0_stage0_iter3;
reg    ap_block_pp0_stage0_subdone;
wire   [0:0] icmp_ln177_fu_96_p2;
reg    ap_condition_exit_pp0_iter1_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    gmem13_blk_n_W;
wire    ap_block_pp0_stage0;
reg    s_kernel_network_out80_blk_n;
reg    ap_block_pp0_stage0_11001;
reg   [511:0] tmp_reg_143;
reg    ap_condition_exit_pp0_iter2_stage0;
reg    ap_block_pp0_stage0_01001;
reg   [35:0] indvar_flatten_fu_54;
wire   [35:0] add_ln177_fu_101_p2;
wire    ap_loop_init;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg   [0:0] ap_NS_fsm;
wire    ap_ext_blocking_cur_n;
wire    ap_int_blocking_cur_n;
wire    ap_enable_pp0;
wire    ap_start_int;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 1'd1;
#0 ap_enable_reg_pp0_iter1 = 1'b0;
#0 ap_enable_reg_pp0_iter2 = 1'b0;
#0 ap_enable_reg_pp0_iter3 = 1'b0;
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
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter1_stage0),
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
        end else if (((ap_loop_exit_ready_pp0_iter2_reg == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if ((1'b1 == ap_condition_exit_pp0_iter1_stage0)) begin
            ap_enable_reg_pp0_iter1 <= 1'b0;
        end else if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
            ap_enable_reg_pp0_iter1 <= ap_start_int;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter2 <= 1'b0;
    end else begin
        if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter2 <= ap_enable_reg_pp0_iter1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter3 <= 1'b0;
    end else begin
        if ((1'b1 == ap_condition_exit_pp0_iter2_stage0)) begin
            ap_enable_reg_pp0_iter3 <= 1'b0;
        end else if ((1'b0 == ap_block_pp0_stage0_subdone)) begin
            ap_enable_reg_pp0_iter3 <= ap_enable_reg_pp0_iter2;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        if ((ap_loop_init == 1'b1)) begin
            indvar_flatten_fu_54 <= 36'd0;
        end else if (((ap_enable_reg_pp0_iter1 == 1'b1) & (icmp_ln177_fu_96_p2 == 1'd0))) begin
            indvar_flatten_fu_54 <= add_ln177_fu_101_p2;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready;
        icmp_ln177_reg_134 <= icmp_ln177_fu_96_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln177_reg_134 == 1'd0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        tmp_reg_143 <= s_kernel_network_out80_dout;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (icmp_ln177_fu_96_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
        ap_condition_exit_pp0_iter1_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter1_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln177_reg_134 == 1'd1) & (ap_enable_reg_pp0_iter2 == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
        ap_condition_exit_pp0_iter2_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter2_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_exit_ready_pp0_iter2_reg == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_idle_pp0 == 1'b1) & (ap_start_int == 1'b0) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter3 == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
        ap_idle_pp0 = 1'b1;
    end else begin
        ap_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter3 == 1'b1) & (1'b0 == ap_block_pp0_stage0))) begin
        gmem13_blk_n_W = m_axi_gmem13_WREADY;
    end else begin
        gmem13_blk_n_W = 1'b1;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter3 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        m_axi_gmem13_WVALID = 1'b1;
    end else begin
        m_axi_gmem13_WVALID = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln177_reg_134 == 1'd0) & (ap_enable_reg_pp0_iter2 == 1'b1) & (1'b0 == ap_block_pp0_stage0))) begin
        s_kernel_network_out80_blk_n = s_kernel_network_out80_empty_n;
    end else begin
        s_kernel_network_out80_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((icmp_ln177_reg_134 == 1'd0) & (ap_enable_reg_pp0_iter2 == 1'b1) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        s_kernel_network_out80_read = 1'b1;
    end else begin
        s_kernel_network_out80_read = 1'b0;
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

assign add_ln177_fu_101_p2 = (indvar_flatten_fu_54 + 36'd1);

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd0];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_01001 = ((icmp_ln177_reg_134 == 1'd0) & (s_kernel_network_out80_empty_n == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b1));
end

always @ (*) begin
    ap_block_pp0_stage0_11001 = (((icmp_ln177_reg_134 == 1'd0) & (s_kernel_network_out80_empty_n == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b1)) | ((m_axi_gmem13_WREADY == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b1)));
end

always @ (*) begin
    ap_block_pp0_stage0_subdone = (((icmp_ln177_reg_134 == 1'd0) & (s_kernel_network_out80_empty_n == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b1)) | ((m_axi_gmem13_WREADY == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b1)));
end

assign ap_block_state1_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state2_pp0_stage0_iter1 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state3_pp0_stage0_iter2 = ((icmp_ln177_reg_134 == 1'd0) & (s_kernel_network_out80_empty_n == 1'b0));
end

assign ap_block_state4_pp0_stage0_iter3 = ~(1'b1 == 1'b1);

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_enable_reg_pp0_iter0 = ap_start_int;

assign ap_ext_blocking_cur_n = gmem13_blk_n_W;

assign ap_ext_blocking_n = (ap_ext_blocking_cur_n & 1'b1);

assign ap_int_blocking_cur_n = s_kernel_network_out80_blk_n;

assign ap_int_blocking_n = (ap_int_blocking_cur_n & 1'b1);

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter1_stage0;

assign ap_str_blocking_n = (1'b1 & 1'b1);

assign icmp_ln177_fu_96_p2 = ((indvar_flatten_fu_54 == mul_ln177_1) ? 1'b1 : 1'b0);

assign m_axi_gmem13_ARADDR = 64'd0;

assign m_axi_gmem13_ARBURST = 2'd0;

assign m_axi_gmem13_ARCACHE = 4'd0;

assign m_axi_gmem13_ARID = 1'd0;

assign m_axi_gmem13_ARLEN = 32'd0;

assign m_axi_gmem13_ARLOCK = 2'd0;

assign m_axi_gmem13_ARPROT = 3'd0;

assign m_axi_gmem13_ARQOS = 4'd0;

assign m_axi_gmem13_ARREGION = 4'd0;

assign m_axi_gmem13_ARSIZE = 3'd0;

assign m_axi_gmem13_ARUSER = 1'd0;

assign m_axi_gmem13_ARVALID = 1'b0;

assign m_axi_gmem13_AWADDR = 64'd0;

assign m_axi_gmem13_AWBURST = 2'd0;

assign m_axi_gmem13_AWCACHE = 4'd0;

assign m_axi_gmem13_AWID = 1'd0;

assign m_axi_gmem13_AWLEN = 32'd0;

assign m_axi_gmem13_AWLOCK = 2'd0;

assign m_axi_gmem13_AWPROT = 3'd0;

assign m_axi_gmem13_AWQOS = 4'd0;

assign m_axi_gmem13_AWREGION = 4'd0;

assign m_axi_gmem13_AWSIZE = 3'd0;

assign m_axi_gmem13_AWUSER = 1'd0;

assign m_axi_gmem13_AWVALID = 1'b0;

assign m_axi_gmem13_BREADY = 1'b0;

assign m_axi_gmem13_RREADY = 1'b0;

assign m_axi_gmem13_WDATA = tmp_reg_143;

assign m_axi_gmem13_WID = 1'd0;

assign m_axi_gmem13_WLAST = 1'b0;

assign m_axi_gmem13_WSTRB = 64'd18446744073709551615;

assign m_axi_gmem13_WUSER = 1'd0;

endmodule //vadd_DRAM_output_stream_Pipeline_VITIS_LOOP_177_1_VITIS_LOOP_178_2
