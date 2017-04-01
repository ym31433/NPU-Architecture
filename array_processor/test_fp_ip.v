/*
module test_fp_ip(
		input  wire [31:0] a,      //      a.a
		input  wire        acc,    //    acc.acc
		input  wire        areset, // areset.reset
		input  wire [31:0] b,      //      b.b
		input  wire        clk,    //    clk.clk
		input  wire [0:0]  en,     //     en.en
		output wire [31:0] q       //      q.q
		);
		
		fp_mac mac(.a(a), .b(b), .clk(clk), .areset(areset), .en(en), .q(q), .acc(acc));

endmodule

*/

module test_fp_ip(
	input wire clk,
	input wire rst,
	input wire [2:0] ctrl,
	input wire output_ctrl,
	inout wire [31:0] data,
	input wire do_act,
	output [4:0] counter,
	output [11:0] ArrWgt_Rd,
	output [11:0] ArrWgt_Wr, 
	output [31:0] ArrWeights_1,
	output fp_mac_acc,
	output [31:0] fp_mac_a,
	output [31:0] fp_mac_b,
	output [31:0] fp_mac_q,
	output fp_add_en,
	output [31:0] fp_add_a,
	output [31:0] fp_add_b,
	output [31:0] fp_add_output,
	output fp_div_en,
	output [31:0] fp_div_a,
	output [31:0] fp_div_b,
	output [31:0] fp_div_output
	);
	pe PE(.Clock(clk), .Reset(rst), .Ctrl(ctrl), .OutputCtrl(output_ctrl), .Data(data), .EnableAct(do_act), .counter(counter), .ArrWgt_Rd(ArrWgt_Rd), .ArrWgt_Wr(ArrWgt_Wr), .ArrWeights_1(ArrWeights_1), .fp_mac_acc(fp_mac_acc), .fp_mac_a(fp_mac_a), .fp_mac_b(fp_mac_b), .fp_mac_output(fp_mac_q), .fp_add_en(fp_add_en), .fp_add_a(fp_add_a), .fp_add_b(fp_add_b), .fp_add_output(fp_add_output), .fp_div_en(fp_div_en), .fp_div_a(fp_div_a), .fp_div_b(fp_div_b), .fp_div_output(fp_div_output));
endmodule
