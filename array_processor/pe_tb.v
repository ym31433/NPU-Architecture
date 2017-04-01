`timescale 100ps/1ps
`define CYCLE 50

module pe_tb();
	reg clk, rst;
	reg [2:0] pe_state;
	reg pe_oe;
	reg [31:0] data;
	wire [31:0] data_w;
	reg do_act;
	wire [3:0] counter;
	wire [11:0] ArrWgt_Rd;
	wire [11:0] ArrWgt_Wr;
	wire [31:0] ArrWeights_1;
	wire fp_mac_acc;
	wire [31:0] fp_mac_b;
	
	//reg in_flag;
	//assign data_w = (in_flag) data : 32'bz;
	assign data_w = data;
	
	parameter PE_IDLE    = 3'd0;
	parameter PE_LOAD    = 3'd1;  //load weights and biases
	parameter PE_MA      = 3'd2;  //multiply & add
	parameter PE_MAB     = 3'd3;  //multiply & add (data from buffer)
	parameter PE_MABO    = 3'd4;
	parameter PE_BIAS    = 3'd5;  //adding biases (multiply 1 & add)
	parameter PE_ACT     = 3'd6;  //activation function
	parameter PE_ACT_CLR = 3'd7;  //activation function with clearing buffer


	test_fp_ip fp_ip(.clk(clk), .rst(rst), .ctrl(pe_state), .output_ctrl(pe_oe), .data(data_w), .do_act(do_act), .counter(counter), .ArrWgt_Rd(ArrWgt_Rd), .ArrWgt_Wr(ArrWgt_Wr), .ArrWeights_1(ArrWeights_1), .fp_mac_acc(fp_mac_acc), .fp_mac_b(fp_mac_b));

	always begin
		#(`CYCLE/2) clk = ~clk;
	end
	
	initial begin
		#0;
		clk = 1'b1;
		rst = 1'b0;
		data = 32'bz;
		//in_flag = 1'b0;
		
		#(`CYCLE);
		rst = 1'b1;
		
		#(`CYCLE);
		rst = 1'b0;
		pe_state = PE_LOAD;
		pe_oe = 1'b0;
		data = {1'b0,1'b1,30'b0};  //2
		//in_flag = 1'b1;
		do_act = 0;
		
		#(`CYCLE);
		data = 32'h44800000; //1024
		
		#(`CYCLE);
		pe_state = PE_MA;
		data = 32'h45800000; //4096
		//data = 32'h44800000; //1024
		
		#(`CYCLE);
		data = 32'bz;
		
		//in_flag = 1'b0;
		
		#(`CYCLE*3);
		pe_state = PE_BIAS;
		//data = 32'h44800000; //1024
		//data = 32'h3f800000;	// 1
				
		#(`CYCLE);
		data = 32'bz;
		
		#(`CYCLE*3);
		pe_state = PE_ACT;
		
		#(`CYCLE);
		pe_state = PE_IDLE;
		pe_oe = 1'b1;
		
		#(`CYCLE*5);
		
		$finish;
		
	end
endmodule