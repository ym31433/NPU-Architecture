`timescale 100ps/1ps
`define CYCLE 50

module npu_tb();
	reg clk, rst;
	reg we, oe;
	reg [31:0] data;
	wire [31:0] data_w;
	wire ready;
	wire [2:0] pe_state;
	wire [3:0] state;
	wire [1:0] num_layers;
	wire [4:0] num_neurons_3;
	wire [5:0] num_multadds;
   wire [4:0] state_count;
   wire [5:0] multadd_count;
	/*
	reg pe_oe;
	reg do_act;
	wire [4:0] counter;
	wire [11:0] ArrWgt_Rd;
	wire [11:0] ArrWgt_Wr;
	wire [31:0] ArrWeights_1;
	wire fp_mac_acc;
	*/
	wire [31:0] fp_mac_a;
	wire [31:0] fp_mac_b;
	wire [31:0] fp_mac_output;
	wire [4:0]  counter;
	wire fp_mac_acc;
	wire [4:0] InBuf_Rd;
	wire [4:0] InBuf_Wr;
	wire [4:0] OutBuf_Rd;
	wire [4:0] OutBuf_Wr;
	wire [11:0] ArrWgt_Rd; // = {ARR_WGT_IND_SIZE{1'b0}}; 	// read index
	wire [11:0] ArrWgt_Wr; // = {ARR_WGT_IND_SIZE{1'b0}}; 	// write index 
	wire pe_oe_0;
	wire [31:0] OutBuf_i0;
	wire [31:0] OutBuf_n_i0;
	/*
	wire fp_add_en;
	wire [31:0] fp_add_a;
	wire [31:0] fp_add_b;
	wire [31:0] fp_add_output;
	wire fp_div_en;
	wire [31:0] fp_div_a;
	wire [31:0] fp_div_b;
	wire [31:0] fp_div_output;
	*/
	
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


	//test_fp_ip fp_ip(.clk(clk), .rst(rst), .ctrl(pe_state), .output_ctrl(pe_oe), .data(data_w), .do_act(do_act), .counter(counter), .ArrWgt_Rd(ArrWgt_Rd), .ArrWgt_Wr(ArrWgt_Wr), .ArrWeights_1(ArrWeights_1), .fp_mac_acc(fp_mac_acc), .fp_mac_a(fp_mac_a), .fp_mac_b(fp_mac_b), .fp_mac_q(fp_mac_q), .fp_add_en(fp_add_en), .fp_add_a(fp_add_a), .fp_add_b(fp_add_b), .fp_add_output(fp_add_output), .fp_div_en(fp_div_en), .fp_div_a(fp_div_a), .fp_div_b(fp_div_b), .fp_div_output(fp_div_output));
	//npu NPU(.rst(rst), .clk(clk), .we(we), .oe(oe), .data(data_w), .ready(ready));
	npu NPU(.rst(rst), .clk(clk), .we(we), .oe(oe), .data(data_w), .ready(ready), .pe_state_r0(pe_state), .state_r(state), .num_layers_r(num_layers), .num_neurons_r3(num_neurons_3), .num_multadds(num_multadds), .state_count_r(state_count), .multadd_count_r(multadd_count), .fp_mac_a(fp_mac_a), .fp_mac_b(fp_mac_b), .fp_mac_output(fp_mac_output), 
	.counter(counter), 
	.fp_mac_acc(fp_mac_acc), 
	.ArrWgt_Rd(ArrWgt_Rd), 
	.ArrWgt_Wr(ArrWgt_Wr), 
	.InBuf_Rd(InBuf_Rd), 
	.InBuf_Wr(InBuf_Wr), 
	.OutBuf_Rd(OutBuf_Rd), 
	.OutBuf_Wr(OutBuf_Wr),
	.pe_oe_0(pe_oe_0),
	.OutBuf_i0(OutBuf_i0),
	.OutBuf_n_i0(OutBuf_n_i0));
	
	
	always begin
		#(`CYCLE/2) clk = ~clk;
	end
	
	initial begin
		#0;
		clk = 1'b1;
		rst = 1'b0;
		data = 32'bz;
		we = 1'b0;
		oe = 1'b0;
		
		#(`CYCLE);
		rst = 1'b1;
		
		#(`CYCLE);
		rst = 1'b0;
		
		#(`CYCLE/2);
		we = 1'b1;
		
		#(`CYCLE);
		data = 32'd0; // 2 layers
		
		#(`CYCLE);
		data = 32'd0; // 1 input neuron
		
		#(`CYCLE);
		data = 32'd0;
		
		#(`CYCLE);
		data = 32'd0;
		
		#(`CYCLE);
		data = 32'd0; // 1 output neuron
		
		#(`CYCLE);
		data = 32'd0; // no activation function
		
		#(`CYCLE);
		data = {1'b0,1'b1,30'b0};  //weight: 2
		
		#(`CYCLE);
		data = 32'h45800000; //bias: 4096
		
		#(`CYCLE);
		data = 32'h44800000; //intput1: 1024
		
		#(`CYCLE);
		we = 1'b0;
		
		#(`CYCLE*5);
		oe = 1;
		
		#(`CYCLE*5);
		
		/*
		pe_state = PE_LOAD;
		pe_oe = 1'b0;
		data = {1'b0,1'b1,30'b0};  //2
		//in_flag = 1'b1;
		do_act = 0;

		#(`CYCLE);
		data = 32'h44800000; //1024
		
		#(`CYCLE);
		data = 32'h45800000; //4096
		
		
		#(`CYCLE);
		data = {1'b0,1'b1,30'b0};  //2
		#(`CYCLE);
		data = 32'h45800000; //4096
		#(`CYCLE);
		data = 32'h44800000; //1024
		
		#(`CYCLE);
		pe_state = PE_MA;
		data = 32'h45800000; //4096
		//data = 32'h44800000; //1024
		
		#(`CYCLE);
		data = {1'b0,1'b1,30'b0};  //2
		//data = 32'bz;
		
		//in_flag = 1'b0;
		
		//#(`CYCLE*3);
		#(`CYCLE);
		pe_state = PE_BIAS;
		data = 32'h44800000; //1024
		//data = 32'h3f800000;	// 1
				
		#(`CYCLE);
		data = 32'bz;
		
		#(`CYCLE*3);
		pe_state = PE_ACT;
		
		#(`CYCLE);
		pe_state = PE_MA;
		data = 32'h45800000; //4096
		#(`CYCLE);
		data = {1'b0,1'b1,30'b0};  //2
		#(`CYCLE);
		pe_state = PE_BIAS;
		data = 32'h44800000; //1024
		#(`CYCLE);
		data = 32'bz;
		
		
		do_act = 1;
		#(`CYCLE*3);
		pe_state = PE_ACT;
		
		#(`CYCLE*23);
		pe_state = PE_IDLE;
		pe_oe = 1'b1;
		
		#(`CYCLE*5);
		*/
		
		$finish;
		
	end
endmodule
