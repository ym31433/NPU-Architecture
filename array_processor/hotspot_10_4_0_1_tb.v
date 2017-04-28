`timescale 100ps/1ps
`define CYCLE 50
`define NUM_LAYERS 1 // 3 layers
`define NUM_IN 9 // 10 neurons
`define NUM_H1 3 // 4 neurons
`define NUM_H2 0
`define NUM_OUT 0 // 1 neuron
`define ACT 1 // hidden layer 1 activation function: 001
`define NUM_W 49 // number of weights and biases
//`define NUM_MA 10 // 10 multiply-adds (without bias)
`define NUM_CALC 40 // number of cycles to calculate
`define NUM_DATA 65536 // number of data sets

`define IN_FILE "/home/cosine/spring2017/cs533/project/benchmark/hotspot/data/pipelined_vector/input_hex.dat"
`define W_FILE "/home/cosine/spring2017/cs533/project/benchmark/hotspot/nn_config/pipelined_vector/10_4_0_1_hex.dat"
`define OUT_FILE "/home/cosine/spring2017/cs533/project/benchmark/hotspot/data/pipelined_vector/10_4_0_1_hex.dat"

module npu_tb();
	reg clk, rst;
	reg we, oe;
	reg [31:0] data;
	wire [31:0] data_w;
	wire ready;
	
	/*
	wire [2:0] pe_state_0, pe_state_1;
	wire pe_oe_0, pe_oe_1;
	*/
	wire [2:0] pe_state_0;
	wire [3:0] state;
	wire do_act;
	wire [4:0] pe_count;
	/*
	wire [1:0] num_layers;
	wire [4:0] num_neurons_3;
	wire [5:0] num_multadds;
   wire [4:0] state_count;
   wire [5:0] multadd_count;
	*/
	
	reg signed [31:0] in[0:`NUM_IN];
	reg signed [31:0] w[0:`NUM_W-1];
	integer i;
	integer infile, wfile, outfile;
	integer iteration;
	
	/*
	reg pe_oe;
	reg do_act;
	wire [4:0] counter;
	wire [11:0] ArrWgt_Rd;
	wire [11:0] ArrWgt_Wr;
	wire [31:0] ArrWeights_1;
	wire fp_mac_acc;
	
	wire [31:0] fp_mac_a_0;
	wire [31:0] fp_mac_b_0;
	wire [31:0] fp_mac_output_0;
	wire [31:0] fp_mac_a_1;
	wire [31:0] fp_mac_b_1;
	wire [31:0] fp_mac_output_1;
	/
	wire [4:0]  counter;
	wire fp_mac_acc;
	wire [4:0] InBuf_Rd;
	wire [4:0] InBuf_Wr;
	wire [4:0] OutBuf_Rd;
	wire [4:0] OutBuf_Wr;
	wire [11:0] ArrWgt_Rd; // = {ARR_WGT_IND_SIZE{1'b0}}; 	// read index
	wire [11:0] ArrWgt_Wr; // = {ARR_WGT_IND_SIZE{1'b0}}; 	// write index 
	wire [31:0] InBuf_i0;
	wire [31:0] ArrWeights_i0;
	wire [31:0] ArrWeights_i1;
	wire [31:0] OutBuf_i0;
	wire [31:0] OutBuf_n_i0;
	wire OutBuf_Full;
	
	wire fp_add_en;
	wire [31:0] fp_add_a;
	wire [31:0] fp_add_b;
	wire [31:0] fp_add_output;
	wire fp_div_en;
	wire [31:0] fp_div_a;
	wire [31:0] fp_div_b;
	wire [31:0] fp_div_output;
	*/
	
	assign data_w = data;
	
	parameter PE_IDLE    = 3'd0;
	parameter PE_LOAD    = 3'd1;  //load weights and biases
	parameter PE_MA      = 3'd2;  //multiply & add
	parameter PE_MAB     = 3'd3;  //multiply & add (data from buffer)
	parameter PE_MABO    = 3'd4;
	parameter PE_BIAS    = 3'd5;  //adding biases (multiply 1 & add)
	parameter PE_ACT     = 3'd6;  //activation function
	parameter PE_ACT_CLR = 3'd7;  //activation function with clearing buffer


	
	//npu NPU(.rst(rst), .clk(clk), .we(we), .oe(oe), .data(data_w), .ready(ready));
	npu NPU(.rst(rst), .clk(clk), .we(we), .oe(oe), .data(data_w), .ready(ready), .state_r(state), .pe_state_r0(pe_state_0), .do_act(do_act), .pe_count_r(pe_count));
	/*
	npu NPU(.rst(rst), .clk(clk), .we(we), .oe(oe), .data(data_w), .ready(ready),
	.pe_state_r0(pe_state_0), .pe_state_r1(pe_state_1), .pe_oe_0(pe_oe_0), .pe_oe_1(pe_oe_1),
	.state_r(state), .num_layers_r(num_layers), .num_neurons_r3(num_neurons_3), .num_multadds(num_multadds), .state_count_r(state_count), .multadd_count_r(multadd_count),
	.fp_mac_a_0(fp_mac_a_0), .fp_mac_b_0(fp_mac_b_0), .fp_mac_output_0(fp_mac_output_0),
	.fp_mac_a_1(fp_mac_a_1), .fp_mac_b_1(fp_mac_b_1), .fp_mac_output_1(fp_mac_output_1)
	);
	*/
	/*
	npu NPU(.rst(rst), .clk(clk), .we(we), .oe(oe), .data(data_w), .ready(ready),
	.pe_state_r0(pe_state),
	.state_r(state), .num_layers_r(num_layers), .num_neurons_r3(num_neurons_3), .num_multadds(num_multadds), .state_count_r(state_count), .multadd_count_r(multadd_count),
	.fp_mac_a(fp_mac_a), .fp_mac_b(fp_mac_b), .fp_mac_output(fp_mac_output), 
	.counter(counter), 
	.fp_mac_acc(fp_mac_acc), 
	.ArrWgt_Rd(ArrWgt_Rd), 
	.ArrWgt_Wr(ArrWgt_Wr), 
	.InBuf_Rd(InBuf_Rd), 
	.InBuf_Wr(InBuf_Wr), 
	.OutBuf_Rd(OutBuf_Rd), 
	.OutBuf_Wr(OutBuf_Wr),
	.pe_oe_0(pe_oe_0),
	.InBuf_i0(InBuf_i0),
	.ArrWeights_i0(ArrWeights_i0),
	.ArrWeights_i1(ArrWeights_i1),
	.OutBuf_i0(OutBuf_i0),
	.OutBuf_n_i0(OutBuf_n_i0),
	.OutBuf_Full(OutBuf_Full));
	*/
	
	always begin
		#(`CYCLE/2) clk = ~clk;
	end
	
	initial begin
      /**** open files ****/
	   //$readmemh (`IN_FILE, in);
	   //$readmemh (`W_FILE, w);

		infile = $fopen(`IN_FILE, "r");
		wfile = $fopen(`W_FILE, "r");
		outfile = $fopen(`OUT_FILE, "a");
		
      /**** reset the NPU ****/
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
		
		/**** start the loop ****/
		//for(iteration = 0; iteration < NUM_DATA; iteration = iteration + 1) begin
		for(iteration = 0; iteration < 1; iteration = iteration + 1) begin
			/**** read input file and weight file ****/
			for(i = 0; i <= `NUM_IN; i = i + 1) begin
				//$fgets(in[i], infile);
				$fscanf(infile, "%08x", in[i]);
			end
			for(i = 0; i < `NUM_W; i = i + 1) begin
				//$fgets(w[i], wfile);
				$fscanf(wfile, "%08x", w[i]);
			end
		
			we = 1'b1;
			/**** send configurations ****/
			#(`CYCLE);
			data = `NUM_LAYERS; // 2 layers
			
			#(`CYCLE);
			data = `NUM_IN; // 2 input neuron
			
			#(`CYCLE);
			data = `NUM_H1;
			
			#(`CYCLE);
			data = `NUM_H2;
			
			#(`CYCLE);
			data = `NUM_OUT; // 2 output neuron
			
			#(`CYCLE);
			data = `ACT; // no activation function
			
			
			/**** send wieghts ****/
			for(i = 0; i < `NUM_W; i = i+1) begin
				#(`CYCLE);
				data = w[i];
			end
			
			
			/**** first layer: send inputs ****/
			for(i = 0; i <= `NUM_IN; i = i+1) begin
				#(`CYCLE);
				data = in[i];
			end
			#(`CYCLE);
			we = 1'b0;
			data = 32'bz;
			
			
			/**** rest of the layers calculation ****/
			#(`CYCLE*`NUM_CALC);
			
			
			/**** receive outputs and write output data ****/
			oe = 1;
			for(i = 0; i <= `NUM_OUT; i = i+1) begin
				$fstrobe(outfile, "%x", data_w);
				 #(`CYCLE);
			end
			oe = 0;
		end
		
		$fclose(infile);
		$fclose(wfile);
		$fclose(outfile);

		$finish;
		
	end
endmodule
