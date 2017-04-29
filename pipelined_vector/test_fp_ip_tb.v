`timescale 100ps/1ps
`define CYCLE 50
`define INFILE "/home/cosine/altera/16.0/test_fp/input_hex.dat"
module fp_tb();
	reg clk, rst, en;
	reg [31:0] a, b;
	wire signed [31:0] q;
	reg acc;
	
	reg signed [31:0] in[0:5];
	
	test_fp_ip fp_ip (.a(a), .b(b), .clk(clk), .areset(rst), .en(en), .q(q), .acc(acc));
	
	integer in_file;
	
	always begin
		#(`CYCLE/2) clk = ~clk;
	end
	
	initial $readmemh (`INFILE, in);
/*	
	initial begin
		in_file = $fopen("/home/cosine/altera/16.0/test_fp/input_hex.dat", "r");
		if(in_file == 0) begin
			$display("File open error!");
			$finish;
		end
	
		x = $readmemh(in_file, "%f %f\n", a, b);
		
		$display("%f %f", a, b);
		
		$fclose(in_file);
	end
*/
	initial begin
		#0;
		clk  = 1'b1;
		rst  = 1'b0;
		en   = 1'b1;
		acc  = 1'b0;

		//a = 32'f1;
		//b = 32'f2;
		//a = {2'b0,7'b1,23'b0};	// 4
		//a = {1'b0, 1'b1, 6'b0, 1'b1, 23'b0};
		//b = {1'b0,1'b1,30'b0};	// 2
		//a = 1;
		//b = 2;
		//a = {1'b0, 1'b1, 2'b0, 1'b1, 1'b0, 1'b1, 25'b0};  // 2^21
		//b = {1'b0, 1'b1, 2'b0, 1'b1, 3'b0, 1'b1, 23'b0}; // 2^18
		
		#(`CYCLE) rst = 1'b1;
		#(`CYCLE) rst = 1'b0;
		acc = 1'b1;
		a = in[0];
		b = in[1];
		#(`CYCLE); 
		a = 0;
		b = 0;
		
		
		//#(`CYCLE) en = 1'b0;
		
		#(`CYCLE*10);
		en  = 1'b1;
		a = in[2];
		b = in[3];
		
		//#(`CYCLE) en = 1'b0;
		#(`CYCLE); 
		a = 0;
		b = 0;
		
		#(`CYCLE*10);
		en  = 1'b1;
		a = in[4];
		b = in[5];
		#(`CYCLE);
		a = 0;
		b = 0;
		
		//#(`CYCLE) en = 1'b0;
		#(`CYCLE) acc = 1'b0;
		
		#(`CYCLE*10);
		
		$finish;
	end
	
endmodule