//==========================================================
//         Title: pe.v
//   Description: Processing Element (PE Module)
// 				  This module implements a single processing 
// 				  element to be used in a vector architecture 
//				  design for neural algorithms. 
//         Class: CS 533 - Parallel Architectures
// 		   Names: Elizabeth Reed,
// 				  Gowthami Manikandan, 
// 				  Yu-Hsuan Tseng
//  Date Created: 2017-Mar-01
// Last Modified: 2017-Mar-21
//==========================================================

//**********************************************************/
// Module: PE 
// Description: Processing element which has the capability of 
// 		performing all operations expected of a single neuron 
// 		node in a neural network.
// Inputs:
// 		- Clock 	 : Clock signal from controller.
// 		- Reset 	 : Clears the internal weight array, 
// 					   the input buffer, output buffer,
// 					   the stored multiply-add value, and 
// 					   all buffer/array index pointers.
// 		- Ctrl		 : Signal for the operation to be 
// 					   performed by the PE in current cycle.
// 		- OutputCtrl : Flag that indicates if the PE should 
// 					   push a value from its output buffer 
// 					   onto the output pins.
// 		- DataIn 	 : Input data supplied to the PE.
// 					   Interpretation of this data depends 
// 					   on the ctrl signal. Could either be 
// 					   a weight to be stored within the PE, 
// 					   or a current data value to be used 
// 					   in a multiply-add operation (in this 
// 					   case it should also be stored in the 
// 					   input buffer).
// Output:
// 		- DataOut 	 : Output data value which is updated 
// 					   when the OutputCtrl signal is set 
// 					   to 1.
//**********************************************************/

//module PE(Clock, Reset, Ctrl, OutputCtrl, DataIn, DataOut);
//module pe(Clock, Reset, Ctrl, OutputCtrl, EnableAct, Data);
module pe(Clock, Reset, Ctrl, OutputCtrl, EnableAct, Data, counter, ArrWgt_Rd, ArrWgt_Wr, ArrWeights_1, fp_mac_acc, fp_mac_a, fp_mac_b, fp_mac_output, fp_add_a, fp_add_b, fp_add_en, fp_add_output, fp_div_a, fp_div_b, fp_div_en, fp_div_output);

output [4:0]  counter;
output [11:0] ArrWgt_Rd;
output [11:0] ArrWgt_Wr;
output [31:0] ArrWeights_1;
output fp_mac_acc;
output [31:0] fp_mac_a;
output [31:0] fp_mac_b;
output [31:0] fp_mac_output;
output fp_add_en;
output [31:0] fp_add_a;
output [31:0] fp_add_b;
output [31:0] fp_add_output;
output fp_div_en;
output [31:0] fp_div_a;
output [31:0] fp_div_b;
output [31:0] fp_div_output;


//--------------------------------
// GENERAL PARAMETER VALUES:
//--------------------------------
// States for operation control in the PE:
parameter PE_IDLE    = 3'd0;
parameter PE_LOAD    = 3'd1;  //load weights and biases
parameter PE_MA      = 3'd2;  //multiply & add
parameter PE_MAB     = 3'd3;  //multiply & add (data from buffer)
parameter PE_MABO    = 3'd4;
parameter PE_BIAS    = 3'd5;  //adding biases (multiply 1 & add)
parameter PE_ACT     = 3'd6;  //activation function
parameter PE_ACT_CLR = 3'd7;  //activation function with clearing buffer

parameter PE_RESET = 8; 	// NOT used as an input value, but used for state.

parameter CTRL_SIZE = 3; 	// number of bits for the ctrl signal

parameter DATA_SIZE = 32;	// number of bits in the data input 

// Counter values for FP operations
parameter COUNT_FP_ADD = 2;	// 3 cycles
parameter COUNT_FP_DIV = 18;	// 19 cycles 
parameter COUNT_MA = 3; 	// FP multiply add operation; 4 cycles
parameter COUNT_ACT = COUNT_FP_ADD + COUNT_FP_DIV + 2;	// FP activation function operation 
parameter COUNTER_SIZE = 5;	// number of bits in the counter value

// Neural network and system configuration info:
parameter MAX_NEURONS_PER_STAGE = 32;	// maximum neurons in a single stage
parameter MAX_NUM_STAGES = 3;	// maximum number of total stages (including start and output)
parameter NUM_TOTAL_PE = 8;		// total number of PEs in the system

// Buffer Index Lengths
// Note: Since these calculations are log2(x), they are hard-coded here.
parameter IN_BUF_IND_SIZE = 5; // log2(32)
parameter OUT_BUF_IND_SIZE = 5; // log2(32)
	// Note: Actually expected to be log2(32/8) = log2(4) = 2; so this could be reduced.
parameter ARR_WGT_IND_SIZE = 12;
	// log2(32*32*3) = log2(3072) = 12
	// Actually, expect to have 8 PEs in the system, so this would be:
	// log2(32*32*3/8) = log2(384) = 9

// Buffer and internal array lengths:
parameter IN_BUF_LEN = MAX_NEURONS_PER_STAGE; 	// length of the input buffer 
	// Note: This input buffer size must be as large as the maximum number 
	// 		 of neurons in a single stage.
parameter OUT_BUF_LEN = MAX_NEURONS_PER_STAGE / NUM_TOTAL_PE; // length of the output buffer
	// Note: This output buffer size must be as large as the maximum number 
	// 		 of neurons in a stage, divided by the number of PEs.
	// 		 (The ceiling quotient of neurons per stage divided by the  
	// 		 total PEs in the system.)
parameter WGT_ARR_LEN = MAX_NEURONS_PER_STAGE * MAX_NEURONS_PER_STAGE * (MAX_NUM_STAGES-1) / NUM_TOTAL_PE;	// length of the array of weight values to be used with multiply-add operations
	// Note: This has to support the worst-case where the maximum number of 
	// 		 neurons is present in every stage. (Consider num_stages-1, because 
	//		 computation is performed at each stage AFTER the "input stage".)  
	// 		 The total number of weights required for the entire computation of all 
	//		 stages is divided by the number of PEs in the system to determine how 
	// 		 many values each individual PE must store in the worst case.

// Basic Boolean values:
parameter TRUE = 1'b1;
parameter FALSE = 1'b0;
//--------------------------------

input Clock;
input Reset;
input [CTRL_SIZE-1:0] Ctrl;		// control signal (for PE operation)
input OutputCtrl;				// provides output from buffer if set to 1
input EnableAct; 				// flag to enable the use of the activation function
inout signed [DATA_SIZE-1:0] Data; 	// data

reg signed [DATA_SIZE-1:0] CurrVal_n;
reg signed [DATA_SIZE-1:0] CurrVal; // the current accumulated value used in the multiply-add calculation

reg signed [DATA_SIZE-1:0] InBuf [0:IN_BUF_LEN-1]; 	// input buffer
reg signed [DATA_SIZE-1:0] OutBuf [0:OUT_BUF_LEN-1]; 	// output buffer
reg signed [DATA_SIZE-1:0] InBuf_n [0:IN_BUF_LEN-1]; 	// input buffer
reg signed [DATA_SIZE-1:0] OutBuf_n [0:OUT_BUF_LEN-1]; 	// output buffer

// Counter used for the calculation stages.
reg [COUNTER_SIZE-1:0] counter;
reg [COUNTER_SIZE-1:0] counter_n;	// next value

// Pointers for read/write within each FIFO buffer, and flags for empty/full conditions 
reg [IN_BUF_IND_SIZE-1:0] InBuf_Wr_n; // = {IN_BUF_IND_SIZE{1'b0}}; 	// write index 
reg [IN_BUF_IND_SIZE-1:0] InBuf_Rd_n; // = {IN_BUF_IND_SIZE{1'b0}}; 	// read index 
reg [IN_BUF_IND_SIZE-1:0] InBuf_Wr; // = {IN_BUF_IND_SIZE{1'b0}}; 	// write index 
reg [IN_BUF_IND_SIZE-1:0] InBuf_Rd; // = {IN_BUF_IND_SIZE{1'b0}}; 	// read index
reg [IN_BUF_IND_SIZE-1:0] InBuf_RdStart_n; // = {IN_BUF_IND_SIZE{1'b0}}; 	// read index start bookmark
reg [IN_BUF_IND_SIZE-1:0] InBuf_RdStart; // = {IN_BUF_IND_SIZE{1'b0}}; 	// read index start bookmark
reg InBuf_Full_n; // flag to indicate buffer is full (next value)
reg InBuf_Full; // flag to indicate buffer is full 

reg [OUT_BUF_IND_SIZE-1:0] OutBuf_Wr_n; // = {OUT_BUF_IND_SIZE{1'b0}}; 	// write index 
reg [OUT_BUF_IND_SIZE-1:0] OutBuf_Rd_n; // = {OUT_BUF_IND_SIZE{1'b0}}; 	// read index
reg [OUT_BUF_IND_SIZE-1:0] OutBuf_Wr; // = {OUT_BUF_IND_SIZE{1'b0}}; 	// write index 
reg [OUT_BUF_IND_SIZE-1:0] OutBuf_Rd; // = {OUT_BUF_IND_SIZE{1'b0}}; 	// read index
reg OutBuf_Full_n; // flag to indicate buffer is full (next value)
reg OutBuf_Full; // flag to indicate buffer is full 
reg OutBuf_Empty_n; // flag to indicate buffer is empty 
reg OutBuf_Empty; // flag to indicate buffer is empty

reg signed [DATA_SIZE-1:0] ArrWeights [0:WGT_ARR_LEN-1];	// array of weights to be used in multiply-add operations
reg signed [DATA_SIZE-1:0] ArrWeights_n [0:WGT_ARR_LEN-1];	// array of weights to be used in multiply-add operations

// DEBUG
assign ArrWeights_1 = ArrWeights[1];

// Pointers for read/write in the array of weights; and flags for full/empty (?).
reg [ARR_WGT_IND_SIZE-1:0] ArrWgt_Wr_n; // {ARR_WGT_IND_SIZE{1'b0}}; 	// write index 
reg [ARR_WGT_IND_SIZE-1:0] ArrWgt_Rd_n; // = {ARR_WGT_IND_SIZE{1'b0}}; 	// read index
reg [ARR_WGT_IND_SIZE-1:0] ArrWgt_Wr; // = {ARR_WGT_IND_SIZE{1'b0}}; 	// write index 
reg [ARR_WGT_IND_SIZE-1:0] ArrWgt_Rd; // = {ARR_WGT_IND_SIZE{1'b0}}; 	// read index
reg ArrWgt_Full; // flag to indicate buffer is full (next value)
reg ArrWgt_Full_n; // flag to indicate buffer is full 

wire signed [DATA_SIZE-1:0] output_val;

assign Data = (((Ctrl == PE_MABO) || (Ctrl == PE_IDLE)) && (OutputCtrl == TRUE)) ? output_val : 32'bz;
assign output_val = OutBuf[OutBuf_Rd];

integer i1, i2, i3;

// value connections for the fp_mac module
wire signed [DATA_SIZE-1:0] fp_mac_output;
reg signed [DATA_SIZE-1:0] fp_mac_a;
reg signed [DATA_SIZE-1:0] fp_mac_b;
reg fp_mac_acc;
wire fp_mac_en;
assign fp_mac_en = 1;
// instantiate fp_mac module
fp_mac fp_mac_mod(.a(fp_mac_a), .acc(fp_mac_acc), .areset(Reset), .b(fp_mac_b), .clk(Clock), .q(fp_mac_output), .en(fp_mac_en));

// Link the fp_add module and the fp_div module to perform 
// the fast sigmoid activation function.
// Connections for fp_add module: 
wire signed [DATA_SIZE-1:0] fp_add_output;
reg signed [DATA_SIZE-1:0] fp_add_a;
wire signed [DATA_SIZE-1:0] fp_add_b;
assign fp_add_b = 32'h3f800000;	// fp value of 1
wire fp_add_en;
assign fp_add_en = 1;
// instantiate fp_add module 
fp_add fp_add_mod(.a(fp_add_a), .areset(Reset), .b(fp_add_b), .clk(Clock), .q(fp_add_output), .en(fp_add_en));

// connections for fp_div module 
wire signed [DATA_SIZE-1:0] fp_div_output;
reg signed [DATA_SIZE-1:0] fp_div_a;	// numerator
reg signed [DATA_SIZE-1:0] fp_div_b;	// denominator
wire fp_div_en;
assign fp_div_en = 1;
// instantiate fp_div module 
fp_div fp_div_mod(.a(fp_div_a), .areset(Reset), .b(fp_div_b), .clk(Clock), .q(fp_div_output), .en(fp_div_en));

integer k1, k2, k3;

// Combinational logic according to current state
always @(*)
begin
	for (k1=0; k1<IN_BUF_LEN; k1=k1+1)
	begin 
		InBuf_n[k1] = InBuf[k1];
	end
	
	for (k2=0; k2<OUT_BUF_LEN; k2=k2+1)
	begin 
		OutBuf_n[k2] = OutBuf[k2];
	end
	
	for (k3=0; k3<WGT_ARR_LEN; k3=k3+1)
	begin 
		ArrWeights_n[k3] = ArrWeights[k3];
	end 

//	fp_mac_en = 1'b1;	// enabled by default
//	fp_add_en = 1'b1;
//	fp_div_en = 1'b1;
	// set the second operand in the add for the activation function to 1.
	//fp_add_b = 32'd1;
	fp_mac_acc = 1;
	fp_mac_a = 32'd0;
	fp_mac_b = 32'd0;
	fp_div_a = 32'd0;
	fp_div_b = 32'd0;
	fp_add_a = 32'd0;

	// By default, maintain all current values:
	CurrVal_n = CurrVal;
	InBuf_Wr_n = InBuf_Wr;
	InBuf_Rd_n = InBuf_Rd;
	InBuf_RdStart_n = InBuf_RdStart;
	InBuf_Full_n = InBuf_Full;
	OutBuf_Wr_n = OutBuf_Wr;
	OutBuf_Rd_n = OutBuf_Rd;
	OutBuf_Full_n = OutBuf_Full;
	ArrWgt_Wr_n = ArrWgt_Wr;
	ArrWgt_Rd_n = ArrWgt_Rd;
	ArrWgt_Full_n = ArrWgt_Full;
	
	/*
	// Don't use shadow copies for the buffers?
	for (j1=0; j1<IN_BUF_LEN; j1=j1+1)
	begin 
		InBuf_n[j1] = InBuf[j1];
	end
	
	for (j2=0; j2<OUT_BUF_LEN; j2=j2+1)
	begin 
		OutBuf_n[j2] = OutBuf[j2];
	end
	
	for (j3=0; j3<WGT_ARR_LEN; j3=j3+1)
	begin 
		ArrWeights_n[j3] = ArrWeights[j3];
	end
	*/
	
	case (Ctrl)
	  PE_MA: // perform multiply-add calculation
	  begin	
	   fp_mac_acc = 1; // by default, turn on the accumulate flag
		
		fp_mac_a = 0;
		fp_mac_b = 0;
		
		// set up the inputs 
		fp_mac_a = Data;
		fp_mac_b = ArrWeights[ArrWgt_Rd];
		
		// store the provided data in the input buffer 
		if (InBuf_Full == FALSE)
		begin 
			InBuf_n[InBuf_Wr] = Data;
		end

		// Insert floating point operation here 
		//CurrVal_n = fp_mac_output;
		
		// Increment the Read pointer for the array of weights 
		// Note: Does not handle ArrWgt being empty.
		if (ArrWgt_Rd + 1 < WGT_ARR_LEN)
		begin 
			ArrWgt_Rd_n = ArrWgt_Rd + 12'd1;
		end 
		else 
		begin 
			ArrWgt_Rd_n = {ARR_WGT_IND_SIZE{1'b0}};
		end 
		
		// Increment the write pointer for the Input Buffer
		if (InBuf_Full == FALSE)
		begin				
			if (InBuf_Wr + 1 < IN_BUF_LEN)
			begin 
				InBuf_Wr_n = InBuf_Wr + 5'd1;
			end 
			else 
			begin 
				InBuf_Wr_n = {IN_BUF_IND_SIZE{1'b0}};
			end 
			
			// Check for condition where NEW write ptr value matches the read ptr --> set flag to indicate buffer is full 
			if (InBuf_Rd == InBuf_Wr_n)
				InBuf_Full_n = TRUE;
			
		end 
	  end
			
	  PE_MAB:
	  begin	
		fp_mac_acc = 1; 	// by default turn on accumulate flag
		fp_mac_a = 0;
		fp_mac_b = 0;
		
		// want to perform this operation:
		// CurrVal <= CurrVal + ( InBuf[InBuf_Rd] * ArrWeights[ArrWgt_Rd] );
		
		// set up the inputs 
		fp_mac_a = InBuf[InBuf_Rd];
		fp_mac_b = ArrWeights[ArrWgt_Rd];

		// Increment the Read ptr
		if (InBuf_Rd + 1 < IN_BUF_LEN)
		begin 
			InBuf_Rd_n = InBuf_Rd + 5'd1;
		end 
		else 
		begin 
			InBuf_Rd_n = {IN_BUF_IND_SIZE{1'b0}};
		end 
		
		// Increment the Read pointer for the array of weights 
		if (ArrWgt_Rd + 1 < WGT_ARR_LEN)
		begin 
			ArrWgt_Rd_n = ArrWgt_Rd + 12'd1;
		end 
		else 
		begin 
			ArrWgt_Rd_n = {ARR_WGT_IND_SIZE{1'b0}};
		end 
	  end
	  
	  PE_MABO:
	  begin	
		fp_mac_acc = 1; 	// by default turn on accumulate flag
		fp_mac_a = 0;
		fp_mac_b = 0;
		
		// want to perform this operation:
		// CurrVal <= CurrVal + ( OutBuf[OutBuf_Rd] * ArrWeights[ArrWgt_Rd] );
			
		// set up the inputs 
		fp_mac_a = OutBuf[OutBuf_Rd];
		fp_mac_b = ArrWeights[ArrWgt_Rd];
		
		// Copy the value from the output buffer into the input buffer 
		InBuf_n[InBuf_Wr] = OutBuf[OutBuf_Rd];
			
		// Increment the Read ptr for the input buffer?
		if (InBuf_Rd + 1 < IN_BUF_LEN)
		begin 
			InBuf_Rd_n = InBuf_Rd + 5'd1;
		end 
		else 
		begin 
			InBuf_Rd_n = {IN_BUF_IND_SIZE{1'b0}};
		end 
		
		// Increment the Read ptr for the output buffer 
		if (OutBuf_Rd + 1 < OUT_BUF_LEN)
		begin
			OutBuf_Rd_n = OutBuf_Rd + 5'd1;
		end 
		else 
		begin 
			OutBuf_Rd_n = {OUT_BUF_IND_SIZE{1'b0}};
		end 

		// Increment the Read pointer for the array of weights 
		if (ArrWgt_Rd + 1 < WGT_ARR_LEN)
		begin 
			ArrWgt_Rd_n = ArrWgt_Rd + 12'd1;
		end 
		else 
		begin 
			ArrWgt_Rd_n = {ARR_WGT_IND_SIZE{1'b0}};
		end 
		
		// If this PE supplied the data value to other PEs from its output buffer, increment the output RD pointer 		
		// Increment the read pointer for the buffer
		// Don't need to use empty flag.
		if (OutBuf_Rd + 1 < OUT_BUF_LEN)
		begin 
			OutBuf_Rd_n = OutBuf_Rd + 5'd1;
		end 
		else 
		begin 
			OutBuf_Rd_n = {OUT_BUF_IND_SIZE{1'b0}};
		end
	  end
			
	  PE_BIAS:
	  begin 
		fp_mac_acc = 1; 	// by default, accumulate flag must be on during bias
		fp_mac_a = 0;
		fp_mac_b = 0;
		
		if (counter == 0) 
		begin 
			// want to perform this operation:
			// CurrVal <= CurrVal + (Data * ArrWeights[ArrWgt_Rd]);
			
			// set up the inputs 
			fp_mac_a = 32'h3f800000;	// fp value of 1
			fp_mac_b = ArrWeights[ArrWgt_Rd];
			
			// leave the accumulate flag on for first cycle only during BIAS stage.
			//fp_mac_acc = 1;
		end 
		else if (counter == COUNT_MA)
		begin
			// calculation should be complete, pull the output value
		
			// Retrieve floating point operation here 
			//CurrVal_n = fp_mac_output;
			
			// turn off accumulate when output is acquired
			//fp_mac_acc = 0;
			
			// Increment the Read pointer for the array of weights 
			if (ArrWgt_Rd + 1 < WGT_ARR_LEN)
			begin 
				ArrWgt_Rd_n = ArrWgt_Rd + 12'd1;
			end 
			else 
			begin 
				ArrWgt_Rd_n = {ARR_WGT_IND_SIZE{1'b0}};
			end 
		end 
	  end 
			
	  PE_ACT:
	  begin 
		if (EnableAct == FALSE)
		begin // no activation function
			// do not perform any activation function operation;
			// just write current output to buffer 
			// clear CurrVal for next neuron calculation
			//CurrVal_n = 0;
			
			// Put the calculated value into the output buffer
			if (OutBuf_Full == FALSE)
			begin 
				OutBuf_n[OutBuf_Wr] = fp_mac_output;
				
				// Update write ptr
				if (OutBuf_Wr + 1 < OUT_BUF_LEN)
				begin 
					OutBuf_Wr_n = OutBuf_Wr + 5'd1;
				end 
				else 
				begin 
					OutBuf_Wr_n = {OUT_BUF_IND_SIZE{1'b0}};
				end 
				
				// set full flag if needed 
				if (OutBuf_Wr_n == OutBuf_Rd)
					OutBuf_Full_n = TRUE;
				
				// set non-empty
				OutBuf_Empty_n = FALSE;
			end 
			
			// Reset the Read ptr for the input buffer
			InBuf_Rd_n = InBuf_RdStart;
			
			// turn off the accumulate function
			fp_mac_acc = 0;
		end 
		else // fast sigmoid activation function
		begin 
			// turn off accumulate by default
			fp_mac_acc = 0;
			fp_div_a = CurrVal;
			
			if (counter == 0) 
			begin 
				// leave accumulate on for the first cycle, just in case 
				fp_mac_acc = 1;
				
				// want to perform this operation:
				// CurrVal <= CurrVal / (1 + {1'b0,CurrVal[DATA_SIZE-2:0]});
									
				// set up the inputs 
				//fp_add_a = {1'b0,CurrVal[DATA_SIZE-2:0]};
				//fp_div_a = CurrVal;
				
				fp_add_a = {1'b0,fp_mac_output[DATA_SIZE-2:0]};
				//fp_div_a = fp_mac_output;
				CurrVal_n = fp_mac_output;
			end
			else if (counter == (COUNT_FP_ADD + 1))
			begin 
				// Note: connect output of the add module to the divisor for the div module. (For the activation function calculation.)
				fp_div_b = fp_add_output;	// only do this when cycle counter == 3 (duration of ADD operation).
			end
			else if (counter == COUNT_ACT)
			begin
				// calculation should be complete, pull the output value of the activation function 
				// Fast Sigmoid Computation:
				// x / (1 + abs(x))
			
				// clear CurrVal for next neuron calculation
				CurrVal_n = 0;
				
				// Note: Tied with the activation function is placement of this value into the output buffer. 
				if (OutBuf_Full == FALSE)
				begin 
					OutBuf_n[OutBuf_Wr] = fp_div_output;
					
					// Update write ptr
					if (OutBuf_Wr + 1 < OUT_BUF_LEN)
					begin 
						OutBuf_Wr_n = OutBuf_Wr + 1;
					end 
					else 
					begin 
						OutBuf_Wr_n = {OUT_BUF_IND_SIZE{1'b0}};
					end 
					
					// set full flag if needed 
					if (OutBuf_Wr_n == OutBuf_Rd)
						OutBuf_Full_n = TRUE;
					
					// set non-empty
					OutBuf_Empty_n = FALSE;
				end 
				
				// Reset the Read ptr for the input buffer
				InBuf_Rd_n = InBuf_RdStart;
			end	
		end // end fast sigmoid activation			
	  end
			
	  PE_ACT_CLR:
	  begin
		if (EnableAct == FALSE)
		begin 	// no activation function operation
			// do not perform any activation function operation;
			// just write current output to buffer 
			// clear CurrVal for next neuron calculation
			CurrVal_n = 0;
			
			// Put the calculated value into the output buffer
			if (OutBuf_Full == FALSE)
			begin 
				OutBuf_n[OutBuf_Wr] = fp_mac_output;
				
				// Update write ptr
				if (OutBuf_Wr + 1 < OUT_BUF_LEN)
				begin 
					OutBuf_Wr_n = OutBuf_Wr + 5'd1;
				end 
				else 
				begin 
					OutBuf_Wr_n = {OUT_BUF_IND_SIZE{1'b0}};
				end 
				
				// set full flag if needed 
				if (OutBuf_Wr_n == OutBuf_Rd)
					OutBuf_Full_n = TRUE;
				
				// set non-empty
				OutBuf_Empty_n = FALSE;
			end 
			
			// Reset the Read ptr for the input buffer
			InBuf_Rd_n = InBuf_RdStart;
			
			// turn off the accumulate function
			fp_mac_acc = 0;
		end 
		else	// fast sigmoid activation function 
		begin
			// turn off accumulate 
			fp_mac_acc = 0;
			fp_div_a = CurrVal;
			
			if (counter == 0) 
			begin 
				// leave accumulate on for the first cycle, just in case 
				fp_mac_acc = 1;
				
				// want to perform this operation:
				// CurrVal <= CurrVal / (1 + {1'b0,CurrVal[DATA_SIZE-2:0]});
				
				// set up the inputs 
				fp_add_a = {1'b0,fp_mac_output[DATA_SIZE-2:0]};
				//fp_div_a = fp_mac_output;		
				CurrVal_n = fp_mac_output;	
			end 
			else if (counter == (COUNT_FP_ADD+1))
			begin 
				// Note: connect output of the add module to the divisor for the div module. (For the activation function calculation.)
				fp_div_b = fp_add_output;	// only do this when cycle counter == 3 (duration of ADD operation).
			end
			else if (counter == COUNT_ACT)
			begin 
				// computation is complete; obtain the value 
				CurrVal_n = 0;
				
				// NOTE: This is a duplicate of the operations 
				// 		 performed in the PE_ACT section, EXCEPT with 
				// 		 the additional function of clearing all the 
				// 		 values in the input buffer.
					
				// Effectively "clear" the input buffer by moving 
				// the read pointer to the write pointer position.
				InBuf_Rd_n = InBuf_Wr;
				// Save this read ptr position
				InBuf_RdStart_n = InBuf_Wr;
				// Specify that the buffer is NOT full 
				InBuf_Full_n = FALSE;
				
				// Store result in output buffer
				if (OutBuf_Full == FALSE)
				begin 
					OutBuf_n[OutBuf_Wr] = fp_div_output;
					
					// Update write ptr
					if (OutBuf_Wr + 1 < OUT_BUF_LEN)
					begin 
						OutBuf_Wr_n = OutBuf_Wr + 1;
					end 
					else 
					begin 
						OutBuf_Wr_n = {OUT_BUF_IND_SIZE{1'b0}};
					end 
					
					// set full flag if needed 
					if (OutBuf_Wr_n == OutBuf_Rd)
						OutBuf_Full_n = TRUE;
					
					// set non-empty
					OutBuf_Empty_n = FALSE;
				end 
							
			end 
		end // end fast sigmoid activation section
	  end 
			
	  PE_LOAD:	// load weight value to array
	  begin 
		// if array is full, do not update the array 
		if (ArrWgt_Full == FALSE)
		begin 
			// Store the value provided on input data line to the array.
			ArrWeights_n[ArrWgt_Wr] = Data;
			
			// Also, update the write pointer for the array.
			if (ArrWgt_Wr + 1 < WGT_ARR_LEN)
			begin 
				ArrWgt_Wr_n = ArrWgt_Wr + 12'd1;
			end 
			else 
			begin 
				ArrWgt_Wr_n = {ARR_WGT_IND_SIZE{1'b0}};
			end 
			
			if (ArrWgt_Rd == ArrWgt_Wr_n)
				ArrWgt_Full_n = TRUE;
		end
	  end 
	
	  
	  PE_IDLE:
	  begin 
		// Update output buffer read pointer on last cycle of calculation
		//if ((OutputCtrl == TRUE) && (counter == COUNT_MA))
		if (OutputCtrl == TRUE)
		begin 
		
			// Only increment the pointer if the buffer is not empty 
			if (OutBuf_Empty == FALSE)
			begin
				// in the last cycle when the output ctrl is set to TRUE, 
				// the output buffer rd pointer needs to be updated.
				if ((OutBuf_Rd + 1) < OUT_BUF_LEN)
				begin 
					OutBuf_Rd_n = OutBuf_Rd + 5'd1;
				end 
				else 
				begin 
					OutBuf_Rd_n = {OUT_BUF_IND_SIZE{1'b0}};
				end 
				
				if (OutBuf_Rd_n == OutBuf_Wr)
				begin
					OutBuf_Empty_n = TRUE;
				end
			end
		end 
	  
	  end 
	  
	  /*
	  default: 
	  begin 
		// do nothing 
	  end 
	  */
	endcase // end case statement on ctrl signal
end


// Counters 
always @(*)
begin
	//if ((((Ctrl == PE_MA) || (Ctrl == PE_MAB) || (Ctrl == PE_MABO) || (Ctrl == PE_BIAS) || (Ctrl == PE_IDLE)) && (counter < COUNT_MA)) || (((Ctrl == PE_ACT) || (Ctrl == PE_ACT_CLR)) && counter < COUNT_ACT))
	if (((Ctrl == PE_BIAS) && (counter < COUNT_MA)) || (((Ctrl == PE_ACT) || (Ctrl == PE_ACT_CLR)) && counter < COUNT_ACT))
	begin 
		// increment counter if the maximum has not yet been reached in either the MA or the ACT stage, depending on the current stage.
		counter_n = counter + 5'd1;
	end
	else
	begin
		counter_n = 5'd0;	// default value for counter is 0
	end
end

// State Transitions and Output Buffer Changes
always @(posedge(Clock), posedge(Reset))
begin
	// Check first for reset operation
	if (Reset == 1'b1)	// Reset is active high 
	begin
		// Clear current internal accumulated value 
		CurrVal <= {DATA_SIZE{1'b0}};
		//CurrVal_n <= {DATA_SIZE{1'b0}};
		
		// Reset counter 
		counter <= 0;
		//counter_n <= 0;
				
		// NOTE: These FIFO arrays may not need to be cleared to 0.
		//  	 Updating the pointers may be sufficient.
		// Clear all arrays/buffers 
		
		for (i1=0; i1<IN_BUF_LEN; i1=i1+1)
		begin 
			InBuf[i1] <= {DATA_SIZE{1'b0}};
		end
		
		for (i2=0; i2<OUT_BUF_LEN; i2=i2+1)
		begin 
			OutBuf[i2] <= {DATA_SIZE{1'b0}};
		end
		
		for (i3=0; i3<WGT_ARR_LEN; i3=i3+1)
		begin 
			ArrWeights[i3] <= {DATA_SIZE{1'b0}};
		end 
		
		
		// Clear all index pointers 
		InBuf_Rd <= {IN_BUF_IND_SIZE{1'b0}};
		InBuf_Wr <= {IN_BUF_IND_SIZE{1'b0}};
		InBuf_RdStart <= {IN_BUF_IND_SIZE{1'b0}};
		InBuf_Full <= FALSE;
		OutBuf_Rd <= {OUT_BUF_IND_SIZE{1'b0}};
		OutBuf_Wr <= {OUT_BUF_IND_SIZE{1'b0}};
		OutBuf_Full <= FALSE;
		OutBuf_Empty <= TRUE;
		ArrWgt_Rd <= {ARR_WGT_IND_SIZE{1'b0}};
		ArrWgt_Wr <= {ARR_WGT_IND_SIZE{1'b0}};
		ArrWgt_Full <= FALSE;
		
		/*
		//InBuf_Rd_n <= {IN_BUF_IND_SIZE{1'b0}};
		//InBuf_Wr_n <= {IN_BUF_IND_SIZE{1'b0}};
		//InBuf_RdStart_n <= {IN_BUF_IND_SIZE{1'b0}};
		//InBuf_Full_n <= FALSE;
		OutBuf_Rd_n <= {OUT_BUF_IND_SIZE{1'b0}};
		OutBuf_Wr_n <= {OUT_BUF_IND_SIZE{1'b0}};
		OutBuf_Full_n <= FALSE;
		ArrWgt_Rd_n <= {ARR_WGT_IND_SIZE{1'b0}};
		ArrWgt_Wr_n <= {ARR_WGT_IND_SIZE{1'b0}};
		ArrWgt_Full_n <= FALSE;
		*/
		//fp_mac_acc <= 0;	// clear accumulator
	end // end Reset
	else 
	begin 
		// update current values to the next values 
		CurrVal <= CurrVal_n;
		counter <= counter_n;
		InBuf_Wr <= InBuf_Wr_n;
		InBuf_Rd <= InBuf_Rd_n;
		InBuf_RdStart <= InBuf_RdStart_n;
		InBuf_Full <= InBuf_Full_n;
		OutBuf_Wr <= OutBuf_Wr_n;
		OutBuf_Rd <= OutBuf_Rd_n;
		OutBuf_Full <= OutBuf_Full_n;
		OutBuf_Empty <= OutBuf_Empty_n;
		ArrWgt_Wr <= ArrWgt_Wr_n;
		ArrWgt_Rd <= ArrWgt_Rd_n;
		ArrWgt_Full <= ArrWgt_Full_n;
		
		
		for (i1=0; i1<IN_BUF_LEN; i1=i1+1)
		begin 
			InBuf[i1] <= InBuf_n[i1];
		end
		
		for (i2=0; i2<OUT_BUF_LEN; i2=i2+1)
		begin 
			OutBuf[i2] <= OutBuf_n[i2];
		end
		
		for (i3=0; i3<WGT_ARR_LEN; i3=i3+1)
		begin 
			ArrWeights[i3] <= ArrWeights_n[i3];
		end 
		
		/*
		// Supply output value if needed.
		if (((Ctrl == PE_MABO) || (Ctrl == PE_IDLE)) && (OutputCtrl == TRUE))
		begin 
			//Data = OutBuf[OutBuf_Rd];
			output_val = OutBuf[OutBuf_Rd];
		end
		*/		
	end // end else for normal operation 
end

endmodule