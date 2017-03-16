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
// Last Modified: 2017-Mar-14
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

/*
// test for fp_mac connections 
module fp_mac (
        input  wire [31:0] a,      //      a.a
        input  wire        acc,    //    acc.acc
        input  wire        areset, // areset.reset
        input  wire [31:0] b,      //      b.b
        input  wire        clk,    //    clk.clk
        output wire [31:0] q       //      q.q
    );
endmodule

module fp_div (
        input  wire [31:0] a,      //      a.a
        input  wire        areset, // areset.reset
        input  wire [31:0] b,      //      b.b
        input  wire        clk,    //    clk.clk
        output wire [31:0] q       //      q.q
    );
endmodule
*/

//module PE(Clock, Reset, Ctrl, OutputCtrl, DataIn, DataOut);
module PE(Clock, Reset, Ctrl, OutputCtrl, Data);

//--------------------------------
// GENERAL PARAMETER VALUES:
//--------------------------------
// States for operation control in the PE:
parameter PE_MA = 0;		// standard multiply-add operation (take input from data in)
parameter PE_MAB = 1;		// multiply-add op using buffered value
parameter PE_BIAS = 2;
parameter PE_ACT = 3;		// activation function
parameter PE_ACT_CLR = 4;	// activation function and clear input buffer
parameter PE_LOAD = 5;		// load value into weight array
parameter PE_IDLE = 6;		// no operation

parameter CTRL_SIZE = 3; 	// number of bits for the ctrl signal

parameter DATA_SIZE = 32;	// number of bits in the data input 

// Counter values for FP operations 
parameter COUNT_MA = 4; 	// FP multiply add operation
parameter COUNT_ACT = 8;	// FP activation function operation 
parameter COUNTER_SIZE = 4;	// number of bits in the counter value

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
parameter TRUE = 1;
parameter FALSE = 0;
//--------------------------------

input Clock;
input Reset;
input [CTRL_SIZE-1:0] Ctrl;		// control signal (for PE operation)
input OutputCtrl;				// provides output from buffer if set to 1
inout [DATA_SIZE-1:0] Data; 	// data

reg [DATA_SIZE-1:0] CurrVal; // the current accumulated value used in the multiply-add calculation

reg [DATA_SIZE-1:0] InBuf [0:IN_BUF_LEN-1]; 	// input buffer
reg [DATA_SIZE-1:0] OutBuf [0:OUT_BUF_LEN-1]; 	// output buffer

// Counter used for the calculation stages.
reg [COUNTER_SIZE-1:0] counter;

// Pointers for read/write within each FIFO buffer, and flags for empty/full conditions 
reg [IN_BUF_IND_SIZE-1:0] InBuf_Wr = {IN_BUF_IND_SIZE{1'b0}}; 	// write index 
reg [IN_BUF_IND_SIZE-1:0] InBuf_Rd = {IN_BUF_IND_SIZE{1'b0}}; 	// read index
reg InBuf_Full = FALSE; // flag to indicate buffer is full 
reg InBuf_Empty = TRUE; // flag to indicate buffer is empty.

reg [OUT_BUF_IND_SIZE-1:0] OutBuf_Wr = {OUT_BUF_IND_SIZE{1'b0}}; 	// write index 
reg [OUT_BUF_IND_SIZE-1:0] OutBuf_Rd = {OUT_BUF_IND_SIZE{1'b0}}; 	// read index
reg OutBuf_Full = FALSE; // flag to indicate buffer is full 
reg OutBuf_Empty = TRUE; // flag to indicate buffer is empty.

reg [DATA_SIZE-1:0] ArrWeights [0:WGT_ARR_LEN-1];	// array of weights to be used in multiply-add operations

// Pointers for read/write in the array of weights; and flags for full/empty (?).
reg [ARR_WGT_IND_SIZE-1:0] ArrWgt_Wr = {ARR_WGT_IND_SIZE{1'b0}}; 	// write index 
reg [ARR_WGT_IND_SIZE-1:0] ArrWgt_Rd = {ARR_WGT_IND_SIZE{1'b0}}; 	// read index
reg ArrWgt_Full = FALSE; // flag to indicate buffer is full 
reg ArrWgt_Empty = TRUE; // flag to indicate buffer is empty.

integer i1, i2, i3;

// value connections for the fp_mac module
wire [DATA_SIZE-1:0] fp_mac_output;
reg [DATA_SIZE-1:0] fp_mac_a;
reg [DATA_SIZE-1:0] fp_mac_b;
reg fp_mac_acc;
// instantiate fp_mac module
fp_mac(.a(fp_mac_a), .acc(fp_mac_acc), .areset(Reset), .b(fp_mac_b), .clk(Clock), .q(fp_mac_output));

// Link the fp_add module and the fp_div module to perform 
// the fast sigmoid activation function.
// Connections for fp_add module: 
wire [DATA_SIZE-1:0] fp_add_output;
reg [DATA_SIZE-1:0] fp_add_a;
reg [DATA_SIZE-1:0] fp_add_b = 1;
// instantiate fp_add module 
fp_add(.a(fp_add_a), .areset(Reset), .b(fp_add_b), .clk(Clock), .q(fp_add_output));

// connections for fp_div module 
wire [DATA_SIZE-1:0] fp_div_output;
reg [DATA_SIZE-1:0] fp_div_a;
reg [DATA_SIZE-1:0] fp_div_b;
// instantiate fp_div module 
fp_add(.a(fp_div_a), .areset(Reset), .b(fp_div_b), .clk(Clock), .q(fp_div_output));


// Check first for reset operation
always @(posedge(Clock), negedge(Reset))
begin
	if (Reset == 1'b0)	// Reset is active low 
	begin 
		// Clear current internal accumulated value 
		CurrVal <= {DATA_SIZE{1'b0}};
		
		// Reset counter 
		counter <= 0;
		
		// NOTE: These FIFO arrays may not need to be cleared to 0.
		//  	 The pointers may be sufficient.
		// Clear all arrays/buffers 
		///*
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
		//*/
		
		// Clear all index pointers 
		InBuf_Rd <= {IN_BUF_IND_SIZE{1'b0}};
		InBuf_Wr <= {IN_BUF_IND_SIZE{1'b0}};
		InBuf_Empty <= TRUE;
		InBuf_Full <= FALSE;
		OutBuf_Rd <= {OUT_BUF_IND_SIZE{1'b0}};
		OutBuf_Wr <= {OUT_BUF_IND_SIZE{1'b0}};
		OutBuf_Empty <= TRUE;
		OutBuf_Full <= FALSE;
		ArrWgt_Rd <= {ARR_WGT_IND_SIZE{1'b0}};
		ArrWgt_Wr <= {ARR_WGT_IND_SIZE{1'b0}};
		ArrWgt_Empty <= TRUE;
		ArrWgt_Full <= FALSE;
		
		// make sure that the second operand in the add for the activation function is still set to 1.
		fp_add_b = 1;
	end 
 
	else 
	begin 
		// Normal Operation 
		// based on the Ctrl signal 
		case(Ctrl)
			PE_MA: // perform multiply-add calculation
			  begin
			  
				if (counter == 0) 
				begin
					// Perform the operation:
					//CurrVal + (Data * ArrWeights[ArrWgt_Rd]);
				
					// set up the inputs 
					fp_mac_a <= Data;
					fp_mac_b <= ArrWeights[ArrWgt_Rd];
					fp_mac_acc <= 1;	// perform accumulate
					
					// increment counter 
					counter <= counter + 1;
				end 
				else if (counter < COUNT_MA) 
				begin
					// increment the counter 
					counter <= counter + 1;
				end 
				else
				begin
					// Reset the counter value 
					counter <= 0;
					
					// Insert floating point operation here 
					CurrVal <= fp_mac_output;
					
					// Increment the Read pointer for the array of weights 
					if (ArrWgt_Rd + 1 < WGT_ARR_LEN)
					begin 
						ArrWgt_Rd <= ArrWgt_Rd + 1;
					end 
					else 
					begin 
						ArrWgt_Rd <= {ARR_WGT_IND_SIZE{1'b0}};
					end 
					
					// TODO: Check for ArrWgt empty?
					
					// Also, load the current data input to the input buffer for future reference. 
					// Note: This should only be done if the buffer is not full.
					if (InBuf_Full == FALSE)
					begin
						// load value into buffer
						InBuf[InBuf_Wr] <= Data;
						
						// increment write pointer
						if (InBuf_Wr + 1 < IN_BUF_LEN)
						begin 
							InBuf_Wr <= InBuf_Wr + 1;
						end 
						else 
						begin 
							InBuf_Wr <= {IN_BUF_IND_SIZE{1'b0}};
						end 
						
						// Check for condition where NEW write ptr value matches the read ptr --> set flag to indicate buffer is full 
						if (InBuf_Rd == (InBuf_Wr + 1))
						begin 
							InBuf_Full <= TRUE;
						end 
						else 
						begin 
							InBuf_Full <= InBuf_Full;
						end 
					end 
					else 
					begin 
						// otherwise, do not load the value
						InBuf[InBuf_Wr] <= InBuf[InBuf_Wr];
						
						// do not increment write pointer 
						InBuf_Wr <= InBuf_Wr;
						
						// no change to the full flag 
						InBuf_Full <= InBuf_Full;
					end 
				end // end else 
			  end
			
			PE_MAB:
			  begin
				
				if (counter == 0)
				begin 
					// want to perform this operation:
					// CurrVal <= CurrVal + ( InBuf[InBuf_Rd] * ArrWeights[ArrWgt_Rd] );
					
					// set up the inputs 
					fp_mac_a <= InBuf[InBuf_Rd];
					fp_mac_b <= ArrWeights[ArrWgt_Rd];
					fp_mac_acc <= 1;	// perform accumulate
					
					// increment counter 
					counter <= counter + 1;
				end
				else if (counter < COUNT_MA) 
				begin
					// increment the counter 
					counter <= counter + 1;
				end 
				else
				begin
					// calculation should be complete, pull the output value
					counter <= 0;
					
					// Insert floating point operation here 
					CurrVal <= fp_mac_output;

					// NOTE: It is possible that the buffer could be empty: what should be the correct behavior in this case? (Currently the PE module has no error flag signal to indicate something went wrong.)
					// For now, just perform the operation on the value indicated by the Read ptr.
					
					// if the buffer is not empty, increment the Read ptr
					if (InBuf_Empty == FALSE)
					begin 
						if (InBuf_Rd + 1 < IN_BUF_LEN)
						begin 
							InBuf_Rd <= InBuf_Rd + 1;
							// set empty flag if needed 
							if (InBuf_Wr == (InBuf_Rd + 1))
								InBuf_Empty <= TRUE;
							else 
								InBuf_Empty <= FALSE;
						end 
						else 
						begin 
							InBuf_Rd <= {IN_BUF_IND_SIZE{1'b0}};
							// set empty flag if needed 
							if (InBuf_Wr == {IN_BUF_IND_SIZE{1'b0}})
								InBuf_Empty <= TRUE;
							else 
								InBuf_Empty <= FALSE;
						end 
					end 
					else 
					begin 
						// otherwise, do not update the read ptr 
						InBuf_Rd <= InBuf_Rd;
					end 

					// Increment the Read pointer for the array of weights 
					if (ArrWgt_Rd + 1 < WGT_ARR_LEN)
					begin 
						ArrWgt_Rd <= ArrWgt_Rd + 1;
					end 
					else 
					begin 
						ArrWgt_Rd <= {ARR_WGT_IND_SIZE{1'b0}};
					end 
					
					// TODO: Check for ArrWgt empty?  This case can be ignored.
					
					// Also, load the current data input to the input buffer for future reference. 
					// Note: This should only be done if the buffer is not full.
					if (InBuf_Full == FALSE)
					begin
						// load value into buffer
						InBuf[InBuf_Wr] <= Data;
						
						// increment write pointer
						if (InBuf_Wr + 1 < IN_BUF_LEN)
						begin 
							InBuf_Wr <= InBuf_Wr + 1;
							// set flag for full buffer if needed 
							// - Does the RD pointer match the new WR ptr val?
							if (InBuf_Rd == (InBuf_Wr + 1))
								InBuf_Full <= TRUE;
							else 
								InBuf_Full <= FALSE;
						end 
						else 
						begin 
							InBuf_Wr <= {IN_BUF_IND_SIZE{1'b0}};
							
							// set flag for full buffer if needed 
							if (InBuf_Rd == {IN_BUF_IND_SIZE{1'b0}})
								InBuf_Full <= TRUE;
							else 
								InBuf_Full <= FALSE;
						end
						
					end 
					else 
					begin 
						// otherwise, do not load the value
						InBuf[InBuf_Wr] <= InBuf[InBuf_Wr];
						
						// do not increment write pointer 
						InBuf_Wr <= InBuf_Wr;
						
						// no change to full flag 
						InBuf_Full <= InBuf_Full;
					end 
				end 
			  end
			
			PE_BIAS:
			  begin 
				if (counter == 0) 
				begin 
					// want to perform this operation:
					// CurrVal <= CurrVal + (Data * ArrWeights[ArrWgt_Rd]);
					
					// set up the inputs 
					fp_mac_a <= Data;
					fp_mac_b <= ArrWeights[ArrWgt_Rd];
					fp_mac_acc <= 1;	// perform accumulate
					
					// increment counter 
					counter <= counter + 1;
				end 
				else if (counter < COUNT_MA)
				begin 
					// increment the counter 
					counter <= counter + 1;
				end 
				else
				begin
					// calculation should be complete, pull the output value
					counter <= 0;
				
					// Retrieve floating point operation here 
					CurrVal <= fp_mac_output;
					
					// Increment the Read pointer for the array of weights 
					if (ArrWgt_Rd + 1 < WGT_ARR_LEN)
					begin 
						ArrWgt_Rd <= ArrWgt_Rd + 1;
					end 
					else 
					begin 
						ArrWgt_Rd <= {ARR_WGT_IND_SIZE{1'b0}};
					end 
				end 
			  end 
			
			PE_ACT:
			  begin 
				if (counter == 0) 
				begin 
					// want to perform this operation:
					// CurrVal <= CurrVal / (1 + {1'b0,CurrVal[DATA_SIZE-2:0]});
										
					// set up the inputs 
					fp_add_a <= {1'b0,CurrVal[DATA_SIZE-2:0]};
					fp_div_a <= CurrVal;
					
					// Clear the value used in the accumulate operation
					fp_mac_acc <= 0;
					
					// increment counter 
					counter <= counter + 1;
				end 
				else if (counter < COUNT_ACT)
				begin 
					// increment the counter 
					counter <= counter + 1;
				end 
				else
				begin
					// calculation should be complete, pull the output value
					counter <= 0;

					// perform the activation function 
					// Fast Sigmoid Computation:
					// x / (1 + abs(x))
				
					// set output here
					CurrVal <= fp_div_output;
								
					// Note: Tied with the activation function is placement of this value into the output buffer. 
					// TODO: What should be the correct behavior if the buffer is full?  Do we overwrite? Or drop the value?
					if (OutBuf_Full == TRUE)
					begin 
						// if full, do not change value
						OutBuf[OutBuf_Wr] <= OutBuf[OutBuf_Wr];
					end 
					else if ((OutBuf_Empty == TRUE) && (OutputCtrl == TRUE))
					begin 
						// if the buffer if empty AND a value needs to be 
						// placed on output for next cycle, do NOT store in 
						// buffer 
						OutBuf[OutBuf_Wr] <= OutBuf[OutBuf_Wr];
					end 
					else 
					begin 
						// TODO: Duplicate the calculation above here too, since non-blocking, have to specify explicitly.
						OutBuf[OutBuf_Wr] <= CurrVal / (1 + {1'b0,CurrVal[DATA_SIZE-2:0]});
					end
				end				
			  end
			
			PE_ACT_CLR:
			  begin 
				if (counter == 0) 
				begin 
					// want to perform this operation:
					// CurrVal <= CurrVal / (1 + {1'b0,CurrVal[DATA_SIZE-2:0]});
					
					// set up the inputs 
					fp_add_a <= {1'b0,CurrVal[DATA_SIZE-2:0]};
					fp_div_a <= CurrVal;
					
					// Clear the value used in the accumulate operation
					fp_mac_acc <= 0;
					
					// increment counter 
					counter <= counter + 1;
				end 
				else if (counter < COUNT_ACT)
				begin 
					// increment the counter 
					counter <= counter + 1;
				end 
				else
				begin 
					// computation is complete, so reset the counter 
					counter <= 0;
					
					// obtain the value 
					//CurrVal <= fp_div_output;
					
					// NOTE: This is a duplicate of the operations 
					// 		 performed in the PE_ACT section, EXCEPT with 
					// 		 the additional function of clearing all the 
					// 		 values in the input buffer.
						
					// Effectively "clear" the input buffer by moving 
					// the read pointer to the write pointer position.
					InBuf_Rd <= InBuf_Wr;
					// Specify that the buffer is NOT full 
					InBuf_Full <= FALSE;
										
					// Check if the output flag has been set.  If so, 
					// overwrite the first entry of the input buffer 
					// with the next value to be pushed from the output
					// buffer.
					
					if (OutputCtrl == TRUE) 
					begin 
						// put result of computation in the Input Buffer as well
						InBuf[InBuf_Wr] <= CurrVal / (1 + {1'b0,CurrVal[DATA_SIZE-2:0]});
						// update the Wr pointer 
						//InBuf_Wr <= InBuf_Wr + 1;
						if (InBuf_Full == FALSE) 
						begin 
							// increment WR pointer 
							if (InBuf_Wr + 1 < IN_BUF_LEN)
							begin 
								InBuf_Wr <= InBuf_Wr + 1;
								InBuf_Full <= FALSE;
								// Note: don't set full flag here, because this is ACT_CLR, and it has just been cleared, so will not be full.
							end 
							else 
							begin 
								InBuf_Wr <= {IN_BUF_IND_SIZE{1'b0}};
								InBuf_Full <= FALSE;
							end
						end 
						else
						begin 
							// do not make changes to the input buffer if it is full
						end
						
						// Specify buffer as non-empty
						InBuf_Empty <= FALSE;
					end 
					else 
					begin 
						InBuf_Empty <= TRUE;
					end 
				end 
				
			  end 
			
			PE_LOAD:	// load weight value to array
			  begin 
				if (ArrWgt_Full == TRUE)
				begin 
					// if array is full, do not update the array 
					ArrWeights[ArrWgt_Wr] <= ArrWeights[ArrWgt_Wr];
				end 
				else
				begin 
					// Store the value provided on input data line to the array.
					ArrWeights[ArrWgt_Wr] <= Data;
					
					// Also, update the write pointer for the array.
					if (ArrWgt_Wr + 1 < WGT_ARR_LEN)
					begin 
						ArrWgt_Wr <= ArrWgt_Wr + 1;
						if (ArrWgt_Rd == (ArrWgt_Wr + 1))
							ArrWgt_Full <= TRUE;
						else 
							ArrWgt_Full <= FALSE;
					end 
					else 
					begin 
						ArrWgt_Wr <= {ARR_WGT_IND_SIZE{1'b0}};
						if (ArrWgt_Rd == {ARR_WGT_IND_SIZE{1'b0}})
							ArrWgt_Full <= TRUE;
						else 
							ArrWgt_Full <= FALSE;
					end 
				end
			  end 
			
			//PE_IDLE:
			default: // Note, this includes PE_IDLE case
			  begin 
			    // do nothing 
			  end 
		endcase // end case statement on ctrl signal

		// In addition to the ctrl signal, need to handle output
		// of a value if the OutputCtrl flag has been set. 
		// Note: Even if the OutputCtrl flag is set, need to 
		// 		 skip this section IF one of the following 
		// 		 conditions are true:
		// 			1. Ctrl == IDLE 
		// 			2. OutBuf is Empty 
		// 			3. (Ctrl == ACT || Ctrl == ACT_CLR) && (counter < COUNT_ACT)
		// 			4. (Ctrl in {MA, MAB, BIAS}) && (counter < COUNT_MA)
		
		// Simplified:
		// Only output if the flag is set and the output buffer is not empty, and the ctrl signal is not idle
		// Updated to include check for counters in ACT and MA operations; output should only be applied on the last cycle of the operation.
		//if ((OutputCtrl == TRUE) && (OutBuf_Empty == FALSE) && (Ctrl != PE_IDLE) && ((((Ctrl == PE_ACT) || (Ctrl == PE_ACT_CLR)) && (counter >= COUNT_ACT)) || (((Ctrl == PE_MA) || (Ctrl == PE_MAB) || (Ctrl == PE_BIAS)) && (counter >= COUNT_MA))))
		
		// Updated: 3/15/2017
		// Changed this condition to allow output to be sent when idle.
		if ((OutputCtrl == TRUE) && (OutBuf_Empty == FALSE) && ((((Ctrl == PE_ACT) || (Ctrl == PE_ACT_CLR)) && (counter >= COUNT_ACT)) || (((Ctrl == PE_MA) || (Ctrl == PE_MAB) || (Ctrl == PE_BIAS)) && (counter >= COUNT_MA))))
		begin 
			// If the OutputCtrl flag is set and the special 
			// exceptions are not met, then need to push a value 
			// from the output buffer to the output. 
			
			// Increment the read pointer for the buffer
			// and set empty flag if needed.
			if (OutBuf_Rd + 1 < OUT_BUF_LEN)
			begin 
				OutBuf_Rd <= OutBuf_Rd + 1;
				
				if (OutBuf_Wr == OutBuf_Rd + 1)
					OutBuf_Empty <= TRUE;
				else 
					OutBuf_Empty <= FALSE;
			end 
			else 
			begin 
				OutBuf_Rd <= {OUT_BUF_IND_SIZE{1'b0}};
				
				if (OutBuf_Wr == {OUT_BUF_IND_SIZE{1'b0}})
					OutBuf_Empty <= TRUE;
				else 
					OutBuf_Empty <= FALSE;
			end 
			
		end 
		else 
		begin 
			// otherwise, make no changes to the output buffer.
		end 
		
		// Note: connect output of the add module to the divisor for the div module. (For the activation function calculation.)
		fp_div_b <= fp_add_output;
	end 
end 

endmodule