module npu(
	rst, clk,
	we, oe,
	data,
	ready,
	//debug
	
	pe_state_r0,
	pe_state_r1,
	pe_oe_0,
	pe_oe_1,
	state_r,
	num_layers_r,
	num_neurons_r3,
	num_multadds,
	state_count_r,
	multadd_count_r,
	fp_mac_a_0, fp_mac_b_0, fp_mac_output_0,
	fp_mac_a_1, fp_mac_b_1, fp_mac_output_1
	/*
	counter, fp_mac_acc, fp_mac_a, fp_mac_b, fp_mac_output, ArrWgt_Rd, ArrWgt_Wr, InBuf_Rd, InBuf_Wr, OutBuf_Rd, OutBuf_Wr,  OutBuf_i0, OutBuf_n_i0, OutBuf_Full, InBuf_i0, ArrWeights_i0, ArrWeights_i1
	*/
	);
	
//debug
parameter IN_BUF_IND_SIZE = 5; // log2(32)
parameter OUT_BUF_IND_SIZE = 5; // log2(32)
parameter ARR_WGT_IND_SIZE = 12;

output [2:0] pe_state_r0;
output [2:0] pe_state_r1;
output pe_oe_0;
output pe_oe_1;

output [3:0] state_r;
output [1:0] num_layers_r;
output [4:0] num_neurons_r3;
output [5:0] num_multadds;
output [4:0] state_count_r;
output [5:0] multadd_count_r;

output [31:0] fp_mac_output_0;
output [31:0] fp_mac_a_0;
output [31:0] fp_mac_b_0;
output [31:0] fp_mac_output_1;
output [31:0] fp_mac_a_1;
output [31:0] fp_mac_b_1;
/*
output [4:0]  counter;
output fp_mac_acc;
output [IN_BUF_IND_SIZE-1:0] InBuf_Rd;
output [IN_BUF_IND_SIZE-1:0] InBuf_Wr;
output [OUT_BUF_IND_SIZE-1:0] OutBuf_Rd;
output [OUT_BUF_IND_SIZE-1:0] OutBuf_Wr;
output [ARR_WGT_IND_SIZE-1:0] ArrWgt_Rd; // = {ARR_WGT_IND_SIZE{1'b0}}; 	// read index
output [ARR_WGT_IND_SIZE-1:0] ArrWgt_Wr; // = {ARR_WGT_IND_SIZE{1'b0}}; 	// write index 
output [31:0] InBuf_i0;
output [31:0] ArrWeights_i0;
output [31:0] ArrWeights_i1;
output [31:0] OutBuf_i0;
output [31:0] OutBuf_n_i0;
output OutBuf_Full;
*/

input         rst, clk, we, oe;
inout signed  [31:0] data;
output        ready;


// ===== PE i/o =====
	parameter PE_IDLE    = 3'd0;
	parameter PE_LOAD    = 3'd1;  //load weights and biases
	parameter PE_MA      = 3'd2;  //multiply & add
	parameter PE_MAB     = 3'd3;  //multiply & add (data from buffer)
	parameter PE_MABO    = 3'd4;
	parameter PE_BIAS    = 3'd5;  //adding biases (multiply 1 & add)
	parameter PE_ACT     = 3'd6;  //activation function
	parameter PE_ACT_CLR = 3'd7;  //activation function with clearing buffer


// PE i/o
reg [2:0]   pe_state_r[0:7],
			pe_state_w[0:7];
wire        pe_oe[0:7];

// ===== state & counter =====
parameter IDLE     = 4'd0;
parameter CONFIG   = 4'd1;  // load configuration
parameter LOAD_W1  = 4'd2;  // load weights (& biases) for hidden layer one
parameter LOAD_W2  = 4'd3;  // load weights (& biases) for hidden layer two
parameter LOAD_WO  = 4'd4;  // load weights (& biases) for output layer
parameter LAYER_H1 = 4'd5;  // hidden layer 1
parameter LAYER_H2 = 4'd6;  // hidden layer 2
parameter LAYER_O  = 4'd7;  // output layer
parameter SEND_O   = 4'd8;  // send output

reg [3:0] state_r, state_w;
reg [4:0] state_count_r, state_count_w;

reg [5:0] multadd_count_r, multadd_count_w;  // counts how many mult_add to do and how many weights(bias) to load

//TODO: don't know the actual cycles yet
parameter CYCLES_MA  = 5'd3; // used in MA, MAB, BIAS. mac: 4 cycles
parameter CYCLES_ACT = 5'd20; // used in ACT, ACT_CLR. add: 3 cycles; divide: 19 cycles
reg [4:0] pe_count_r, pe_count_w; // counts the cycles to do FP MULT_ACC, DIV, ACT

// ===== internal registers & wires =====
parameter ZERO_HIDDEN = 2'd0;
parameter ONE_HIDDEN  = 2'd1;
parameter TWO_HIDDEN  = 3'd2;

reg  [1:0] num_layers_r;
wire [1:0] num_layers_w;
//num_neurons[0] is the # of neurons in input layer
//range: 0 ~ 31, which corresponds to number 1 ~ 32
reg  [4:0] num_neurons_r[0:3];
wire [4:0] num_neurons_w[0:3];
//whether to pass activation function in a layer
reg [2:0] do_act_r;
wire [2:0] do_act_w;


reg [5:0] num_multadds;

//PE ID for loading weights(biases)
wire [4:0] pe_w_id;

reg [4:0] current_num_neurons;
//num_iterations means the iterations needed per layer
wire [1:0] num_iterations;
//num_busy_pes is the number of non-idle pes in the last iteration of the layer
wire [2:0] num_busy_pes;

reg do_act;

// ======== input & output ========
assign ready = (state_w == SEND_O)? 1'b1: 1'b0;

// ======== debug =======
assign pe_state_r0 = pe_state_r[0];
assign pe_state_r1 = pe_state_r[1];
assign pe_oe_0 = pe_oe[0];
assign pe_oe_1 = pe_oe[1];
assign num_neurons_r3 = num_neurons_r[3];

// ===== internal registers & wires =====
// num_layers & num_neurons
assign num_layers_w = (state_r == CONFIG && state_count_r == 5'd0)? data[1:0]: num_layers_r;
genvar i_nn;
generate
for(i_nn = 0; i_nn < 4; i_nn = i_nn+1) begin: num_neurons
	assign num_neurons_w[i_nn] = (state_r == CONFIG && state_count_r == i_nn+1)? data[4:0]: num_neurons_r[i_nn];
end
endgenerate
assign do_act_w = (state_r == CONFIG && state_count_r == 5'd5)? data[2:0]: do_act_r;
/*
always@(*) begin
	num_neurons_w[0] = num_neurons_r[0];
	num_neurons_w[1] = num_neurons_r[1];
	num_neurons_w[2] = num_neurons_r[2];
	num_neurons_w[3] = num_neurons_r[3];
	if(state_r == CONFIG) begin
		case(state_count_r) begin
			5'd1: begin
				num_neurons_w[0] = data_i;
			end
			5'd2: begin
				num_neurons_w[1] = data_i;
			end
			5'd3: begin
				num_neurons_w[2] = data_i;
			end
			5'd4: begin
				num_neurons_w[3] = data_i;
			end
		end
	end
end
*/

// num_multadds
// in loading weights, this includes bias
// in layers, this excludes bias
always@(*) begin
	num_multadds = 6'd0;
	case(state_r)
		LOAD_W1: begin
			num_multadds = {1'b0, num_neurons_r[0]} + 6'd1;
		end
		LOAD_W2: begin
			num_multadds = {1'b0, num_neurons_r[1]} + 6'd1;
		end
		LOAD_WO: begin
			case(num_layers_r)
				ZERO_HIDDEN: begin
					num_multadds = {1'b0, num_neurons_r[0]} + 6'd1;
				end
				ONE_HIDDEN: begin
					num_multadds = {1'b0, num_neurons_r[1]} + 6'd1;
				end
				TWO_HIDDEN: begin
					num_multadds = {1'b0, num_neurons_r[2]} + 6'd1;
				end
			endcase
		end
		LAYER_H1: begin
			num_multadds = {1'b0, num_neurons_r[0]};
		end
		LAYER_H2: begin
			num_multadds = {1'b0, num_neurons_r[1]};
		end
		LAYER_O: begin
			case(num_layers_r)
				ZERO_HIDDEN: begin
					num_multadds = {1'b0, num_neurons_r[0]};
				end
				ONE_HIDDEN: begin
					num_multadds = {1'b0, num_neurons_r[1]};
				end
				TWO_HIDDEN: begin
					num_multadds = {1'b0, num_neurons_r[2]};
				end
			endcase

		end
	endcase
end

// pe_w_id
assign pe_w_id = state_count_w & 5'b00111;

// num_iterations& num_busy_pes & do_act
assign num_iterations = (current_num_neurons >> 3);
assign num_busy_pes = (current_num_neurons & 5'b00111);
always@(*) begin
	current_num_neurons = 5'd0;
	do_act = 1'b0;
	case(state_r)
		LAYER_H1: begin
			current_num_neurons = num_neurons_r[1];
			do_act = do_act_r[0];
		end
		LAYER_H2: begin
			current_num_neurons = num_neurons_r[2];
			do_act = do_act_r[1];
		end
		LAYER_O: begin
			current_num_neurons = num_neurons_r[3];
			do_act = do_act_r[2];
		end
	endcase
end

// TODO: change signal names
// ===== connection to PEs =====
/*
//debug
pe PE0(.Clock(clk), .Reset(rst),
	.Ctrl(pe_state_r[0]),
	.OutputCtrl(pe_oe[0]),
	.Data(data),
	.EnableAct(do_act),
	.fp_mac_output(fp_mac_output),
	.fp_mac_a(fp_mac_a),
	.fp_mac_b(fp_mac_b),
	.counter(counter), 
	.fp_mac_acc(fp_mac_acc), 
	.ArrWgt_Rd(ArrWgt_Rd), 
	.ArrWgt_Wr(ArrWgt_Wr), 
	.InBuf_Rd(InBuf_Rd), 
	.InBuf_Wr(InBuf_Wr), 
	.OutBuf_Rd(OutBuf_Rd), 
	.OutBuf_Wr(OutBuf_Wr),
	.InBuf_i0(InBuf_i0),
	.ArrWeights_i0(ArrWeights_i0),
	.ArrWeights_i1(ArrWeights_i1),
	.OutBuf_i0(OutBuf_i0),
	.OutBuf_n_i0(OutBuf_n_i0),
	.OutBuf_Full(OutBuf_Full));
	*/
pe PE0(.Clock(clk), .Reset(rst),
	.Ctrl(pe_state_r[0]),
	.OutputCtrl(pe_oe[0]),
	.Data(data),
	.EnableAct(do_act),
	.fp_mac_output(fp_mac_output_0),
	.fp_mac_a(fp_mac_a_0),
	.fp_mac_b(fp_mac_b_0));
	
pe PE1(.Clock(clk), .Reset(rst),
	.Ctrl(pe_state_r[1]),
	.OutputCtrl(pe_oe[1]),
	.Data(data),
	.EnableAct(do_act),
	.fp_mac_output(fp_mac_output_1),
	.fp_mac_a(fp_mac_a_1),
	.fp_mac_b(fp_mac_b_1));
/*
pe PE0(.Clock(clk), .Reset(rst),
	.Ctrl(pe_state_r[0]),
	.OutputCtrl(pe_oe[0]),
	.Data(data),
	.EnableAct(do_act));

pe PE1(.Clock(clk), .Reset(rst),
	.Ctrl(pe_state_r[1]),
	.OutputCtrl(pe_oe[1]),
	.Data(data),
	.EnableAct(do_act));

pe PE2(.Clock(clk), .Reset(rst),
	.Ctrl(pe_state_r[2]),
	.OutputCtrl(pe_oe[2]),
	.Data(data),
	.EnableAct(do_act));

pe PE3(.Clock(clk), .Reset(rst),
	.Ctrl(pe_state_r[3]),
	.OutputCtrl(pe_oe[3]),
	.Data(data),
	.EnableAct(do_act));

pe PE4(.Clock(clk), .Reset(rst),
	.Ctrl(pe_state_r[4]),
	.OutputCtrl(pe_oe[4]),
	.Data(data),
	.EnableAct(do_act));

pe PE5(.Clock(clk), .Reset(rst),
	.Ctrl(pe_state_r[5]),
	.OutputCtrl(pe_oe[5]),
	.Data(data),
	.EnableAct(do_act));

pe PE6(.Clock(clk), .Reset(rst),
	.Ctrl(pe_state_r[6]),
	.OutputCtrl(pe_oe[6]),
	.Data(data),
	.EnableAct(do_act));

pe PE7(.Clock(clk), .Reset(rst),
	.Ctrl(pe_state_r[7]),
	.OutputCtrl(pe_oe[7]),
	.Data(data),
	.EnableAct(do_act));
*/
// ========= combinational =========
// state
always@(*) begin
	state_w = state_r;
	case(state_r)
		IDLE: begin
			if(we == 1'b1) begin
				state_w = CONFIG;
			end
		end
		CONFIG: begin
			if(state_count_r == 5'd5) begin
				if(num_layers_r == ZERO_HIDDEN) begin
					state_w = LOAD_WO;
				end
				else begin
					state_w = LOAD_W1;
				end
			end
		end
		LOAD_W1: begin
			if(state_count_r == num_neurons_r[1] &&
			   multadd_count_r == num_multadds) begin
				if(num_layers_r == ONE_HIDDEN) begin
					state_w = LOAD_WO;
				end
				else begin
					state_w = LOAD_W2;
				end
			end
		end
		LOAD_W2: begin
			if(state_count_r == num_neurons_r[2] &&
			   multadd_count_r == num_multadds) begin
				state_w = LOAD_WO;
			end
		end
		LOAD_WO: begin
			if(state_count_r == num_neurons_r[3] &&
			   multadd_count_r == num_multadds) begin
				if(num_layers_r == ZERO_HIDDEN) begin
					state_w = LAYER_O;
				end
				else begin
					state_w = LAYER_H1;
				end
			end
		end
		LAYER_H1: begin
			if(pe_state_r[0] == PE_ACT_CLR) begin
				if(num_layers_r == ONE_HIDDEN) begin
					state_w = LAYER_O;
				end
				else begin
					state_w = LAYER_H2;
				end
			end
		end
		LAYER_H2: begin
			if(pe_state_r[0] == PE_ACT_CLR) begin
				state_w = LAYER_O;
			end
		end
		LAYER_O: begin
			if(pe_state_r[0] == PE_ACT_CLR) begin
				state_w = SEND_O;
			end
		end
		SEND_O: begin
			if(state_count_r == num_neurons_r[3] && oe == 1'b1) begin
				state_w = IDLE;
			end
		end
	endcase
end

// state_count
always@(*) begin
	state_count_w = state_count_r;
	if(state_w != state_r) begin
		state_count_w = 5'd0;
	end
	else if( state_r == CONFIG ||
		     ( (state_r == LOAD_W1 || state_r == LOAD_W2 || state_r == LOAD_WO) && multadd_count_r == num_multadds ) ||
		     (pe_state_r[0] == PE_ACT && pe_count_r == CYCLES_ACT) ||
		     (state_r == SEND_O && oe == 1'b1) ) begin
		state_count_w = state_count_r + 5'd1;
	end
end

// multadd_count
always@(*) begin
	multadd_count_w = 6'd0;
/*
	multadd_count_w = multadd_count_r;
	if(state_w != state_r ||
	   ( (state_r == LOAD_W1 || state_r == LOAD_W2 || state_r == LOAD_WO) && multadd_count_r == num_multadds ) ||
	   (pe_state_r[0] == PE_ACT && pe_count_r == CYCLES_ACT) ) begin
		multadd_count_w = 6'd0;
	end
*/
	if( (state_r == LOAD_W1 || state_r == LOAD_W2 || state_r == LOAD_WO ||
		 pe_state_r[0] == PE_MA || pe_state_r[0] == PE_MAB || pe_state_r[0] == PE_MABO) &&
		multadd_count_r != num_multadds ) begin
		multadd_count_w = multadd_count_r + 6'd1;
	end
end

// pe_count
always@(*) begin
	pe_count_w = 5'd0;
	if( (pe_state_r[0] == PE_BIAS && pe_count_r != CYCLES_MA) ||
		((pe_state_r[0] == PE_ACT || pe_state_r[0] == PE_ACT_CLR) && pe_count_r != CYCLES_ACT) ) begin
		pe_count_w = pe_count_r + 5'd1;
	end
end

// pe_state
integer i;
always@(*) begin
for(i = 0; i < 8; i = i+1) begin
	pe_state_w[i] = pe_state_r[i];
	case(pe_state_r[i])
		PE_IDLE: begin
			if((state_w == LOAD_W1 || state_w == LOAD_W2 || state_w == LOAD_WO) && pe_w_id == i) begin
				pe_state_w[i] = PE_LOAD;
			end
			else if(state_w != state_r && (state_w == LAYER_H1 || state_w == LAYER_H2 || state_w == LAYER_O) && (i <= num_busy_pes || num_iterations > 2'd0)) begin
				pe_state_w[i] = PE_MA;
			end
		end
		PE_LOAD: begin
			if(state_w == LAYER_H1 || state_w == LAYER_O) begin
				pe_state_w[i] = PE_MA;
			end
			else if((state_w == LOAD_W1 || state_w == LOAD_W2 || state_w == LOAD_WO) && pe_w_id != i) begin
				pe_state_w[i] = PE_IDLE;
			end
		end
		PE_MA: begin
			if(multadd_count_r == num_multadds) begin
				pe_state_w[i] = PE_BIAS;
			end
			else if( (state_w == LAYER_H2 || state_w == LAYER_O) && num_layers_r[1] != 2'd0 &&
				     state_count_w == 5'd0 &&
				     (multadd_count_w & 6'b000111) == i) begin
				pe_state_w[i] = PE_MABO;
			end
		end
		PE_MABO: begin
			if(multadd_count_r == num_multadds) begin
				pe_state_w[i] = PE_BIAS;
			end
			else begin
				pe_state_w[i] = PE_MA;
			end
		end
		PE_MAB: begin
			if(multadd_count_r == num_multadds) begin
				pe_state_w[i] = PE_BIAS;
			end
		end
		PE_BIAS: begin
			if(pe_count_r == CYCLES_MA) begin
				if(state_count_r == num_iterations || (i > num_busy_pes && state_count_r == num_iterations - 5'd1)) begin
					pe_state_w[i] = PE_ACT_CLR;
				end
				else begin
					pe_state_w[i] = PE_ACT;
				end
			end
		end
		PE_ACT: begin
			if((do_act == 1'b1 && pe_count_r == CYCLES_ACT) || do_act == 1'b0) begin
				pe_state_w[i] = PE_MAB;
			end
		end
		PE_ACT_CLR: begin
			if((do_act == 1'b1 && pe_count_r == CYCLES_ACT) || do_act == 1'b0) begin
				if(i == 0 && state_w != SEND_O) begin
					pe_state_w[i] = PE_MABO;
				end
				else if(state_w != state_r && state_w != SEND_O && (i <= num_busy_pes || num_iterations > 0)) begin // if enters next layer and should be working
					pe_state_w[i] = PE_MA;
				end
				else begin
					pe_state_w[i] = PE_IDLE;
				end
			end
		end
	endcase
end
end

// pe_oe
// two cases to be 1'b1:
// 1. broadcasting output to PEs representing neurons in the next layer
// 2. SEND_O state
genvar i_po;
generate
for(i_po = 0; i_po < 8; i_po = i_po+1) begin: output_en
	assign pe_oe[i_po] = (pe_state_r[i_po] == PE_MABO ||
						  (pe_state_r[i_po] == PE_IDLE &&
						   (state_r == LAYER_H2 || state_r == LAYER_O) && num_layers_r[1] != 2'd0 &&
					       state_count_r == 5'd0 &&
					       (multadd_count_w & 6'b000111) == i_po) ||
						  (state_r == SEND_O && oe == 1'b1 && state_count_r == i_po))? 1'b1: 1'b0;
end
endgenerate

// ========= sequential    =========
always@(posedge clk or posedge rst) begin
	if(rst == 1'b1) begin
		pe_state_r[0]     <= PE_IDLE;
		pe_state_r[1]     <= PE_IDLE;
		pe_state_r[2]     <= PE_IDLE;
		pe_state_r[3]     <= PE_IDLE;
		pe_state_r[4]     <= PE_IDLE;
		pe_state_r[5]     <= PE_IDLE;
		pe_state_r[6]     <= PE_IDLE;
		pe_state_r[7]     <= PE_IDLE;
		state_r           <= IDLE;
		state_count_r     <= 5'd0;
		multadd_count_r   <= 6'd0;
		pe_count_r        <= 5'd0;
		num_layers_r      <= 2'd0;
		num_neurons_r[0]  <= 5'd0;
		num_neurons_r[1]  <= 5'd0;
		num_neurons_r[2]  <= 5'd0;
		num_neurons_r[3]  <= 5'd0;
		do_act_r          <= 3'd0;
	end
	else begin
		pe_state_r[0]     <= pe_state_w[0];
		pe_state_r[1]     <= pe_state_w[1];
		pe_state_r[2]     <= pe_state_w[2];
		pe_state_r[3]     <= pe_state_w[3];
		pe_state_r[4]     <= pe_state_w[4];
		pe_state_r[5]     <= pe_state_w[5];
		pe_state_r[6]     <= pe_state_w[6];
		pe_state_r[7]     <= pe_state_w[7];
		state_r           <= state_w;
		state_count_r     <= state_count_w;
		multadd_count_r   <= multadd_count_w;
		pe_count_r        <= pe_count_w;
		num_layers_r      <= num_layers_w;
		num_neurons_r[0]  <= num_neurons_w[0];
		num_neurons_r[1]  <= num_neurons_w[1];
		num_neurons_r[2]  <= num_neurons_w[2];
		num_neurons_r[3]  <= num_neurons_w[3];
		do_act_r          <= do_act_w;
	end
end

endmodule
