module fvm #(
    parameter HASH_BITS = 32,
    parameter DEPTH = 16,
    parameter DATA_BITS = 128
) (
    input wire clk,
    input wire rst,
    input wire lookup_valid,
    input wire [HASH_BITS-1:0] lookup_hash,
    output reg lookup_hit,
    output reg [DATA_BITS-1:0] lookup_data,
    input wire write_valid,
    input wire [HASH_BITS-1:0] write_hash,
    input wire [DATA_BITS-1:0] write_data
);
reg [HASH_BITS-1:0] tag_mem [0:DEPTH-1];
reg [DATA_BITS-1:0] data_mem [0:DEPTH-1];
reg valid_mem [0:DEPTH-1];
reg [$clog2(DEPTH)-1:0] wr_ptr;
integer i;
always @(posedge clk) begin
    if (rst) begin
        lookup_hit <= 1'b0;
        lookup_data <= {DATA_BITS{1'b0}};
        wr_ptr <= {($clog2(DEPTH)){1'b0}};
        for (i = 0; i < DEPTH; i = i + 1) begin
            tag_mem[i] <= {HASH_BITS{1'b0}};
            data_mem[i] <= {DATA_BITS{1'b0}};
            valid_mem[i] <= 1'b0;
        end
    end else begin
        lookup_hit <= 1'b0;
        if (lookup_valid) begin
            for (i = 0; i < DEPTH; i = i + 1) begin
                if (valid_mem[i] && tag_mem[i] == lookup_hash) begin
                    lookup_hit <= 1'b1;
                    lookup_data <= data_mem[i];
                end
            end
        end
        if (write_valid) begin
            tag_mem[wr_ptr] <= write_hash;
            data_mem[wr_ptr] <= write_data;
            valid_mem[wr_ptr] <= 1'b1;
            wr_ptr <= wr_ptr + 1'b1;
        end
    end
end
endmodule
