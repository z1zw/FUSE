module tars #(
    parameter ID_BITS = 8,
    parameter DEPTH = 8
) (
    input wire clk,
    input wire rst,
    input wire push_valid,
    input wire [ID_BITS-1:0] push_id,
    output reg push_ready,
    input wire pop_ready,
    output reg pop_valid,
    output reg [ID_BITS-1:0] pop_id
);
reg [ID_BITS-1:0] fifo [0:DEPTH-1];
reg [$clog2(DEPTH):0] count;
reg [$clog2(DEPTH)-1:0] head;
reg [$clog2(DEPTH)-1:0] tail;
always @(posedge clk) begin
    if (rst) begin
        count <= {($clog2(DEPTH)+1){1'b0}};
        head <= {($clog2(DEPTH)){1'b0}};
        tail <= {($clog2(DEPTH)){1'b0}};
        push_ready <= 1'b1;
        pop_valid <= 1'b0;
        pop_id <= {ID_BITS{1'b0}};
    end else begin
        push_ready <= (count < DEPTH);
        pop_valid <= (count > 0);
        if (push_valid && push_ready) begin
            fifo[tail] <= push_id;
            tail <= tail + 1'b1;
            count <= count + 1'b1;
        end
        if (pop_ready && pop_valid) begin
            pop_id <= fifo[head];
            head <= head + 1'b1;
            count <= count - 1'b1;
        end
    end
end
endmodule
