module lrc #(
    parameter TEMPLATE_BITS = 8,
    parameter SWITCH_CYCLES = 16
) (
    input wire clk,
    input wire rst,
    input wire req_valid,
    input wire [TEMPLATE_BITS-1:0] req_template,
    output reg req_ready,
    output reg done,
    output reg [TEMPLATE_BITS-1:0] active_template
);
reg [7:0] counter;
reg busy;
always @(posedge clk) begin
    if (rst) begin
        req_ready <= 1'b1;
        done <= 1'b0;
        active_template <= {TEMPLATE_BITS{1'b0}};
        counter <= 8'd0;
        busy <= 1'b0;
    end else begin
        done <= 1'b0;
        if (req_valid && req_ready) begin
            if (req_template == active_template) begin
                done <= 1'b1;
            end else begin
                busy <= 1'b1;
                req_ready <= 1'b0;
                counter <= SWITCH_CYCLES[7:0];
                active_template <= req_template;
            end
        end else if (busy) begin
            if (counter == 8'd0) begin
                busy <= 1'b0;
                req_ready <= 1'b1;
                done <= 1'b1;
            end else begin
                counter <= counter - 8'd1;
            end
        end
    end
end
endmodule
