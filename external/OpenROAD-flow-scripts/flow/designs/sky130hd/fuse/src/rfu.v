module rfu #(
    parameter TEMPLATE_BITS = 8,
    parameter LAT_BITS = 16
) (
    input wire clk,
    input wire rst,
    input wire issue_valid,
    input wire [TEMPLATE_BITS-1:0] issue_template,
    input wire [LAT_BITS-1:0] issue_latency,
    output reg issue_ready,
    output reg busy,
    output reg done,
    output reg [TEMPLATE_BITS-1:0] loaded_template
);
reg [LAT_BITS-1:0] remaining;
always @(posedge clk) begin
    if (rst) begin
        issue_ready <= 1'b1;
        busy <= 1'b0;
        done <= 1'b0;
        remaining <= {LAT_BITS{1'b0}};
        loaded_template <= {TEMPLATE_BITS{1'b0}};
    end else begin
        done <= 1'b0;
        if (issue_valid && issue_ready) begin
            busy <= 1'b1;
            issue_ready <= 1'b0;
            remaining <= issue_latency;
            loaded_template <= issue_template;
        end else if (busy) begin
            if (remaining == {LAT_BITS{1'b0}}) begin
                busy <= 1'b0;
                issue_ready <= 1'b1;
                done <= 1'b1;
            end else begin
                remaining <= remaining - 1'b1;
            end
        end
    end
end
endmodule
