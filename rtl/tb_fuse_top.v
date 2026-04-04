module tb_fuse_top;
reg clk;
reg rst;
reg task_valid;
reg [7:0] task_id;
reg [7:0] task_template;
reg [15:0] task_latency;
reg [31:0] task_hash;
reg [127:0] task_data;
wire task_ready;
wire done;
fuse_top dut(
    .clk(clk),
    .rst(rst),
    .task_valid(task_valid),
    .task_id(task_id),
    .task_template(task_template),
    .task_latency(task_latency),
    .task_hash(task_hash),
    .task_data(task_data),
    .task_ready(task_ready),
    .done(done)
);
initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
end
initial begin
    rst = 1'b1;
    task_valid = 1'b0;
    task_id = 8'd0;
    task_template = 8'd0;
    task_latency = 16'd0;
    task_hash = 32'd0;
    task_data = 128'd0;
    #30;
    rst = 1'b0;
    #20;
    task_valid = 1'b1;
    task_id = 8'd1;
    task_template = 8'd3;
    task_latency = 16'd12;
    task_hash = 32'h0000beef;
    task_data = 128'h00010002000300040005000600070008;
    #10;
    task_valid = 1'b0;
    #300;
    $finish;
end
endmodule
