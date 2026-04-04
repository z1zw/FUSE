module fuse_top #(
    parameter TEMPLATE_BITS = 8,
    parameter HASH_BITS = 32,
    parameter DATA_BITS = 128,
    parameter LAT_BITS = 16,
    parameter RFU_COUNT = 4
) (
    input wire clk,
    input wire rst,
    input wire task_valid,
    input wire [7:0] task_id,
    input wire [TEMPLATE_BITS-1:0] task_template,
    input wire [LAT_BITS-1:0] task_latency,
    input wire [HASH_BITS-1:0] task_hash,
    input wire [DATA_BITS-1:0] task_data,
    output wire task_ready,
    output wire done
);
wire sched_valid;
wire [7:0] sched_id;
wire sched_ready;
wire lrc_ready;
wire lrc_done;
wire [TEMPLATE_BITS-1:0] active_template;
wire lookup_hit;
wire [DATA_BITS-1:0] lookup_data;
wire rfu_ready [0:RFU_COUNT-1];
wire rfu_done [0:RFU_COUNT-1];
wire rfu_busy [0:RFU_COUNT-1];
assign task_ready = sched_ready;
assign done = rfu_done[0];
tars u_tars(
    .clk(clk),
    .rst(rst),
    .push_valid(task_valid),
    .push_id(task_id),
    .push_ready(sched_ready),
    .pop_ready(lrc_ready & rfu_ready[0]),
    .pop_valid(sched_valid),
    .pop_id(sched_id)
);
lrc u_lrc(
    .clk(clk),
    .rst(rst),
    .req_valid(sched_valid),
    .req_template(task_template),
    .req_ready(lrc_ready),
    .done(lrc_done),
    .active_template(active_template)
);
fvm u_fvm(
    .clk(clk),
    .rst(rst),
    .lookup_valid(sched_valid),
    .lookup_hash(task_hash),
    .lookup_hit(lookup_hit),
    .lookup_data(lookup_data),
    .write_valid(rfu_done[0]),
    .write_hash(task_hash),
    .write_data(task_data)
);
genvar i;
generate
for (i = 0; i < RFU_COUNT; i = i + 1) begin : g_rfu
    rfu u_rfu(
        .clk(clk),
        .rst(rst),
        .issue_valid(lrc_done),
        .issue_template(active_template),
        .issue_latency(task_latency),
        .issue_ready(rfu_ready[i]),
        .busy(rfu_busy[i]),
        .done(rfu_done[i]),
        .loaded_template()
    );
end
endgenerate
endmodule
