create_clock -period 20.000 [get_ports clk]
set_input_delay 2.000 -clock clk [all_inputs]
set_output_delay 2.000 -clock clk [all_outputs]
set_load 0.05 [all_outputs]
