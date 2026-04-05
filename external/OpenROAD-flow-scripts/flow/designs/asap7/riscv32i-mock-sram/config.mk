export DESIGN_NICKNAME = riscv32i-mock-sram
export BLOCKS=fakeram7_256x32

# Override platform default (BLOCKS_grid_strategy.tcl) which defines an
# ElementGrid macro grid with no stripes, producing empty PDN grids
# (PDN-0232/0233). BLOCK_grid_strategy.tcl uses M4-M5 connections that
# match the block's MAX_ROUTING_LAYER=M4 constraint.
export PDN_TCL = $(PLATFORM_DIR)/openRoad/pdn/BLOCK_grid_strategy.tcl

include designs/asap7/riscv32i/config.mk

