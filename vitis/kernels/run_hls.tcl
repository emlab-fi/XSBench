# Create project
open_project -reset proj_xs_kernel

# Add design files
add_files xs_kernel.cpp

# Set the top-level function
set_top xs_lookup_kernel

# ########################################################
# Create a solution
open_solution -reset solution1 -flow_target vitis
# Define technology and clock rate
set_part  {xcvu9p-flga2104-2-i}
create_clock -period 10

# Set variable to select which steps to execute
set hls_exec 1

csim_design
# Set any optimization directives
# End of directives

if {$hls_exec == 1} {
	# Run Synthesis and Exit
	csynth_design
} elseif {$hls_exec == 2} {
	# Run Synthesis, RTL Simulation and Exit
	csynth_design
	cosim_design
} elseif {$hls_exec == 3} {
	# Run Synthesis, RTL Simulation, RTL implementation and Exit
	csynth_design
	cosim_design
	export_design
} else {
	# Default is to exit after running csynth
	csynth_design
}

exit
