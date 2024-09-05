#===============================================================================
# User Options
#===============================================================================

TARGET_HW   ?= xilinx_u55c_gen3x16_xdma_3_202210_1
TARGET_OUTPUT ?= hardware
XILINX_XRT ?= /opt/xilinx/xrt/

#===============================================================================
# Program name & source code list
#===============================================================================

program = XSBench

source = \
Main.cpp \
io.cpp \
Simulation.cpp \
GridInit.cpp \
XSutils.cpp \
Materials.cpp

obj = $(source:.cpp=.o)


#===============================================================================
# Sets Flags
#===============================================================================

# compiler to use
CC = g++

# Compile flags
CFLAGS = -std=c++17 -Wall -O3 -lxrt_coreutil -pthread --I$(XILINX_XRT)/include -L$X(ILINX_XRT)/lib
LDFLAGS = -lm -lxrt_coreutil -pthread

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) XSbench_header.h Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp XSbench_header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run:
	./$(program)
