#===============================================================================
# User Options
#===============================================================================

TARGET_HW   ?= Agilex7
TARGET_OUTPUT ?= hardware

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
# Select hardware
# These paths and part numbers are hardcoded for Intel Devcloud
#===============================================================================
ifeq ($(TARGET_HW), Agilex7)
HW_DEVICE = Agilex7
HW_FULL_PATH = /glob/development-tools/versions/oneapi/2024.0.2.1/oneapi/de10_agilex:2B2E_8GBx4
endif

ifeq ($(TARGET_HW), Stratix10)
HW_DEVICE = Stratix10
HW_FULL_PATH = /glob/development-tools/versions/oneapi/2024.0.2.1/oneapi/intel_s10sx_pac:pac_s10
endif

ifeq ($(TARGET_HW), Arria10)
HW_DEVICE = Arria10
HW_FULL_PATH = /glob/development-tools/versions/oneapi/2024.0.2.1/oneapi/intel_s10gx_pac:pac_a10
endif

#===============================================================================
# Sets Flags
#===============================================================================

# compiler to use
CC = icpx

# Compile flags
CFLAGS = -std=c++17 -Wall -O3 -fsycl -fintelfpga
LDFLAGS = -lm -fsycl -fintelfpga -Xsv

# Linker Flags

ifeq ($(TARGET_OUTPUT), emulator)
LDFLAGS += -Xsemulator -Xstarget=$(HW_DEVICE)
endif

ifeq ($(TARGET_OUTPUT), simulation)
LDFLAGS += -Xssimulation -Xstarget=$(HW_DEVICE)
endif

ifeq ($(TARGET_OUTPUT), report)
LDFLAGS += -Xshardware -fsycl-link=early -Xstarget=$(HW_FULL_PATH)
endif

ifeq ($(TARGET_OUTPUT), hardware)
LDFLAGS += -Xshardware -Xstarget=$(HW_FULL_PATH)
endif


#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) XSbench_header.h Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp XSbench_header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

edit:
	vim -p $(source) XSbench_header.h

run:
	./$(program)
