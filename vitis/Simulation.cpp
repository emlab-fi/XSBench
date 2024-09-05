#include "XSbench_header.h"
#include <xrt/xrt_device.h>
#include <experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

////////////////////////////////////////////////////////////////////////////////////
// BASELINE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
// All "baseline" code is at the top of this file. The baseline code is a simple
// implementation of the algorithm, with only minor CPU optimizations in place.
// Following these functions are a number of optimized variants,
// which each deploy a different combination of optimizations strategies. By
// default, XSBench will only run the baseline implementation. Optimized variants
// are not yet implemented in this SYCL port.
////////////////////////////////////////////////////////////////////////////////////


unsigned long long run_event_based_simulation(Inputs in, SimulationData SD, int mype, double * kernel_init_time)
{

	////////////////////////////////////////////////////////////////////////////////
	// SUMMARY: Simulation Data Structure Manifest for "SD" Object
	// Here we list all heap arrays (and lengths) in SD that would need to be
	// offloaded manually if using an accelerator with a seperate memory space
	////////////////////////////////////////////////////////////////////////////////
	// int * num_nucs;                     // Length = length_num_nucs;
	// double * concs;                     // Length = length_concs
	// int * mats;                         // Length = length_mats
	// double * unionized_energy_array;    // Length = length_unionized_energy_array
	// int * index_grid;                   // Length = length_index_grid
	// NuclideGridPoint * nuclide_grid;    // Length = length_nuclide_grid
	//
	// Note: "unionized_energy_array" and "index_grid" can be of zero length
	//        depending on lookup method.
	//
	// Note: "Lengths" are given as the number of objects in the array, not the
	//       number of bytes.
	////////////////////////////////////////////////////////////////////////////////

	// Let's create an extra verification array to reduce manually later on
	// aligned at 4k boundary for the XRT runtime buffer
	if( mype == 0 ) printf("Allocating an additional %.1lf MB of memory for verification arrays...\n", in.lookups * sizeof(int) /1024.0/1024.0);
	int * verification_host = (int *) aligned_alloc(4096, in.lookups * sizeof(int));

	// Timers
	double start = get_time();
	double init_done = 0;
	double move_data_done = 0;
	double stop = 0;


	///Upload kernel to FPGA
	unsigned int device_index = 0;
	auto device = xrt::device(device_index);
	printf("Device name: %s\n", device.get_info<xrt::info::device::name>().c_str());
	printf("Device bdf: %s\n", device.get_info<xrt::info::device::bdf>().c_str());
	auto kernel_uuid = device.load_xclbin("xs_kernel.xclbin");

	printf("Kernel uploaded\n");

	auto kernel = xrt::kernel(device, kernel_uuid, "xs_lookup_krnl");

	// Allocate memory on FPGA device
	auto num_nucs_b = xrt::bo(device, SD.num_nucs, SD.length_num_nucs, kernel.group_id(1));
	auto concs_b = xrt::bo(device, SD.concs, SD.length_concs, kernel.group_id(2));
	auto unionized_energy_array_b = xrt::bo(device, SD.unionized_energy_array, SD.length_unionized_energy_array, kernel.group_id(3));
	auto index_grid_b = xrt::bo(device, SD.index_grid, SD.length_index_grid, kernel.group_id(4));
	auto nuclide_grid_b = xrt::bo(device, SD.nuclide_grid, SD.length_nuclide_grid, kernel.group_id(5));
	auto mats_b = xrt::bo(device, SD.mats, SD.length_mats, kernel.group_id(6));
	auto verification_b = xrt::bo(device,verification_host, in.lookups * sizeof(int), kernel.group_id(8));

	init_done = get_time();
	printf("Kernel init and allocation took: %.3lf seconds\n", init_done - start);

	// move data to fpga
	num_nucs_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	concs_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	unionized_energy_array_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	index_grid_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	nuclide_grid_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	mats_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	verification_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

	move_data_done = get_time();

	// simulation kernel
	printf("Kernel moving data took: %.3lf seconds\n", move_data_done - init_done);
	printf("Running kernel\n");
	auto run = kernel(in.lookups, num_nucs_b, concs_b, unionized_energy_array_b, index_grid_b,
					  nuclide_grid_b, mats_b, SD.max_num_nucs, in.n_isotopes, in.n_gridpoints);
	run.wait();

	// copy verification back to host
	verification_b.sync(XCL_BO_SYNC_BO_FROM_DEVICE);


	stop = get_time();
	printf("Kernel finished, took %.3lf seconds (including verification copy back)\n", stop - move_data_done);

	// for calculations
	*kernel_init_time = move_data_done;

	// Host reduces the verification array
	unsigned long long verification_scalar = 0;
	for( int i = 0; i < in.lookups; i++ )
		verification_scalar += verification_host[i];

	return verification_scalar;
}