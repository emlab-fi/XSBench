#include "XSbench_header.h"
#include <sycl/ext/intel/fpga_extensions.hpp>

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

// use SYCL namespace to reduce symbol names
using namespace sycl;

//simulation kernel functor, templated, so we can spawn multiple identical pipelines
template<int COUNT, int INDEX>
struct simulation_kernel {

	// data required for running the kernel
	Inputs in;
	int* num_nucs_d;
	float* concs_d;
	float* unionized_energy_array_d;
	int* index_grid_d;
	NuclideGridPoint* nuclide_grid_d;
	int* mats_d;
	int max_num_nucs;
	int* verification_d;

	//constructor
	simulation_kernel(
		Inputs i, int* num_nucs, float* concs, float* unionized_energy_array, int* index_grid,
		NuclideGridPoint* nuclide_grid, int* mats, int max_nm, int* verification
	) : in(i), num_nucs_d(num_nucs), concs_d(concs), unionized_energy_array_d(unionized_energy_array),
		index_grid_d(index_grid), nuclide_grid_d(nuclide_grid), mats_d(mats), max_num_nucs(max_nm),
		verification_d(verification) {}

	void operator()(id<1> idx) const {

		uint64_t seed = STARTING_SEED;

		size_t i = idx[0] + INDEX * (in.lookups / COUNT);

		// Forward seed to lookup index (we need 2 samples per lookup)
		seed = fast_forward_LCG(seed, 2*i);

		// Randomly pick an energy and material for the particle
		float p_energy = LCG_random_float(&seed);
		int mat         = pick_mat(&seed);

		float macro_xs_vector[5] = {0};

		// Perform macroscopic Cross Section Lookup
		calculate_macro_xs(
				p_energy,        // Sampled neutron energy (in lethargy)
				mat,             // Sampled material type index neutron is in
				in.n_isotopes,   // Total number of isotopes in simulation
				in.n_gridpoints, // Number of gridpoints per isotope in simulation
				num_nucs_d,     // 1-D array with number of nuclides per material
				concs_d,        // Flattened 2-D array with concentration of each nuclide in each material
				unionized_energy_array_d, // 1-D Unionized energy array
				index_grid_d,   // Flattened 2-D grid holding indices into nuclide grid for each unionized energy level
				nuclide_grid_d, // Flattened 2-D grid holding energy levels and XS_data for all nuclides in simulation
				mats_d,         // Flattened 2-D array with nuclide indices defining composition of each type of material
				macro_xs_vector, // 1-D array with result of the macroscopic cross section (5 different reaction channels)
				in.grid_type,    // Lookup type (nuclide, hash, or unionized)
				in.hash_bins,    // Number of hash bins used (if using hash lookup type)
				max_num_nucs  // Maximum number of nuclides present in any material
				);

		// For verification, and to prevent the compiler from optimizing
		// all work out, we interrogate the returned macro_xs_vector array
		// to find its maximum value index, then increment the verification
		// value by that index. In this implementation, we store to a global
		// array that will get tranferred back and reduced on the host.
		float max = -1.0;
		int max_idx = 0;
		for(int j = 0; j < 5; j++ )
		{
			if( macro_xs_vector[j] > max )
			{
				max = macro_xs_vector[j];
				max_idx = j;
			}
		}
		verification_d[i] = max_idx+1;
	}

};


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
	if( mype == 0 ) printf("Allocating an additional %.1lf MB of memory for verification arrays...\n", in.lookups * sizeof(int) /1024.0/1024.0);
	int * verification_host = (int *) malloc(in.lookups * sizeof(int));

	// Timers
	double start = get_time();
	double stop;

	// Scope here is important, as when we exit this blocl we will automatically sync with device
	// to ensure all work is done and that we can read from verification_host array.
	{
		// create a fpga queue

		queue sycl_q{ext::intel::fpga_selector_v};
		if(mype == 0 ) printf("Running on: %s\n", sycl_q.get_device().get_info<sycl::info::device::name>().c_str());
		if(mype == 0 ) printf("Initializing device buffers and JIT compiling kernel...\n");

		////////////////////////////////////////////////////////////////////////////////
		// Allocate memory on FPGA device
		////////////////////////////////////////////////////////////////////////////////

		int* num_nucs_d = malloc_device<int>(SD.length_num_nucs, sycl_q);
		float* concs_d = malloc_device<float>(SD.length_concs, sycl_q);
		int* mats_d = malloc_device<int>(SD.length_mats, sycl_q);
		NuclideGridPoint* nuclide_grid_d = malloc_device<NuclideGridPoint>(SD.length_nuclide_grid, sycl_q);
		int* verification_d = malloc_device<int>(in.lookups, sycl_q);

		// Cannot create empty alocations
		if( SD.length_unionized_energy_array == 0 )
		{
			SD.length_unionized_energy_array = 1;
			SD.unionized_energy_array = (float *) malloc(sizeof(float));
		}
		float* unionized_energy_array_d = malloc_device<float>(SD.length_unionized_energy_array, sycl_q);

		if( SD.length_index_grid == 0 )
		{
			SD.length_index_grid = 1;
			SD.index_grid = (int *) malloc(sizeof(int));
		}
		// Check whether we can allocate it
		size_t index_grid_allocation_sz = ceil((SD.length_index_grid * sizeof(int)));
		assert( index_grid_allocation_sz <= sycl_q.get_device().get_info<sycl::info::device::max_mem_alloc_size>() );
		int* index_grid_d = malloc_device<int>(SD.length_index_grid, sycl_q);

		////////////////////////////////////////////////////////////////////////////////
		// Define Device Kernels
		////////////////////////////////////////////////////////////////////////////////

		// move data to fpga
		sycl_q.submit([&](handler &cgh) {
				cgh.memcpy(num_nucs_d, SD.num_nucs, SD.length_num_nucs * sizeof(int)); });
		sycl_q.submit([&](handler &cgh) {
				cgh.memcpy(concs_d, SD.concs, SD.length_concs * sizeof(float)); });
		sycl_q.submit([&](handler &cgh) {
				cgh.memcpy(mats_d, SD.mats, SD.length_mats * sizeof(int)); });
		sycl_q.submit([&](handler &cgh) {
				cgh.memcpy(nuclide_grid_d, SD.nuclide_grid, SD.length_nuclide_grid * sizeof(NuclideGridPoint)); });
		sycl_q.submit([&](handler &cgh) {
				cgh.memcpy(unionized_energy_array_d, SD.unionized_energy_array, SD.length_unionized_energy_array * sizeof(float)); });
		sycl_q.submit([&](handler &cgh) {
				cgh.memcpy(index_grid_d, SD.index_grid, index_grid_allocation_sz); });

		sycl_q.wait();

		// simulation kernels - we submit several parallel kernels
		sycl_q.submit([&](handler &cgh) {
			cgh.parallel_for(range<1>(in.lookups),
				simulation_kernel<1, 0>(in, num_nucs_d, concs_d, unionized_energy_array_d,
					index_grid_d, nuclide_grid_d, mats_d, SD.max_num_nucs, verification_d));
		});



		sycl_q.wait();

		// copy verification back to host
		sycl_q.submit([&](handler &cgh)
				{
				cgh.memcpy(verification_host, verification_d, in.lookups);
				});

		sycl_q.wait();

		stop = get_time();
		if(mype==0) printf("Kernel initialization, compilation, and launch took %.2lf seconds.\n", stop-start);
		if(mype==0) printf("Beginning event based simulation...\n");
	}

	// Host reduces the verification array
	unsigned long long verification_scalar = 0;
	for( int i = 0; i < in.lookups; i++ )
		verification_scalar += verification_host[i];

	return verification_scalar;
}


// binary search for energy on unionized energy grid
// returns lower index
template <class T>
long grid_search( long n, float quarry, T A)
{
	long lowerLimit = 0;
	long upperLimit = n-1;
	long examinationPoint;
	long length = upperLimit - lowerLimit;

	while( length > 1 )
	{
		examinationPoint = lowerLimit + ( length / 2 );

		if( A[examinationPoint] > quarry )
			upperLimit = examinationPoint;
		else
			lowerLimit = examinationPoint;

		length = upperLimit - lowerLimit;
	}

	return lowerLimit;
}

// Calculates the microscopic cross section for a given nuclide & energy
template <class Double_Type, class Int_Type, class NGP_Type>
void calculate_micro_xs(   float p_energy, int nuc, long n_isotopes,
		long n_gridpoints,
		Double_Type  egrid, Int_Type  index_data,
		NGP_Type  nuclide_grids,
		long idx, float *  xs_vector, int grid_type, int hash_bins ){
	// Variables
	float f;
	NuclideGridPoint low, high;
	long low_idx, high_idx;

  if( grid_type == UNIONIZED) // Unionized Energy Grid - we already know the index, no binary search needed.
	{
		// pull ptr from energy grid and check to ensure that
		// we're not reading off the end of the nuclide's grid
		if( index_data[idx * n_isotopes + nuc] == n_gridpoints - 1 )
			low_idx = nuc*n_gridpoints + index_data[idx * n_isotopes + nuc] - 1;
		else
		{
			low_idx = nuc*n_gridpoints + index_data[idx * n_isotopes + nuc];
		}
	}
	else // nothing else supported, use default 0 index
	{
		low_idx = 0;
		high_idx = 0;
	}

	high_idx = low_idx + 1;
	low = nuclide_grids[low_idx];
	high = nuclide_grids[high_idx];

	// calculate the re-useable interpolation factor
	f = (high.energy - p_energy) / (high.energy - low.energy);

	// Total XS
	xs_vector[0] = high.total_xs - f * (high.total_xs - low.total_xs);

	// Elastic XS
	xs_vector[1] = high.elastic_xs - f * (high.elastic_xs - low.elastic_xs);

	// Absorbtion XS
	xs_vector[2] = high.absorbtion_xs - f * (high.absorbtion_xs - low.absorbtion_xs);

	// Fission XS
	xs_vector[3] = high.fission_xs - f * (high.fission_xs - low.fission_xs);

	// Nu Fission XS
	xs_vector[4] = high.nu_fission_xs - f * (high.nu_fission_xs - low.nu_fission_xs);
}

// Calculates macroscopic cross section based on a given material & energy
template <class Double_Type, class Int_Type, class NGP_Type, class E_GRID_TYPE, class INDEX_TYPE>
void calculate_macro_xs( float p_energy, int mat, long n_isotopes,
		long n_gridpoints, Int_Type  num_nucs,
		Double_Type  concs,
		E_GRID_TYPE  egrid, INDEX_TYPE  index_data,
		NGP_Type  nuclide_grids,
		Int_Type  mats,
		float * macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs ){
	int p_nuc; // the nuclide we are looking up
	long idx = -1;
	float conc; // the concentration of the nuclide in the material

	// cleans out macro_xs_vector
	#pragma unroll
	for( int k = 0; k < 5; k++ )
		macro_xs_vector[k] = 0;

	// If we are using the unionized energy grid (UEG), we only
	// need to perform 1 binary search per macroscopic lookup.
	// If we are using the nuclide grid search, it will have to be
	// done inside of the "calculate_micro_xs" function for each different
	// nuclide in the material.
	if( grid_type == UNIONIZED )
		idx = grid_search( n_isotopes * n_gridpoints, p_energy, egrid);


	// Once we find the pointer array on the UEG, we can pull the data
	// from the respective nuclide grids, as well as the nuclide
	// concentration data for the material
	// Each nuclide from the material needs to have its micro-XS array
	// looked up & interpolatied (via calculate_micro_xs). Then, the
	// micro XS is multiplied by the concentration of that nuclide
	// in the material, and added to the total macro XS array.
	// (Independent -- though if parallelizing, must use atomic operations
	//  or otherwise control access to the xs_vector and macro_xs_vector to
	//  avoid simulataneous writing to the same data structure)
	for( int j = 0; j < num_nucs[mat]; j++ )
	{
		float xs_vector[5];
		p_nuc = mats[mat*max_num_nucs + j];
		conc = concs[mat*max_num_nucs + j];
		calculate_micro_xs( p_energy, p_nuc, n_isotopes,
				n_gridpoints, egrid, index_data,
				nuclide_grids, idx, xs_vector, grid_type, hash_bins );

		#pragma unroll
		for( int k = 0; k < 5; k++ )
			macro_xs_vector[k] += xs_vector[k] * conc;
	}
}

// picks a material based on a probabilistic distribution
int pick_mat( unsigned long * seed )
{
	// I have a nice spreadsheet supporting these numbers. They are
	// the fractions (by volume) of material in the core. Not a
	// *perfect* approximation of where XS lookups are going to occur,
	// but this will do a good job of biasing the system nonetheless.

	// Also could be argued that doing fractions by weight would be
	// a better approximation, but volume does a good enough job for now.

	float dist[12];
	dist[0]  = 0.140;	// fuel
	dist[1]  = 0.052;	// cladding
	dist[2]  = 0.275;	// cold, borated water
	dist[3]  = 0.134;	// hot, borated water
	dist[4]  = 0.154;	// RPV
	dist[5]  = 0.064;	// Lower, radial reflector
	dist[6]  = 0.066;	// Upper reflector / top plate
	dist[7]  = 0.055;	// bottom plate
	dist[8]  = 0.008;	// bottom nozzle
	dist[9]  = 0.015;	// top nozzle
	dist[10] = 0.025;	// top of fuel assemblies
	dist[11] = 0.013;	// bottom of fuel assemblies

	//precalculated cumulative data to unroll a loop and remove unecessary calculations
	// THIS DOES NOT BEHAVE AS NORMAL SIMULATOR! WE CANNOT PRECALCULATE THE CUMULATIVE
	float dist_cumulative[12];
	dist_cumulative[0] = 0.140;
	dist_cumulative[1] = 0.192;
	dist_cumulative[2] = 0.467;
	dist_cumulative[3] = 0.601;
	dist_cumulative[4] = 0.755;
	dist_cumulative[5] = 0.819;
	dist_cumulative[6] = 0.885;
	dist_cumulative[7] = 0.940;
	dist_cumulative[8] = 0.948;
	dist_cumulative[9] = 0.963;
	dist_cumulative[10] = 0.988;
	dist_cumulative[11] = 1.000;

	float roll = LCG_random_float(seed);

	// makes a pick based on the distro

	#pragma unroll
	for( int i = 0; i < 12; i++ )
	{
        float running = 0;
        #pragma unroll 12
        for ( int j = i; j > 0; j-- ) {
            running += dist[j]
        }
		if (roll < running) {
			return i;
		}
	}

	return 0;
}

double LCG_random_double(uint64_t * seed)
{
	// LCG parameters
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return (double) (*seed) / (double) m;
}

float LCG_random_float(uint64_t * seed)
{
	// LCG parameters
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return (float) (*seed) / (float) m;
}

uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
	// LCG parameters
	const uint64_t m = 9223372036854775808ULL; // 2^63
	uint64_t a = 2806196910506780709ULL;
	uint64_t c = 1ULL;

	n = n % m;

	uint64_t a_new = 1;
	uint64_t c_new = 0;

	while(n > 0)
	{
		if(n & 1)
		{
			a_new *= a;
			c_new = c_new * a + c;
		}
		c *= (a + 1);
		a *= a;

		n >>= 1;
	}

	return (a_new * seed + c_new) % m;

}
