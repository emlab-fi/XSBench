#ifndef __XSBENCH_HEADER_H__
#define __XSBENCH_HEADER_H__

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<strings.h>
#include<math.h>
#include<assert.h>
#include<stdint.h>
#include <chrono> 

// Papi Header
#ifdef PAPI
#include "papi.h"
#endif

// Grid types
#define UNIONIZED 0
#define NUCLIDE 1
#define HASH 2

// Simulation types
#define HISTORY_BASED 1
#define EVENT_BASED 2

// Binary Mode Type
#define NONE 0
#define READ 1
#define WRITE 2

// Starting Seed
#define STARTING_SEED 1070

// Structures
typedef struct{
	double energy;
	double total_xs;
	double elastic_xs;
	double absorbtion_xs;
	double fission_xs;
	double nu_fission_xs;
} NuclideGridPoint;

typedef struct{
	int nthreads;
	long n_isotopes;
	long n_gridpoints;
	int lookups;
	char * HM;
	int grid_type; // 0: Unionized Grid (default)    1: Nuclide Grid
	int hash_bins;
	int particles;
	int simulation_method;
	int binary_mode;
	int kernel_id;
} Inputs;

typedef struct{
	int * num_nucs;                     // Length = length_num_nucs;
	double * concs;                     // Length = length_concs
	int * mats;                         // Length = length_mats
	double * unionized_energy_array;    // Length = length_unionized_energy_array
	int * index_grid;                   // Length = length_index_grid
	NuclideGridPoint * nuclide_grid;    // Length = length_nuclide_grid
	long length_num_nucs;
	long length_concs;
	long length_mats;
	long length_unionized_energy_array;
	long length_index_grid;
	long length_nuclide_grid;
	int max_num_nucs;
	double * p_energy_samples;
	long length_p_energy_samples;
	int * mat_samples;
	long length_mat_samples;
} SimulationData;

// io.c
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int(long a);
Inputs read_CLI( int argc, char * argv[] );
void print_CLI_error(void);
void print_inputs(Inputs in, int nprocs, int version);
int print_results( Inputs in, int mype, double runtime, int nprocs, unsigned long long vhash, double time );
void binary_write( Inputs in, SimulationData SD );
SimulationData binary_read( Inputs in );

// Simulation.c
unsigned long long run_event_based_simulation(Inputs in, SimulationData SD, int mype, double * kernel_init_time);

// GridInit.c
SimulationData grid_init_do_not_profile( Inputs in, int mype );

// XSutils.c
int NGP_compare( const void * a, const void * b );
int double_compare(const void * a, const void * b);
size_t estimate_mem_usage( Inputs in );
double get_time(void);

// Materials.c
int * load_num_nucs(long n_isotopes);
int * load_mats( int * num_nucs, long n_isotopes, int * max_num_nucs );
double * load_concs( int * num_nucs, int max_num_nucs );

// binary search for energy on nuclide energy grid
// This funciton is defined in the header, as it is also used by the
// initialization region of the program.
template <class T>
long grid_search_nuclide( long n, double quarry, T A, long low, long high)
{
	long lowerLimit = low;
	long upperLimit = high;
	long examinationPoint;
	long length = upperLimit - lowerLimit;

	while( length > 1 )
	{
		examinationPoint = lowerLimit + ( length / 2 );

		if( A[examinationPoint].energy > quarry )
			upperLimit = examinationPoint;
		else
			lowerLimit = examinationPoint;

		length = upperLimit - lowerLimit;
	}

	return lowerLimit;
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
#endif
