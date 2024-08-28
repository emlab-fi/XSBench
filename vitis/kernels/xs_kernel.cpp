#include "xs_kernel.hpp"
#include <stdint.h>

namespace { // annonymous namespace

constexpr uint64_t STARTING_SEED = 1070;

long grid_search(long n, double querry, double* grid)
{
	long lowerLimit = 0;
	long upperLimit = n-1;
	long examinationPoint;
	long length = upperLimit - lowerLimit;

	while( length > 1 )
	{
		examinationPoint = lowerLimit + ( length / 2 );

		if( grid[examinationPoint] > querry )
			upperLimit = examinationPoint;
		else
			lowerLimit = examinationPoint;

		length = upperLimit - lowerLimit;
	}

	return lowerLimit;
}


void calculate_micro_xs(double energy, int nuclide, long n_isotopes, long n_gridpoints, double* energy_grid,
                        int* index_grid, NuclideGridPoint* nuclide_grid, long index, double* xs_vector) {

    long low_index, high_index;

    if( index_grid[index * n_isotopes + nuclide] == n_gridpoints - 1 )
        low_index = nuclide*n_gridpoints + index_grid[index * n_isotopes + nuclide] - 1;
    else
    {
        low_index = nuclide*n_gridpoints + index_grid[index * n_isotopes + nuclide];
    }

    high_index = low_index + 1;

    NuclideGridPoint low = nuclide_grid[low_index];
    NuclideGridPoint high = nuclide_grid[high_index];

    double f = (high.energy - energy) / (high.energy - low.energy);

    xs_vector[0] = high.total_xs - f * (high.total_xs - low.total_xs);
    xs_vector[1] = high.elastic_xs - f * (high.elastic_xs - low.elastic_xs);
    xs_vector[2] = high.absorbtion_xs - f * (high.absorbtion_xs - low.absorbtion_xs);
    xs_vector[3] = high.fission_xs - f * (high.fission_xs - low.fission_xs);
    xs_vector[4] = high.nu_fission_xs - f * (high.nu_fission_xs - low.nu_fission_xs);

}


void calculate_macro_xs(double energy, int mat, long n_isotopes, long n_gridpoints, int* num_nucs,
                        double* concs, double* energy_grid, int* index_grid, NuclideGridPoint* nuclide_grid,
                        int* materials, double* macro_xs_vector, int max_nuclides) {

    for( int i = 0; i < 5; ++i ) {
        macro_xs_vector[i] = 0;
    }

    long idx = grid_search(n_isotopes * n_gridpoints, energy, energy_grid);

    for ( int i = 0; i < num_nucs[mat]; i++ ) {
        double xs_vector[5];

        int p_nuc =materials[mat * max_nuclides + i];
        double conc = concs[mat * max_nuclides + i];

        calculate_micro_xs(energy, p_nuc, n_isotopes, n_gridpoints, energy_grid,
                           index_grid, nuclide_grid, idx, xs_vector);

        for (int j = 0; j < 5; ++j ) {
            macro_xs_vector[j] = xs_vector[j] * conc;
        }
    }
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

// picks a material based on a probabilistic distribution
int pick_mat( unsigned long * seed )
{
	double dist[12];
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
	double dist_cumulative[12];
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

	double roll = LCG_random_double(seed);

	// makes a pick based on the distro

	for( int i = 0; i < 12; i++ )
	{
        double running = 0;
        for ( int j = i; j > 0; j-- ) {
            running += dist[j];
        }
		if (roll < running) {
			return i;
		}
	}

	return 0;
}

} // anonymous namespace

void xs_lookup_krnl(int lookups,                    // how many lookups to do
                    int* num_nucs,                  // nuclides per material, 1-D array
                    double* concs,                   // nuclide concentrations per material, flattened 2-D
                    double* unionized_energy_grid,  // unionized energy grid, 1-D array
                    int* index_grid,                // index grid for unionized grid, flatenned 2-D
                    NuclideGridPoint* nuclide_grid, // XS data for nuclides, flatenned 2-D
                    int* materials,                 // nuclide indices for material definition, flatenned 2-D
                    int max_nuclides,               // max nuclides in any material
                    int* verification,              // verification array
                    long n_isotopes,                // number of isotopes in simulation
                    long n_gridpoints)              // number of grindpoints per isotope
{
    uint64_t seed = STARTING_SEED;

    for (int i = 0; i < lookups; ++i) {

        seed = fast_forward_LCG(seed, 2*i);
        double energy = LCG_random_double(&seed);
        int mat = pick_mat(&seed);

        double macro_xs_vector[5] = {0};

        calculate_macro_xs(
            energy, mat, n_isotopes, n_gridpoints, num_nucs, concs, unionized_energy_grid,
            index_grid, nuclide_grid, materials, macro_xs_vector, max_nuclides
        );

        double max = -1.0;
		int max_idx = 0;
		for(int j = 0; j < 5; ++j )
		{
			if( macro_xs_vector[j] > max )
			{
				max = macro_xs_vector[j];
				max_idx = j;
			}
		}
		verification[i] = max_idx+1;
    }
}