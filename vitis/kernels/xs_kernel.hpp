#ifndef _KRNL_VADD_H_
#define _KRNL_VADD_H_

extern "C" {

struct NuclideGridPoint {
	double energy;
	double total_xs;
	double elastic_xs;
	double absorbtion_xs;
	double fission_xs;
	double nu_fission_xs;
};

void xs_lookup_krnl(int lookups,                    // how many lookups to do
                    int* num_nucs,                  // nuclides per material, 1-D array
                    double* conc,                   // nuclide concentrations per material, flattened 2-D
                    double* unionized_energy_grid,  // unionized energy grid, 1-D array
                    int* index_grid,                // index grid for unionized grid, flatenned 2-D
                    NuclideGridPoint* nuclide_grid, // XS data for nuclides, flatenned 2-D
                    int* materials,                 // nuclide indices for material definition, flatenned 2-D
                    int max_nuclides,               // max nuclides in any material
                    int* verification,              // verification array
                    long n_isotopes,                // number of isotopes in simulation
                    long n_gridpoints);             // number of grindpoints per isotope
} // extern "C"
#endif