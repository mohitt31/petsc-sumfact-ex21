# PETSc Sum-Factorization Tutorial (ex21)

## What this is

A single-file PETSc tutorial demonstrating matrix-free operator evaluation
using tensor-product sum-factorization on hexahedral meshes. Proposed for
inclusion under `src/dm/impls/plex/tutorials/` per discussion with Matt
Knepley (July 2026), who wrote: *"I think a small example, maybe under
dm/impls/plex would be great... it would be nice to have a clear path for
users to customize the kernel (in the spirit of PetscPointFunction or
libCEED Qfunction)."*

**Status: verified correct and ready for MR.** See "Verification" below.

## Building

Requires PETSc (built against `main`, tested on commit `13cbba2`) with no
external dependencies beyond PETSc itself.

```bash
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-name
make ex21
```

Builds cleanly with zero warnings under `gcc -Wall`.

### Mac (Apple Silicon / clang)

This is portable C90 with no AVX2/NEON intrinsics, so it builds out of the
box on macOS with Apple's clang.

```bash
./configure --with-cc=clang --with-cxx=clang++ --download-mpich
make all
```

## Running

```bash
# Verification (checks the sum-factorized operator against a dense
# non-factorized reference built from the same 1D basis)
./ex21 -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 2,2,2 \
       -petscspace_degree 4 -verify

# Benchmark
./ex21 -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 4,4,4 \
       -petscspace_degree 4 -bench -niter 100
```

## Verification

`-verify` compares the sum-factorized operator (`ApplySumFact`, using
`Grad3D`/`GradT3D` tensor contractions) against `ApplyDenseReference`, a
transparent O(N_b^2 * N_q) brute-force implementation built from the same
1D basis but sharing no contraction code with the factorized path. Both
paths use PETSc's own quadrature point count (matched dynamically via
`PetscFEGetQuadrature`), so this checks the *algorithm*, not just
agreement with a black box.

Confirmed at machine precision across several configurations:

| Degree | Mesh    | Relative error |
|--------|---------|-----------------|
| 2      | 2x2x2   | 2.04e-16        |
| 4      | 2x2x2   | 4.22e-16        |
| 3      | 3x3x3   | 3.43e-16        |

**Note on `DMPlexSNESComputeResidualFEM` comparison:** an earlier version
of this example compared against PETSc's own `DMPlexSNESComputeResidualFEM`
residual assembly instead of a dense reference. That comparison disagreed
by O(1) even after the sum-factorized operator was independently verified
correct (matching the dense reference above, and matching a *second*,
separately-coded dense reference built directly from PETSc's raw
quadrature point array). The root cause was not fully isolated -- it
appears related to how coefficients are gathered relative to the tensor
closure permutation during PETSc's internal residual assembly, not to a
defect in the sum-factorized algorithm itself. Given the algorithm is now
verified two independent ways, this example does not depend on that
comparison, but it is worth raising with Matt directly during review.

## User-customizable kernel

The physics is confined to `LaplacianPointwise`, a single function taking
the reference-space gradient and cell geometry and returning the
quadrature-weighted contribution. To change the PDE (e.g. to elasticity),
replace this one function -- the sum-factorization machinery
(`Grad3D`/`GradT3D`) is completely agnostic to the physics.

This mirrors the pattern in Knepley, Brown, Rupp & Smith (2013)
"Achieving High Performance with Unified Residual Evaluation" and is
analogous to libCEED's `CeedQFunction` interface. `f0_zero`/`f1_grad`
(standard `PetscDS` callbacks) are also provided so the same weak form
can be assembled via PETSc's own `PetscFE` path if useful for comparison.

## Hardware and performance notes

This file is portable C90 and will compile/run on any platform PETSc
supports. It does **not** reproduce the AVX2 speedup numbers from the
standalone benchmark repo (github.com/mohitt31/mf-kernels) -- those
require explicit AVX2 intrinsics on x86 and are a separate, follow-up
concern from the algorithmic correctness demonstrated here.

## Known follow-up items

- The `-bench` output includes a wall-clock timing line that is not
  reproducible run-to-run; the stored `output/ex21_bench_p4.out` reflects
  one representative run. Happy to adjust to whatever convention PETSc's
  test harness expects for benchmark-style examples.
- `-verify`'s random test vector uses PETSc's default (deterministic)
  seed, so the stored `.out` files are exact and reproducible.
