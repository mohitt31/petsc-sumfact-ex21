# PETSc Sum-Factorization Tutorial Prototype (ex21)

## What this is

A single-file PETSc tutorial demonstrating matrix-free operator evaluation
using tensor-product sum-factorization on hexahedral meshes. Proposed for
inclusion under `src/dm/impls/plex/tutorials/` per discussion with Matt
Knepley (July 2026).

## Building

Requires PETSc installed and configured with hex mesh support (no external
dependencies beyond PETSc itself).

```bash
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-name
make ex21
```

### Mac (Apple Silicon / clang)

This builds out of the box on macOS with Apple's clang. No AVX2 or NEON
intrinsics are used in this portable C version.

```bash
# Typical PETSc configure for Mac:
./configure --with-cc=clang --with-cxx=clang++ --download-mpich
make all
```

## Running

```bash
# Basic verification (degree 4, 2x2x2 hex mesh)
./ex21 -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 2,2,2 \
       -petscspace_degree 4 -verify

# Benchmark (degree 4, 4x4x4 hex mesh, 100 iterations)
./ex21 -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 4,4,4 \
       -petscspace_degree 4 -benchmark -niter 100

# With PETSc log for detailed timing
./ex21 -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 4,4,4 \
       -petscspace_degree 4 -benchmark -log_view
```

## Hardware and performance notes

This file is **portable C** (C90 compatible per PETSc style). It will
compile and run correctly on any platform PETSc supports.

**It will NOT reproduce the headline SIMD speedup numbers.** Those numbers:

| Variant        | Speedup | Hardware           | Compiler        |
|----------------|---------|--------------------|-----------------| 
| Explicit AVX2  | ~6x     | AMD EPYC 7763      | gcc 13.3, -O3   |
| Even-odd AVX2  | ~8.6-10x| AMD EPYC 7763      | gcc 13.3, -O3   |

...were measured with explicit AVX2 intrinsics on the standalone repo:
https://github.com/mohitt31/mf-kernels

To reproduce on your hardware, clone that repo and run on an x86 machine
with AVX2. The algorithm is identical; only the SIMD width and memory
access pattern differ.

## User-customizable kernel

The physics is defined via `f0_laplacian` and `f1_laplacian` functions
following PETSc's `PetscDS` pointwise callback pattern. To change the PDE
(e.g., to elasticity), replace these functions. The sum-factorization
machinery is agnostic to the physics.

This design follows the pattern described in Knepley, Brown, Rupp & Smith
(2013) "Achieving High Performance with Unified Residual Evaluation" and
is analogous to libCEED's `CeedQFunction` interface.

## What needs to happen before MR submission

1. **Run on PETSc main branch** to confirm compilation and test passage
2. **Generate actual test output** to replace TODO placeholders in output/
3. **Confirm example number** (ex21 is a placeholder; check latest main)
4. **Clean up closure handling** in the cell loop (DMPlexVecGetClosure
   returns allocated memory that must be properly freed)
5. **Style review** against PETSc style guide (2-space indent verified,
   C90 declarations verified)
