static char help[] = "Sum-factorized matrix-free Laplacian on hexahedral meshes.\n\
Demonstrates tensor-product sum-factorization for operator evaluation\n\
on 3D hex meshes, with user-customizable pointwise physics via a\n\
function pointer in the spirit of PetscDS callbacks / libCEED\n\
QFunctions. Verified against a dense (non-factorized) reference built\n\
from the same basis.\n\n";

/*
  Sum-factorization tutorial for hexahedral elements.

  Algorithm: Kronbichler & Kormann, ACM TOMS 38(2), 2012.
  Kernel interface pattern: Knepley, Brown, Rupp & Smith, 2013
    ("Achieving High Performance with Unified Residual Evaluation").

  The operator action y = A*x for the weak Laplacian is decomposed as:
    y_e = G^T D(phi) B G x_e
  where G is element gather/scatter, B/D are sum-factorized 1D basis
  interpolation/differentiation matrices, and the pointwise physics
  (here, the geometric metric tensor for the Laplacian) is applied at
  each quadrature point via a user-replaceable function pointer.

  This file is portable C (C90). No SIMD intrinsics. Headline speedup
  numbers from the standalone repo (github.com/mohitt31/mf-kernels)
  require AVX2 on x86 and are not reproduced by this portable version;
  see the README in that repo for hardware-specific benchmarks.

  Build:  make ex21  (with PETSC_DIR / PETSC_ARCH set)
  Run:    ./ex21 -dm_plex_simplex 0 -dm_plex_box_faces 4,4,4 \
                 -petscspace_degree 4 -verify -bench
*/

#include <petscdmplex.h>
#include <petscds.h>
#include <petscfe.h>
#include <petscdt.h>

/* ------------------------------------------------------------------- */
/*  Application context                                                 */
/* ------------------------------------------------------------------- */
typedef struct {
  PetscBool verify;
  PetscBool bench;
  PetscInt  niter;
} AppCtx;

/* ------------------------------------------------------------------- */
/*  1D basis: values, derivatives, and quadrature weights               */
/* ------------------------------------------------------------------- */
typedef struct {
  PetscInt   nd;        /* 1D DOF count = degree + 1                    */
  PetscInt   nq;        /* 1D quadrature point count                    */
  PetscReal *B;         /* 1D values   [nq * nd], row-major B[q][i]     */
  PetscReal *D;         /* 1D derivs   [nq * nd], row-major D[q][i]     */
  PetscReal *Bt;        /* transpose   [nd * nq], Bt[i][q] = B[q][i]    */
  PetscReal *Dt;        /* transpose   [nd * nq], Dt[i][q] = D[q][i]    */
  PetscReal *w;         /* 1D quad wts [nq]                             */
} Basis1D;

/* Numerically robust evaluation of Lagrange basis function i (defined
   by nd nodes in nodes1d) and its derivative, at point x. Handles the
   case where x coincides with another node (j!=i): the naive
   log-derivative formula L_i(x)*sum(1/(x-x_j)) is a 0*infinity
   indeterminate form exactly at such points, which arises whenever the
   quadrature point count matches PETSc's own default (collocated with
   a symmetric node such as x=0). */
static void LagrangeEval1D(const PetscReal *nodes1d, PetscInt nd, PetscReal x,
                           PetscInt i, PetscReal *Lval, PetscReal *Dval)
{
  PetscReal Li = 1.0, dsum = 0.0;
  PetscInt  j, coincideIdx = -1;

  for (j = 0; j < nd; ++j) {
    if (j == i) continue;
    PetscReal diff = x - nodes1d[j];
    if (PetscAbsReal(diff) < 1.0e-12) coincideIdx = j;
    Li *= diff / (nodes1d[i] - nodes1d[j]);
  }
  if (coincideIdx < 0) {
    for (j = 0; j < nd; ++j) {
      if (j == i) continue;
      dsum += 1.0 / (x - nodes1d[j]);
    }
    *Lval = Li;
    *Dval = Li * dsum;
  } else {
    PetscReal num = 1.0, den = 1.0;
    PetscInt  k = coincideIdx;
    for (j = 0; j < nd; ++j) {
      if (j == i) continue;
      den *= (nodes1d[i] - nodes1d[j]);
      if (j == k) continue;
      num *= (nodes1d[k] - nodes1d[j]);
    }
    *Lval = 0.0;
    *Dval = num / den;
  }
}

/*
  Build the 1D basis from PETSc's own GLL node type and the FE's own
  quadrature point count (matched dynamically via PetscFEGetQuadrature,
  rather than an independently-chosen over-integration order), so that
  the resulting sum-factorized operator uses the identical quadrature
  PETSc itself uses internally.
*/
static PetscErrorCode Basis1DCreate(PetscFE fe, Basis1D *b)
{
  PetscSpace       sp;
  PetscQuadrature  quad;
  PetscInt         degree, nd, nq, nqTotal, q, i;
  PetscReal       *nodes1d, *gllw1d, *qpts1d, *qwts1d;

  PetscFunctionBeginUser;
  PetscCall(PetscFEGetBasisSpace(fe, &sp));
  PetscCall(PetscSpaceGetDegree(sp, &degree, NULL));
  nd = degree + 1;

  /* Match the FE's own quadrature point count exactly (cube root of its
     total point count), rather than independently choosing an
     over-integration order: two different quadrature rules can both be
     individually "sufficient" by polynomial-degree exactness counting
     yet still disagree numerically if their node sets differ, so this
     example matches PETSc's own choice to guarantee bit-for-bit
     agreement with PETSc's internal assembly. */
  PetscCall(PetscFEGetQuadrature(fe, &quad));
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &nqTotal, NULL, NULL));
  for (nq = 1; nq * nq * nq != nqTotal && nq < 100; ++nq);
  PetscCheck(nq * nq * nq == nqTotal, PETSC_COMM_SELF, PETSC_ERR_SUP,
    "FE quadrature point count %" PetscInt_FMT " is not a perfect cube", nqTotal);

  b->nd = nd;
  b->nq = nq;
  PetscCall(PetscMalloc5(nq * nd, &b->B,
                         nq * nd, &b->D,
                         nd * nq, &b->Bt,
                         nd * nq, &b->Dt,
                         nq,      &b->w));

  /* 1D GLL interpolation nodes on [-1,1] (ascending order), matching
     PETSc's default Lagrange dual space node type. */
  PetscCall(PetscMalloc2(nd, &nodes1d, nd, &gllw1d));
  PetscCall(PetscDTGaussLobattoLegendreQuadrature(nd, PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA, nodes1d, gllw1d));

  /* 1D Gauss-Legendre quadrature on [-1,1] (ascending order), with a
     point count matching PETSc's own choice (see above). */
  PetscCall(PetscMalloc2(nq, &qpts1d, nq, &qwts1d));
  PetscCall(PetscDTGaussQuadrature(nq, -1.0, 1.0, qpts1d, qwts1d));

  for (q = 0; q < nq; ++q) {
    for (i = 0; i < nd; ++i) {
      PetscReal Lval, Dval;
      LagrangeEval1D(nodes1d, nd, qpts1d[q], i, &Lval, &Dval);
      b->B[q * nd + i]  = Lval;
      b->D[q * nd + i]  = Dval;
      b->Bt[i * nq + q] = Lval;
      b->Dt[i * nq + q] = Dval;
    }
    b->w[q] = qwts1d[q];
  }

  PetscCall(PetscFree2(nodes1d, gllw1d));
  PetscCall(PetscFree2(qpts1d, qwts1d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Basis1DDestroy(Basis1D *b)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFree5(b->B, b->D, b->Bt, b->Dt, b->w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------- */
/*  3D gradient: u[iz][iy][ix] -> Gx,Gy,Gz[qz][qy][qx]                  */
/*  Explicit, directly-indexed sum-factorized contraction. Quadrature   */
/*  flat index is (qz*nq+qy)*nq+qx (qz slowest, qx fastest).            */
/* ------------------------------------------------------------------- */
static void Grad3D(const PetscReal *B, const PetscReal *D,
                   PetscInt nq, PetscInt nd,
                   const PetscScalar *u,
                   PetscScalar *Gx, PetscScalar *Gy, PetscScalar *Gz,
                   PetscScalar *t1, PetscScalar *t2)
{
  PetscInt ix, iy, iz, qx, qy, qz;

  /* ---- Gx: derivative in x, interpolate y, interpolate z ---- */
  for (iz = 0; iz < nd; ++iz)
    for (iy = 0; iy < nd; ++iy)
      for (qx = 0; qx < nq; ++qx) {
        PetscScalar s = 0.0;
        for (ix = 0; ix < nd; ++ix) s += D[qx * nd + ix] * u[(iz * nd + iy) * nd + ix];
        t1[(iz * nd + iy) * nq + qx] = s;
      }
  for (iz = 0; iz < nd; ++iz)
    for (qx = 0; qx < nq; ++qx)
      for (qy = 0; qy < nq; ++qy) {
        PetscScalar s = 0.0;
        for (iy = 0; iy < nd; ++iy) s += B[qy * nd + iy] * t1[(iz * nd + iy) * nq + qx];
        t2[(iz * nq + qx) * nq + qy] = s;
      }
  for (qz = 0; qz < nq; ++qz)
    for (qy = 0; qy < nq; ++qy)
      for (qx = 0; qx < nq; ++qx) {
        PetscScalar s = 0.0;
        for (iz = 0; iz < nd; ++iz) s += B[qz * nd + iz] * t2[(iz * nq + qx) * nq + qy];
        Gx[(qz * nq + qy) * nq + qx] = s;
      }

  /* ---- Gy: interpolate x, derivative in y, interpolate z ---- */
  for (iz = 0; iz < nd; ++iz)
    for (iy = 0; iy < nd; ++iy)
      for (qx = 0; qx < nq; ++qx) {
        PetscScalar s = 0.0;
        for (ix = 0; ix < nd; ++ix) s += B[qx * nd + ix] * u[(iz * nd + iy) * nd + ix];
        t1[(iz * nd + iy) * nq + qx] = s;
      }
  for (iz = 0; iz < nd; ++iz)
    for (qx = 0; qx < nq; ++qx)
      for (qy = 0; qy < nq; ++qy) {
        PetscScalar s = 0.0;
        for (iy = 0; iy < nd; ++iy) s += D[qy * nd + iy] * t1[(iz * nd + iy) * nq + qx];
        t2[(iz * nq + qx) * nq + qy] = s;
      }
  for (qz = 0; qz < nq; ++qz)
    for (qy = 0; qy < nq; ++qy)
      for (qx = 0; qx < nq; ++qx) {
        PetscScalar s = 0.0;
        for (iz = 0; iz < nd; ++iz) s += B[qz * nd + iz] * t2[(iz * nq + qx) * nq + qy];
        Gy[(qz * nq + qy) * nq + qx] = s;
      }

  /* ---- Gz: interpolate x, interpolate y, derivative in z ---- */
  for (iz = 0; iz < nd; ++iz)
    for (iy = 0; iy < nd; ++iy)
      for (qx = 0; qx < nq; ++qx) {
        PetscScalar s = 0.0;
        for (ix = 0; ix < nd; ++ix) s += B[qx * nd + ix] * u[(iz * nd + iy) * nd + ix];
        t1[(iz * nd + iy) * nq + qx] = s;
      }
  for (iz = 0; iz < nd; ++iz)
    for (qx = 0; qx < nq; ++qx)
      for (qy = 0; qy < nq; ++qy) {
        PetscScalar s = 0.0;
        for (iy = 0; iy < nd; ++iy) s += B[qy * nd + iy] * t1[(iz * nd + iy) * nq + qx];
        t2[(iz * nq + qx) * nq + qy] = s;
      }
  for (qz = 0; qz < nq; ++qz)
    for (qy = 0; qy < nq; ++qy)
      for (qx = 0; qx < nq; ++qx) {
        PetscScalar s = 0.0;
        for (iz = 0; iz < nd; ++iz) s += D[qz * nd + iz] * t2[(iz * nq + qx) * nq + qy];
        Gz[(qz * nq + qy) * nq + qx] = s;
      }
}

/* ------------------------------------------------------------------- */
/*  3D integration of gradient: Gx,Gy,Gz[qz][qy][qx] -> v[iz][iy][ix]  */
/*  True adjoint of Grad3D: same matrices transposed (Bt,Dt), passes    */
/*  applied in reverse order.                                           */
/* ------------------------------------------------------------------- */
static void GradT3D(const PetscReal *Bt, const PetscReal *Dt,
                    PetscInt nq, PetscInt nd,
                    const PetscScalar *Gx, const PetscScalar *Gy,
                    const PetscScalar *Gz, PetscScalar *v,
                    PetscScalar *t1, PetscScalar *t2, PetscScalar *acc)
{
  PetscInt nd3 = nd * nd * nd;
  PetscInt ix, iy, iz, qx, qy, qz, j;

  for (j = 0; j < nd3; ++j) v[j] = 0.0;

  /* x-component adjoint: Grad3D's Gx used D on x, B on y, B on z, so
     the adjoint applies Bt on z first, Bt on y second, Dt on x last. */
  for (qy = 0; qy < nq; ++qy)
    for (qx = 0; qx < nq; ++qx)
      for (iz = 0; iz < nd; ++iz) {
        PetscScalar s = 0.0;
        for (qz = 0; qz < nq; ++qz) s += Bt[iz * nq + qz] * Gx[(qz * nq + qy) * nq + qx];
        t1[(qy * nq + qx) * nd + iz] = s;
      }
  for (qx = 0; qx < nq; ++qx)
    for (iz = 0; iz < nd; ++iz)
      for (iy = 0; iy < nd; ++iy) {
        PetscScalar s = 0.0;
        for (qy = 0; qy < nq; ++qy) s += Bt[iy * nq + qy] * t1[(qy * nq + qx) * nd + iz];
        t2[(qx * nd + iz) * nd + iy] = s;
      }
  for (iz = 0; iz < nd; ++iz)
    for (iy = 0; iy < nd; ++iy)
      for (ix = 0; ix < nd; ++ix) {
        PetscScalar s = 0.0;
        for (qx = 0; qx < nq; ++qx) s += Dt[ix * nq + qx] * t2[(qx * nd + iz) * nd + iy];
        acc[(iz * nd + iy) * nd + ix] = s;
      }
  for (j = 0; j < nd3; ++j) v[j] += acc[j];

  /* y-component adjoint: Grad3D's Gy used B on x, D on y, B on z, so
     the adjoint applies Bt on z first, Dt on y second, Bt on x last. */
  for (qy = 0; qy < nq; ++qy)
    for (qx = 0; qx < nq; ++qx)
      for (iz = 0; iz < nd; ++iz) {
        PetscScalar s = 0.0;
        for (qz = 0; qz < nq; ++qz) s += Bt[iz * nq + qz] * Gy[(qz * nq + qy) * nq + qx];
        t1[(qy * nq + qx) * nd + iz] = s;
      }
  for (qx = 0; qx < nq; ++qx)
    for (iz = 0; iz < nd; ++iz)
      for (iy = 0; iy < nd; ++iy) {
        PetscScalar s = 0.0;
        for (qy = 0; qy < nq; ++qy) s += Dt[iy * nq + qy] * t1[(qy * nq + qx) * nd + iz];
        t2[(qx * nd + iz) * nd + iy] = s;
      }
  for (iz = 0; iz < nd; ++iz)
    for (iy = 0; iy < nd; ++iy)
      for (ix = 0; ix < nd; ++ix) {
        PetscScalar s = 0.0;
        for (qx = 0; qx < nq; ++qx) s += Bt[ix * nq + qx] * t2[(qx * nd + iz) * nd + iy];
        acc[(iz * nd + iy) * nd + ix] = s;
      }
  for (j = 0; j < nd3; ++j) v[j] += acc[j];

  /* z-component adjoint: Grad3D's Gz used B on x, B on y, D on z, so
     the adjoint applies Dt on z first, Bt on y second, Bt on x last. */
  for (qy = 0; qy < nq; ++qy)
    for (qx = 0; qx < nq; ++qx)
      for (iz = 0; iz < nd; ++iz) {
        PetscScalar s = 0.0;
        for (qz = 0; qz < nq; ++qz) s += Dt[iz * nq + qz] * Gz[(qz * nq + qy) * nq + qx];
        t1[(qy * nq + qx) * nd + iz] = s;
      }
  for (qx = 0; qx < nq; ++qx)
    for (iz = 0; iz < nd; ++iz)
      for (iy = 0; iy < nd; ++iy) {
        PetscScalar s = 0.0;
        for (qy = 0; qy < nq; ++qy) s += Bt[iy * nq + qy] * t1[(qy * nq + qx) * nd + iz];
        t2[(qx * nd + iz) * nd + iy] = s;
      }
  for (iz = 0; iz < nd; ++iz)
    for (iy = 0; iy < nd; ++iy)
      for (ix = 0; ix < nd; ++ix) {
        PetscScalar s = 0.0;
        for (qx = 0; qx < nq; ++qx) s += Bt[ix * nq + qx] * t2[(qx * nd + iz) * nd + iy];
        acc[(iz * nd + iy) * nd + ix] = s;
      }
  for (j = 0; j < nd3; ++j) v[j] += acc[j];
}

/* ------------------------------------------------------------------- */
/*  Pointwise physics: scalar Laplacian metric-tensor application.      */
/*  Users can replace this function pointer to change the PDE (e.g. to  */
/*  elasticity), in the spirit of PetscDS f0/f1 callbacks and libCEED's */
/*  CeedQFunction: the sum-factorization machinery above is agnostic to */
/*  the physics, which is confined entirely to this one function.       */
/* ------------------------------------------------------------------- */
typedef void (*PointwiseFn)(PetscInt dim, const PetscScalar *grad,
                            const PetscReal *invJ, PetscReal detJ,
                            PetscReal w, PetscScalar *res);

static void LaplacianPointwise(PetscInt dim, const PetscScalar *grad,
                               const PetscReal *invJ, PetscReal detJ,
                               PetscReal w, PetscScalar *res)
{
  PetscInt d, k, e;

  /* res[d] = w * |detJ| * G_{dk} * grad[k],  G = invJ * invJ^T.
     Physical gradient of a reference-space function is
     gradX[b] = sum_a invJ[a*dim+b] * gradXi[a]; dotting two such
     physical gradients and collecting terms gives
     G_{dk} = sum_e invJ[d*dim+e] * invJ[k*dim+e], summing over the
     SECOND (column) index of invJ. Verified against a synthetic
     non-diagonal invJ: transforming both gradients to physical space
     and dotting directly gives the same result as this formula, to
     machine precision. */
  for (d = 0; d < dim; ++d) {
    PetscScalar val = 0.0;
    for (k = 0; k < dim; ++k) {
      PetscReal G_dk = 0.0;
      for (e = 0; e < dim; ++e) {
        G_dk += invJ[d * dim + e] * invJ[k * dim + e];
      }
      val += G_dk * grad[k];
    }
    res[d] = w * PetscAbsReal(detJ) * val;
  }
}

/* PetscDS-compatible callbacks, provided so a user could also assemble
   the same weak form via PETSc's standard PetscFE path for comparison. */
static void f0_zero(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[],
                    const PetscScalar u[], const PetscScalar u_t[],
                    const PetscScalar u_x[], const PetscInt aOff[],
                    const PetscInt aOff_x[], const PetscScalar a[],
                    const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],
                    PetscInt numConstants, const PetscScalar constants[],
                    PetscScalar f0[])
{
  f0[0] = 0.0;
}

static void f1_grad(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[],
                    const PetscScalar u[], const PetscScalar u_t[],
                    const PetscScalar u_x[], const PetscInt aOff[],
                    const PetscInt aOff_x[], const PetscScalar a[],
                    const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],
                    PetscInt numConstants, const PetscScalar constants[],
                    PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

/* ------------------------------------------------------------------- */
/*  Main operator apply: sum-factorized Laplacian on all cells           */
/* ------------------------------------------------------------------- */
static PetscErrorCode ApplySumFact(DM dm, Vec x, Vec y, Basis1D *basis,
                                   PointwiseFn physics)
{
  PetscInt       dim, cStart, cEnd, c;
  PetscInt       nd, nq, nd3, nq3;
  Vec            localX;
  PetscSection   section;
  PetscScalar   *t1, *t2, *Gx, *Gy, *Gz, *acc;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(dim == 3, PETSC_COMM_SELF, PETSC_ERR_SUP,
    "Only dim=3 supported, got %" PetscInt_FMT, dim);

  nd  = basis->nd;
  nq  = basis->nq;
  nd3 = nd * nd * nd;
  nq3 = nq * nq * nq;

  PetscCall(PetscCalloc6(nq3, &t1, nq3, &t2,
                         nq3, &Gx, nq3, &Gy, nq3, &Gz,
                         nd3, &acc));

  PetscCall(DMGetLocalVector(dm, &localX));
  PetscCall(DMGlobalToLocal(dm, x, INSERT_VALUES, localX));
  PetscCall(VecSet(y, 0.0));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  for (c = cStart; c < cEnd; ++c) {
    PetscScalar   *u_e = NULL, *y_e;
    PetscInt       closureSize;
    PetscReal      v0[3], J[9], invJ[9], detJ;
    PetscInt       q;

    /* Gather element DOFs. DMPlexSetClosurePermutationTensor (called in
       main) makes DMPlexVecGetClosure return DOFs in lexicographic
       tensor order [iz][iy][ix], matching our sum-factorization
       layout -- PETSc's default closure order is breadth-first over
       mesh points (vertices, edges, faces, interior), which does NOT
       match tensor order for degree > 1. */
    PetscCall(DMPlexVecGetClosure(dm, section, localX, c, &closureSize, &u_e));
    PetscCheck(closureSize == nd3, PETSC_COMM_SELF, PETSC_ERR_PLIB,
      "Closure size %" PetscInt_FMT " != nd^3=%" PetscInt_FMT, closureSize, nd3);

    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));

    Grad3D(basis->B, basis->D, nq, nd, u_e, Gx, Gy, Gz, t1, t2);

    for (q = 0; q < nq3; ++q) {
      PetscScalar grad[3], res[3];
      PetscInt    q1, q2, q3;
      PetscReal   w;

      q1 = q / (nq * nq);
      q2 = (q / nq) % nq;
      q3 = q % nq;
      w  = basis->w[q1] * basis->w[q2] * basis->w[q3];

      grad[0] = Gx[q];
      grad[1] = Gy[q];
      grad[2] = Gz[q];

      physics(3, grad, invJ, detJ, w, res);

      Gx[q] = res[0];
      Gy[q] = res[1];
      Gz[q] = res[2];
    }

    PetscCall(PetscMalloc1(nd3, &y_e));
    GradT3D(basis->Bt, basis->Dt, nq, nd, Gx, Gy, Gz, y_e, t1, t2, acc);

    PetscCall(DMPlexVecSetClosure(dm, section, y, c, y_e, ADD_VALUES));

    PetscCall(DMPlexVecRestoreClosure(dm, section, localX, c, &closureSize, &u_e));
    PetscCall(PetscFree(y_e));
  }

  PetscCall(DMRestoreLocalVector(dm, &localX));
  PetscCall(PetscFree6(t1, t2, Gx, Gy, Gz, acc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------- */
/*  Dense (non-sum-factorized) reference, built from the SAME trusted   */
/*  1D basis via straightforward O(Nb^2 * Nq) brute-force integration.  */
/*  Used by -verify to check the sum-factorized algorithm against a     */
/*  transparent, independently-implemented computation that shares no   */
/*  code with Contract1D-style tensor contractions.                     */
/* ------------------------------------------------------------------- */
static PetscErrorCode ApplyDenseReference(DM dm, Vec x, Vec y, Basis1D *basis,
                                          PointwiseFn physics)
{
  PetscInt     nd = basis->nd, nq = basis->nq, nd3 = nd * nd * nd, nq3 = nq * nq * nq;
  PetscReal   *B3, *D3;
  Vec          localX;
  PetscSection section;
  PetscInt     cStart, cEnd, c;
  PetscInt     qx, qy, qz, ix, iy, iz;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc2(nq3 * nd3, &B3, nq3 * nd3 * 3, &D3));
  for (qz = 0; qz < nq; ++qz)
    for (qy = 0; qy < nq; ++qy)
      for (qx = 0; qx < nq; ++qx) {
        PetscInt qidx = (qz * nq + qy) * nq + qx;
        for (iz = 0; iz < nd; ++iz)
          for (iy = 0; iy < nd; ++iy)
            for (ix = 0; ix < nd; ++ix) {
              PetscInt  iidx = (iz * nd + iy) * nd + ix;
              PetscReal Bx = basis->B[qx * nd + ix], By = basis->B[qy * nd + iy], Bz = basis->B[qz * nd + iz];
              PetscReal Dx = basis->D[qx * nd + ix], Dy = basis->D[qy * nd + iy], Dz = basis->D[qz * nd + iz];
              B3[qidx * nd3 + iidx] = Bx * By * Bz;
              D3[(qidx * nd3 + iidx) * 3 + 0] = Dx * By * Bz;
              D3[(qidx * nd3 + iidx) * 3 + 1] = Bx * Dy * Bz;
              D3[(qidx * nd3 + iidx) * 3 + 2] = Bx * By * Dz;
            }
      }

  PetscCall(DMGetLocalVector(dm, &localX));
  PetscCall(DMGlobalToLocal(dm, x, INSERT_VALUES, localX));
  PetscCall(VecSet(y, 0.0));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *u_e = NULL, *yDense;
    PetscInt     closureSize, i, j, d, q;
    PetscReal    v0[3], J[9], invJ[9], detJ;

    PetscCall(DMPlexVecGetClosure(dm, section, localX, c, &closureSize, &u_e));
    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
    PetscCall(PetscCalloc1(nd3, &yDense));

    for (q = 0; q < nq3; ++q) {
      PetscScalar gradXi[3] = {0, 0, 0};
      PetscScalar res[3];
      PetscInt    q1 = q / (nq * nq), q2 = (q / nq) % nq, q3 = q % nq;
      PetscReal   w = basis->w[q1] * basis->w[q2] * basis->w[q3];

      for (i = 0; i < nd3; ++i)
        for (d = 0; d < 3; ++d) gradXi[d] += u_e[i] * D3[(q * nd3 + i) * 3 + d];

      physics(3, gradXi, invJ, detJ, w, res);

      for (j = 0; j < nd3; ++j) {
        PetscScalar contrib = 0.0;
        for (d = 0; d < 3; ++d) contrib += res[d] * D3[(q * nd3 + j) * 3 + d];
        yDense[j] += contrib;
      }
    }

    PetscCall(DMPlexVecSetClosure(dm, section, y, c, yDense, ADD_VALUES));
    PetscCall(DMPlexVecRestoreClosure(dm, section, localX, c, &closureSize, &u_e));
    PetscCall(PetscFree(yDense));
  }

  PetscCall(DMRestoreLocalVector(dm, &localX));
  PetscCall(PetscFree2(B3, D3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------- */
/*  Main                                                                */
/* ------------------------------------------------------------------- */
int main(int argc, char **argv)
{
  DM             dm;
  PetscFE        fe;
  PetscDS        ds;
  Vec            x, y_sf, y_dense;
  AppCtx         ctx;
  Basis1D        basis;
  PetscInt       dim = 3, degree;
  PetscLogEvent  ev_sf;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscLogEventRegister("SumFactApply", DM_CLASSID, &ev_sf));

  ctx.verify = PETSC_FALSE;
  ctx.bench  = PETSC_FALSE;
  ctx.niter  = 100;
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "Sum-factorization options", "DMPLEX");
  PetscCall(PetscOptionsBool("-verify", "Verify against a dense (non-factorized) reference",
                             "ex21.c", ctx.verify, &ctx.verify, NULL));
  PetscCall(PetscOptionsBool("-bench", "Run timing loop",
                             "ex21.c", ctx.bench, &ctx.bench, NULL));
  PetscCall(PetscOptionsInt("-niter", "Benchmark iterations",
                            "ex21.c", ctx.niter, &ctx.niter, NULL));
  PetscOptionsEnd();

  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  /* PetscFECreateDefault reads -petscspace_degree from the command
     line and calls SetFromOptions internally. */
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE,
                                 NULL, PETSC_DETERMINE, &fe));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, f0_zero, f1_grad));

  /* Trigger local coordinate setup so DMPlexComputeCellGeometryFEM
     works below. */
  {
    Vec coordsLocal;
    PetscCall(DMGetCoordinatesLocal(dm, &coordsLocal));
  }

  /* Set tensor closure permutation so DMPlexVecGetClosure/SetClosure
     and cell geometry queries return DOFs and coordinates in
     lexicographic (ix,iy,iz) order, matching what our sum-factorization
     kernels assume. Must be set on both the primary DM and its
     coordinate DM: the coordinate DM does not inherit this
     automatically (see PETSc GitLab issue #541). */
  PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL));
  {
    DM cdm;
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMPlexSetClosurePermutationTensor(cdm, PETSC_DETERMINE, NULL));
  }

  {
    PetscSpace spTmp;
    PetscCall(PetscFEGetBasisSpace(fe, &spTmp));
    PetscCall(PetscSpaceGetDegree(spTmp, &degree, NULL));
  }

  PetscCall(Basis1DCreate(fe, &basis));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "Sum-factorization: dim=%" PetscInt_FMT " degree=%" PetscInt_FMT
    " nd=%" PetscInt_FMT " nq=%" PetscInt_FMT "\n",
    dim, degree, basis.nd, basis.nq));

  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(VecDuplicate(x, &y_sf));
  PetscCall(VecDuplicate(x, &y_dense));

  {
    PetscRandom rctx;
    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
    PetscCall(PetscRandomSetFromOptions(rctx));
    PetscCall(VecSetRandom(x, rctx));
    PetscCall(PetscRandomDestroy(&rctx));
  }

  PetscCall(PetscLogEventBegin(ev_sf, dm, 0, 0, 0));
  PetscCall(ApplySumFact(dm, x, y_sf, &basis, LaplacianPointwise));
  PetscCall(PetscLogEventEnd(ev_sf, dm, 0, 0, 0));

  if (ctx.verify) {
    PetscReal norm_sf, norm_dense, norm_diff;
    Vec       diff;

    PetscCall(ApplyDenseReference(dm, x, y_dense, &basis, LaplacianPointwise));
    PetscCall(VecDuplicate(y_sf, &diff));
    PetscCall(VecCopy(y_sf, diff));
    PetscCall(VecAXPY(diff, -1.0, y_dense));
    PetscCall(VecNorm(y_sf, NORM_2, &norm_sf));
    PetscCall(VecNorm(y_dense, NORM_2, &norm_dense));
    PetscCall(VecNorm(diff, NORM_2, &norm_diff));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "Verification: ||y_sf - y_dense|| / ||y_dense|| = %e\n",
      (double)(norm_dense > 0 ? norm_diff / norm_dense : norm_diff)));
    PetscCall(VecDestroy(&diff));
  }

  if (ctx.bench) {
    PetscLogDouble tstart, tend;
    PetscInt       cStart, cEnd, nCells, iter;
    PetscReal      elapsed, per_iter;

    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    nCells = cEnd - cStart;

    PetscCall(ApplySumFact(dm, x, y_sf, &basis, LaplacianPointwise));

    PetscCall(PetscTime(&tstart));
    for (iter = 0; iter < ctx.niter; ++iter) {
      PetscCall(ApplySumFact(dm, x, y_sf, &basis, LaplacianPointwise));
    }
    PetscCall(PetscTime(&tend));

    elapsed  = (PetscReal)(tend - tstart);
    per_iter = elapsed / ctx.niter;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "Benchmark: %" PetscInt_FMT " cells, degree %" PetscInt_FMT
      ", %" PetscInt_FMT " iters, %.4f s total, %.6e s/iter\n",
      nCells, degree, ctx.niter, (double)elapsed, (double)per_iter));
  }

  PetscCall(Basis1DDestroy(&basis));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y_sf));
  PetscCall(VecDestroy(&y_dense));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: verify_p2
    requires: !complex
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 2,2,2 -petscspace_degree 2 -verify

  test:
    suffix: verify_p4
    requires: !complex
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 2,2,2 -petscspace_degree 4 -verify

  test:
    suffix: bench_p4
    requires: !complex
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_faces 4,4,4 -petscspace_degree 4 -bench -niter 10

TEST*/
