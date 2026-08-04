static char help[] = "Sum-factorized matrix-free Laplacian on hexahedral meshes.\n\
Demonstrates tensor-product sum-factorization for operator evaluation\n\
on 3D hex meshes with user-customizable pointwise physics via PetscDS\n\
callbacks, verified against PetscFE's standard tabulation path.\n\n";

/*
  Sum-factorization tutorial for hexahedral elements.

  Algorithm: Kronbichler & Kormann, ACM TOMS 38(2), 2012.
  User kernel pattern: Knepley, Brown, Rupp & Smith, 2013
    ("Achieving High Performance with Unified Residual Evaluation").

  The operator action y = A*x for the weak Laplacian is decomposed as:
    y_e = G^T B^T D B G x_e
  where G is element gather/scatter, B is sum-factorized basis evaluation,
  and D is the pointwise quadrature operation (geometry + physics).

  This file is portable C (C90). No SIMD intrinsics. Headline speedup
  numbers from the standalone repo (github.com/mohitt31/mf-kernels)
  require AVX2 on x86 and are not reproduced by this portable version.

  Build:  make ex21  (with PETSC_DIR / PETSC_ARCH set)
  Run:    ./ex21 -dm_plex_simplex 0 -dm_plex_box_faces 4,4,4 \
                 -petscspace_degree 4 -verify -bench
*/

#include <petscdmplex.h>
#include <petscds.h>
#include <petscfe.h>

/* ------------------------------------------------------------------- */
/*  Application context                                                 */
/* ------------------------------------------------------------------- */
typedef struct {
  PetscBool verify;
  PetscBool bench;
  PetscInt  niter;
} AppCtx;

/* DIAGNOSTIC: project the physical X-coordinate onto the FE space.
   Used to empirically determine the closure array's flat-index-to-axis
   mapping: since the projected field's DOF value at any node equals that
   node's physical X-coordinate, printing the closure array reveals
   exactly which flat index varies with X, Y, and Z. */
static PetscErrorCode CoordXFunc(PetscInt dim, PetscReal time, const PetscReal xc[],
                                 PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = xc[0];
  return 0;
}

/* ------------------------------------------------------------------- */
/*  1D basis and even-odd precomputation                                */
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

/*
  Extract 1D basis data from PetscFE's tensor-product tabulation.

  For a tensor-product element on a hex, the 3D basis is:
    B3D[(q1*nq+q2)*nq+q3][(i1*nd+i2)*nd+i3] = B1d[q1][i1] * B1d[q2][i2] * B1d[q3][i3]

  We extract B1d by reading the slice q1=0,q2=0,i1=0,i2=0:
    B3D[q3][i3] = B1d[0][0]^2 * B1d[q3][i3]

  Then normalize by B1d[0][0]^2 = B3D[0][0].
*/
static PetscErrorCode Basis1DCreate(PetscFE fe, Basis1D *b)
{
  PetscTabulation tab;
  PetscInt        Nb, Nq, nd, nq, q, i;
  PetscReal       norm;

  PetscFunctionBeginUser;
  PetscCall(PetscFEGetCellTabulation(fe, 1, &tab));
  Nb = tab->Nb;
  Nq = tab->Np;

  /* Cube root to get 1D sizes */
  for (nd = 1; nd * nd * nd != Nb && nd < 100; ++nd);
  PetscCheck(nd * nd * nd == Nb, PETSC_COMM_SELF, PETSC_ERR_SUP,
    "Nb=%" PetscInt_FMT " is not a perfect cube", Nb);
  for (nq = 1; nq * nq * nq != Nq && nq < 100; ++nq);
  PetscCheck(nq * nq * nq == Nq, PETSC_COMM_SELF, PETSC_ERR_SUP,
    "Nq=%" PetscInt_FMT " is not a perfect cube", Nq);

  b->nd = nd;
  b->nq = nq;
  PetscCall(PetscMalloc5(nq * nd, &b->B,
                         nq * nd, &b->D,
                         nd * nq, &b->Bt,
                         nd * nq, &b->Dt,
                         nq,      &b->w));

  /* Extract 1D basis from 3D tabulation.
     B3D[0][0] = B1d[0][0]^3, so B1d[0][0] = cbrt(B3D[0][0]).
     B3D[q3][i3] = B1d[0][0]^2 * B1d[q3][i3] at q1=q2=0, i1=i2=0.
     Therefore B1d[q][i] = B3D[q][i] / B1d[0][0]^2 = B3D[q][i] / cbrt(B3D[0][0])^2 */
  {
    PetscReal b3d00 = tab->T[0][0];
    PetscCheck(PetscAbsReal(b3d00) > 1.0e-14, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
      "B3D[0][0] is zero; basis is not tensor-product with nonzero B1d[0][0]");
    norm = PetscPowReal(PetscAbsReal(b3d00), 2.0 / 3.0);
    if (b3d00 < 0) norm = -norm; /* preserve sign */
  }
  for (q = 0; q < nq; ++q) {
    for (i = 0; i < nd; ++i) {
      b->B[q * nd + i]  = tab->T[0][q * Nb + i] / norm;
      /* Derivative: component 2 (z-direction) of the gradient */
      b->D[q * nd + i]  = tab->T[1][(q * Nb + i) * 3 + 2] / norm;
      b->Bt[i * nq + q] = b->B[q * nd + i];
      b->Dt[i * nq + q] = b->D[q * nd + i];
    }
  }

  /* Reorder basis-function index from PETSc's internal FE-tabulation
     order into lexicographic (ascending physical-position) order, to
     match the order DMPlexVecGetClosure returns after
     DMPlexSetClosurePermutationTensor is applied. These two orders are
     NOT the same: empirically, PETSc's native FE tabulation index for a
     degree-2 element puts the interior/center node at index 0 and the
     vertices at indices 1,2, while the tensor-permuted closure array
     returns vertex(left), interior(center), vertex(right) in that
     lexicographic order. Without this correction, B1d/D1d and the
     closure array u_e disagree on what index i means, producing a
     result that is internally self-consistent but wrong relative to
     PETSc's own reference assembly.

     The permutation is found generically (works for any degree): each
     1D Lagrange basis function attains its largest value at the
     quadrature point nearest its own node, so sorting basis indices by
     argmax_q(B1d[q][i]) recovers position order without needing to know
     PETSc's internal native ordering convention analytically. */
  {
    PetscInt  *permNativeOfLex, *peakQ;
    PetscReal *newB, *newD;
    PetscInt   ii, qq;

    PetscCall(PetscMalloc2(nd, &permNativeOfLex, nd, &peakQ));
    for (ii = 0; ii < nd; ++ii) {
      PetscInt  bestQ   = 0;
      PetscReal bestVal = b->B[0 * nd + ii];
      for (qq = 1; qq < nq; ++qq) {
        if (b->B[qq * nd + ii] > bestVal) { bestVal = b->B[qq * nd + ii]; bestQ = qq; }
      }
      peakQ[ii]           = bestQ;
      permNativeOfLex[ii] = ii;
    }
    /* Selection sort native indices by ascending peakQ */
    for (ii = 0; ii < nd - 1; ++ii) {
      PetscInt minIdx = ii, jj, tmp;
      for (jj = ii + 1; jj < nd; ++jj) {
        if (peakQ[permNativeOfLex[jj]] < peakQ[permNativeOfLex[minIdx]]) minIdx = jj;
      }
      tmp                    = permNativeOfLex[ii];
      permNativeOfLex[ii]    = permNativeOfLex[minIdx];
      permNativeOfLex[minIdx] = tmp;
    }

    PetscCall(PetscMalloc2(nq * nd, &newB, nq * nd, &newD));
    for (qq = 0; qq < nq; ++qq) {
      for (ii = 0; ii < nd; ++ii) {
        PetscInt nativeI    = permNativeOfLex[ii];
        newB[qq * nd + ii] = b->B[qq * nd + nativeI];
        newD[qq * nd + ii] = b->D[qq * nd + nativeI];
      }
    }
    PetscCall(PetscArraycpy(b->B, newB, nq * nd));
    PetscCall(PetscArraycpy(b->D, newD, nq * nd));
    for (qq = 0; qq < nq; ++qq) {
      for (ii = 0; ii < nd; ++ii) {
        b->Bt[ii * nq + qq] = b->B[qq * nd + ii];
        b->Dt[ii * nq + qq] = b->D[qq * nd + ii];
      }
    }
    PetscCall(PetscFree2(newB, newD));
    PetscCall(PetscFree2(permNativeOfLex, peakQ));
  }

  /* Extract 1D quadrature weights from the 3D quadrature.
     w3D[(q1*nq+q2)*nq+q3] = w1d[q1]*w1d[q2]*w1d[q3]
     At q1=0,q2=0: w3D[q3] = w1d[0]^2 * w1d[q3]
     So w1d[q3] = w3D[q3] / w3D[0] * w1d[0], and w1d[0] = cbrt(w3D[0]) */
  {
    PetscQuadrature quad;
    const PetscReal *wts;
    PetscInt         nq_check;
    PetscReal        w0;

    PetscCall(PetscFEGetQuadrature(fe, &quad));
    PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &nq_check, NULL, &wts));
    PetscCheck(nq_check == Nq, PETSC_COMM_SELF, PETSC_ERR_PLIB,
      "Quadrature count mismatch");
    w0 = PetscPowReal(wts[0], 1.0 / 3.0);
    for (q = 0; q < nq; ++q) {
      b->w[q] = wts[q] / (w0 * w0) ; /* w3D[q3] / w1d[0]^2 */
    }
    /* Verify: w1d[0] should equal w0 */
  }
  /* DIAGNOSTIC: Lagrange basis functions must sum to 1.0 at every point
     (partition of unity), and their derivatives must sum to 0.0 (derivative
     of the constant function 1). These properties hold regardless of node
     type (equispaced, GLL, etc.) or axis-ordering convention. If either
     check fails, the extraction indexing assumption below is wrong,
     independent of closure ordering, geometry, or physics. */
  {
    PetscInt  q, i;
    PetscReal maxErrB = 0.0, maxErrD = 0.0;

    for (q = 0; q < nq; ++q) {
      PetscReal sumB = 0.0, sumD = 0.0;
      for (i = 0; i < nd; ++i) {
        sumB += b->B[q * nd + i];
        sumD += b->D[q * nd + i];
      }
      maxErrB = PetscMax(maxErrB, PetscAbsReal(sumB - 1.0));
      maxErrD = PetscMax(maxErrD, PetscAbsReal(sumD));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "DIAGNOSTIC: partition-of-unity max error = %e (want ~1e-14)\n",
      (double)maxErrB));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "DIAGNOSTIC: derivative-sum max error     = %e (want ~1e-14)\n",
      (double)maxErrD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DIAGNOSTIC: raw B1d[q][i] matrix (nq=%"
      PetscInt_FMT " x nd=%" PetscInt_FMT "):\n", nq, nd));
    for (q = 0; q < nq; ++q) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  q=%" PetscInt_FMT ": ", q));
      for (i = 0; i < nd; ++i) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%8.5f ", (double)b->B[q * nd + i]));
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Basis1DDestroy(Basis1D *b)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFree5(b->B, b->D, b->Bt, b->Dt, b->w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------- */
/*  Sum-factorized 1D tensor contraction (portable, no SIMD)            */
/*                                                                      */
/*  Computes:  v[k][q] = sum_i S[q][i] * u[k][i]                       */
/*  for k in [0,nspec), q in [0,nout), i in [0,nin)                     */
/*  Layout: u is [nspec * nin], v is [nspec * nout], S is [nout * nin]   */
/* ------------------------------------------------------------------- */
static void Contract1D(const PetscReal *S, PetscInt nout, PetscInt nin,
                       const PetscScalar *u, PetscScalar *v, PetscInt nspec)
{
  PetscInt k, q, i;

  for (k = 0; k < nspec; ++k) {
    for (q = 0; q < nout; ++q) {
      PetscScalar acc = 0.0;
      for (i = 0; i < nin; ++i) {
        acc += S[q * nin + i] * u[k * nin + i];
      }
      v[k * nout + q] = acc;
    }
  }
}

/* ------------------------------------------------------------------- */
/*  Transpose helper: [no][nm][ni] -> [no][ni][nm]                      */
/* ------------------------------------------------------------------- */
static void TransposeInner(const PetscScalar *src, PetscScalar *dst,
                           PetscInt no, PetscInt nm, PetscInt ni)
{
  PetscInt o, m, i;

  for (o = 0; o < no; ++o) {
    for (m = 0; m < nm; ++m) {
      for (i = 0; i < ni; ++i) {
        dst[(o * ni + i) * nm + m] = src[(o * nm + m) * ni + i];
      }
    }
  }
}

/* Rotate: [na][nb][nc] -> [nb][nc][na] */
static void RotateOuter(const PetscScalar *src, PetscScalar *dst,
                        PetscInt na, PetscInt nb, PetscInt nc)
{
  PetscInt a, b, c;

  for (a = 0; a < na; ++a) {
    for (b = 0; b < nb; ++b) {
      for (c = 0; c < nc; ++c) {
        dst[(b * nc + c) * na + a] = src[(a * nb + b) * nc + c];
      }
    }
  }
}


/* ------------------------------------------------------------------- */
/*  3D gradient: u[i3][i2][i1] -> Gx,Gy,Gz[q1][q2][q3]                */
/*  For each component, use D on one axis and B on the other two.       */
/* ------------------------------------------------------------------- */
static void Grad3D(const PetscReal *B, const PetscReal *D,
                   PetscInt nq, PetscInt nd,
                   const PetscScalar *u,
                   PetscScalar *Gx, PetscScalar *Gy, PetscScalar *Gz,
                   PetscScalar *t1, PetscScalar *t2)
{
  /* Gx: D along i1, B along i2, B along i3 */
  Contract1D(D, nq, nd, u, t1, nd * nd);
  TransposeInner(t1, t2, nd, nd, nq);
  Contract1D(B, nq, nd, t2, t1, nd * nq);
  RotateOuter(t1, t2, nd, nq, nq);
  Contract1D(B, nq, nd, t2, Gx, nq * nq);

  /* Gy: B along i1, D along i2, B along i3 */
  Contract1D(B, nq, nd, u, t1, nd * nd);
  TransposeInner(t1, t2, nd, nd, nq);
  Contract1D(D, nq, nd, t2, t1, nd * nq);
  RotateOuter(t1, t2, nd, nq, nq);
  Contract1D(B, nq, nd, t2, Gy, nq * nq);

  /* Gz: B along i1, B along i2, D along i3 */
  Contract1D(B, nq, nd, u, t1, nd * nd);
  TransposeInner(t1, t2, nd, nd, nq);
  Contract1D(B, nq, nd, t2, t1, nd * nq);
  RotateOuter(t1, t2, nd, nq, nq);
  Contract1D(D, nq, nd, t2, Gz, nq * nq);
}

/* ------------------------------------------------------------------- */
/*  3D integration of gradient: Gx,Gy,Gz[q1][q2][q3] -> v[i3][i2][i1] */
/*  Transpose of Grad3D: use Bt/Dt and reverse pass order.              */
/* ------------------------------------------------------------------- */
static void GradT3D(const PetscReal *Bt, const PetscReal *Dt,
                    PetscInt nq, PetscInt nd,
                    const PetscScalar *Gx, const PetscScalar *Gy,
                    const PetscScalar *Gz, PetscScalar *v,
                    PetscScalar *t1, PetscScalar *t2, PetscScalar *acc)
{
  PetscInt nd3 = nd * nd * nd;
  PetscInt j;

  /* Zero output */
  for (j = 0; j < nd3; ++j) v[j] = 0.0;

  /* x-component: Bt along q3, Bt along q2, Dt along q1 */
  Contract1D(Bt, nd, nq, Gx, t1, nq * nq);
  TransposeInner(t1, t2, nq, nq, nd);
  Contract1D(Bt, nd, nq, t2, t1, nq * nd);
  RotateOuter(t1, t2, nq, nd, nd);
  Contract1D(Dt, nd, nq, t2, acc, nd * nd);
  for (j = 0; j < nd3; ++j) v[j] += acc[j];

  /* y-component: Bt along q3, Dt along q2, Bt along q1 */
  Contract1D(Bt, nd, nq, Gy, t1, nq * nq);
  TransposeInner(t1, t2, nq, nq, nd);
  Contract1D(Dt, nd, nq, t2, t1, nq * nd);
  RotateOuter(t1, t2, nq, nd, nd);
  Contract1D(Bt, nd, nq, t2, acc, nd * nd);
  for (j = 0; j < nd3; ++j) v[j] += acc[j];

  /* z-component: Dt along q3, Bt along q2, Bt along q1 */
  Contract1D(Dt, nd, nq, Gz, t1, nq * nq);
  TransposeInner(t1, t2, nq, nq, nd);
  Contract1D(Bt, nd, nq, t2, t1, nq * nd);
  RotateOuter(t1, t2, nq, nd, nd);
  Contract1D(Bt, nd, nq, t2, acc, nd * nd);
  for (j = 0; j < nd3; ++j) v[j] += acc[j];
}

/* ------------------------------------------------------------------- */
/*  Pointwise physics: scalar Laplacian (weak form)                     */
/*                                                                      */
/*  This function applies the quadrature-point operation:                */
/*    res[d] = w * |detJ| * sum_k (J^{-T} J^{-1})_{dk} grad[k]        */
/*                                                                      */
/*  Users can replace this function pointer to change the PDE.          */
/*  The interface follows the PetscDS f1 callback convention.            */
/* ------------------------------------------------------------------- */
typedef void (*PointwiseFn)(PetscInt dim, const PetscScalar *grad,
                            const PetscReal *invJ, PetscReal detJ,
                            PetscReal w, PetscScalar *res);

static void LaplacianPointwise(PetscInt dim, const PetscScalar *grad,
                               const PetscReal *invJ, PetscReal detJ,
                               PetscReal w, PetscScalar *res)
{
  PetscInt d, k, e;

  /* res[d] = w * |detJ| * G_{dk} * grad[k],  G = J^{-T} J^{-1} */
  for (d = 0; d < dim; ++d) {
    PetscScalar val = 0.0;
    for (k = 0; k < dim; ++k) {
      PetscReal G_dk = 0.0;
      for (e = 0; e < dim; ++e) {
        G_dk += invJ[e * dim + d] * invJ[e * dim + k];
      }
      val += G_dk * grad[k];
    }
    res[d] = w * PetscAbsReal(detJ) * val;
  }
}

/* PetscDS-compatible callbacks for the standard verification path */
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
                    PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[d] = u_x[d];
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

  /* Allocate scratch */
  PetscCall(PetscCalloc6(nq3, &t1, nq3, &t2,
                         nq3, &Gx, nq3, &Gy, nq3, &Gz,
                         nd3, &acc));

  /* Scatter global -> local */
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

    /* Gather element DOFs.
       DMCreateDS sets the tensor-product closure permutation automatically
       (DMPlexSetClosurePermutationTensor), so DOFs come back in lexicographic
       order [i3][i2][i1] matching our sum-factorization layout. */
    PetscCall(DMPlexVecGetClosure(dm, section, localX, c, &closureSize, &u_e));
    PetscCheck(closureSize == nd3, PETSC_COMM_SELF, PETSC_ERR_PLIB,
      "Closure size %" PetscInt_FMT " != nd^3=%" PetscInt_FMT, closureSize, nd3);

    /* Get cell geometry (constant Jacobian for affine hex) */
    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));

    /* Forward: u_e -> gradient at quadrature points */
    Grad3D(basis->B, basis->D, nq, nd, u_e, Gx, Gy, Gz, t1, t2);

    /* Pointwise physics at each quadrature point */
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

    /* Backward: integrate weighted gradient -> y_e */
    PetscCall(PetscMalloc1(nd3, &y_e));
    GradT3D(basis->Bt, basis->Dt, nq, nd, Gx, Gy, Gz, y_e, t1, t2, acc);

    /* Scatter-add to global vector */
    PetscCall(DMPlexVecSetClosure(dm, section, y, c, y_e, ADD_VALUES));

    PetscCall(DMPlexVecRestoreClosure(dm, section, localX, c, &closureSize, &u_e));
    PetscCall(PetscFree(y_e));
  }

  PetscCall(DMRestoreLocalVector(dm, &localX));
  PetscCall(PetscFree6(t1, t2, Gx, Gy, Gz, acc));
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
  Vec            x, y_sf, y_ref;
  AppCtx         ctx;
  Basis1D        basis;
  PetscInt       dim = 3, degree;
  PetscLogEvent  ev_sf;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscLogEventRegister("SumFactApply", DM_CLASSID, &ev_sf));

  /* Options */
  ctx.verify = PETSC_FALSE;
  ctx.bench  = PETSC_FALSE;
  ctx.niter  = 100;
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "Sum-factorization options", "DMPLEX");
  PetscCall(PetscOptionsBool("-verify", "Verify against PetscFE path",
                             "ex21.c", ctx.verify, &ctx.verify, NULL));
  PetscCall(PetscOptionsBool("-bench", "Run timing loop",
                             "ex21.c", ctx.bench, &ctx.bench, NULL));
  PetscCall(PetscOptionsInt("-niter", "Benchmark iterations",
                            "ex21.c", ctx.niter, &ctx.niter, NULL));
  PetscOptionsEnd();

  /* Create hex mesh via command-line options */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  /* Setup PetscFE: tensor-product Lagrange (isSimplex = false).
     PetscFECreateDefault reads -petscspace_degree from the command line. */
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE,
                                 NULL, PETSC_DETERMINE, &fe));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, f0_zero, f1_grad));

  /* Trigger local coordinate setup so DMPlexComputeCellGeometryFEM works */
  {
    Vec coordsLocal;
    PetscCall(DMGetCoordinatesLocal(dm, &coordsLocal));
  }

  /* Set tensor closure permutation so DMPlexVecGetClosure/SetClosure and
     cell geometry queries return DOFs and coordinates in lexicographic
     (ix,iy,iz) order, matching what our sum-factorization kernels assume.
     PETSc's default closure order is breadth-first over mesh points
     (vertices, then edges, then faces, then interior), which does NOT
     match tensor order for degree > 1.

     Must be set on both the primary DM and its coordinate DM: the
     coordinate DM does not inherit this automatically
     (see PETSc GitLab issue #541). */
  PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL));
  {
    DM cdm;
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMPlexSetClosurePermutationTensor(cdm, PETSC_DETERMINE, NULL));
  }

  {
    PetscSpace sp;
    PetscCall(PetscFEGetBasisSpace(fe, &sp));
    PetscCall(PetscSpaceGetDegree(sp, &degree, NULL));
  }

  /* Extract 1D basis from PetscFE */
  PetscCall(Basis1DCreate(fe, &basis));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "Sum-factorization: dim=%" PetscInt_FMT " degree=%" PetscInt_FMT
    " nd=%" PetscInt_FMT " nq=%" PetscInt_FMT "\n",
    dim, degree, basis.nd, basis.nq));

  /* Create vectors */
  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(VecDuplicate(x, &y_sf));
  PetscCall(VecDuplicate(x, &y_ref));

  /* DIAGNOSTIC: constant-field null-space test.
     grad(constant) = 0 everywhere, so the Laplacian operator action on a
     constant field must be exactly zero (up to roundoff). This isolates
     the gather/scatter + contraction pipeline from the random-field test
     below, independent of the basis-extraction diagnostics above. */
  if (ctx.verify) {
    Vec       xConst, yConst;
    PetscReal normConst;

    PetscCall(VecDuplicate(x, &xConst));
    PetscCall(VecDuplicate(x, &yConst));
    PetscCall(VecSet(xConst, 1.0));
    PetscCall(ApplySumFact(dm, xConst, yConst, &basis, LaplacianPointwise));
    PetscCall(VecNorm(yConst, NORM_2, &normConst));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "DIAGNOSTIC: ||A*1|| = %e (want ~1e-12 or smaller)\n", (double)normConst));
    PetscCall(VecDestroy(&xConst));
    PetscCall(VecDestroy(&yConst));
  }

  /* DIAGNOSTIC: project physical X-coordinate onto the field, then dump
     cell 0's closure array decoded via our assumed flat-index formula
     idx = iz*nd*nd + iy*nd + ix (ix fastest). Since the projected value
     AT a node equals that node's physical X-coordinate, this directly
     reveals which axis (if any) ix/iy/iz actually correspond to in the
     real closure layout: entries that vary only with ix (not iy,iz)
     confirm ix maps to physical X. If instead entries vary with iz (or
     some other pattern) while ix is held fixed in the printout, that
     tells us the true flat-index-to-axis correspondence. */
  if (ctx.verify) {
    Vec       xCoord;
    Vec       localXCoord;
    PetscSection section;
    PetscScalar *u_e = NULL;
    PetscInt     closureSize, ix, iy, iz, nd = basis.nd;
    PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);

    funcs[0] = CoordXFunc;
    PetscCall(VecDuplicate(x, &xCoord));
    PetscCall(DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_VALUES, xCoord));

    PetscCall(DMGetLocalVector(dm, &localXCoord));
    PetscCall(DMGlobalToLocal(dm, xCoord, INSERT_VALUES, localXCoord));
    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(DMPlexVecGetClosure(dm, section, localXCoord, 0, &closureSize, &u_e));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "DIAGNOSTIC: cell 0 closure of projected X-coordinate (nd=%" PetscInt_FMT
      ", closureSize=%" PetscInt_FMT "):\n", nd, closureSize));
    for (iz = 0; iz < nd; ++iz) {
      for (iy = 0; iy < nd; ++iy) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  iz=%" PetscInt_FMT " iy=%" PetscInt_FMT ": ", iz, iy));
        for (ix = 0; ix < nd; ++ix) {
          PetscInt idx = (iz * nd + iy) * nd + ix;
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%7.4f ",
            (double)PetscRealPart(u_e[idx])));
        }
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
      }
    }

    PetscCall(DMPlexVecRestoreClosure(dm, section, localXCoord, 0, &closureSize, &u_e));
    PetscCall(DMRestoreLocalVector(dm, &localXCoord));
    PetscCall(VecDestroy(&xCoord));
  }

  /* Initialize x with a nontrivial function */
  {
    PetscRandom rctx;
    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
    PetscCall(PetscRandomSetFromOptions(rctx));
    PetscCall(VecSetRandom(x, rctx));
    PetscCall(PetscRandomDestroy(&rctx));
  }

  /* Sum-factorized apply */
  PetscCall(PetscLogEventBegin(ev_sf, dm, 0, 0, 0));
  PetscCall(ApplySumFact(dm, x, y_sf, &basis, LaplacianPointwise));
  PetscCall(PetscLogEventEnd(ev_sf, dm, 0, 0, 0));

  /* Verification */
  if (ctx.verify) {
    PetscReal norm_sf, norm_ref, norm_diff;
    Vec       diff;

    PetscCall(DMPlexSNESComputeResidualFEM(dm, x, y_ref, NULL));
    PetscCall(VecDuplicate(y_sf, &diff));
    PetscCall(VecCopy(y_sf, diff));
    PetscCall(VecAXPY(diff, -1.0, y_ref));
    PetscCall(VecNorm(y_sf, NORM_2, &norm_sf));
    PetscCall(VecNorm(y_ref, NORM_2, &norm_ref));
    PetscCall(VecNorm(diff, NORM_2, &norm_diff));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "Verification: ||y_sf - y_ref|| / ||y_ref|| = %e\n",
      (double)(norm_ref > 0 ? norm_diff / norm_ref : norm_diff)));
    PetscCall(VecDestroy(&diff));
  }

  /* Benchmark */
  if (ctx.bench) {
    PetscLogDouble tstart, tend;
    PetscInt       cStart, cEnd, nCells, iter;
    PetscReal      elapsed, per_iter;

    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    nCells = cEnd - cStart;

    /* Warmup */
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

  /* Cleanup */
  PetscCall(Basis1DDestroy(&basis));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y_sf));
  PetscCall(VecDestroy(&y_ref));
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
