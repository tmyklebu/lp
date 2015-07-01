/*
Tor Myklebust's LP solver
Copyright (C) 2013-2015 Tor Myklebust

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <qd/dd_real.h>
#include <fenv.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <ieee754.h>
#include <assert.h>
#include <math.h>
#include "lp.h"
#include <vector>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <set>
#include <gmpxx.h>
using namespace std;

bool flag_no_stale = false;
double flag_mehrotra_neigh = 0.9;
double flag_stale_neigh = 0.5;
int flag_max_stale = 16;
int flag_stale_corrector = 1;

// Solve min(c x : A x = b, x >= 0).
// The dual is min(-b y : A^T y + s = c, s >= 0).
//
// A self-dual formulation (Ye-Todd-Mizuno) is:
// min h0 theta
// st  Ax - b tau + b0 theta = 0             (y)
//     A^T y + s - c tau - c0 theta = 0      (x; s is a slack)
//     b^T y - c^T x - g0 theta - kappa = 0  (tau; kappa is a slack)
//     -b0 y - c0 x + g0 tau = -h0           (theta)
//     x, s, tau, kappa >= 0
//     y, theta free.
//
// Complementary slackness here is XS = 0 and tau kappa = 0.
//
// The happy fun system of equations defining a MTY direction is this:
//
// 0     A     0     -b    b0    0   | 0
// -A^T  0     -I    c     c0    0   | 0
// b^T   -c^T  0     0     -g0   -1  | 0
// -b0^T -c0^T 0     g0    0     0   | 0
// 0     S     X     0     0     0   | -Xs + gammamu e
// 0     0     0     kappa 0     tau | -kappatau + gammamu
//
// You can block-reduce this to the following to remove kappa and s:
//
// 0     A     0     -b    b0    0   | 0
// -A^T  S/X   0     c     c0    0   | -s + gammamu / x
// b^T   -c^T  0     K/T   -g0   0   | kappa - gammamu/tau
// -b0^T -c0^T 0     g0    0     0   | 0
// 0     S     X     0     0     0   | -Xs + gammamu e
// 0     0     0     kappa 0     tau | -kappatau + gammamu
//
// Finding dx, dy, dtheta, and dtau is then the following system:
//
// 0       A        -b      b0      | 0
// -A^T    S/X      c       c0      | -s + gammamu / x
// b^T     -c^T     K/T     -g0     | -kappa + gammamu/tau
// -b0^T   -c0^T    g0      0       | 0
//
// Using the second equation block to eliminate dx:
//
// -X/S A^T            I      X/S c            -X/S c0          | -x + gammamu / s
// A X/S A^T           0      -A X/S c - b     b0 + A X/S c0    | A x - A gammamu / s
// (b - A X/S c)^T     0      K/T + c^T X/S c  -c^T X/S c0 - g0 | c^T x - c^T gammamu / s - kappa + gammamu/tau
// -(A X/S c0 - b0)^T  0      g0 - c0^T X/S c  c0^T X/S c0      | c0^T x - c0^T gammamu / s
//
// This doesn't give you a positive definite system, though; b0, c0, and g0
// ruin everything.
//
// At this point, I looked at YTM1994.  They suggest that, instead of trying to
// wrestle the thing into a form where you can solve for dy and dtheta
// directly, you write dy as a function of dtheta and dtau using the second
// block of equations above.
//
// Xu-Hung-Ye (1996) give a simpler form that is what's actually used here.


struct gradients {
  double dx[MAXN];
  double dy[MAXN];
  double ds[MAXN];
  double dkappa, dtau;
};

long long get_cpu_usecs() {
  timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec * 1000000LL + ts.tv_nsec/1000;
}

unordered_map<const char *, double> scope_usecs_top, scope_usecs_bot;
ScopeTimer::ScopeTimer(const char *p) : p(p), before(get_cpu_usecs()) {}
ScopeTimer::~ScopeTimer() {
  long long after = get_cpu_usecs();
  scope_usecs_top[p] += after-before;
  scope_usecs_bot[p]++;
}

double x[MAXN], y[MAXN], s[MAXN];
double kappa, tau;

double objadjust;
double b0[MAXN], c0[MAXN], g0, h0;
double b[MAXN];
double c[MAXN];
int m, n;

int iterno;
char last_step_kind;
double mu0;

double Ax[MAXN], ATy[MAXN];
double mu;
double comprat;

static double rho = 1;

void find_unbalanced_steplen(const gradients &pg, double &plen, double &dlen);
double eval_unbalanced_step(const gradients &pg, double plen, double dlen);
double eval_step(const gradients &pg, double plen, double dlen);
double eval_step(const gradients &pg);

void my_frexp(double &d, long long &k) {
  union {
    long long a;
    double b;
  } u;
  u.b = d;
  int expo = u.a >> 52 & 2047;
  k += expo - 1023;
  u.a += (1023ll - expo) << 52;
  d = u.b;
}

double potential() {
  double obj = tau*kappa;
  FOR(i,n) obj += s[i]*x[i];
  double ans = log(obj) * n * (1+rho);
  double nans = 0;
  long long expo = 0;
  double c1 = 1.0, c2 = 1.0, c3 = 1.0, c4 = 1.0;
  int ii = 0;
  for (; ii + 63 < n; ii += 32) {
#define ONE(c,z) c *= (x[ii+z] * s[ii+z]);
#define FOUR(z) ONE(c1,z) ONE(c2, z+1) ONE(c3, z+2) ONE(c4, z+3)
#define SIXTEEN(z) FOUR(z) FOUR(z+4) FOUR(z+8) FOUR(z+12)
    SIXTEEN(0) SIXTEEN(16)
    my_frexp(c1, expo);
    my_frexp(c2, expo);
    my_frexp(c3, expo);
    my_frexp(c4, expo);
  }
  for (; ii < n; ii++) {
    ONE(c1, 0)
    my_frexp(c1, expo);
  }
#undef SIXTEEN
#undef FOUR
#undef ONE
  nans -= expo * log(2.0);
  nans -= log(c1 * c2 * c3 * c4);
  nans -= log(tau*kappa);
  return ans + nans;
}

double potential(const gradients &g, double a) {
  double obj = (tau+g.dtau*a)*(kappa+g.dkappa*a);
  FOR(i,n) obj += (s[i]+g.ds[i]*a)*(x[i]+g.dx[i]*a);
  double ans = log(obj) * n * (1+rho);
  double nans = 0;
  long long expo = 0;
  double c1 = 1.0, c2 = 1.0, c3 = 1.0, c4 = 1.0;
  int ii = 0;
  for (; ii + 63 < n; ii += 32) {
#define ONE(c,z) c *= (x[ii+z] + g.dx[ii+z] * a) * (s[ii+z] + g.ds[ii+z] * a);
#define FOUR(z) ONE(c1,z) ONE(c2, z+1) ONE(c3, z+2) ONE(c4, z+3)
#define SIXTEEN(z) FOUR(z) FOUR(z+4) FOUR(z+8) FOUR(z+12)
    SIXTEEN(0) SIXTEEN(16)// SIXTEEN(32) SIXTEEN(48)
    my_frexp(c1, expo);
    my_frexp(c2, expo);
    my_frexp(c3, expo);
    my_frexp(c4, expo);
  }
  for (; ii < n; ii++) {
    ONE(c1, 0)
    my_frexp(c1, expo);
  }
#undef SIXTEEN
#undef FOUR
#undef ONE
  nans -= expo * log(2.0);
  nans -= log(c1 * c2 * c3 * c4);
  nans -= log((tau+g.dtau*a)*(kappa+g.dkappa*a));
  return ans + nans;
}

double mumutilde() {
  double mu = 0, mutilde = 0;
  FOR(i,n) mu += x[i]*s[i], mutilde += 1/(x[i]*s[i]);
  mu += kappa*tau; mutilde += 1/(kappa*tau);
  return mu*mutilde/((n+1)*(double)(n+1));
}

// YTM errors
double primal_error, dual_error, duality_error, scaling_error;
// standard scaled infeasibilities and duality gap.
double pinfeas, dinfeas, duality_gap;
void verify_current_iterate(long long usecs_taken) {
// st  Ax - b tau + b0 theta = 0             (y)
//     A^T y + s - c tau - c0 theta = 0      (x; s is a slack)
//     b^T y - c^T x - g0 theta - kappa = 0  (tau; kappa is a slack)
//     -b0 y - c0 x + g0 tau = -h0           (theta)
  vecmd xxx;
  vecnd yyy;
  FOR(i, m) xxx[i] = Ax[i] - b[i] * tau + b0[i];
  primal_error = 0;
  FOR(i, m) primal_error += xxx[i]*xxx[i];

  FOR(i,n) yyy[i] = ATy[i] - c[i]*tau - c0[i] + s[i];
  dual_error = 0;
  FOR(i,n) dual_error += yyy[i]*yyy[i];

  duality_error = 0;
  FOR(i,n) duality_error -= c[i]*x[i];
  FOR(i,m) duality_error += b[i]*y[i];
  duality_error -= g0;
  duality_error -= kappa;

  scaling_error = h0;
  FOR(i,n) scaling_error -= c0[i]*x[i];
  FOR(i,m) scaling_error -= b0[i]*y[i];
  scaling_error += g0*tau;

  double cx = 0;
  FOR(i,n) cx += c[i] * x[i];
  cx += objadjust;
  double by = 0;
  FOR(i,m) by += b[i] * y[i];
  cx /= tau; by /= tau;

  duality_gap = 0;
  FOR(i,n) duality_gap += c[i]*x[i];
  FOR(i,m) duality_gap -= b[i]*y[i];
  duality_gap /= (tau + fabs(by));

  pinfeas = 0, dinfeas = 0;
  FOR(i,m) pinfeas = max(pinfeas, fabs(b[i]*tau - Ax[i]));
  FOR(i,n) dinfeas = max(dinfeas, fabs(c[i]*tau - s[i] - ATy[i]));
  //FOR(i,m) pinfeas += (b[i]*tau - Ax[i]) * (b[i]*tau - Ax[i]);
  //FOR(i,n) dinfeas += (c[i]*tau - s[i] - ATy[i]) * (c[i]*tau - s[i] - ATy[i]);
  //double xnorm = 0, snorm = 0;
  double maxcomp = kappa*tau, mincomp = kappa*tau;
  FOR(i,n) {
    //xnorm += x[i]*x[i], snorm += s[i]*s[i];
    mincomp = min(mincomp, x[i]*s[i]);
    maxcomp = max(maxcomp, x[i]*s[i]);
    comprat = maxcomp / mincomp;
  }
  double bnorm=0, cnorm=0;
  FOR(i,m) bnorm = max(bnorm, fabs(b[i]));
  FOR(i,n) cnorm = max(cnorm, fabs(c[i]));
  pinfeas /= tau * (1 + bnorm);
  dinfeas /= tau * (1 + cnorm);
  //pinfeas = sqrt(pinfeas) / (tau + sqrt(xnorm));
  //dinfeas = sqrt(dinfeas) / (tau + sqrt(snorm));
  if (1) {
    double mu = 0, mutilde = 0;
    FOR(i, n) mu += x[i]*s[i], mutilde += 1/(x[i]*s[i]);
    mu += kappa*tau; mutilde += 1/(kappa*tau);
    mu /= n+1; mutilde /= n+1;
    printf("%c%5i|%8.1e %8.1e|%8.1e %8.1e %8.1e|%5.2f %5.2f %5.2f|%10lli\n",
        last_step_kind, iterno, cx, by, pinfeas, dinfeas, fabs(duality_gap),
        -log10(mu/mu0), log10(maxcomp/mincomp), log10(kappa/tau),
        usecs_taken);
  } else {
    printf("kappa %15.10g tau %15.10g\n", kappa, tau);
    printf("-------------------------------\n");
    printf("primal infeasibility: %15.10g\n", pinfeas);
    printf("dual   infeasibility: %15.10g\n", dinfeas);
    printf("first  system norm:   %15.10g\n", primal_error);
    printf("second system norm:   %15.10g\n", dual_error);
    printf("third  ineq   value:  %15.10g\n", duality_error);
    printf("fourth ineq   value:  %15.10g\n", scaling_error);
    printf("Primal obj    value:  %15.10g\n", cx);
    printf("Dual   obj    value:  %15.10g\n", by);
    printf("-------------------------------\n");
  }
}

void init_naive_initial_point() {
  // This is SeDuMi's initialisation.
  double cnorm = 0, bnorm = 0;
  FOR(i,n) cnorm = max(cnorm, fabs(c[i]));
  FOR(i,m) bnorm = max(bnorm, fabs(b[i]));
  double targ = sqrt((1+cnorm) * (1+bnorm));
  FOR(i,n) x[i] = 1.;
  FOR(i,m) y[i] = 0;
  FOR(i,n) s[i] = 1.;
  comprat = 1;
  tau = 1/targ;
  kappa = targ/n;
  mu0 = 1.;
}

void compute_sd_coeffs() {
  FOR(i,m) b0[i] = (b[i]*tau - Ax[i]);
  FOR(i,n) c0[i] = (-c[i]*tau + s[i] + ATy[i]);

  g0 = 0;
  FOR(i,m) g0 += b[i] * y[i];
  FOR(i,n) g0 -= c[i] * x[i];
  g0 -= kappa;

  h0 = 0;
  FOR(i,n) h0 += c0[i] * x[i];
  FOR(i,m) h0 += b0[i] * y[i];
  h0 -= g0 * tau;
}

double best_mu(double *a, double *b, double tau, double kappa) {
  double ans = 0;
  FOR(i,n) ans += a[i]*b[i];
  return (ans + tau*kappa) / (n + 1);
}

double dot(const double *v, const double *w, int n) {
  double a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;
  for (int i = 0; i < n-7; i += 8) {
    a0 += v[i+0] * w[i+0];
    a1 += v[i+1] * w[i+1];
    a2 += v[i+2] * w[i+2];
    a3 += v[i+3] * w[i+3];
    a4 += v[i+4] * w[i+4];
    a5 += v[i+5] * w[i+5];
    a6 += v[i+6] * w[i+6];
    a7 += v[i+7] * w[i+7];
  }
  for (int i = n - n%8; i < n; i++) a0 += v[i] * w[i];
  return a0+a1+a2+a3+a4+a5+a6+a7;
}

void do_step(double lambda, const gradients &g) {
  FOR(i,n) x[i] += g.dx[i] * lambda;
  FOR(i,m) y[i] += g.dy[i] * lambda;
  FOR(i,n) s[i] += g.ds[i] * lambda;
  tau += g.dtau * lambda;
  kappa += g.dkappa * lambda;
}

void do_aggressive_step(const gradients &pg, double tube) {
  double maxlen = 1e75;
  FOR(i, n) if (pg.dx[i] < 0) maxlen = min(maxlen, -x[i] / pg.dx[i]);
  FOR(i, n) if (pg.ds[i] < 0) maxlen = min(maxlen, -s[i] / pg.ds[i]);
  if (pg.dtau < 0) maxlen = min(maxlen, -tau/pg.dtau);
  if (pg.dkappa < 0) maxlen = min(maxlen, -kappa/pg.dkappa);
  maxlen *= tube;
  do_step(maxlen, pg);
}

void find_unbalanced_steplen(const gradients &pg, double &plen, double &dlen) {
  plen = 1, dlen = 1;
  FOR(i, n) if (pg.dx[i] < 0) plen = min(plen, -x[i] / pg.dx[i]);
  FOR(i, n) if (pg.ds[i] < 0) dlen = min(dlen, -s[i] / pg.ds[i]);
  if (pg.dtau < 0) {
    plen = min(plen, -tau/pg.dtau);
    dlen = min(dlen, -tau/pg.dtau);
  }
  if (pg.dkappa < 0) {
    plen = min(plen, -kappa/pg.dkappa);
    dlen = min(dlen, -kappa/pg.dkappa);
  }
}

double eval_unbalanced_step(const gradients &pg, double plen, double dlen) {
  double ptau = tau + plen * pg.dtau;
  double dtau = tau + dlen * pg.dtau;
  double newtau = max(ptau, dtau);

  // estimate the new mu; if it's worse, reject the step.
  double newmu = 0;
  FOR(i,n) newmu += (x[i] + plen * pg.dx[i]) * (s[i] + dlen * pg.ds[i]);
  if (newtau == ptau) {
    newmu *= ptau / dtau;
    newmu += (kappa + plen * pg.dkappa) * ptau;
  } else {
    newmu *= dtau / ptau;
    newmu += (kappa + dlen * pg.dkappa) * dtau;
  }
  newmu /= n+1;
  return newmu;
}

double eval_step(const gradients &pg, double plen, double dlen) {
  double minlen = min(plen, dlen);
  double ans = eval_unbalanced_step(pg, plen, dlen);
  if (ans >= mu) ans = eval_unbalanced_step(pg, minlen, minlen);
  return ans;
}

double eval_step(const gradients &pg) {
  double plen, dlen;
  find_unbalanced_steplen(pg, plen, dlen);
  plen *= .99; dlen *= .99;
  return eval_step(pg, plen, dlen);
}

// Yinyu Ye's step
int do_unbalanced_aggressive_step(const gradients &pg, double tube) {
  double plen, dlen;
  find_unbalanced_steplen(pg, plen, dlen);
  plen *= tube; dlen *= tube;

  if (eval_unbalanced_step(pg, plen, dlen) >= mu) {
    // oookay, so separate primal and dual steps is a bad idea.
    plen = dlen = min(plen, dlen);
    if (eval_unbalanced_step(pg, plen, dlen) >= mu) {
      printf("Rejecting unbalanced step that worsens mu by %.16g\n",
          eval_unbalanced_step(pg, plen, dlen) / mu);
      return 1;
    }
  }

  double ptau = tau + plen * pg.dtau;
  double dtau = tau + dlen * pg.dtau;
  double newtau = min(ptau, dtau);

  FOR(i,n) x[i] += plen * pg.dx[i];
  FOR(i,m) y[i] += dlen * pg.dy[i];
  FOR(i,n) s[i] += dlen * pg.ds[i];
  if (newtau == ptau) {
    kappa += dlen * pg.dkappa;
    FOR(i,m) y[i] *= newtau/dtau;
    FOR(i,n) s[i] *= newtau/dtau;
  } else {
    kappa += plen * pg.dkappa;
    FOR(i,n) x[i] *= newtau/ptau;
  }
  tau = newtau;
  return 0;
}

void check_residuals(const scaled_system &scaling, const double *dx, const double *dy, const double *ds, const double *rhs1, const double *rhs2, const double *rhs3) {
    vecmd resid1; vecnd resid2, resid3;
    mul_A(resid1, dx);
    mul_Atran(resid2, dy);
    FOR(i,m) {
      dd_real ai = 0;
      FOR(j, sparse_A[i].size()) {
        ai += dd_real(sparse_A[i][j].second) * dx[sparse_A[i][j].first];
      }
      dd_real diff = ai - resid1[i];
      double reldiff = fabs(to_double(diff / (1 + fabs(ai)))) / tau;
      if (0 && reldiff > 1e-4) {
        printf("%16a %16a\n", to_double(ai), resid1[i]);
        FOR(j, sparse_A[i].size())
          printf("%4i %16a %16a\n", sparse_A[i][j].first, sparse_A[i][j].second, dx[sparse_A[i][j].first]);
      }
    }
    FOR(i,m) resid1[i] = rhs1[i] - resid1[i];
    FOR(i,n) resid2[i] = rhs2[i] - resid2[i] - ds[i];
    FOR(i,n) resid3[i] = ds[i];
    scaling.scale(resid3);
    FOR(i,n) resid3[i] += dx[i] - rhs3[i];
    double re1norm = 0, rh1norm = 0;
    double re3norm = 0, rh3norm = 0;
    FOR(i,m) re1norm = max(re1norm, fabs(resid1[i]));
    FOR(i,m) rh1norm = max(rh1norm, fabs(rhs1[i]));
    FOR(i,n) re3norm = max(re3norm, fabs(resid3[i]));
    FOR(i,n) rh3norm = max(rh3norm, fabs(rhs3[i]));
    //printf("%g %g %g\n", dot(resid1,resid1,m), dot(resid2,resid2,n), dot(resid3,resid3,n));
    printf("%12g %12g -- %12g %12g\n", re1norm, rh1norm, re3norm, rh3norm);
}

// Solve
//
//   A dx = rhs1
//   A^T dy + ds = rhs2
//   dx + T^2 ds = rhs3
//
// for dx, dy, and ds.
//
// I use the normal equations thusly:
//
//   A dx = rhs1
//   A T^2 A^T dy + A T^2 ds = A T^2 rhs2
//   A dx + A T^2 ds = A rhs3
//
// dy = (A T^2 A^T)^(-1) (A T^2 rhs2 - A rhs3 + rhs1)
// ds = rhs2 - A^T dy
// dx = rhs3 - T^2 ds
void find_xys(const scaled_system &scaling, double *dx, double *dy, double *ds,
    const double *rhs1, const double *rhs2, const double *rhs3, int nrhs) {
  ScopeTimer _st("find_xys");
  vector<double> rhs(nrhs * m);
  vector<double> tmp(nrhs * n);
  FOR(i,n*nrhs) tmp[i] = rhs2[i];
  FOR(i, nrhs) scaling.scale(&tmp[0] + n * i);
  FOR(i,n*nrhs) tmp[i] -= rhs3[i];
  FOR(i, nrhs) mul_A(&rhs[0] + m * i, &tmp[0] + n * i);
  FOR(i,m*nrhs) rhs[i] += rhs1[i];
  scaling.get_dy(dy, &rhs[0], nrhs);

  FOR(i, nrhs) mul_Atran(&tmp[0] + n * i, dy + m * i);
  FOR(i,n*nrhs) ds[i] = rhs2[i] - tmp[i];

  FOR(i,n*nrhs) dx[i] = ds[i];
  FOR(i, nrhs) scaling.scale(dx + n * i);
  FOR(i,n*nrhs) dx[i] = rhs3[i] - dx[i];

  // check residuals
  if (0) {
    FOR(i, nrhs)
      check_residuals(scaling, dx+n*i,dy+m*i,ds+n*i,rhs1+m*i,rhs2+n*i,rhs3+n*i);
  }
}


struct xhy_logic {
  vecmd rP; vecnd rD; double rG;
  vecnd xhy_x1; vecmd xhy_y1; vecnd xhy_s1;
  const scaled_system &scaling;

  xhy_logic(const scaled_system &scaling) : scaling(scaling) {}

  void find_xys(double *dx, double *dy, double *ds,
      const double *rhs1, const double *rhs2, const double *rhs3, int nrhs) {
    ::find_xys(scaling, dx, dy, ds, rhs1, rhs2, rhs3, nrhs);
  }

  double find_t(const double *x0, const double *y0,
      const double *x1, const double *y1, double eta, double mu,
      double *dtau, double *dkappa) {
    double gamma = 1-eta;
    double dk0 = dot(b, y0, m) - dot(c, x0, n) - eta * rG;
    double dkm = dot(b, y1, m) - dot(c, x1, n);
    double e0 = tau * dk0 - gamma * mu + kappa * tau;
    double m0 = tau * (dkm - kappa);
    double t = -e0/m0;
    *dtau = -t*tau;
    *dkappa = dk0 + t * dkm;
    return t;
  }
  
  // Xiaojie Xu, Pi-Fang Hung, Yinyu Ye, A simplified homogeneous self-dual
  // linear programming algorithm and its implementation (1996).
  void find_dirs(gradients *pg, gradients *cg) {
    ScopeTimer _st("xhy::find_dirs");
    FOR(i,m) rP[i] = (b[i] * tau - Ax[i]);
    FOR(i,n) rD[i] = (c[i] * tau - (ATy[i] + s[i]));
    rG = dot(c,x,n) - dot(b,y,m) + kappa;
  
    // 0   A	0	-b	0	eta rP
    // A^T 0	I	-c	0	eta rD
    // 0   I	T^2	0	0	-x + gammamu/s
    // b   -c	0	0	-1	eta rG
    // 0   0	0	kappa	tau	gammamu - kappatau
    // Work out dx, dy, and ds in the case where dtau is 0 and -tau to satisfy
    // the first three equations.  Work out dkappa in each case from the fourth
    // equation.  Then work out the right dtau using the fifth equation.  (I use
    // slope-intercept form instead of two-point form, though.)
  
    vector<double> tmp1(3*m);
    vector<double> dx(3*n), dy(3*m), ds(3*n);
    vecnd x0, s0; vecmd y0;
    double *x1 = xhy_x1, *y1 = xhy_y1, *s1 = xhy_s1;
    #if 0 // "Naive" method.
    vector<double> tmp2(3*n), tmp3(3*n);
    FOR(i,m) tmp1[i] = rP[i];
    FOR(i,n) tmp2[i] = rD[i];
    FOR(i,n) tmp3[i] = -x[i];
    FOR(i,m) tmp1[m+i] = 0;
    FOR(i,n) tmp2[n+i] = 0;
    FOR(i,n) tmp3[n+i] = -x[i] + mu/s[i];
    FOR(i,m) tmp1[2*m+i] = -b[i];
    FOR(i,n) tmp2[2*n+i] = -c[i];
    FOR(i,n) tmp3[2*n+i] = 0;
    find_xys(&dx[0],&dy[0],&ds[0], &tmp1[0],&tmp2[0],&tmp3[0], 3);

    FOR(i,n) x1[i] = (dx[i+2*n])*tau;
    FOR(i,m) y1[i] = (dy[i+2*m])*tau;
    FOR(i,n) s1[i] = (ds[i+2*n])*tau;
    #else // Somewhat more careful method.
    vecnd tmp2;
    FOR(i,n) tmp2[i] = rD[i];
    scaling.scale(&tmp2[0]);
    mul_A(&tmp1[0], &tmp2[0]); // tmp1 == A T^2 rD
    FOR(i,m) tmp1[2*m+i] = -(rP[i] + tmp1[i] + 2*Ax[i]);
    FOR(i,m) tmp1[i] = tmp1[i] + b[i] * tau;
    FOR(i,n) tmp2[i] = x[i] - mu/s[i];
    mul_A(&tmp1[m], &tmp2[0]);
    scaling.get_dy(&dy[0], &tmp1[0], 3);
    FOR(i, 3) mul_Atran(&ds[i*n], &dy[i*m]);
    FOR(i,n) dx[i] = ds[i] = rD[i] - ds[i];
    FOR(i,n) dx[i+n] = ds[i+n] = -ds[i+n];
    FOR(i,n) x1[i] = -(s1[i] = -s[i] - (rD[i] + ds[i+2*n]));
    scaling.scale(x1);
    FOR(i,m) y1[i] = dy[i+2*m] - y[i];
    FOR(i,2) scaling.scale(&dx[i*n]);
    FOR(i,n) dx[i] = -x[i] - dx[i];
    FOR(i,n) dx[i+n] = (-x[i]+mu/s[i]) - dx[i+n];
    #endif

    if (0) {
      vecmd r1; vecnd r2, r3;
      mul_A(r1, &dx[0]);
      FOR(i,m) r1[i] -= rP[i];
      mul_Atran(r2, &dy[0]);
      FOR(i,n) r2[i] = r2[i] + ds[i] - rD[i];
      FOR(i,n) r3[i] = ds[i];
      scaling.scale(r3);
      FOR(i,n) r3[i] += dx[i] + x[i];
      printf("%g %g %g\n", dot(r1,r1,m), dot(r2,r2,n), dot(r3,r3,n));

      mul_A(r1, &dx[n]);
      mul_Atran(r2, &dy[m]);
      FOR(i,n) r2[i] = r2[i] + ds[n+i];
      FOR(i,n) r3[i] = ds[n+i];
      scaling.scale(r3);
      FOR(i,n) r3[i] += dx[n+i] + x[i] - mu/s[i];
      printf("%g %g %g\n", dot(r1,r1,m), dot(r2,r2,n), dot(r3,r3,n));

      mul_A(r1, x1);
      FOR(i,m) r1[i] += tau*b[i];
      mul_Atran(r2, y1);
      FOR(i,n) r2[i] = r2[i] + s1[i] + tau*c[i];
      FOR(i,n) r3[i] = s1[i];
      scaling.scale(r3);
      FOR(i,n) r3[i] += x1[i];
      printf("%g %g %g\n", dot(r1,r1,m), dot(r2,r2,n), dot(r3,r3,n));
    }
  
    double t = find_t(&dx[0], &dy[0], x1, y1, 1, mu, &pg->dtau, &pg->dkappa);
    //printf("predictor tee is %g\n", t);
    FOR(i,n) pg->dx[i] = dx[i] + t * x1[i];
    FOR(i,m) pg->dy[i] = dy[i] + t * y1[i];
    FOR(i,n) pg->ds[i] = ds[i] + t * s1[i];
    if (0) {
      vecmd asdf;
      mul_A(asdf, pg->dx);
      FOR(i,m) printf("%12g %12g %12g\n", asdf[i], rP[i] - t * b[i] * tau,
          asdf[i] - rP[i] - pg->dtau * b[i]);
    }

    t = find_t(&dx[n], &dy[m], x1, y1, 0, mu, &cg->dtau, &cg->dkappa);
    //printf("corrector tee is %g\n", t);
    FOR(i,n) cg->dx[i] = dx[n+i] + t * x1[i];
    FOR(i,m) cg->dy[i] = dy[m+i] + t * y1[i];
    FOR(i,n) cg->ds[i] = ds[n+i] + t * s1[i];
  
    /*
    if (0) {
      vecmd resid1; vecnd resid2, resid3;
      double resid4, resid5;
      mul_A(resid1, g->dx);
      FOR(i,m) resid1[i] -= b[i] * g->dtau;
      FOR(i,m) resid1[i] -= eta * (b[i] * tau - Ax[i]);
      mul_Atran(resid2, g->dy);
      FOR(i,n) resid2[i] += g->ds[i];
      FOR(i,n) resid2[i] -= c[i] * g->dtau;
      FOR(i,n) resid2[i] -= eta * (c[i] * tau - ATy[i] - s[i]);
      FOR(i,n) resid3[i] = g->ds[i];
      scaling.scale(resid3);
      FOR(i,n) resid3[i] += g->dx[i];
      FOR(i,n) resid3[i] -= -x[i] + gamma*mu/s[i];
      resid4 = dot(b, g->dy, m) - dot(c, g->dx, n) - g->dkappa - eta * rG;
      resid5 = kappa * g->dtau + tau * g->dkappa - gamma*mu + kappa*tau;
      double r1norm = dot(resid1, resid1, m), r2norm = dot(resid2, resid2, n),
          r3norm = dot(resid3, resid3, n);
      printf("%g %g %g %g %g %g\n", t, r1norm, r2norm, r3norm, resid4, resid5);
    }
    */
  }
  
  int hsd_gondzio(gradients *g) {
    ScopeTimer _st("hsd_gondzio");
    double delta_alpha = 0.1, gamma = 0.1, betamin = 0.2, betamax = 5.0;
    double alphap = 1, alphad = 1;
    find_unbalanced_steplen(*g, alphap, alphad);
    double cur_goodness = eval_step(*g, alphap * 0.99, alphad * 0.99);
  
    double alp = min(1.08*alphap + 0.08, 1.0);
    double ald = min(1.08*alphad + 0.08, 1.0); 
  
    vecmd rhs1;
    vecnd rhs2, rhs3;
    FOR(i,m) rhs1[i] = 0;
    FOR(i,n) rhs2[i] = 0;
    vecnd cx, cs;
    vecmd cy;
  
    double tgtmu = mu;
  
    FOR(i,n) {
      double cpi = (x[i] + alp * g->dx[i]) * (s[i] + ald * g->ds[i]);
      if      (cpi < betamin * tgtmu) rhs3[i] = (betamin*tgtmu - cpi) / s[i];
      else if (cpi > betamax * tgtmu) rhs3[i] = (-betamax/2 * tgtmu) / s[i];
      else rhs3[i] = 0;
    }
    double ala = sqrt(alp * ald);
    double cpi = (kappa + ala * g->dkappa) * (tau + ala * g->dtau);
    double rhs5;
    if      (cpi < betamin * tgtmu) rhs5 = betamin * tgtmu - cpi;
    else if (cpi > betamax * tgtmu) rhs5 = -betamax/2 * tgtmu;
    else rhs5 = 0;
  
    find_xys(&cx[0], &cy[0], &cs[0], &rhs1[0], &rhs2[0], &rhs3[0], 1);
  
    // 0   A	0	-b	0	0
    // A^T 0	I	-c	0	0
    // 0   I	T^2	0	0	rhs3
    // b   -c	0	0	-1	0
    // 0   0	0	kappa	tau	rhs5
    double dtau = (dot(b, &cy[0], m) - dot(c, &cx[0], n) - rhs5/tau)
                / (dot(b, xhy_y1, m) - dot(c, xhy_x1, n) - kappa);
    double dkappa = rhs5/tau - kappa*dtau;
    FOR(i,n) cx[i] -= dtau * xhy_x1[i];
    FOR(i,m) cy[i] -= dtau * xhy_y1[i];
    FOR(i,n) cs[i] -= dtau * xhy_s1[i];
    dtau *= tau;
  
    if (0) {
      vecmd r1; vecnd r2, r3;
  
      mul_A(&r1[0], &cx[0]);
      FOR(i,m) r1[i] -= b[i] * dtau;
  
      mul_Atran(&r2[0], &cy[0]);
      FOR(i,n) r2[i] += cs[i];
      FOR(i,n) r2[i] -= c[i] * dtau;
  
      FOR(i,n) r3[i] = cs[i];
      scaling.scale(&r3[0]);
      FOR(i,n) r3[i] += cx[i] - rhs3[i];
  
      double r4 = dot(b, &cy[0], m) - dot(c, &cx[0], n) - dkappa;
      double r5 = kappa * dtau + tau * dkappa - rhs5;
  
      printf("%g %g %g %g %g\n", dot(&r1[0], &r1[0], m),
          dot(&r2[0], &r2[0], n), dot(&r3[0], &r3[0], n), r4*r4, r5*r5);
    }
  
    static gradients *gg = new gradients;
    FOR(i,n) gg->dx[i] = g->dx[i] + cx[i];
    FOR(i,m) gg->dy[i] = g->dy[i] + cy[i];
    FOR(i,n) gg->ds[i] = g->ds[i] + cs[i];
    gg->dkappa = g->dkappa + dkappa;
    gg->dtau = g->dtau + dtau;
  
    double new_goodness = eval_step(*gg);
  
    if (new_goodness < cur_goodness * 0.99) {
      FOR(i, n) g->dx[i] += cx[i], g->ds[i] += cs[i];
      FOR(i, m) g->dy[i] += cy[i];
      g->dkappa += dkappa;
      g->dtau += dtau;
      return 2;
    }
    return 0;
  }
  
  void hsd_gondzio(gradients *g, int power) {
    while (power-- && hsd_gondzio(g));
  }
  
  void hsd_find_stale_corrector(gradients *pg, int power = 1) {
    ScopeTimer _st("hsd_find_stale_corrector");
    vecmd stale_r1, cy;
    vecnd stale_r2, stale_r3, cx, cs;
  
    #define CAREFUL_STALE 1
  
    #if CAREFUL_STALE
    double plen, dlen;
    find_unbalanced_steplen(*pg, plen, dlen);
    plen *= .99; dlen *= .99;
    double bestmu = eval_step(*pg, plen, dlen);
    #endif
  
    for (int iter = 0; iter < power; iter++) {
      FOR(i, m) stale_r1[i] = 0;
      FOR(i, n) stale_r2[i] = 0;
      const double *rhs3 = iter ? &cs[0] : pg->ds;
      FOR(i, n) stale_r3[i] = rhs3[i];
      scaling.scale(&stale_r3[0]);
      FOR(i, n) stale_r3[i] = (stale_r3[i] - x[i] * rhs3[i] / s[i]);
      find_xys(&cx[0], &cy[0], &cs[0], &stale_r1[0], &stale_r2[0], &stale_r3[0], 1);
     
      double rhs5 = 0;
      double dtau = (dot(b, &cy[0], m) - dot(c, &cx[0], n) - rhs5/tau)
                  / (dot(b, xhy_y1, m) - dot(c, xhy_x1, n) - kappa);
      double dkappa = rhs5/tau - kappa*dtau;
      FOR(i,n) cx[i] -= dtau * xhy_x1[i];
      FOR(i,m) cy[i] -= dtau * xhy_y1[i];
      FOR(i,n) cs[i] -= dtau * xhy_s1[i];
      dtau *= tau;
  
      FOR(i, n) pg->dx[i] += cx[i];
      FOR(i, m) pg->dy[i] += cy[i];
      FOR(i, n) pg->ds[i] += cs[i];
      pg->dtau += dtau; pg->dkappa += dkappa;
      #if CAREFUL_STALE
      find_unbalanced_steplen(*pg, plen, dlen);
      plen *= .99; dlen *= .99;
      double thismu = eval_step(*pg, plen, dlen);
      if (thismu > bestmu) {
        FOR(i, n) pg->dx[i] -= cx[i];
        FOR(i, m) pg->dy[i] -= cy[i];
        FOR(i, n) pg->ds[i] -= cs[i];
        pg->dtau -= dtau; pg->dkappa -= dkappa;
        printf("rejected stale corrector\n");
        break;
      }
      bestmu = thismu;
      #endif
    }
    #undef CAREFUL_STALE
  }
  
  gradients predictor, corrector, mehrotra, predcorr, mod_mehrotra;
  double mehrotra_gamma;

  void check_step_goodness(const gradients &g) {
      /*
      mul_A(resid1, g->dx);
      FOR(i,m) resid1[i] -= b[i] * g->dtau;
      FOR(i,m) resid1[i] -= eta * (b[i] * tau - Ax[i]);
      */
      /*
    if (0) {
      vecmd asdf;
      mul_A(asdf, pg->dx);
      FOR(i,m) printf("%12g %12g %12g\n", asdf[i], rP[i] - t * b[i] * tau,
          asdf[i] - rP[i] - pg->dtau * b[i]);
    }
          */
    vecmd rP, drP; vecnd rD, drD;
    mul_A(rP, x); mul_A(drP, g.dx);
    FOR(i,m) rP[i] = tau * b[i] - rP[i];
    FOR(i,m) drP[i] -= b[i] * g.dtau;
    mul_Atran(rD, y); mul_Atran(drD, g.dy);
    FOR(i,n) rD[i] = tau * c[i] - rD[i] - s[i];
    FOR(i,n) drD[i] += g.ds[i] - c[i] * g.dtau;
    FOR(i,m) drP[i] += 1e-200;
    FOR(i,n) drD[i] += 1e-200;
    FOR(i,m) printf("p %6i %12g %12g %12g\n", i, rP[i], drP[i], rP[i] / drP[i]);
    FOR(i,n) printf("d %6i %12g %12g %12g\n", i, rD[i], drD[i], rD[i] / drD[i]);
  }

  int find_mehrotra_dirs() {
    ScopeTimer _st("xhy::find_mehrotra_dirs");
    find_dirs(&predictor, &corrector);
  
    double alphap = 1, alphad = 1;
    find_unbalanced_steplen(predictor, alphap, alphad);
  
    double new_value = eval_step(predictor, .99*alphap, .99*alphad);
    double newmu = (tau + min(alphap, alphad) * predictor.dtau)
        * (kappa + min(alphap, alphad) * predictor.dkappa);
    FOR(i,n) newmu += (x[i] + alphap * predictor.dx[i])
                    * (s[i] + alphad * predictor.ds[i]);
    newmu /= n+1;
    if (newmu > mu) {
      alphap = alphad = min(alphap, alphad);
      newmu = (tau + min(alphap, alphad) * predictor.dtau)
          * (kappa + min(alphap, alphad) * predictor.dkappa);
      FOR(i,n) newmu += (x[i] + alphap * predictor.dx[i])
                      * (s[i] + alphad * predictor.ds[i]);
      newmu /= n+1;
      if (newmu > mu) {
        printf("Something has gone horribly wrong.\n");
        abort();
      }
    }
    double gamma = newmu/mu; gamma *= gamma*gamma;
    mehrotra_gamma = gamma;
  
    vector<double> rhs1(3*m), rhs2(3*n), rhs3(3*n);
    vector<double> dxout(3*n), dyout(3*m), dsout(3*n);
    FOR(i,m) rhs1[0+i] = (1-gamma) * rP[i];
    FOR(i,m) rhs1[m+i] = (1-gamma) * rP[i];
    FOR(i,m) rhs1[2*m+i] = 0;
    FOR(i,n) rhs2[0+i] = (1-gamma) * rD[i];
    FOR(i,n) rhs2[n+i] = (1-gamma) * rD[i];
    FOR(i,n) rhs2[2*n+i] = 0;
    FOR(i,n) rhs3[0+i] = -x[i] + gamma * mu / s[i]
                       - predictor.dx[i] * predictor.ds[i] / s[i];
    FOR(i,n) rhs3[n+i] = -x[i] + gamma * mu / s[i];
    FOR(i,n) rhs3[2*n+i] = -predictor.dx[i] * predictor.ds[i] / s[i];
    double rhs4 = (1-gamma) * rG;
    double rhs5 = gamma*mu - kappa*tau;
    find_xys(&dxout[0], &dyout[0], &dsout[0], &rhs1[0], &rhs2[0], &rhs3[0], 3);
    FOR(i,n) mehrotra.dx[i] = dxout[i];
    FOR(i,m) mehrotra.dy[i] = dyout[i];
    FOR(i,n) mehrotra.ds[i] = dsout[i];
    double t = find_t(mehrotra.dx, mehrotra.dy, xhy_x1, xhy_y1, 1-gamma, mu,
        &mehrotra.dtau, &mehrotra.dkappa);
    //printf("mehrotra tee is %g\n", t);
    FOR(i,n) mehrotra.dx[i] += t * xhy_x1[i];
    FOR(i,m) mehrotra.dy[i] += t * xhy_y1[i];
    FOR(i,n) mehrotra.ds[i] += t * xhy_s1[i];
  
    double meh_value = eval_step(mehrotra);
  
    if (meh_value > mu) {
      return 3;
      // If Mehrotra makes things worse or isn't even as good as the predictor
      // step, don't do it.
      ScopeTimer _st("mehrotra_fixup");
      FOR(i, n) {
        predcorr.dx[i] = dxout[n+i];
        mehrotra.dx[i] = dxout[2*n+i];
      }
      FOR(i, m) {
        predcorr.dy[i] = dyout[m+i];
        mehrotra.dy[i] = dyout[2*m+i];
      }
      FOR(i, n) {
        predcorr.ds[i] = dsout[n+i];
        mehrotra.ds[i] = dsout[2*n+i];
      }
      double tpc = find_t(predcorr.dx, predcorr.dy, xhy_x1, xhy_y1, 1-gamma,
          mu, &predcorr.dtau, &predcorr.dkappa);
      FOR(i,n) predcorr.dx[i] += tpc * xhy_x1[i];
      FOR(i,m) predcorr.dy[i] += tpc * xhy_y1[i];
      FOR(i,n) predcorr.ds[i] += tpc * xhy_s1[i];
      double tm = find_t(mehrotra.dx, mehrotra.dy, xhy_x1, xhy_y1, 0, mu,
          &mehrotra.dtau, &mehrotra.dkappa);
      FOR(i,n) mehrotra.dx[i] += tm * xhy_x1[i];
      FOR(i,m) mehrotra.dy[i] += tm * xhy_y1[i];
      FOR(i,n) mehrotra.ds[i] += tm * xhy_s1[i];

      double pcp, pcd;
      find_unbalanced_steplen(predcorr, pcp, pcd);
      pcp *= 0.99; pcd *= 0.99;
      double pc_value = eval_step(predcorr, pcp, pcd);

      double xms=0, pxms=0, smx=0, psmx=0, xps=0, spx=0;
      FOR(i, n) {
        xms += x[i] * mehrotra.ds[i];
        pxms += predcorr.dx[i] * mehrotra.ds[i];
        smx += s[i] * mehrotra.dx[i];
        psmx += predcorr.ds[i] * mehrotra.dx[i];
        xps += x[i] * predcorr.ds[i];
        spx += s[i] * predcorr.dx[i];
      }
      double xeffect = smx + pcd * psmx, seffect = xms + pcp * pxms;

      vector<int> xinds, sinds;
      FOR(i, n) {
        double xx = x[i] + pcp * predcorr.dx[i];
        if (xx < fabs(pcp * mehrotra.dx[i])) xinds.push_back(i);
        double ss = s[i] + pcp * predcorr.ds[i];
        if (ss < fabs(pcp * mehrotra.ds[i])) sinds.push_back(i);
      }
      double xcoeff = 0, bestx = pcp * spx;
      double scoeff = 0, bests = pcd * xps;
      for (double beta = -1; beta <= 1; beta += 0.125) {
        double lp=1, ld=1;
        FOR(ii, xinds.size()) {
          int i = xinds[ii];
          double dx = predcorr.dx[i] + beta * mehrotra.dx[i];
          if (dx < 0) lp = min(lp, -x[i] / dx);
        }
        FOR(ii, sinds.size()) {
          int i = sinds[ii];
          double ds = predcorr.ds[i] + beta * mehrotra.ds[i];
          if (ds < 0) ld = min(lp, -s[i] / ds);
        }
        double pval = lp * (spx + beta * xeffect);
        double dval = ld * (xps + beta * seffect);
        if (pval < bestx) bestx = pval, xcoeff = beta;
        if (dval < bests) bests = dval, scoeff = beta;
      }

      FOR(i, n) mod_mehrotra.dx[i] = predcorr.dx[i] + xcoeff * mehrotra.dx[i];
      FOR(i, m) mod_mehrotra.dy[i] = predcorr.dy[i] + scoeff * mehrotra.dy[i];
      FOR(i, n) mod_mehrotra.ds[i] = predcorr.ds[i] + scoeff * mehrotra.ds[i];
      mod_mehrotra.dtau = predcorr.dtau
          + min(xcoeff * mehrotra.dtau, scoeff * mehrotra.dtau);
      mod_mehrotra.dkappa = predcorr.dkappa
          + min(xcoeff * mehrotra.dkappa, scoeff * mehrotra.dkappa);

      double mod_value = eval_step(mod_mehrotra);
      double bestval = min(mod_value, min(pc_value, min(new_value, mu)));
      if (bestval == mod_value) {
        return 1;
      } else if (bestval == pc_value) {
        return 2;
      } else if (bestval == new_value) {
        return 2;
      } else if (bestval == mu) {
        return 4;
      } else abort();
    }
    return 0;
  }
};

long long iter_begin;
static double chol_gap_top = 0, chol_gap_bot = 1e-20;
static double chol_pot_top = 0, chol_pot_bot = 1e-20;
static double chol_usecs_top = 0, chol_usecs_bot = 1e-20;
static double stale_gap_top = -1e-20, stale_gap_bot = 1e-20;
static double stale_pot_top = -1e-20, stale_pot_bot = 1e-20;
static double stale_usecs_top = 0, stale_usecs_bot = 1e-20;

int max_gondzio = 4;

struct stepping_strategy {
  dfp_scaled_system dss;
  int stale_cholesky;
  int last_step_weak = 1;
  xhy_logic xhy;

  stepping_strategy() : xhy(dss) {
  }

  void dfp_reduce(double *updown, int n) {
    double vv, vw, ww;
    vv=vw=ww=0;
    FOR(i,n) {
      vv += updown[i  ] * updown[i  ];
      vw += updown[i  ] * updown[i+n];
      ww += updown[i+n] * updown[i+n];
    }
    // "Rotate" things so that the update and downdate are orthogonal.
    double A = vw, B = vv+ww;
    double ess;
    double c2 = B*B-4*A*A;
    double c0 = -A*A;
    double disc = c2*c2 - 4*c0*c2;
    if (c2 <= 1e-8 * B*B || disc < 0) {
      // In practice, this "optimal rotation" shouldn't destroy many
      // accurate digits, so this branch should be taken rarely.  (If it is
      // taken, though, you can kiss at least 3 digits of accuracy goodbye.)
      ess = 1e8;
    } else {
      disc = sqrt(disc);
      ess = (disc - c2) / c2 / 2;
      if (ess < 1) ess = 1;
    }
    double tee = sqrt(ess-1);
    ess = sqrt(ess);
    if (vw > 0) tee = -tee;

    int testcoord = rand() % n;
    vecnd testvec;
    FOR(i,n) testvec[i] = updown[i] * updown[testcoord] - updown[n+i] * updown[n+testcoord];

    if (tee*tee < 1) {
      FOR(i, n) {
        double tmp  = ess * updown[i] + tee * updown[n+i];
        updown[n+i] = tee * updown[i] + ess * updown[n+i];
        updown[i] = tmp;
      }
    } else if (tee > 0) {
      double rest = 1 / (ess + tee);
      FOR(i, n) {
        double tmp = tee * (updown[i] + updown[n+i]);
        updown[i]   = tmp + rest * updown[i];
        updown[n+i] = tmp + rest * updown[n+i];
      }
    } else {
      double rest = 1 / (ess - tee);
      FOR(i, n) {
        double tmp = tee * (updown[i] - updown[n+i]);
        updown[i]   = -tmp + rest * updown[i];
        updown[n+i] =  tmp + rest * updown[n+i];
      }
    }

    vecnd testvec2;
    FOR(i,n) testvec2[i] = updown[i] * updown[testcoord] - updown[n+i] * updown[n+testcoord];
    double dif = 0;
    FOR(i,n) dif += fabs(testvec[i] - testvec2[i]);
  }

  vecnd scal;

  void iteration() {
    ScopeTimer _st("iteration");
    stale_cholesky++;
    if (last_step_weak > 0 || stale_cholesky > flag_max_stale || flag_no_stale) {
      stale_cholesky = 0;
      last_step_weak = 0;
      revive_rows();
      if (((diag_scaled_system *)dss.base.get())->cs.handle_rechol()) {
        mul_A(Ax, x);
        compute_sd_coeffs();
      }
      ScopeTimer _st("dfp::update_root");
      dss.update_root();
      FOR(i,n) scal[i] = x[i]/s[i];

    }
    dss.prep();

    mu = best_mu(x,s,tau,kappa);

    static double neigh = flag_mehrotra_neigh, badneigh = 0.99;
    double oldpot = potential();
    if (!stale_cholesky) { // Fresh Cholesky factor; do a Mehrotra step
      switch (xhy.find_mehrotra_dirs()) {
        case 0: {
          xhy.hsd_gondzio(&xhy.mehrotra, max_gondzio);
          if (do_unbalanced_aggressive_step(xhy.mehrotra, neigh)) {
            throw "Couldn't take a Mehrotra step even though it was best.";
          }
          last_step_kind = 'm';
        } break;
        case 1: {
          xhy.hsd_gondzio(&xhy.mod_mehrotra, max_gondzio);
          if (do_unbalanced_aggressive_step(xhy.mod_mehrotra, neigh)) {
            throw "Couldn't take a modified Mehrotra step even though it was best.";
          }
          last_step_kind = 'M';
        } break;
        case 2: {
          xhy.hsd_gondzio(&xhy.predcorr, max_gondzio);
          if (do_unbalanced_aggressive_step(xhy.predcorr, badneigh)) {
            throw "Couldn't take a pred-corr step even though it was best.";
          }
          last_step_kind = 'C';
        } break;
        case 3: {
          xhy.hsd_gondzio(&xhy.predictor, max_gondzio);
          if (do_unbalanced_aggressive_step(xhy.predictor, badneigh)) {
            throw "Couldn't take a predictor step even though it was best.";
          }
          last_step_kind = 'P';
        } break;
        case 4: {
          printf("Something has gone horribly wrong.\n");
          abort();
        } break;
        default: {
          printf("Unknown return from xhy::find_mehrotra_dirs.\n");
          abort();
        } break;
      }
      double got = oldpot - potential();
      chol_gap_top += -log(best_mu(x,s,tau,kappa) / mu);
      chol_gap_bot++;
      chol_pot_top += got;
      chol_pot_bot++;
      long long after = get_cpu_usecs();
      chol_usecs_top += after-iter_begin;
      chol_usecs_bot++;
    } else { // Stale Cholesky factor.  Do affine scaling.
      xhy.find_dirs(&xhy.predictor, &xhy.corrector);
      if (flag_stale_corrector) xhy.hsd_find_stale_corrector(&xhy.predictor);
      double alphap = 1, alphad = 1;
      find_unbalanced_steplen(xhy.predictor, alphap, alphad);
      double gamma;
      {
        double newmu = (tau + min(alphap, alphad) * xhy.predictor.dtau)
            * (kappa + min(alphap, alphad) * xhy.predictor.dkappa);
        FOR(i,n) newmu += (x[i] + alphap * xhy.predictor.dx[i])
                        * (s[i] + alphad * xhy.predictor.ds[i]);
        newmu /= n+1;
        gamma = newmu / mu;
        gamma *= gamma;
      }
      FOR(i,n) xhy.predictor.dx[i] = (1-gamma) * xhy.predictor.dx[i] + gamma * xhy.corrector.dx[i];
      FOR(i,m) xhy.predictor.dy[i] = (1-gamma) * xhy.predictor.dy[i] + gamma * xhy.corrector.dy[i];
      FOR(i,n) xhy.predictor.ds[i] = (1-gamma) * xhy.predictor.ds[i] + gamma * xhy.corrector.ds[i];
      xhy.predictor.dtau = (1-gamma) * xhy.predictor.dtau + gamma * xhy.corrector.dtau;
      xhy.predictor.dkappa += (1-gamma) * xhy.predictor.dkappa + gamma * xhy.corrector.dkappa;
      if (do_unbalanced_aggressive_step(xhy.predictor, flag_stale_neigh))
        last_step_weak = 1;
  
      double got = oldpot - potential();
      stale_gap_top += -log(best_mu(x,s,tau,kappa) / mu);
      stale_gap_bot++;
      stale_pot_top += got;
      stale_pot_bot++;
      long long after = get_cpu_usecs();
      stale_usecs_top += after-iter_begin;
      stale_usecs_bot++;
      if (got / (after-iter_begin+1) < 2 * chol_pot_top / chol_usecs_top) {
        //last_step_weak++;
      }
      last_step_kind = 's';
    }
  }

  void revive_rows() {
    return;
    // Rows that were numerically linearly dependent in the past can become
    // relevant as the scaling eliminates the apparent linear dependence.
    // We revive rows whose infeasibility starts causing a problem.
    if (!stale_cholesky) {
      double maxpinf = 0;
      FOR(i, m) if (!dead_rows[i])
        maxpinf = max(maxpinf, fabs(Ax[i] - tau * b[i]));
      FOR(i, m) if (dead_rows[i]) {
        double here = fabs(Ax[i] - tau * b[i]);
        if (here > maxpinf * .05) {
          revive_row(i);
        }
      }
    }
  }
};

void read_problem() {
  scanf("%i %i", &m, &n);
  FOR(i, m) scanf("%lf", b+i);
  FOR(i, n) scanf("%lf", c+i);

  int i, j; double d;
  while (3 == scanf("%i %i %lf ", &i, &j, &d))
    sparse_A[i].push_back(make_pair(j, d));
}

static double best_invariant = 1e99;
static double first_invariant = 1e99;
static double best_rpinf = 1e99, best_rdinf = 1e99;

vector<double> best_x, best_y, best_s; double best_tau, best_kappa;

int converged() {
  #if 0
  // "Classical" convergence metric.
  double rp2 = 0, rd2 = 0, rg = 0, b2 = 0, c2 = 0;
  FOR(i,m) rp2 += (b[i]-Ax[i]/tau) * (b[i]-Ax[i]/tau);
  rp2 = sqrt(rp2);
  FOR(i,m) b2 += b[i]*b[i];
  b2 = sqrt(b2);
  FOR(i,n) rd2 += (c[i]-(ATy[i]+s[i])/tau) * (c[i]-(ATy[i]+s[i])/tau);
  rd2 = sqrt(rd2);
  FOR(i,n) c2 += c[i]*c[i];
  c2 = sqrt(c2);
  double cx = 0, by = 0;
  FOR(i,n) cx += c[i]*x[i];
  FOR(i,m) by += b[i]*y[i];
  double pbadness = rp2 / (1 + b2), dbadness = rd2 / (1 + c2);
  double gbadness = mu / (1 + fabs(cx));
  return pbadness < 1e-8 && dbadness < 1e-8 && gbadness < 1e-8;
  #else
  // Freund's metric.
  double rpinf = 0, rdinf = 0, binf = 0, cinf = 0, cx = 0, by = 0;
  static double best_rpinf = 1e30, best_rdinf = 1e30, best_rg = 1e30;
  double invariant = 0;
  FOR(i,n) {
    double here = ATy[i] + s[i] - tau * c[i];
    rdinf = max(rdinf, fabs(here));
    cx += c[i] * x[i];
    cinf = max(cinf, fabs(c[i]));
    invariant += here*here;
  }
  FOR(i,m) {
    double here = Ax[i] - tau * b[i];
    rpinf = max(rpinf, fabs(here));
    by += b[i] * y[i];
    binf = max(binf, fabs(b[i]));
    invariant += here*here;
  }
  {
    double g0 = cx-by+kappa;
    invariant += g0*g0;
  }
  double rg = (cx - by) / tau;
  rdinf /= tau; rpinf /= tau; by /= tau; cx /= tau;

  if (invariant < best_invariant) {
    best_invariant = invariant;
    best_x.resize(n); best_y.resize(m); best_s.resize(n);
    FOR(i,n) best_x[i] = x[i];
    FOR(i,m) best_y[i] = y[i];
    FOR(i,n) best_s[i] = s[i];
    best_tau = tau; best_kappa = kappa;
  }
  best_rpinf = min(rpinf, best_rpinf);
  best_rdinf = min(rdinf, best_rdinf);

  // Use best_rpinf here because it may be hard to lower primal infeasibility
  // when numerical error dominates the dx part of the search direction.
  double metric = 2 * best_rpinf / (1 + binf) + 2 * rdinf / (1 + cinf)
      + max(rg, 0.0) / max(1.0, max(fabs(cx), fabs(by)));
  return metric < 1e-8;
  #endif
}

extern "C" void openblas_set_num_threads(int);

int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--no-stale")) {
      flag_no_stale = true;
      flag_mehrotra_neigh = 0.99;
    } else if (!strncmp(argv[i], "--max-stale=", 12)) {
      sscanf(argv[i], "--max-stale=%d", &flag_max_stale);
    } else if (!strncmp(argv[i], "--meh-neigh=", 12)) {
      sscanf(argv[i], "--meh-neigh=%lf", &flag_mehrotra_neigh);
    } else if (!strncmp(argv[i], "--stale-neigh=", 14)) {
      sscanf(argv[i], "--stale-neigh=%lf", &flag_stale_neigh);
    } else if (!strncmp(argv[i], "--no-stale-corr", 15)) {
      flag_stale_corrector = 0;
    } else if (!strncmp(argv[i], "--stale-corr", 12)) {
      flag_stale_corrector = 1;
    } else if (!strncmp(argv[i], "--no-rebuild-factor", 19)) {
      flag_rebuild_factor = 0;
    } else if (!strncmp(argv[i], "--rebuild-factor", 16)) {
      flag_rebuild_factor = 1;
    } else if (!strncmp(argv[i], "--no-simplicial-factor", 22)) {
      flag_simplicial_factor = 0;
    } else if (!strncmp(argv[i], "--simplicial-factor", 19)) {
      flag_simplicial_factor = 1;
    }
  }

  feenableexcept(FE_INVALID);
  openblas_set_num_threads(1);
  setlinebuf(stdout);
  read_problem();
  do_cholesky_init();
  init_naive_initial_point();
  stepping_strategy *strat = new stepping_strategy;
  long long before = get_cpu_usecs();
  mul_A(Ax, x);
  mul_Atran(ATy, y);
  try {
    while (1) {
      iter_begin = get_cpu_usecs();
      strat->iteration();
      long long after = get_cpu_usecs();
      ++iterno;
      mul_A(Ax, x);
      mul_Atran(ATy, y);
      verify_current_iterate(after-before);
      compute_sd_coeffs();
      if (converged()) break;
      if (iterno > 5000) abort();
    }
  } catch (const char *errstring) {
    printf("%s; restoring best iterate\n", errstring);
    FOR(i,n) x[i] = best_x[i];
    FOR(i,m) y[i] = best_y[i];
    FOR(i,n) s[i] = best_s[i];
    tau = best_tau;
    kappa = best_kappa;
    abort();
  }

  FOR(i,n) x[i] /= tau, s[i] /= tau;
  FOR(i,m) y[i] /= tau;
  kappa /= tau;
  tau = 1;
  mul_A(Ax, x);
  mul_Atran(ATy, y);
  double cx = 0, by = 0, pinf = 0, dinf = 0;
  FOR(i,n) cx += c[i] * x[i], dinf = max(dinf, fabs(ATy[i] + s[i] - c[i]));
  FOR(i,m) by += b[i] * y[i], pinf = max(pinf, fabs(Ax[i] - b[i]) / (1+fabs(b[i])));
  printf("%g %g %g %g\n", cx, by, pinf, dinf);
  printf("Did %i Choleskies, %i halfsolves\n",
      stat_num_choleskies, stat_num_halfsolves);
  printf("Average Mehrotra potential reduction:  %g (%g per usec)\n", chol_pot_top / chol_pot_bot, chol_pot_top / chol_usecs_top);
  printf("Average    stale potential reduction:  %g (%g per usec)\n", stale_pot_top / stale_pot_bot, stale_pot_top / stale_usecs_top);

  vector<string> scope_records;
  FORALL(it, scope_usecs_top) {
    const char *p = it->first;
    char buf[4096];
    sprintf(buf, "%s: %8.0f %12.0f %12.0f", p, scope_usecs_bot[p], scope_usecs_top[p], scope_usecs_top[p] / scope_usecs_bot[p]);
    scope_records.push_back(buf);
  }
  sort(scope_records.begin(), scope_records.end());
  FOR(i, scope_records.size()) printf("%80s\n", scope_records[i].c_str());
}
