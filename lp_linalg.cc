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

#include "lp.h"
#include <stdio.h>
#include <stdlib.h>
#include <bitset>
#include <set>
#include <math.h>
#include <random>


static std::default_random_engine randn_generator;
static std::normal_distribution<double> randn_distribution(0.0,1.0);

static double __attribute__((unused)) randn() {
  return randn_distribution(randn_generator);
}


void diag_scaled_system::update_root() {
  prep();
}

void diag_scaled_system::prep() {
  scal.resize(n);
  FOR(i, n) scal[i] = x[i]/s[i];
  cs.build(&scal[0]);
}

void diag_scaled_system::get_dy(double *ans, const double *rhs, int nrhs) const {
  FOR(i, m * nrhs) ans[i] = rhs[i];
  cs.lower_half_solve(ans, nrhs);
  cs.upper_half_solve(ans, nrhs);
  FOR(z,4) if (1) { // One step of iterative refinement.
    FOR(r, nrhs) {
      vecnd atx; vecmd aatx;
      mul_Atran(atx, ans + r*m);
      FOR(i,n) atx[i] *= scal[i];
      mul_A(aatx, atx);
      FOR(i,m) aatx[i] = rhs[i + r*m] - aatx[i];
      double len = 0;
      FOR(i,m) len += aatx[i]*aatx[i];
      printf("%g\n", len);
      cs.lower_half_solve(aatx, 1);
      cs.upper_half_solve(aatx, 1);
      FOR(i,m) ans[i + r*m] += aatx[i];
    }
  }
}

void diag_scaled_system::scale(double *v) const {
  FOR(i,n) v[i] *= scal[i];
}

void diag_scaled_system::lower_half_solve_T(double *v, int nrhs) const {
  cs.lower_half_solve_T(v, nrhs);
}

void diag_scaled_system::lower_half_solve(double *v, int nrhs) const {
  cs.lower_half_solve(v, nrhs);
}

void diag_scaled_system::upper_half_solve(double *v, int nrhs) const {
  cs.upper_half_solve(v, nrhs);
}


void dfp_scaled_system::lower_half_solve_T(double *v, int nrhs) const {
  base->lower_half_solve_T(v, nrhs);
}

void dfp_scaled_system::lower_half_solve(double *v, int nrhs) const {
  base->lower_half_solve(v, nrhs);
}

void dfp_scaled_system::upper_half_solve(double *v, int nrhs) const {
  base->upper_half_solve(v, nrhs);
}

void dfp_scaled_system::update_root() {
  base->prep();
  base_initialised = 1;
  x.resize(n); s.resize(n);
  FOR(i,n) x[i] = ::x[i], s[i] = ::s[i];
  fresh_cholesky = 1;
}

void dfp_scaled_system::prep() {
  if (!base_initialised) update_root();
  base->update();

  x.resize(n); s.resize(n);
  FOR(i,n) if (x[i] != ::x[i] || s[i] != ::s[i]) {
    fresh_cholesky = 0;
    x[i] = ::x[i], s[i] = ::s[i];
  }
  if (fresh_cholesky) return;

  ScopeTimer st("dss::prep");

  e.resize(n); f.resize(n); Hs.resize(n); H1e.resize(n);
  xs = 0;
  FOR(i,n) xs += x[i] * s[i];
  mu = xs / n;
  sHs = 0;
  sHe = 0; ef = 0;
  FOR(i,n) {
    e[i] = s[i] - mu/x[i];
    f[i] = x[i] - mu/s[i];
    Hs[i] = s[i];
    ef += e[i] * f[i];
  }
  base->scale(&Hs[0]);
  FOR(i,n) {
    sHs += s[i] * Hs[i];
    sHe += Hs[i] * e[i];
  }

  vvv.resize(4*n+4); vv = &vvv[0];
  while ((intptr_t)vv & 31) vv++;
  uuu.resize(4*m+4); uu = &uuu[0];
  while ((intptr_t)uu & 31) uu++;

  build_vv();
  mul_A_v4(uu, vv);
  base->lower_half_solve_T(uu, 4);

  FOR(i,4) FOR(j,4) M[i][j] = 0;
  FOR(i,m) FOR(j,4) FOR(k,j+1) M[k][j] += uu[4*i+j] * uu[4*i+k];
  FOR(j,4) FOR(k,j) M[j][k] = M[k][j];
  build_M();
}

void dfp_scaled_system::build_vv() {
  FOR(i,n) H1e[i] = e[i];
  base->scale(&H1e[0]);
  FOR(i,n) H1e[i] -= sHe/sHs * Hs[i];
  eH1e = 0;
  FOR(i,n) eH1e += e[i] * H1e[i];

  double scal[4] = {1/sqrt(xs), 1/sqrt(sHs), 0.0, 0.0};
  if (ef > 0) scal[2] = 1/sqrt(ef);
  if (eH1e > 0) scal[3] = 1/sqrt(eH1e);
  FOR(i,n) {
    vv[4*i+0] = scal[0] * x[i] + scal[1] * Hs[i];
    vv[4*i+1] = scal[0] * x[i] - scal[1] * Hs[i];
    vv[4*i+2] = scal[2] * f[i] + scal[3] * H1e[i];
    vv[4*i+3] = scal[2] * f[i] - scal[3] * H1e[i];
  }
}

void dfp_scaled_system::build_M() {
  M[0][1] += 2;
  M[1][0] += 2;
  M[2][3] += 2;
  M[3][2] += 2;
}

void dfp_scaled_system::scale(double *w) const {
  if (fresh_cholesky) return base->scale(w);
  double dots[4] = {0,0,0,0};
  FOR(i,n) FOR(j,4) dots[j] += w[i] * vv[4*i+j];
  FOR(i,4) dots[i] *= .5;
  base->scale(w);
  FOR(i,n) w[i] += 
      + ((dots[0] * vv[4*i+1] + dots[1] * vv[4*i+0])
      +  (dots[2] * vv[4*i+3] + dots[3] * vv[4*i+2]));
}


void bfgs_scaled_system::build_vv() {
  double xe = 0;
  FOR(i,n) xe += x[i]*e[i];
  FOR(i,n) H1e[i] = e[i] - xe/xs * s[i];
  base->scale(&H1e[0]);
  double sH1e = 0;
  FOR(i,n) sH1e += s[i]*H1e[i];
  sH1e -= xe;
  FOR(i,n) H1e[i] -= sH1e/xs * x[i];
  eH1e = 0;
  FOR(i,n) eH1e += e[i] * H1e[i];
  double sef = 0, sxs = 1/sqrt(xs);
  if (ef > 1e-9 * xs) sef = 1/sqrt(ef);

  FOR(i,n) {
    vv[4*i+0] = x[i] * sxs;
    vv[4*i+1] = (Hs[i] - x[i]) * sxs;
    vv[4*i+2] = f[i] * sef;
    vv[4*i+3] = (H1e[i] - f[i]) * sef;
  }
}

void bfgs_scaled_system::build_M() {
  M[1][1] -= (sHs - xs) / xs;
  M[0][1] -= 1;
  M[1][0] -= 1;
  if (ef > 1e-9 * xs) M[3][3] -= (eH1e - ef) / ef;
  M[2][3] -= 1;
  M[3][2] -= 1;
}

void bfgs_scaled_system::scale(double *w) const {
  double xw = 0, fw = 0;
  FOR(i,n) xw += x[i] * w[i], fw += f[i] * w[i];
  double fwef = 0;
  if (ef > 0) fwef = fw/ef;
  FOR(i,n) w[i] -= xw/xs * s[i] + fwef * e[i];
  base->scale(w);
  double sw = 0, ew = 0;
  FOR(i,n) sw += s[i] * w[i], ew += e[i] * w[i];
  double ewef = 0;
  if (ef > 0) ewef = ew/ef;
  FOR(i,n) w[i] -= sw/xs * x[i] + ewef * f[i];
  FOR(i,n) w[i] += xw/xs * x[i] + fwef * f[i];
}

void dfp_scaled_system::get_dy(double *ans, const double *rhs, int nrhs) const {
  FOR(i,m*nrhs) ans[i] = rhs[i];
  lower_half_solve(ans, nrhs);
  if (!fresh_cholesky) {
    FOR(r, nrhs) {
      double *rans = ans + r * m;
      double A[4][5];
      FOR(i,4) A[i][4] = 0;
      FOR(i,4) FOR(j,4) A[i][j] = M[i][j];
      FOR(i,m) {
        A[0][4] += uu[4*i+0] * rans[i];
        A[1][4] += uu[4*i+1] * rans[i];
        A[2][4] += uu[4*i+2] * rans[i];
        A[3][4] += uu[4*i+3] * rans[i];
      }
      FOR(i,4) for (int j = i+1; j < 4; j++) {
        double r = hypot(A[i][i], A[j][i]);
        if (r < 1e-12) continue;
        double c = A[i][i] / r;
        double s = -A[j][i] / r;
        for (int k = i+1; k < 5; k++) {
          double t = A[i][k] * c - A[j][k] * s;
          A[j][k] = A[i][k] * s + A[j][k] * c;
          A[i][k] = t;
        }
        A[i][i] = r;
        A[j][i] = 0;
      }
      for (int i = 3; i >= 0; i--) if (fabs(A[i][i]) > 1e-12) {
        double r = A[i][i];
        FOR(j,5) A[i][j] /= r;
        FOR(j,i) {
          double r = A[j][i];
          FOR(k,5) A[j][k] -= r * A[i][k];
        }
      }
      FOR(i,4) {
        if (A[i][i] > 1e-12) A[i][4] /= A[i][i], A[i][i] = 1;
        else A[i][4] = 0;
      }
      
      FOR(i,m) rans[i] -= (A[0][4] * uu[4*i+0] + A[1][4] * uu[4*i+1])
                       + (A[2][4] * uu[4*i+2] + A[3][4] * uu[4*i+3]);
    }
  }
  upper_half_solve(ans, nrhs);

  FOR(z, 4) if (0) { // One step of iterative refinement.
    FOR(r, nrhs) {
      vecnd atx; vecmd aatx;
      mul_Atran(atx, ans + r*m);
      base->scale(atx);
      mul_A(aatx, atx);
      FOR(i,m) aatx[i] = rhs[i + r*m] - aatx[i];
      double len = 0;
      FOR(i,m) len += aatx[i]*aatx[i];
      printf("%11g ", len);
      base->lower_half_solve(aatx, 1);
      base->upper_half_solve(aatx, 1);
      FOR(i,m) ans[i + r*m] += aatx[i];
    }
    printf("\n");
  }
}

struct gradients {
  double dx[MAXN];
  double dy[MAXN];
  double ds[MAXN];
};

void get_gradients(const double *rhs1, const double *rhs2, const double *rhs3,
    const scaled_system &ss, gradients &g) {
  vecmd rhs;
  vecnd foo;
  FOR(i, n) foo[i] = rhs2[i] - rhs3[i];
  ss.scale(foo);
  mul_A(rhs, foo);
  ss.get_dy(g.dy, rhs, 1);
  mul_Atran(g.ds, g.dy);
  FOR(i,n) g.ds[i] = rhs2[i] - g.ds[i];
  FOR(i,n) g.dx[i] = rhs3[i] - g.ds[i];
  ss.scale(g.dx);
}
