#include "lp.h"
#include <qd/dd_real.h>
#include "v4.h"
#include <amd.h>
#include <colamd.h>
#include <algorithm>
#include <math.h>
#include <queue>
#include <set>

extern int m, n;

vector<pair<int, double> > sparse_A[MAXN];
vector<pair<int, double> > sparse_At[MAXN];

void build_sparse_Atran_from_A() {
  FOR(i,n) sparse_At[i].clear();
  FOR(i,m)
    FOR(jj,sparse_A[i].size())
      sparse_At[sparse_A[i][jj].first].push_back(
              make_pair(i, sparse_A[i][jj].second));
}

void build_sparse_A_from_Atran() {
  FOR(i,m) sparse_A[i].clear();
  FOR(i,n)
    FOR(jj,sparse_At[i].size())
      sparse_A[sparse_At[i][jj].first].push_back(
              make_pair(i, sparse_At[i][jj].second));
}

void dump_problem() {
  FOR(j, n) printf("%8.4f ", c[j]); printf("\n");
  FOR(i, m) {
    int jj = 0;
    FOR(j, n) {
      if (jj >= sparse_A[i].size() || sparse_A[i][jj].first != j)
                printf("         ");
      else printf("%8.4f ", sparse_A[i][jj++].second);
    }
    printf("%8.4f\n", b[i]);
  }
}

void rescale_problem() {
  FOR(zzz,5) {
    FOR(i,m) {
      double maxval = 0;
      FOR(j,sparse_A[i].size()) maxval = max(maxval, fabs(sparse_A[i][j].second));
      FOR(j,sparse_A[i].size()) sparse_A[i][j].second /= maxval;
      b[i] /= maxval;
    }
    static double maxval[MAXN];
    FOR(i,m) {
      FOR(j,sparse_A[i].size()) {
        int k = sparse_A[i][j].first;
        maxval[k] = max(maxval[k], fabs(sparse_A[i][j].second));
      }
    }
    FOR(i,m) {
      FOR(j,sparse_A[i].size()) sparse_A[i][j].second /= maxval[sparse_A[i][j].first];
    }
    FOR(i,n) c[i] /= maxval[i];
  }
}

vector<vector<int> > sparsity;

void build_sparsity_pattern(int sizlim) {
  build_sparse_Atran_from_A();

  vector<set<int> > vsi(m);
  FOR(i,n) if (sparse_At[i].size() <= sizlim) {
    FOR(jj,sparse_At[i].size()) {
      int j = sparse_At[i][jj].first;
      vsi[j].insert(j);
      FOR(kk,jj) {
        int k = sparse_At[i][kk].first;
        vsi[j].insert(k);
        vsi[k].insert(j);
      }
    }
  }

  sparsity.resize(m);
  FOR(i,m) sparsity[i] = vector<int>(vsi[i].begin(), vsi[i].end());

  int totnz = 0;
  FOR(i,m) totnz += sparsity[i].size();
  printf("totnz(%i) = %i\n", sizlim, totnz);
}

void dump_problem();

void reorder_equations() {
  vector<int> vi(m+1);
  {
    vector<int> i0, j0;
    FOR(i, sparsity.size()) {
      i0.push_back(j0.size());
      FOR(j, sparsity[i].size()) 
        j0.push_back(sparsity[i][j]);
    }
    i0.push_back(j0.size());
    if (1) {
      double info[AMD_INFO];
      amd_order(m, &i0[0], &j0[0], &vi[0], 0, info);
      amd_info(info);
    } else {
      int stats[COLAMD_STATS];
      symamd(m, &j0[0], &i0[0], &vi[0], 0, stats, calloc, free);
      symamd_report(stats);
    }
    printf("done ordering\n");
  }

  double b2[m];
  vector<vector<pair<int, double> > > vvv(m);
  FOR(i,m) vvv[i].swap(sparse_A[vi[i]]), b2[i] = b[vi[i]];
  FOR(i,m) sparse_A[i].swap(vvv[i]), b[i] = b2[i];
  build_sparse_Atran_from_A();
}

void insert_var_equalities(pair<int, int> *a, pair<int, int> *b) {
  if (b-a == 0) return;
  pair<int, int> *c = a + (b-a)/2;
  insert_var_equalities(a, c);
  insert_var_equalities(c+1, b);
  sparse_At[c->first].push_back(make_pair(m, 1));
  sparse_At[c->second].push_back(make_pair(m, -1));
  m++;
}

vector<vector<int> > dch_groups;

void do_dch_groups() {
  FOR(ii,dch_groups.size()) {
    vector<int> &vi = dch_groups[ii];
    sort(vi.begin(), vi.end());
    FOR(i,vi.size()-1) {
      sparse_A[m].push_back(make_pair(vi[i+1], 1));
      sparse_A[m].push_back(make_pair(vi[i&i+1], -1));
      b[m++] = 0;
    }
  }
  build_sparse_Atran_from_A();
}

#if 1
static const int dch_threshold = 640;
static const int dch_chunksize = 4;
#else
static const int dch_threshold = 4;
static const int dch_chunksize = 2;
#endif

void do_dense_column_hack() {
  int threshold = dch_threshold;
  int ndense = 0;
  FOR(i,n) if (sparse_At[i].size() > threshold) {
    ndense++;
    int chunksize = dch_chunksize;
    vector<int> grp(1,i);
    for (int ii = chunksize; ii < sparse_At[i].size(); ii += chunksize) {
      int iii = ii + chunksize;
      if (iii > sparse_At[i].size()) iii = sparse_At[i].size();
      sparse_At[n] =
              vector<pair<int, double> >(&sparse_At[i][ii], &sparse_At[i][iii]);
      grp.push_back(n++);
      //printf("created a group\n");
    }
    sparse_At[i].resize(chunksize);
    dch_groups.push_back(grp);
  }
  build_sparse_A_from_Atran();
  printf("Found %i dense columns.\n", ndense);
}

void do_equation_ordering() {
  build_sparsity_pattern(dch_threshold);
  do_dense_column_hack();
  do_dch_groups();
  build_sparsity_pattern(0x7fffffff);
}

void mul_A_v4(double *ans_, const double *x_) {
  ScopeTimer _st("vec4_mul_A");
#ifdef AVX
  const v4df *x = (const v4df *)x_;
  v4df *ans = (v4df *)ans_;
  FOR(i,m) {
    v4df foo = {0.0,0.0,0.0,0.0};
    v4df bar = {0.0,0.0,0.0,0.0};
    FOR(jj, sparse_A[i].size()) {
      int j = sparse_A[i][jj].first;
      double d = sparse_A[i][jj].second;
      foo += v4_broadcast(d) * x[j];
    }
    ans[i] = foo+bar;
  }
#else
  const v2df *x = (const v2df *)x_;
  v2df *ans = (v2df *)ans_;
  FOR(i,m) {
    v2df foo = {0.0,0.0};
    v2df bar = {0.0,0.0};
    FOR(jj, sparse_A[i].size()) {
      int j = sparse_A[i][jj].first;
      v2df d = v2_broadcast(sparse_A[i][jj].second);
      foo += d * x[2*j  ];
      bar += d * x[2*j+1];
    }
    ans[2*i  ] = foo;
    ans[2*i+1] = bar;
  }
#endif
}

void mul_A(dd_real *ans, const dd_real *x) {
  ScopeTimer _st("dd_mul_A");
  FOR(i,m) ans[i] = 0.0;
  FOR(i,m) FOR(jj,sparse_A[i].size()) {
    int j = sparse_A[i][jj].first;
    double d = sparse_A[i][jj].second;
    ans[i] += d * x[j];
  }
}

void mul_Atran(dd_real *ans, const dd_real *y) {
  ScopeTimer _st("dd_mul_Atran");
  FOR(i,n) ans[i] = 0.0;
  FOR(i,m) FOR(jj,sparse_A[i].size()) {
    int j = sparse_A[i][jj].first;
    double d = sparse_A[i][jj].second;
    ans[j] += d * y[i];
  }
}

static inline void twosum(double &a, double &b) {
  double x = a+b;
  double e = x-a;
  b = (a-(x-e)) + (b-e);
  a = x;
}

static inline void split(double &x, double &b) {
  double c = x * (0x1.0p26 + 1);
  double d = c-(c-x);
  b = x-d;
  x = d;
}

static inline void twoprod(double &a, double &b) {
  double a1=a, a2=0;
  double b1=b, b2=0;
  a = a*b;
  split(a1,a2);
  split(b1,b2);
  b = a2*b2 - (((a - a1*b1) - a2*b1) - a1*b2);
}

void mul_A(double *ans, const double *x) {
  ScopeTimer _st("mul_A");
  FOR(i,m) {
    double shi1 = 0, shi2 = 0, shi3 = 0, shi4 = 0, slo1 = 0, slo2 = 0, slo3 = 0, slo4 = 0;
    int maxjj = sparse_A[i].size(), jj;
    for (jj = 0; jj < maxjj-3; jj += 4) {
#define ACCUM(jj,shi,slo) do { \
      int j = sparse_A[i][jj].first; \
      double d = sparse_A[i][jj].second; \
      double hi = d, lo = x[j]; \
      twoprod(hi, lo); \
      twosum(shi, hi); \
      twosum(hi, lo); \
      twosum(slo, hi); \
      slo += lo; \
      } while (0)
      ACCUM(jj,shi1, slo1);
      ACCUM(jj+1,shi2, slo2);
      ACCUM(jj+2,shi3, slo3);
      ACCUM(jj+3,shi4, slo4);
    }
    switch (maxjj % 4) {
      case 3: ACCUM(jj+2, shi3, slo3);
      case 2: ACCUM(jj+1, shi2, slo2);
      case 1: ACCUM(jj, shi1, slo1);
      case 0:;
    }
    #undef ACCUM
    twosum(shi1, shi2); twosum(shi3, shi4); twosum(shi1, shi3);
    twosum(slo1, slo2); twosum(slo3, slo4); twosum(slo1, slo3);
    twosum(shi2, shi3); twosum(shi4, slo1); twosum(shi2, shi4);
    ans[i] = shi1 + shi2;
  }
}

void mul_Atran(double *ans, const double *y) {
  ScopeTimer _st("mul_Atran");
  FOR(i,n) {
    double shi1 = 0, shi2 = 0, shi3 = 0, shi4 = 0, slo1 = 0, slo2 = 0, slo3 = 0, slo4 = 0;
    int maxjj = sparse_At[i].size(), jj;
    for (jj = 0; jj < maxjj-3; jj += 4) {
#define ACCUM(jj,shi,slo) do { \
      int j = sparse_At[i][jj].first; \
      double d = sparse_At[i][jj].second; \
      double hi = d, lo = y[j]; \
      twoprod(hi, lo); \
      twosum(shi, hi); \
      twosum(hi, lo); \
      twosum(slo, hi); \
      slo += lo; \
      } while (0)
      ACCUM(jj,shi1, slo1);
      ACCUM(jj+1,shi2, slo2);
      ACCUM(jj+2,shi3, slo3);
      ACCUM(jj+3,shi4, slo4);
    }
    switch (maxjj % 4) {
      case 3: ACCUM(jj+2, shi3, slo3);
      case 2: ACCUM(jj+1, shi2, slo2);
      case 1: ACCUM(jj, shi1, slo1);
      case 0:;
    }
    #undef ACCUM
    twosum(shi1, shi2); twosum(shi3, shi4); twosum(shi1, shi3);
    twosum(slo1, slo2); twosum(slo3, slo4); twosum(slo1, slo3);
    twosum(shi2, shi3); twosum(shi4, slo1); twosum(shi2, shi4); 
    ans[i] = shi1 + shi2;
  }
}
