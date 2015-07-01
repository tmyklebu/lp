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

#include <cholmod.h>
#include <cblas.h>
#include "v4.h"
#include "lp.h"
#include <math.h>
#include <string.h>
#include <algorithm>
#include <unistd.h>
#include <execinfo.h>
#include <unordered_map>

int stat_num_halfsolves, stat_num_choleskies;
int flag_rebuild_factor = 0;
int flag_simplicial_factor = 0;

static cholmod_dense wrap_matrix(double *x, int m, int n);
static cholmod_dense wrap_vector(double *x, int n);

static void dump_backtrace() {
  static void *p[1024];
  static char buf[1024];
  int k = backtrace(p,1000);
  sprintf(buf, "/usr/bin/addr2line -f -C -e /proc/%i/exe", getpid());
  FILE *f = popen(buf, "w");
  FOR(i,k) fprintf(f, "%p\n", p[i]);
  fflush(f);
  fclose(f);
}

struct supernodal_factorisation_chunk {
  virtual void lower_half_solve(double *x, int nrhs) = 0;
  virtual void upper_half_solve(double *x, int nrhs) = 0;
  virtual void lower_half_solve_T(double *x, int nrhs) = 0;
  //virtual void upper_half_solve_T(double *x, int nrhs) = 0;
};

struct diagonal_triangle : supernodal_factorisation_chunk {
  double *triangle;
  int c0, c1;
  int m;

  void lower_half_solve(double *x, int nrhs) {
    if (nrhs > 1) {
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
          CblasNonUnit,
          c1 - c0, nrhs, 1,
          &triangle[0], c1-c0,
          x + c0, m);
    } else {
      cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
          c1 - c0,
          &triangle[0], c1-c0,
          x + c0, 1);
    }
  }

  void upper_half_solve(double *x, int nrhs) {
    if (nrhs > 1) {
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
          CblasNonUnit,
          c1 - c0, nrhs, 1,
          &triangle[0], c1-c0,
          x + c0, m);
    } else {
      cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
          c1 - c0,
          &triangle[0], c1-c0,
          x + c0, 1);
    }
  }

  void lower_half_solve_T(double *x, int nrhs) {
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
        CblasNonUnit,
        nrhs, c1 - c0, 1,
        &triangle[0], c1-c0,
        x + c0 * nrhs, nrhs);
  }

  void upper_half_solve_T(double *x, int nrhs) {
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
        CblasNonUnit,
        nrhs, c1 - c0, 1,
        &triangle[0], c1-c0,
        x + c0 * nrhs, nrhs);
  }
};

struct sparse_triangle : supernodal_factorisation_chunk {
  double *ar;
  int *ia, *ja;
  int c0, c1, m;

  void lower_half_solve(double *x, int nrhs) {
    if (nrhs > 1) {
      for (int i = 0; i < nrhs; i++) lower_half_solve(x+m*i, 1);
    } else {
      for (int i = c0; i < c1; i++) {
        int j = ia[i-c0];
        if (ja[j] == i) {
          x[i] /= ar[j];
          j++;
        }
        for (; j < ia[i+1-c0]; j++)
          x[ja[j]] -= ar[j] * x[i];
      }
    }
  }

  void lower_half_solve_T(double *x, int nrhs) {
    if (nrhs == 8) {
      for (int i = c0; i < c1; i++) {
        v4df *xx = (v4df *)x;
        int j = ia[i-c0];
        if (ja[j] == i) {
          v4df arj = v4_broadcast(ar[j]);
          xx[2*i] /= arj;
          xx[2*i+1] /= arj;
          j++;
        }
        for (; j < ia[i+1-c0]; j++) {
          v4df arj = v4_broadcast(ar[j]);
          xx[2*ja[j]] -= arj * xx[2*i];
          xx[2*ja[j]+1] -= arj * xx[2*i+1];
        }
      }
    } else if (nrhs == 4) {
      for (int i = c0; i < c1; i++) {
        v4df *xx = (v4df *)x;
        int j = ia[i-c0];
        if (ja[j] == i) {
          xx[i] /= v4_broadcast(ar[j++]);
        }
        for (; j < ia[i+1-c0]; j++) {
          xx[ja[j]] -= v4_broadcast(ar[j]) * xx[i];
        }
      }
    } else if (nrhs == 2) {
      for (int i = c0; i < c1; i++) {
        v2df *xx = (v2df *)x;
        int j = ia[i-c0];
        if (ja[j] == i) {
          xx[i] /= v2_broadcast(ar[j++]);
        }
        for (; j < ia[i+1-c0]; j++) {
          xx[ja[j]] -= v2_broadcast(ar[j]) * xx[i];
        }
      }
    } else {
      for (int i = c0; i < c1; i++)
        for (int r = 0; r < nrhs; r++) {
          int j = ia[i-c0];
          if (ja[j] == i) x[r + i*nrhs] /= ar[j++];
          for (; j < ia[i+1-c0]; j++)
            x[r + ja[j]*nrhs] -= ar[j] * x[r + i*nrhs];
        }
    }
  }

  void upper_half_solve(double *x, int nrhs) {
    if (nrhs > 1) {
      for (int i = 0; i < nrhs; i++) upper_half_solve(x+m*i, 1);
    } else {
      for (int i = c1-1; i >= c0; i--) {
        int j;
        for (j = ia[i-c0+1]-1; j > ia[i-c0]; j--)
          x[i] -= ar[j] * x[ja[j]];
        if (j < ia[i-c0]) continue;
        if (ja[j] == i) x[i] /= ar[j];
        else x[i] -= ar[j] * x[ja[j]];
      }
    }
  }

  void upper_half_solve_T(double *x, int nrhs) {
    abort();
  }
};

static vector<double> gather;

struct packed_subdiagonal_rectangle : supernodal_factorisation_chunk {
  double *rectangle;
  int c0, c1, r0, r1;
  int m;

  void lower_half_solve(double *x, int nrhs) {
    if (nrhs > 1) {
      for (int i = 0; i < nrhs; i++) lower_half_solve(x+m*i, 1);
      /*
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
          r1 - r0, nrhs, c1 - c0, -1.0,
          &rectangle[0], r1 - r0,
          x + c0, m,
          1.0, x + r0, m);
      */
    } else {
      cblas_dgemv(CblasColMajor, CblasNoTrans,
          r1 - r0, c1 - c0, -1.0,
          &rectangle[0], r1 - r0,
          x + c0, 1,
          1.0, x + r0, 1);
    }
  }

  void upper_half_solve(double *x, int nrhs) {
    if (nrhs > 1) {
      for (int i = 0; i < nrhs; i++) upper_half_solve(x+m*i,1);
    } else {
      cblas_dgemv(CblasColMajor, CblasTrans,
          r1 - r0, c1 - c0, -1.0,
          &rectangle[0], r1 - r0,
          x + r0, 1,
          1.0, x + c0, 1);
    }
  }

  void lower_half_solve_T(double *x, int nrhs) {
    if (nrhs == 8) {
      v4df *xx = (v4df *)x;
      int k = 0;
      if ((r1-r0 & 3) == 0) {
        for (int i = c0; i < c1; i++)
          for (int j = r0; j < r1; j += 4) {
            xx[2*j+0] -= v4_broadcast(rectangle[k+0]) * xx[2*i];
            xx[2*j+1] -= v4_broadcast(rectangle[k+0]) * xx[2*i+1];
            xx[2*j+2] -= v4_broadcast(rectangle[k+1]) * xx[2*i];
            xx[2*j+2] -= v4_broadcast(rectangle[k+1]) * xx[2*i+1];
            xx[2*j+4] -= v4_broadcast(rectangle[k+2]) * xx[2*i];
            xx[2*j+5] -= v4_broadcast(rectangle[k+2]) * xx[2*i+1];
            xx[2*j+6] -= v4_broadcast(rectangle[k+3]) * xx[2*i];
            xx[2*j+7] -= v4_broadcast(rectangle[k+3]) * xx[2*i+1];
            k += 4;
          }
      } else 
        for (int i = c0; i < c1; i++)
          for (int j = r0; j < r1; j++) {
            xx[2*j+0] -= v4_broadcast(rectangle[k]) * xx[2*i];
            xx[2*j+1] -= v4_broadcast(rectangle[k]) * xx[2*i+1];
            k++;
          }
    } else if (nrhs == 4) {
      v4df *xx = (v4df *)x;
      int k = 0;
      if ((r1-r0 & 3) == 0) {
        for (int i = c0; i < c1; i++)
          for (int j = r0; j < r1; j += 4) {
            xx[j+0] -= v4_broadcast(rectangle[k+0]) * xx[i];
            xx[j+1] -= v4_broadcast(rectangle[k+1]) * xx[i];
            xx[j+2] -= v4_broadcast(rectangle[k+2]) * xx[i];
            xx[j+3] -= v4_broadcast(rectangle[k+3]) * xx[i];
            k += 4;
          }
      } else 
        for (int i = c0; i < c1; i++)
          for (int j = r0; j < r1; j++)
            xx[j] -= v4_broadcast(rectangle[k++]) * xx[i];
    } else {
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
          nrhs, r1 - r0, c1 - c0, -1.0,
          x + c0 * nrhs, nrhs,
          &rectangle[0], r1 - r0,
          1.0, x + r0 * nrhs, nrhs);
    }
  }

  void upper_half_solve_T(double *x, int nrhs) {}
};

struct subdiagonal_rectangle : supernodal_factorisation_chunk {
  double *rectangle;
  int c0, c1;
  int *rowno;
  int nrow;
  int m;

  void lower_half_solve(double *x, int nrhs) {
    if (nrhs > 1) {
      for (int i = 0; i < nrhs; i++) lower_half_solve(x+m*i, 1);
    } else {
      if (gather.size() < nrow) gather.resize(nrow);
      cblas_dgemv(CblasColMajor, CblasNoTrans,
          nrow, c1 - c0, 1.0,
          &rectangle[0], nrow,
          x + c0, 1,
          0.0, &gather[0], 1);
  
      int k = 0;
      for (int j = 0; j < nrhs; j++) {
        double *p = x + j*m;
        for (int i = 0; i < (int)nrow; i++) {
          p[rowno[i]] -= gather[k++];
        }
      }

    }
  }

  void upper_half_solve(double *x, int nrhs) {
    if (nrhs > 1) {
      for (int i = 0; i < nrhs; i++) upper_half_solve(x+m*i,1);
    } else {
      if (gather.size() < nrow) gather.resize(nrow);
      FOR(i, nrow) {
        gather[i] = x[rowno[i]];
      }

      cblas_dgemv(CblasColMajor, CblasTrans,
          nrow, c1 - c0, -1.0,
          &rectangle[0], nrow,
          &gather[0], 1,
          1.0, x + c0, 1);
    }
  }

  void lower_half_solve_T(double *x, int nrhs) {
    int siz = nrhs * nrow + 10;
    if (gather.size() < siz) gather.resize(siz);
    if (nrhs == 4) {
      v4df *xx = (v4df *)x;
      double *gath = &gather[0];
      while ((intptr_t)gath & 63) gath++;
      v4df *ga = (v4df *)gath;
      int k = 0;
      v4df v4zero = v4_broadcast(0.0);
      for (int i = 0; i < nrow; i++) ga[i] = v4zero;
      if ((nrow & 3) == 0) {
        for (int i = c0; i < c1; i++)
          for (int j = 0; j < nrow; j += 4) {
            ga[j+0] -= v4_broadcast(rectangle[k+0]) * xx[i];
            ga[j+1] -= v4_broadcast(rectangle[k+1]) * xx[i];
            ga[j+2] -= v4_broadcast(rectangle[k+2]) * xx[i];
            ga[j+3] -= v4_broadcast(rectangle[k+3]) * xx[i];
            k += 4;
          }
      } else 
        for (int i = c0; i < c1; i++)
          for (int j = 0; j < nrow; j++)
            ga[j] -= v4_broadcast(rectangle[k++]) * xx[i];
      for (int j = 0; j < nrow; j++)
        xx[rowno[j]] += ga[j];
    } else {
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
          nrhs, nrow, c1 - c0, 1.0,
          x + c0 * nrhs, nrhs,
          &rectangle[0], nrow,
          0.0, &gather[0], nrhs);
  
      int k = 0;
      FOR(i, nrow) {
        double *p = x + nrhs * rowno[i];
        for (int j = 0; j < nrhs; j++) {
          p[j] -= gather[k++];
        }
      }
    }
  }

  void upper_half_solve_T(double *x, int nrhs) {}
};

struct supernodal_factorisation {
  vector<unique_ptr<supernodal_factorisation_chunk> > fac;

  vector<double> doubles;
  vector<int> ints;
  int next_double, next_int;

  void flush(vector<pair<pair<int, int>, double> > &sparse) {
    if (!sparse.size()) return;
    unique_ptr<sparse_triangle> st(new sparse_triangle());
    int lo = 0x3fffffff, hi = -1;

    for (int i = 0; i < sparse.size(); i++) {
      hi = max(hi, sparse[i].first.first);
      lo = min(lo, sparse[i].first.first);
    }
    hi++;
    static vector<double> count;
    if (count.size() < hi) count.resize(hi);
    int *ia = &ints[next_int]; next_int += hi-lo+1;
    for (int i = lo; i < hi; i++) count[i] = 0;
    for (int i = 0; i < sparse.size(); i++)
      count[sparse[i].first.first]++;
    ia[0] = 0;
    for (int i = lo; i < hi; i++) ia[i-lo+1] = ia[i-lo] + count[i];
    static vector<int> ian;
    ian.resize(hi-lo+1);
    FOR(i,hi-lo+1) ian[i] = ia[i];
    int *ja = &ints[next_int]; next_int += sparse.size();
    double *ar = &doubles[next_double]; next_double += sparse.size();
    for (int i = 0; i < sparse.size(); i++) {
      int col = sparse[i].first.first, row = sparse[i].first.second;
      double x = sparse[i].second;
      int j = ian[col-lo]++;
      ja[j] = row; ar[j] = x;
    }

    st->c0 = lo; st->c1 = hi; st->m = m;
    st->ia = ia;
    st->ja = ja;
    st->ar = ar;

    /*
    printf("sparse triangle size %zi\n", sparse.size());
    for (int i = 0; i < st->c1 - st->c0; i++) {
      printf("%4i: ", i + st->c0);
      for (int j = st->ia[i]; j < st->ia[i+1]; j++)
        printf(" %i:%.4f", st->ja[j], st->ar[j]);
      printf("\n");
    }
    printf("\n");
    */

    fac.push_back(move(st));
    sparse.resize(0);
  }

  void decompose_packed_rect(double *nodex, int stride, int c0, int c1, int r0, int r1) {
    if ((r1-r0) * (c1-c0) > 2048) {
      int c2 = (c0+c1)/2;
      int r2 = (r0+r1)/2;
      int longcol = (c1 - c0) > (r1 - r0);
      if (longcol) {
        decompose_packed_rect(nodex, stride, c0, c2, r0, r1);
        decompose_packed_rect(nodex+(c2-c0)*stride, stride, c2, c1, r0, r1);
      } else {
        decompose_packed_rect(nodex, stride, c0, c1, r0, r2);
        decompose_packed_rect(nodex+r2-r0, stride, c0, c1, r2, r1);
      }
      return;
    }
    double *rectangle = &doubles[next_double];
    int nnz = 0;
    FOR(i,(c1 - c0)) FOR(j,r1-r0) nnz += nodex[stride * i + j] != 0;
    next_double += (c1-c0) * (r1-r0);
    for (int i = 0; i < (c1-c0); i++)
      for (int j = 0; j < (r1-r0); j++)
        rectangle[(r1-r0) * i + j] = nodex[stride * i + j];
    unique_ptr<packed_subdiagonal_rectangle> psr(
        new packed_subdiagonal_rectangle());
    psr->rectangle = rectangle;
    psr->c0 = c0; psr->c1 = c1;
    psr->r0 = r0; psr->r1 = r1;
    psr->m = m;
      if (0) if (nnz <= 0.8 * (c1-c0) * (r1-r0)) {
        FOR(i,(c1 - c0)) {
          FOR(j,(r1-r0)) printf("%c", ".*"[rectangle[(r1-r0)*i+j] != 0]);
          printf("\n");
        }
        printf("\n");
      }
    //printf("packed rectangle size %i\n", (r1-r0)*(c1-c0));
    fac.push_back(move(psr));
  }

  void decompose_triangle(vector<pair<pair<int, int>, double> > &sparse,
      double *nodex, int c0, int ncol, int nrow) {
    if (ncol > 16) {
      int p2 = 1; while (p2 < ncol * 0.9) p2 *= 2; p2 /= 2;
      int rest = ncol - p2;
      decompose_triangle(sparse, nodex, c0, p2, nrow);
      flush(sparse);
      decompose_packed_rect(nodex + p2, nrow, c0, c0 + p2, c0 + p2, c0 + ncol);
      decompose_triangle(sparse, nodex + nrow * p2 + p2, c0 + p2, rest, nrow);
    } else {
      flush(sparse);
      double *square = &doubles[next_double];
      next_double += ncol * ncol;
      FOR(i,ncol) for (int j = i; j < ncol; j++) {
        square[ncol * i + j] = nodex[nrow * i + j];
      }
      unique_ptr<diagonal_triangle> dt(new diagonal_triangle());
      dt->triangle = square;
      dt->c0 = c0; dt->c1 = c0 + ncol; dt->m = m;
      //printf("dense triangle size %i\n", ncol * (ncol+1) / 2);
      //FOR(i,ncol) { FOR(j,ncol) printf("%c", ".*"[!!nodex[nrow*i+j]]); printf("\n"); }
      fac.push_back(move(dt));
    }
  }

  void decompose_rectangle(double *nodex, int stride, int height, int c0, int c1, int *rowno) {
    if (height * (c1-c0) > 2048) {
      int c2 = (c0+c1)/2;
      int r2 = height/2;
      int longcol = (c1 - c0) > height;
      if (longcol) {
        decompose_rectangle(nodex, stride, height, c0, c2, rowno);
        decompose_rectangle(nodex + (c2-c0)*stride, stride, height, c2, c1, rowno);
      } else {
        decompose_rectangle(nodex, stride, r2, c0, c1, rowno);
        decompose_rectangle(nodex+r2, stride, height-r2, c0, c1, rowno+r2);
      }
      return;
    }
    double *rectangle = &doubles[next_double];
    int nnz = 0;
    FOR(i,(c1 - c0)) FOR(j,height) nnz += nodex[stride * i + j] != 0;
    next_double += height * (c1 - c0);
    FOR(i,(c1 - c0)) FOR(j,height) {
      rectangle[height * i + j] = nodex[stride * i + j];
    }
    if (rowno[0] + height - 1 != rowno[height - 1]) {
      unique_ptr<subdiagonal_rectangle> sr(new subdiagonal_rectangle());
      sr->rectangle = rectangle;
      sr->c0 = c0; sr->c1 = c1; sr->m = m;
      sr->rowno = &ints[next_int]; next_int += height;
      sr->nrow = height;
      for (int i = 0; i < height; i++) sr->rowno[i] = rowno[i];
      //printf("subdiagonal rect size %i of %i\n", nnz, (c1-c0) * height);
      if (0) if (nnz <= 0.8 * (c1-c0) * height) {
        FOR(i,(c1 - c0)) {
          FOR(j,height) printf("%c", ".*"[rectangle[height*i+j] != 0]);
          printf("\n");
        }
        printf("\n");
      }
      fac.push_back(move(sr));
    } else {
      unique_ptr<packed_subdiagonal_rectangle> sr(new packed_subdiagonal_rectangle());
      sr->rectangle = rectangle;
      sr->c0 = c0; sr->c1 = c1;
      sr->r0 = rowno[0]; sr->r1 = rowno[height - 1] + 1;
      sr->m = m;
      //printf("PACKED rect size %i of %i\n", nnz, (c1-c0) * height);
      if (0) if (nnz <= 0.8 * (c1-c0) * height) {
        FOR(i,(c1 - c0)) {
          FOR(j,height) printf("%c", ".*"[rectangle[height*i+j] != 0]);
          printf("\n");
        }
        printf("\n");
      }
      fac.push_back(move(sr));
    }
  }

  supernodal_factorisation(const cholmod_factor *L) {
    rebuild(L);
  }

  void rebuild(const cholmod_factor *L) __attribute__((noinline)) {
    ScopeTimer st_("supernodal_factorisation::rebuild");
    fac.clear();
    next_double = next_int = 0;

    static vector<pair<pair<int, int>, double> > sparse;
    sparse.resize(0);
    int num_ints=0, num_doubles=0;
    FOR(i, L->nsuper) { // for each supernode
      int *sup = (int *)L->super;
      int *pi = (int *)L->pi;
      int *px = (int *)L->px;
      double *x = (double *)L->x;
      int *ss = (int *)L->s;
  
      int r0 =  pi[i], r1 =  pi[i+1], nrow = r1 - r0;
      int c0 = sup[i], c1 = sup[i+1], ncol = c1 - c0;

      num_doubles += nrow * ncol;
      num_ints += ncol + nrow * ncol + 1;
    }
    doubles.resize(num_doubles);
    ints.resize(num_ints + num_doubles);

    FOR(i, L->nsuper) { // for each supernode
      int *sup = (int *)L->super;
      int *pi = (int *)L->pi;
      int *px = (int *)L->px;
      double *x = (double *)L->x;
      int *ss = (int *)L->s;
  
      int r0 =  pi[i], r1 =  pi[i+1], nrow = r1 - r0;
      int c0 = sup[i], c1 = sup[i+1], ncol = c1 - c0;
      int px0 = px[i];
  
      double *nodex = x + px0;
      int *rowno = ss + r0;

      int snz = 0;
      FOR(i,ncol) for (int j = i; j < ncol; j++) {
        double d = nodex[nrow * i + j];
        snz += d != 0;
      }
     
      if (ncol > 15 && snz > 5 * ncol) {
        decompose_triangle(sparse, nodex, c0, ncol, nrow);
      } else {
        FOR(i, ncol) for (int j = i; j < ncol; j++) {
          double d = nodex[nrow * i + j];
          if (d != 0) sparse.push_back(make_pair(make_pair(c0+i, c0+j), d));
        }
      }

      if (nrow > ncol) {
        int rnz = 0;
        FOR(i,ncol) FOR(j,nrow-ncol) {
          double d = nodex[nrow * i + j + ncol];
          rnz += d != 0;
        }

        double conc = rnz / (double)ncol / (nrow - ncol);
        if (conc > .5 && (nrow-ncol) * ncol > 100) {
          flush(sparse);
          decompose_rectangle(nodex + ncol, nrow, nrow - ncol, c0, c1, rowno + ncol);
        } else {
          FOR(i, ncol) for (int j = ncol; j < nrow; j++) {
            double d = nodex[nrow * i + j];
            if (d != 0)
              sparse.push_back(make_pair(make_pair(c0+i, rowno[j]), d));
          }
        }
      }
    }
    flush(sparse);
  }

  void lower_half_solve(double *x, int nrhs) {
    FOR(i, fac.size()) fac[i]->lower_half_solve(x, nrhs);
  }
  void lower_half_solve_T(double *x, int nrhs) {
    FOR(i, fac.size()) fac[i]->lower_half_solve_T(x, nrhs);
  }
  void upper_half_solve(double *x, int nrhs) {
    for (int i = fac.size()-1; i >= 0; i--) fac[i]->upper_half_solve(x, nrhs);
  }
  /*
  void upper_half_solve_T(double *x, int nrhs) {
    for (int i = fac.size()-1; i>=0; i--) fac[i]->upper_half_solve_T(x, nrhs);
  }
  */
};

void choleskied_system::lower_half_solve(double *ans, const double *rhs) const {
  FOR(i, m) ans[i] = rhs[i];
  lower_half_solve(ans, 1);
}
void choleskied_system::upper_half_solve(double *ans, const double *rhs) const {
  FOR(i, m) ans[i] = rhs[i];
  upper_half_solve(ans, 1);
}

static cholmod_common comm;
static int done_setup = 0;

cholmod_sparse *eh;
cholmod_factor *factor_pattern;

static void errhandle(int stat, const char *file, int line, const char *mess) {
  fprintf(stderr, "chol error: %s:%i(%i): %s\n", file,line,stat,mess);
  dump_backtrace();
  abort();
}

static double potrf_maxdiag;
static int potrf_current_row;
static vector<int> potrf_pivots;

static void fill_potrf_maxdiag(cholmod_sparse *spa) {
  potrf_maxdiag = 0;
  for (int i = 0; i < spa->nrow; i++) {
    int first = ((int *)spa->p)[i];
    int last;
    if (spa->packed) {
      last = ((int *)spa->p)[i+1];
    } else {
      last = first + ((int *)spa->nz)[i];
    }
    for (int jj = first; jj < last; jj++) {
      if (((int *)spa->i)[jj] == i) {
        potrf_maxdiag = max(potrf_maxdiag, ((double *)spa->x)[jj]);
        ((double *)spa->x)[jj] *= 1+1e-14; // regularisation
      }
    }
  }
}

static vector<int> default_options;

static void push_natural_ordering(cholmod_common &comm) {
  default_options.push_back(comm.nmethods);
  default_options.push_back(comm.method[0].ordering);
  default_options.push_back(comm.postorder);
  comm.nmethods = 1;
  comm.method[0].ordering = CHOLMOD_NATURAL;
  comm.postorder = 0;
}

static void pop_options(cholmod_common &comm) {
  comm.postorder = default_options.back();
  default_options.pop_back();
  comm.method[0].ordering = default_options.back();
  default_options.pop_back();
  comm.nmethods = default_options.back();
  default_options.pop_back();
}

static void rebuild_eh() {
  if (eh) cholmod_free_sparse(&eh, &comm);

  int tot = 0;
  FOR(i, m) tot += sparse_A[i].size();
  cholmod_sparse *ehtran = cholmod_allocate_sparse(n, m, tot, 1, 1, 0, CHOLMOD_REAL, &comm);
  if (!ehtran) abort();
  int cur = 0;
  FOR(i, m) {
    ((int *)ehtran->p)[i] = cur;
    FOR(j, sparse_A[i].size()) {
      ((int *)ehtran->i)[cur] = sparse_A[i][j].first;
      ((double *)ehtran->x)[cur] = sparse_A[i][j].second;
      cur++;
    }
  }
  ((int *)ehtran->p)[m] = cur;
  eh = cholmod_transpose(ehtran, 1, &comm);
  cholmod_free_sparse(&ehtran, &comm);
}

vector<int> natural_ordering;

void reorder_equations(cholmod_factor *factor_pattern) {
  vector<vector<pair<int, double> > > sa2(m);
  vector<double> b2(m);
  vector<double> y2(m);
  vector<bool> dr2(m);
  vecmi foo;
  int *perm = (int *)factor_pattern->Perm;
  //FOR(i, m) if (perm[i] != i) printf("%i->%i ", i, perm[i]); printf("\n");
  FOR(i, m) sa2[i] = sparse_A[perm[i]], b2[i] = b[perm[i]], y2[i] = y[perm[i]];
  FOR(i, m) dr2[i] = dead_rows[perm[i]];
  FOR(i, m) foo[i] = natural_ordering[perm[i]];
  FOR(i, m) dead_rows[i] = dr2[i];
  FOR(i, m) sparse_A[i] = sa2[i], b[i] = b2[i], y[i] = y2[i];
  FOR(i, m) natural_ordering[i] = foo[i];
  FOR(i, m) perm[i] = i;
  build_sparse_Atran_from_A();
  rebuild_eh();
}

static cholmod_common::cholmod_method_struct best_ordering_method;

void do_setup() {
  if (done_setup) return;
  dead_rows.resize(m);
  natural_ordering.resize(m);
  cholmod_start(&comm);

  comm.error_handler = errhandle;
  comm.final_ll = 1;
  comm.final_pack = 1;
  comm.supernodal = 2;
  comm.grow1 = 0;
  comm.grow2 = 0;
  comm.dbound = 1e-10;
  comm.nmethods = 9;
  #if CHOLMOD_TEST
  push_natural_ordering(comm);
  #endif

  rebuild_eh();

  factor_pattern = cholmod_analyze(eh, &comm);
  best_ordering_method = comm.method[comm.current];
  printf("one cholesky will take about %g flops\n", comm.fl);
  printf("factor has about %g nonzeros\n", comm.lnz);
  if (!factor_pattern) abort();

  if (comm.nmethods != 1) { // Ensure that the permutation is the identity.
    reorder_equations(factor_pattern);
    push_natural_ordering(comm);
  }

  FOR(i, m) natural_ordering[i] = i;

  done_setup = true;
}

void do_cholesky_init() {
  rescale_problem();
  do_setup();
}

choleskied_system::choleskied_system() {
  factor = 0;
}

choleskied_system::~choleskied_system() {
  cholmod_free_factor(&factor, &comm);
}

static cholmod_dense wrap_matrix(double *x, int m, int n) {
  cholmod_dense ans;
  ans.nrow = m;
  ans.ncol = n;
  ans.nzmax = m*n;
  ans.d = m;
  ans.x = (void *)x;
  ans.xtype = CHOLMOD_REAL;
  ans.dtype = CHOLMOD_DOUBLE;
  return ans;
}

static cholmod_dense wrap_vector(double *x, int n) {
  cholmod_dense ans;
  ans.nrow = n;
  ans.ncol = 1;
  ans.nzmax = n;
  ans.d = n;
  ans.x = (void *)x;
  ans.xtype = CHOLMOD_REAL;
  ans.dtype = CHOLMOD_DOUBLE;
  return ans;
}

static cholmod_dense *copy_vector(const double *x, int n) {
  cholmod_dense *foo = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, &comm);
  FOR(i, n) ((double *)foo->x)[i] = x[i];
  return foo;
}

double diag_scaling = 1e-16;
static int redo = 0;

static void build_errhandle(int, const char *, int, const char *) {
  diag_scaling *= 2;
  redo = 1;
}

vector<bool> old_dead_rows;
vector<bool> dead_rows;
static int killed, revived;

void kill_row(int r) {
  if (!dead_rows[r]) {
    dead_rows[r] = true;
    killed++;
  }
}

void revive_row(int r) {
  if (dead_rows[r]) {
    dead_rows[r] = false;
    revived++;
  }
}


cholmod_sparse *choleskied_system::build_scaled_system(const double *scale) {
  cholmod_sparse *mya = cholmod_copy(eh, 0, 1, &comm);

  { // Remove dead rows and apply the scaling.
    int *myai = (int *)mya->p;
    int *myaj = (int *)mya->i;
    double *myax = (double *)mya->x;
    int *jout = myaj;
    double *xout = myax;
    FOR(i, mya->ncol) {
      double ss = sqrt(scale[i]);
      int j0 = myai[i];
      myai[i] = jout - myaj;
      for (int j = j0; j < myai[i+1]; j++) if (!dead_rows[myaj[j]]) {
        *jout++ = myaj[j], *xout++ = myax[j] * ss;
      }
    }
    myai[mya->ncol] = jout - myaj;
  }

  cholmod_sparse *aat;
  {
    ScopeTimer _st("aat");
    aat = cholmod_aat(mya, 0, 0, 1, &comm);
  }

  { // reintroduce missing diagonal entries.
    int *aati = (int *)aat->p;
    int *&aatj = (int *&)aat->i;
    double *&aatx = (double *&)aat->x;
    aat->nzmax += m;
    double *vd = (double *)malloc(sizeof(double) * (aati[m] + m));
    int *vj = (int *)malloc(sizeof(int) * (aati[m] + m));
    vector<int> vi(m);
    int k = 0;
    FOR(i, m) {
      vi[i] = k;
      if (aati[i] == aati[i+1]) {
        vj[k] = i, vd[k++] = 1e-200;
      } else {
        for (int j = aati[i]; j < aati[i+1]; j++)
          vj[k] = aatj[j], vd[k++] = aatx[j];
      }
    }
    free(aatj); free(aatx);
    aatj = vj; aatx = vd;
    FOR(i, m) aati[i] = vi[i];
    aati[m] = k;
  }

  aat->stype = -1;
  cholmod_free_sparse(&mya, &comm);
  return aat;
}

int first = 1;
static void find_dead_rows(const cholmod_factor *L) {
  return;
  if (first)
  FOR(i, L->nsuper) { // for each supernode
    int *sup = (int *)L->super;
    int *pi = (int *)L->pi;
    int *px = (int *)L->px;
    double *x = (double *)L->x;
    int *ss = (int *)L->s;

    int r0 =  pi[i], r1 =  pi[i+1], nrow = r1 - r0;
    int c0 = sup[i], c1 = sup[i+1], ncol = c1 - c0;
    int px0 = px[i];

    double *nodex = x + px0;
    int *rowno = ss + r0;

    FOR(i, ncol) if (nodex[(nrow + 1) * i] > 1e30 && !dead_rows[c0 + i]) {
      kill_row(i + c0);
    }
  }
  first=0;
}

int choleskied_system::handle_rechol() {
  if ((killed || revived) && old_dead_rows != dead_rows) {
    ScopeTimer _st("rechol");
    vecnd scale;
    FOR(i,n) scale[i] = x[i] / s[i];
    cholmod_sparse *aat = build_scaled_system(scale);
    pop_options(comm);
    cholmod_free_factor(&factor_pattern, &comm);
    {
      cholmod_common::cholmod_method_struct foo = comm.method[2];
      cholmod_common::cholmod_method_struct baz = comm.method[1];
      cholmod_common::cholmod_method_struct qux = comm.method[0];
      int bar = comm.nmethods;
      comm.nmethods = 3;
      comm.method[0].ordering = CHOLMOD_GIVEN;
      comm.method[1].ordering = CHOLMOD_AMD;
      comm.method[2].ordering = CHOLMOD_NATURAL;
      vecmi noinv;
      FOR(i,m) noinv[natural_ordering[i]] = i;
      factor_pattern = cholmod_analyze_p(aat, noinv, 0, 0, &comm);
      comm.method[2] = foo;
      comm.method[1] = baz;
      comm.method[0] = qux;
      comm.nmethods = bar; 
    }
    reorder_equations(factor_pattern);
    push_natural_ordering(comm);
    killed = revived = 0;
    cholmod_free_sparse(&aat, &comm);

    old_dead_rows = dead_rows;

    printf("one cholesky will take about %g flops\n", comm.fl);
    printf("factor has about %g nonzeros\n", comm.lnz);
    return 1;
  }
  return 0;
}

void choleskied_system::factor_scaled_system(cholmod_sparse *aat) {
  fill_potrf_maxdiag(aat);
  stat_num_choleskies++;
  ScopeTimer _st("chol");
  potrf_current_row = 0;
  potrf_pivots.clear();
  cholmod_factorize(aat, factor, &comm);
  find_dead_rows(factor);
}

void choleskied_system::build(const double *scale) {
  ScopeTimer _st("cholbuild");
  cholmod_sparse *aat = build_scaled_system(scale);

  if (factor) cholmod_free_factor(&factor, &comm);
  factor = cholmod_copy_factor(factor_pattern, &comm);

  factor_scaled_system(aat);
  cholmod_free_sparse(&aat, &comm);
  if (flag_rebuild_factor) {
    if (!fac) fac = new supernodal_factorisation(factor);
    else fac->rebuild(factor);
  }
  if (flag_simplicial_factor) {
    ScopeTimer _st("cholmod_change_factor");
    cholmod_change_factor(CHOLMOD_REAL, 1, 0, 1, 1, factor, &comm);
  }
}

void choleskied_system::lower_half_solve_T(double *ans, int nrhs) const {
  ScopeTimer _st("lower_half_solve_T");
  return fac->lower_half_solve_T(ans, nrhs);
}

static cholmod_dense *solve_workspace_ans2;
static cholmod_dense *solve_workspace_ans;
static cholmod_dense *solve_workspace_Y;
static cholmod_dense *solve_workspace_E;
static cholmod_sparse *solve_workspace_Xset;

void choleskied_system::lower_half_solve(double *ans, int nrhs) const {
  static const char *names[] = {"???", "lower_half_solve",
                                "lower_half_solve_2",
                                "lower_half_solve_3",
                                "lower_half_solve_4"};
  ScopeTimer _st(nrhs < 5 ? names[nrhs] : "huge_lower_half_solve");
  if (flag_rebuild_factor) {
    return fac->lower_half_solve(ans, nrhs);
  } else {
    cholmod_dense theans = wrap_matrix(ans, m, nrhs);
    stat_num_halfsolves++;
    cholmod_solve2(7, factor, &theans, 0, &solve_workspace_ans2, 0, &solve_workspace_Y, &solve_workspace_E, &comm);
    cholmod_solve2(4, factor, solve_workspace_ans2, 0, &solve_workspace_ans, 0, &solve_workspace_Y, &solve_workspace_E, &comm);
    FOR(i, m*nrhs) ans[i] = ((double *)solve_workspace_ans->x)[i];
  }
}

void choleskied_system::upper_half_solve(double *ans, int nrhs) const {
  static const char *names[] = {"???", "upper_half_solve",
                                "upper_half_solve_2",
                                "upper_half_solve_3",
                                "upper_half_solve_4"};
  ScopeTimer _st(nrhs < 5 ? names[nrhs] : "huge_upper_half_solve");
  if (flag_rebuild_factor) {
    return fac->upper_half_solve(ans, nrhs);
  } else {
    cholmod_dense theans = wrap_matrix(ans, m, nrhs);
    stat_num_halfsolves++;
    cholmod_solve2(5, factor, &theans, 0, &solve_workspace_ans2, 0, &solve_workspace_Y, &solve_workspace_E, &comm);
    cholmod_solve2(8, factor, solve_workspace_ans2, 0, &solve_workspace_ans, 0, &solve_workspace_Y, &solve_workspace_E, &comm);
    FOR(i, m*nrhs) ans[i] = ((double *)solve_workspace_ans->x)[i];
  }
}

extern "C" {
  // OpenBLAS-specific giant hack.  Does the "diagonal pivoting" thing.
  // Essentially, I copied the dpotf2_L implementation from OpenBLAS and hacked
  // it to suit the needs of an LP solver.
  typedef struct {
    void *a, *b, *c, *d, *alpha, *beta;
    long m, n, k, lda, ldb, ldc, ldd;
  } blas_arg_t;

  int dgemv_n(long, long, long, double, double *, long,
      double *, long, double *, long, double *);
  int dscal_k(long, long, long, double, double *, long,
      double *, long, double *, long);
  double ddot_k(long, double *, long, double *, long);

  int dpotf2_L(blas_arg_t *args, long *range_m, long *range_n,
      double *sa, double *sb, long myid) {
    //ScopeTimer st("dpotf2_L");
    int n = args->n, lda = args->lda;
    double *a = (double *)args->a;
    if (range_n) {
      n = range_n[1] - range_n[0];
      a += range_n[0] * (lda+1);
    }
    double *aoffset = a;
    const double eps = 1e-75;
    FOR(j, n) {
      double ajj = aoffset[j] - ddot_k(j, a+j, lda, a+j, lda);
      double scalby;
      if (ajj < eps) {
        ajj = 1e75;
        scalby = 0;
        potrf_pivots.push_back(potrf_current_row);
      } else {
        ajj = sqrt(ajj);
        scalby = 1 / ajj;
      }
      aoffset[j] = ajj;
  
      int i = n - j - 1;
      if (i > 0) {
        dgemv_n(i, j, 0, -1,
                a + j + 1, lda,
                a + j, lda,
                aoffset + j + 1, 1, sb);
        dscal_k(i, 0, 0, scalby,
                aoffset + j + 1, 1, 0, 0, 0, 0);
      }
      aoffset += lda;
      potrf_current_row++;
    }
    return 0;
  }
}
