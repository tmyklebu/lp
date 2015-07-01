#ifndef LP_H_INCLUDED
#define LP_H_INCLUDED

struct asdfasdfasdfasdfasdfasdfasdf {};
#define __float128 asdfasdfasdfasdfasdfasdfasdf

#include <functional>
#include <vector>
#include <memory>
#include <utility>
#include <cholmod.h>
using namespace std;

#define FOR(i,n) for (int i=0;i<(n);i++)
#define FORALL(it, st) for (typeof(st.begin()) it=st.begin();it!=st.end();it++)

#define MAXN 4000000

extern double x[MAXN], y[MAXN], s[MAXN];
extern double kappa, tau;

extern double b0[MAXN], c0[MAXN], g0, h0;
extern vector<pair<int, double> > sparse_A[MAXN];
extern vector<pair<int, double> > sparse_At[MAXN];
extern double b[MAXN];
extern double c[MAXN];
extern int m, n;
extern vector<vector<int> > sparsity;

void build_sparse_Atran_from_A();
void build_sparse_A_from_Atran();
void dump_problem();
void rescale_problem();
void build_sparsity_pattern(int sizlim);
void dump_problem();
void reorder_equations();
void insert_var_equalities(pair<int, int> *a, pair<int, int> *b);
void do_dch_groups();
void do_dense_column_hack();
void do_equation_ordering();
void mul_A_v4(double *ans, const double *x);
void mul_A(double *ans, const double *x);
void mul_Atran(double *ans, const double *y);
void verify_current_iterate();
void init_naive_initial_point();
void init_compute_sd_coeffs();
void initialise();

void symbolic_build_asat();
void symbolic_chol();

void dump_sparsity_pattern(const vector<vector<pair<int, double> > > &asat);

//void build_asat(const double *scale);
//void factor_asat();
//void solve_asat(double *x, const double *b);

struct gradients;

double best_mu(double *a, double *b, double tau, double kappa);
void prep_find_ytm_dirs();
void find_ytm_dirs(gradients *g, double gamma);
double calc_beta(double mu);
double calc_best_stepped_mu(double lambda, const gradients &g);
double calc_best_stepped_beta(double lambda, const gradients &g);
void do_step(double lambda, const gradients &g);
void do_corrected_predictor_step(const gradients &pg, const gradients &cg,
    double betamax);
void do_mty_iteration();
void read_problem();

void do_cholesky_init();

struct choleskied_system {
  struct supernodal_factorisation *fac;
  vector<double> asat_v;
  cholmod_sparse *mya;
  cholmod_factor *factor;
  choleskied_system();
  ~choleskied_system();

  cholmod_sparse *build_scaled_system(const double *scale);
  void factor_scaled_system(cholmod_sparse *aat);

  void build(const double *scale);

  void lower_half_solve(double *ans, const double *rhs) const;
  void upper_half_solve(double *ans, const double *rhs) const;
  void lower_half_solve(double *ans, int nrhs) const;
  void upper_half_solve(double *ans, int nrhs) const;
  void lower_half_solve_T(double *ans, int nrhs) const;
  void upper_half_solve_T(double *ans, int nrhs) const;

  int handle_rechol();
};

struct scaled_system {
  virtual void update_root() = 0;
  virtual void prep() = 0;
  virtual void update() {}
  virtual void lower_half_solve_T(double *v, int nrhs) const = 0;
  virtual void lower_half_solve(double *v, int nrhs) const = 0;
  virtual void upper_half_solve(double *v, int nrhs) const = 0;
  virtual void get_dy(double *ans, const double *rhs, int nrhs) const = 0;
  virtual void scale(double *v) const = 0;
};

struct diag_scaled_system : scaled_system {
  vector<double> scal;
  choleskied_system cs;
  void update_root() override;
  void prep() override;
  void lower_half_solve_T(double *v, int nrhs) const override;
  void lower_half_solve(double *v, int nrhs) const override;
  void upper_half_solve(double *v, int nrhs) const override;
  void get_dy(double *ans, const double *rhs, int nrhs) const override;
  void scale(double *v) const override;
};

struct dfp_scaled_system : scaled_system {
  unique_ptr<scaled_system> base;
  int base_initialised;
  int fresh_cholesky;
  vector<double> x, s;
  vector<double> e, f, Hs, H1e;
  vector<double> vvv, uuu;
  double *vv, *uu;
  double M[4][4];
  double xs, ef, mu, sHs, sHe, eH1e;
  //dfp_scaled_system() : base(new dpsr1_scaled_system()), base_initialised(0) {}
  dfp_scaled_system() : base(new diag_scaled_system()), base_initialised(0) {}
  void update_root();
  void prep();
  void lower_half_solve_T(double *v, int nrhs) const;
  void lower_half_solve(double *v, int nrhs) const;
  void upper_half_solve(double *v, int nrhs) const;
  void get_dy(double *ans, const double *rhs, int nrhs) const;
  void scale(double *v) const;

  virtual void build_vv();
  virtual void build_M();
};

struct conjgrad_scaled_system : scaled_system {
  vector<double> scaling;
  choleskied_system cs;
  mutable double bad_ratio;
  double cur_ratio;

  conjgrad_scaled_system() : bad_ratio(1e99) {}
  void update_root();
  void prep() {}

  void lower_half_solve_T(double *v, int nrhs) const;
  void lower_half_solve(double *v, int nrhs) const;
  void upper_half_solve(double *v, int nrhs) const;
  void get_dy(double *ans, const double *rhs, int nrhs) const;
  void scale(double *v) const;
};

struct bfgs_scaled_system : dfp_scaled_system {
  void build_vv() override;
  void build_M() override;
  void scale(double *v) const override;
};

void get_gradients(const double *rhs1, const double *rhs2, const double *rhs3,
    const scaled_system &ss, gradients &g);

extern int stat_num_halfsolves, stat_num_choleskies;

struct ScopeTimer {
  const char *p;
  long long before;
  ScopeTimer(const char *p);
  ~ScopeTimer();
};

template <typename T>
struct ptr_wrapper {
  T *p;
  operator T *() { return p; }
  operator const T *() const { return p; }
};

struct vecmi : ptr_wrapper<int> {
  vecmi();
  ~vecmi();
};

struct vecni : ptr_wrapper<int> {
  vecni();
  ~vecni();
};

struct vecmd : ptr_wrapper<double> {
  vecmd();
  ~vecmd();
};

struct vecnd : ptr_wrapper<double> {
  vecnd();
  ~vecnd();
};

extern double diag_scaling;

extern vector<bool> dead_rows;
void kill_row(int r);
void revive_row(int r);

extern int flag_rebuild_factor;
extern int flag_simplicial_factor;

#endif
