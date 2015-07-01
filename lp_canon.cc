
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <string.h>
#include <vector>
#include <math.h>
using namespace std;

#define FOR(i,n) for (int i=0;i<(int)n;i++)

double objective_constant;

void garbled(char *p) {
 printf("garbled input: \"%s\"\n", p);
 abort();
}

struct glpkrow {
 vector<pair<int, double> > coeff;
 double lb, ub;
};

struct glpkcol {
 double lo, up;
 double objcoef;
};

glpkrow r[4000000];
glpkcol c[4000000];
int nr, nc;

// clean up columns with range (-oo, x].
void fixup_ub_vars() {
 int changes = 0;
 FOR(i,nr) FOR(jj, r[i].coeff.size()) {
  int j = r[i].coeff[jj].first;
  if (c[j].lo < -1e75 && c[j].up <= 1e75)
   r[i].coeff[jj].second = -r[i].coeff[jj].second;
 }
 FOR(j, nc) if (c[j].lo < -1e75 && c[j].up <= 1e75) {
  swap(c[j].lo, c[j].up); c[j].lo *= -1; c[j].up *= -1;
  c[j].objcoef *= -1;
  changes++;
 }
 fprintf(stderr, "Fixed up %i upper-bounded variables.\n", changes);
}

// clean up columns with nonzero finite lower bounds.
void fixup_var_lbs() {
 int changes = 0;
 FOR(i, nr) FOR(jj, r[i].coeff.size()) {
  int j = r[i].coeff[jj].first;
  if (c[j].lo >= -1e75) {
   r[i].lb -= c[j].lo * r[i].coeff[jj].second;
   r[i].ub -= c[j].lo * r[i].coeff[jj].second;
  }
 }
 FOR(j, nc) if (c[j].lo >= -1e75 && c[j].lo != 0) {
  objective_constant += c[j].lo * c[j].objcoef;
  c[j].up -= c[j].lo;
  c[j].lo = 0;
  changes++;
 }
 fprintf(stderr, "Fixed up %i lower-bounded variables.\n", changes);
}

// turn all free variables into differences of nonnegative variables.
void fixup_free_vars() {
 int changes = 0;
 static int negcol[4000000];
 memset(negcol, -1, sizeof(negcol));
 int next = nc;
 FOR(i, nc) if (c[i].lo < -1e75 && c[i].up > 1e75) {
  negcol[i] = next++;
  c[i].lo = 0;
  c[negcol[i]].lo = 0;
  c[negcol[i]].up = 1.0/0.0;
  c[negcol[i]].objcoef = -c[i].objcoef;
  changes++;
 }
 FOR(i, nr) {
  FOR(jj, r[i].coeff.size()) {
   int j = r[i].coeff[jj].first;
   if (negcol[j] != -1)
    r[i].coeff.push_back(make_pair(negcol[j], -r[i].coeff[jj].second));
  }
 }
 nc = next;
 fprintf(stderr, "Fixed up %i free variables.\n", changes);
}

// turn all double-bounded variables into nonnegative variables and constraints.
void fixup_db_vars() {
 int changes = 0;
 FOR(i, nc) if (c[i].lo == 0 && c[i].up <= 1e75) {
  r[nr].coeff.push_back(make_pair(i, 1));
  r[nr].lb = -1.0/0.0;
  r[nr].ub = c[i].up;
  c[i].up = 1.0/0.0;
  nr++;
  changes++;
 }
 fprintf(stderr, "Fixed up %i double-bounded variables.\n", changes);
}

// remove all fixed variables.
void fixup_fixed_vars() {
 int changes = 0;
 vector<int> dead(nc, 0);
 FOR(i, nc) if (c[i].lo == c[i].up) {
  dead[i] = 1;
  objective_constant += c[i].objcoef * c[i].lo;
  changes++;
 }

 FOR(i, nr) {
  FOR(jj, r[i].coeff.size()) {
   int j = r[i].coeff[jj].first;
   double d = r[i].coeff[jj].second;
   if (dead[j]) {
    r[i].coeff[jj] = r[i].coeff.back();
    r[i].coeff.pop_back();
    jj--;
    r[i].lb -= d * c[j].lo;
    r[i].ub -= d * c[j].lo;
   }
  }
  sort(r[i].coeff.begin(), r[i].coeff.end());
 }
 fprintf(stderr, "Fixed up %i fixed variables.\n", changes);
}

// remove all unused variables.
void fixup_unused_vars() {
 int changes = 0;
 vector<int> uses(nc, 0);
 FOR(i, nr)
  FOR(jj, r[i].coeff.size())
   uses[r[i].coeff[jj].first]++;

 vector<int> remap_to(nc, -1);
 int next = 0;
 FOR(i, nc) {
  if (uses[i])
   remap_to[i] = next++;
  else {
   if (c[i].objcoef < 0) objective_constant += c[i].objcoef * c[i].up;
   else objective_constant += c[i].objcoef * c[i].lo;
   changes++;
  }
 }
 FOR(i, nr) 
  FOR(jj, r[i].coeff.size()) 
   r[i].coeff[jj].first = remap_to[r[i].coeff[jj].first];
 
 FOR(i, nc) if (remap_to[i] != -1) {
  c[remap_to[i]].lo = c[i].lo;
  c[remap_to[i]].up = c[i].up;
  c[remap_to[i]].objcoef = c[i].objcoef;
 }

 nc = next;
 fprintf(stderr, "Fixed up %i unused variables.\n", changes);
}

// turn all double-bounded rows into pairs of rows.
void fixup_db_rows() {
 int changes = 0;
 FOR(i, nr) if (r[i].lb >= -1e75 && r[i].ub <= 1e75 && r[i].lb < r[i].ub) {
  r[nr].coeff = r[i].coeff;
  r[nr].ub = r[i].ub;
  r[nr].lb = -1.0/0.0;
  r[i].ub = 1.0/0.0;
  nr++;
  changes++;
 }
 fprintf(stderr, "Fixed up %i double-bounded rows.\n", changes);
}

// delete all free rows.
void fixup_free_rows() {
 int changes = 0;
 FOR(i, nr) if (r[i].lb < -1e75 && r[i].ub > 1e75) {
  r[i--] = r[--nr];
  changes++;
 }
 fprintf(stderr, "Fixed up %i free rows.\n", changes);
}

void fixup_empty_rows() {
 int changes = 0;
 FOR(i, nr) if (r[i].coeff.size() == 0) {
  if (r[i].lb > 0 || r[i].ub < 0) {
   fprintf(stderr, "problem is obviously infeasible\n");
   printf("0 0\n"); exit(0);
  }
  r[i--] = r[--nr];
  changes++;
 }
 fprintf(stderr, "Fixed up %i empty rows.\n", changes);
}

// replace <= constraints with >= constraints
void fixup_ub_rows() {
 int changes = 0;
 FOR(i, nr) if (r[i].lb < -1e75 && r[i].ub <= 1e75) {
  FOR(j, r[i].coeff.size()) r[i].coeff[j].second *= -1;
  r[i].lb = -r[i].ub;
  r[i].ub = 1.0/0.0;
  changes++;
 }
 fprintf(stderr, "Fixed up %i upper-bounded rows.\n", changes);
}

// replace >= constraints with equalities by adding slacks.
void fixup_lb_rows() {
 int changes = 0;
 FOR(i, nr) if (r[i].lb >= -1e75 && r[i].ub > 1e75) {
  r[i].coeff.push_back(make_pair(nc, -1));
  c[nc].lo = 0; c[nc].up = 1.0/0.0; c[nc].objcoef = 0;
  nc++;
  r[i].ub = r[i].lb;
  changes++;
 }
 fprintf(stderr, "Fixed up %i lower-bounded rows.\n", changes);
}

void read_glpk() {
 int hasclass = 0;
 while (1) {
  char buf[4096];
  if (!gets(buf)) break;
  switch (buf[0]) {
   case 'c': break;
   case 'p': {
    char kind[16];
    if (3 != sscanf(buf, "%*c %s %*s %i %i %*i", kind, &nr, &nc)) garbled(buf);
    if (nr > 2000000 || nc > 2000000) {
     printf("input too large\n");
     abort();
    }
    FOR(i,nc) c[i].lo = 0, c[i].up = 1.0/0.0, c[i].objcoef = 0;
    FOR(i,nr) r[i].lb = r[i].ub = 0;
    if (!strcmp(kind, "mip")) hasclass = 1;
   } break;
   case 'i': {
    int rowno;
    char kind;
    double lb, ub;
    int tokens = sscanf(buf, "%*c %i %c %lf %lf", &rowno, &kind, &lb, &ub);
    rowno--;
    if (kind == 'f' && tokens != 2 || strchr("lus", kind) && tokens != 3 ||
        kind == 'd' && tokens != 4) garbled(buf);
    r[rowno].lb = -1.0/0.0; r[rowno].ub = 1.0/0.0;
    if (kind == 'f');
    else if (kind == 'l') r[rowno].lb = lb;
    else if (kind == 'u') r[rowno].ub = lb;
    else if (kind == 'd') r[rowno].lb = lb, r[rowno].ub = ub;
    else if (kind == 's') r[rowno].ub = r[rowno].lb = lb;
    else garbled(buf);
   } break;
   case 'j': {
    int colno;
    char kind;
    double lb, ub;
    if (hasclass) {
     if (sscanf(buf, "%*c %i %*s %c %lf %lf", &colno, &kind, &lb, &ub) < 2)
      garbled(buf);
    } else {
     if (sscanf(buf, "%*c %i %c %lf %lf", &colno, &kind, &lb, &ub) < 2)
      garbled(buf);
    }
    colno--;
    c[colno].lo = -1.0/0.0; c[colno].up = 1.0/0.0;
    if (kind == 'f');
    else if (kind == 'l') c[colno].lo = lb;
    else if (kind == 'u') c[colno].up = lb;
    else if (kind == 'd') c[colno].lo = lb, c[colno].up = ub;
    else if (kind == 's') c[colno].up = c[colno].lo = lb;
    else garbled(buf);
   } break;
   case 'a': {
    int i,j; double f;
    if (sscanf(buf, "%*c %i %i %lf", &i, &j, &f) != 3) garbled(buf);
    if (!i) c[j-1].objcoef = f;
    else r[i-1].coeff.push_back(make_pair(j-1, f));
   } break;
   case 'n': break;
   case 'e': break;
   default: garbled(buf);
  }
 }
}

void convert_to_standard_equality_form() {
 fixup_ub_vars();
 fixup_var_lbs();
 fixup_free_vars();
 fixup_db_vars();

 fixup_db_rows();
 fixup_free_rows();
 fixup_ub_rows();
 fixup_lb_rows();

 fixup_fixed_vars();
 fixup_unused_vars();
 fixup_empty_rows();

 FOR(i, nr) if (r[i].lb != r[i].ub) {
  printf("row %i is bad; bnds %g %g\n", i, r[i].lb, r[i].ub); abort();
 }

 FOR(i, nc) if (c[i].lo != 0 || c[i].up <= 1e75) {
  printf("col %i is bad; bnds %.20g %.20g\n", i, c[i].lo, c[i].up); abort();
 }
}

int main() {
 read_glpk();

 convert_to_standard_equality_form();

 fprintf(stderr, "objective constant %g\n", objective_constant);
 printf("%i %i\n", nr, nc);
 FOR(i, nr) printf("%.17f ", r[i].lb); printf("\n");
 FOR(i, nc) printf("%.17f ", c[i].objcoef); printf("\n");
 FOR(i, nr)
  FOR(jj, r[i].coeff.size())
   printf("%i %i %.17f\n", i, r[i].coeff[jj].first, r[i].coeff[jj].second);
}
