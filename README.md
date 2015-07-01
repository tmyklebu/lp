Overview
--------

This is a linear program solver.  It's reasonably quick, but it's decidedly
research-grade.  I wrote it as a graduate student.  It comes in two parts.

 - The `lp` program solves the linear program on stdin using an interior-point
   method and prints to stdout information about its progress.
 - The `lp_canon` program converts the linear program in
   [GNU LP format](https://www.gnu.org/software/glpk) on stdin to the format
   accepted by `lp`.

The `lp` program solves problems of the form

    max        c^T x
    subject to A x  = b
               x >= 0,

and, simultaneously, their duals, of the form

    min        b^T y
    subject to A^T y + s = c
               s >= 0.

The variables in these problems are `x`, `y`, and `s`.  Both `x` and `s` are
n-vectors, while y is an m-vector.  The data of these problems are the m-vector
`b`, the n-vector `c`, and the m x n matrix `A`.

Building
--------

Type `make lp lp_canon` to build both programs.  You will probably need to hack the variables at the top of the `Makefile` to suit your installation.

The solver depends on
[SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) and
[OpenBLAS](https://github.com/xianyi/OpenBLAS).  As configured, the build
process also depends on SuiteSparse being built with
[METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) support;
however, this can be disabled by removing `-lmetis` from the `CHOLMOD_FLAGS`
definition.

The solver *requires* OpenBLAS.  There is a dirty hack at the bottom of
`cholmod.cc` where I override how base-case Cholesky factorisation is done.
The only immediately-visible symptom of linking against a different BLAS might
be an undefined reference error for `openblas_set_num_threads`.  You can remedy
this by deleting the call to that function, but Cholesky factorisation will
behave differently and the solver might bomb out prematurely.

Input format
------------

The solver `lp` accepts its input in the following format

    m n
    b1 b2 ... bm
    c1 c2 ... cn
    i1 j1 a1
    i2 j2 a2
    .
    .
    .

Everything named above is a number.  `m` is the number of linear equations in
`Ax = b` and `n` is the number of variables in `x`.  `b1` through `bm` give
the right-hand sides of `b`.  `c1` through `cn` give the objective coefficients
of the variables in `x`.  Each triple `ik jk ak` specifies that the entry of
`A` in the `ik` row and the `jk` column takes the value `ak`.  All unspecified
entries of `A` are assumed to be zero.

Output format
-------------

The solver `lp` first gives some information on the Cholesky factorisation it's
about to repeatedly compute, such as

    one cholesky will take about 5.40147e+08 flops
    factor has about 961654 nonzeros

Then it prints out an iteration log, which has a number of lines of the form

    m    1| 5.4e+14 -7.0e+08| 7.0e+04  1.0e+00  5.9e+00| 0.39  1.77  9.06|     67044
    s    2| 5.4e+14 -7.6e+08| 7.0e+04  1.0e+00  5.0e+00| 0.46  1.37  9.29|     74575
    s    3| 5.4e+14 -7.8e+08| 7.0e+04  1.0e+00  4.7e+00| 0.46  1.36  9.31|     81995
    s    4| 5.4e+14 -8.0e+08| 7.0e+04  1.0e+00  4.6e+00| 0.47  1.65  9.32|     89451

 - The first column indicates what kind of step was just taken.  `m` means a
   Mehrotra step; `s` means a stale step.
 - The second column gives the iteration number.
 - The third and fourth columns give the primal and dual objective values.
 - The next three columns give the primal and dual residuals and the duality
   gap in a certain reformulation of the problem.
 - The next three columns give the log of the barrier parameter, the (log of
   the) ratio between the largest and smallest complementarity products, and
   the (log of the) ratio between kappa and tau.
 - The last column tells you how many microseconds have been spent since the
   beginning of the iteration loop.

Then it prints out something like

    1.12484e+07 1.12484e+07 5.50353e-08 7.45058e-09
    Did 34 Choleskies, 0 halfsolves
    Average Mehrotra potential reduction:  11244.4 (0.182446 per usec)
    Average    stale potential reduction:  -1 (-inf per usec)
 
Only the first line deserves attention here; the others are misleading or
inaccurate.  The first two entries are the primal and dual objective values at
termination.  The last two entries are the worst violation of a primal and dual
constraint at termination.

Afterward are twenty or so lines of the form

                         aat:       34        68685         2020
                        chol:       34      1330724        39139
                   cholbuild:       34      1536361        45187
            dfp::update_root:       34      1538603        45253
 
This is a kind of performance profile.  The first column gives the name of a
`ScopeTimer` somewhere in the code.  The second column says how many times
that `ScopeTimer` was constructed.  The third column says how much time, in
total, was spend with one of the `ScopeTimer`s with that name alive.  The final
column gives the average time a `ScopeTimer` with that name was alive.  All
times are in microseconds.

If you were hoping to recover the solution to your problem, a little bit of
work needs to be done inside the code to make that possible.  Somewhere in
`cholmod.cc`, the matrix `A` is permuted.  You need to keep track of that
permutation and apply its inverse before reading off the entries of `x`.

Running the code
----------------

Given an MPS file and a (working) installation of GLPK, you can get a GNU LP
file via the incantation

    glpsol --nomip --check --mps foo.mps --wglp foo.glp

Then, after successfully building this package, you can solve it via the
incantation

    lp_canon < foo.glp | lp --no-stale

The solver `lp` accepts a number of arguments:

 - `--no-stale`:  Disables "stale steps," which use an old Cholesky
   factorisation to find search directions at the current iterate.  Also
   sets a longer default Mehrotra step length.
 - `--max-stale=42`:  Do 42 stale steps with each Cholesky factorisation.
 - `--meh-neigh=0.987`:  Go a fraction 0.987 of the distance to the boundary to
   step when doing a Mehrotra step.
 - `--stale-neigh=0.678`: Go a fraction 0.678 of the distance to the boundary
   with each stale step.
 - `--no-stale-corr`/`--stale-corr`:  Enable/disable the "stale corrector."
 - `--no-rebuild-factor`/`--rebuild-factor`:  Enable/disable rebuilding the
   Cholesky factor in blocked form.
 - `--no-simplicial-factor`/`--simplicial-factor`:  Enable/disable converting
   CHOLMOD's supernodal factor to simplicial.

Simply running `lp` with no arguments will crash.  This is because I haven't
yet made the "stale steps", which are enabled by default, work with either of
CHOLMOD's data structures.  Disabling the "stale steps" or building the blocked
supernodal factor remedies this problem.

