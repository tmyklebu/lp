This is a linear program solver.  It comes in two parts.

 - The `lp` program solves the linear program on stdin using an interior-point
   method.
 - The `lp_canon` program convert the linear program in GNU LP format on stdin
   to the format accepted by `lp`.

Type `make lp lp_canon` to build both programs.  You will probably need to hack the variables at the top of the `Makefile` to suit your installation.

The solver depends on SuiteSparse and OpenBLAS.  As configured, the build
process also depends on SuiteSparse being built with METIS support; however,
this can be disabled by removing `-lmetis` from the `CHOLMOD_FLAGS` definition.

The solver *requires* OpenBLAS.  There is a dirty hack at the bottom of
`cholmod.cc` where I override how base-case Cholesky factorisation is done.
The only immediately-visible symptom of linking against a different BLAS might
be an undefined reference error for `openblas_set_num_threads`.  You can remedy
this by deleting the call to that function, but Cholesky factorisation will
behave differently.
