# You probably need to reconfigure these.  They work on my machine.
OPENBLAS_DIR=/opt/openblas
OPENBLAS_LIB_DIR=$(OPENBLAS_DIR)/lib
OPENBLAS_INCLUDE_DIR=$(OPENBLAS_DIR)/include
SUITESPARSE_INCLUDE_DIR=/usr/local/include
SUITESPARSE_LIB_DIR=/usr/local/lib

# If your processor does not support AVX instructions, comment this out.
# Unfortunately, if your processor does not support AVX instructions, you will
# see a substantial slowdown as gcc tries, poorly, to emulate 4-wide vector
# arithmetic.
AVX=-DAVX

CXXFLAGS=-g -march=native -O3 ${AVX} -std=gnu++0x
CXX=/usr/bin/g++

LP_SRCS=lp.cc cholmod.cc lp_linalg.cc sparse_linalg.cc allocation.cc

CHOLMOD_FLAGS=-I$(SUITESPARSE_INCLUDE_DIR) -I$(OPENBLAS_INCLUDE_DIR) -L$(SUITESPARSE_LIB_DIR) -Wl,-rpath,$(SUITESPARSE_LIB_DIR) -lumfpack -lcholmod -lamd -lcolamd -lsuitesparseconfig -L$(OPENBLAS_LIB_DIR) -lopenblas -llapack -Wl,-rpath,$(OPENBLAS_LIB_DIR) -lmetis -lcamd -lccolamd

lp: $(LP_SRCS) lp.h Makefile
	$(CXX) $(CXXFLAGS) -o $@ $(LP_SRCS) $(CHOLMOD_FLAGS)
