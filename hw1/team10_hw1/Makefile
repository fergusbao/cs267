
# We will compile your code with the PrgEnv-intel and PrgEnv-gnu
# programming environments and use the best score.  The PrgEnv-intel
# environment is the default on Cori.  To switch to another environment, use
# something like `module swap PrgEnv-intel PrgEnv-gnu`.
#
# On Cori, we will benchmark your DGEMM's performance against the performance
# of the default vendor-tuned DGEMM. This is done in benchmark-blas.
#
# NERSC's cc and CC compiler wrappers link benchmark-blas's call to dgemm
# to the correct implementation automatically. If you wish to compare with
# other BLAS implementations, check the NERSC documentation.

COMPILER = $(shell cc --version)
IS_ICC = $(findstring icc, $(COMPILER))

CC = cc
OPT = -O3 -std=c99 -march=native -mfpmath=sse -funroll-loops
CFLAGS = -Wall -std=gnu99
LDFLAGS = -Wall
# librt is needed for clock_gettime
LDLIBS = -lrt

ifneq (,$(IS_ICC))
	LDLIBS = -lrt
endif


targets = benchmark-blocked
objects = benchmark.o dgemm-blocked.o

benchmark-blocked: benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)

dgemm-blocked.o: dgemm-blocked.c
	$(CC) -c $(CFLAGS) $(OPT) $<

benchmark.o : benchmark.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
