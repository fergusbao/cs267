
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

# I can use `CC=icc make` in bash and `env CC=icc make` in fish.
CC ?= cc

# My CPU is also haswell. If I want to run intel vtune on my PC, I can forcefully set it to "-march=haswell -xHASWELL -g".
APPEND_OPT ?= 

COMPILER = $(shell $(CC) --version)
IS_ICC = $(findstring icc, $(COMPILER))

IS_CORI = $(shell uname -a | grep cori)

# recolic: I don't know why ucb removes `-lblas` for icc, which causes a compilation time error.
# recolic: ok I know now.
ifneq (,$(IS_ICC))
$(info Using Intel C++ compiler...)
	OPT = -O3 -ansi-alias -restrict
ifneq (,$(IS_CORI))
$(info Using Intel C++ compiler on CORI...)
	OPT = -O3 -ansi-alias -restrict -march=haswell -xHASWELL
endif
else
$(info Warning: Not using Intel C++ compiler...)
	# OPT = -O3 -mavx
	OPT = -O3 -march=native -mavx
endif

CFLAGS = -Wall -std=gnu99 $(OPT) $(APPEND_OPT)
LDFLAGS = -Wall
# librt is needed for clock_gettime
LDLIBS = -lblas -lrt

ifneq (,$(IS_ICC))
ifneq (,$(IS_CORI))
	LDLIBS = -lrt
endif
endif

targets = benchmark-naive benchmark-blocked benchmark-blas dgemm-blocked-memaligned dgemm-mem-thanard target-immintrin
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o dgemm-blocked-memaligned.o  dgemm-mem-thanard.o target-immintrin.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)
dgemm-blocked-memaligned : benchmark.o dgemm-blocked-memaligned.o 
	$(CC) -o $@ $^ $(LDLIBS)
dgemm-mem-thanard : benchmark.o  dgemm-mem-thanard.o
	$(CC) -o $@ $^ $(LDLIBS)
target-immintrin : benchmark.o target-immintrin.o
	$(CC) -o $@ $^ $(LDLIBS)


%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
