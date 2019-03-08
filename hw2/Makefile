#
# Edison - NERSC 
#
# Intel Compilers are loaded by default; for other compilers please check the module list
#
CXX ?= icpc # sorry
MPICXX ?= mpiicpc
OPENMP ?= -fopenmp #Note: this is the flag for Intel compilers. Change this to -fopenmp for GNU compilers. See http://www.nersc.gov/users/computational-systems/edison/programming/using-openmp/
APP ?= 
CFLAGS ?= -O3 -std=c++14 $(APP) -march=haswell -xHASWELL -fast -ipo -fno-alias -fno-fnalias -mtune=haswell -unroll -no-ansi-alias -ipo-seperate -fno-exception
MPICFLAGS ?= -O3 -std=c++14 $(APP)
LIBS ?= -lm
MPILIBS ?= # -rpath-link /opt/intel/compilers_and_libraries_2019.1.144/linux/mpi/intel64/libfabric/lib/libfabric.so.1

LAUNCH_FROM_BUILD_FISH ?= f

TARGETS = serial openmp mpi autograder

.PHONY: serial autograder openmp mpi

all:	$(TARGETS)

serial: 
	@[ $(LAUNCH_FROM_BUILD_FISH) = f ] && echo '||||||| WARNING: YOU MUST ALWAYS USE build.sh OR build.fish TO BUILD MY PROJECT !!!' || echo -n ''
	$(CXX) -o $@ $(CFLAGS) -DDISABLE_MPI serial.cpp common.cpp $(LIBS)
autograder: autograder.o common.o
	$(CXX) -o $@ $(CFLAGS) autograder.o common.o $(LIBS)
openmp: 
	@[ $(LAUNCH_FROM_BUILD_FISH) = f ] && echo '||||||| WARNING: YOU MUST ALWAYS USE build.sh OR build.fish TO BUILD MY PROJECT !!!' || echo -n ''
	$(CXX) -o $@ $(CFLAGS) $(OPENMP) -DDISABLE_MPI openmp.cpp common.cpp $(LIBS)
mpi:
	$(MPICXX) -o $@ $(MPICFLAGS) $(MPILIBS) mpi.cpp common.cpp $(LIBS)

autograder.o: autograder.cpp common.h
	$(CXX) -c $(CFLAGS) autograder.cpp
openmp.o: openmp.cpp common.h
	$(CXX) -c $(OPENMP) $(CFLAGS) openmp.cpp
serial.o: serial.cpp common.h
	$(CXX) -c $(CFLAGS) serial.cpp
mpi.o: mpi.cpp common.h
	$(MPICXX) -c $(CFLAGS) mpi.cpp
common.o $(LIBS): common.cpp common.h
	@[ $(LAUNCH_FROM_BUILD_FISH) = f ] && echo '||||||| WARNING: YOU MUST ALWAYS USE build.sh OR build.fish TO BUILD MY PROJECT !!!' || echo -n ''
	$(CXX) -c $(CFLAGS) common.cpp

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
