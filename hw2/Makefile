#
# Edison - NERSC 
#
# Intel Compilers are loaded by default; for other compilers please check the module list
#
CXX = icpc # sorry
MPCXX ?= CXX
OPENMP ?= -fopenmp #Note: this is the flag for Intel compilers. Change this to -fopenmp for GNU compilers. See http://www.nersc.gov/users/computational-systems/edison/programming/using-openmp/
APP ?= 
CFLAGS ?= -O3 -std=c++14 $(APP) -march=haswell -xHASWELL
LIBS ?= -lm # -lr

LAUNCH_FROM_BUILD_FISH ?= f

TARGETS = serial openmp mpi autograder

all:	$(TARGETS)

serial: serial.o common.o
	$(CXX) -o $@ serial.o common.o $(LIBS)
autograder: autograder.o common.o
	$(CXX) -o $@ autograder.o common.o $(LIBS)
openmp: openmp.o common.o
	$(CXX) -o $@ $(OPENMP) openmp.o common.o $(LIBS)
mpi: mpi.o common.o
	$(MPCXX) -o $@ $(MPILIBS) mpi.o common.o $(LIBS)

autograder.o: autograder.cpp common.h
	$(CXX) -c $(CFLAGS) autograder.cpp
openmp.o: openmp.cpp common.h
	$(CXX) -c $(OPENMP) $(CFLAGS) openmp.cpp
serial.o: serial.cpp common.h
	$(CXX) -c $(CFLAGS) serial.cpp
mpi.o: mpi.cpp common.h
	$(MPCXX) -c $(CFLAGS) mpi.cpp
common.o $(LIBS): common.cpp common.h
	@[ $(LAUNCH_FROM_BUILD_FISH) = f ] && echo '||||||| WARNING: YOU MUST ALWAYS USE build.sh OR build.fish TO BUILD MY PROJECT !!!' || echo -n ''
	$(CXX) -c $(CFLAGS) common.cpp

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
