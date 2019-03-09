#!/bin/bash

function build_target_pgo () {
    target="$1"
    prof_dir="./prof_$target"

    echo "Building target $target with PGO technic..."

    CXX=icpc MPICXX=mpiicpc APP="-prof-gen -prof-dir$prof_dir" OPENMP=-qopenmp LAUNCH_FROM_BUILD_FISH=t make $target &&
    rm -rf $prof_dir &&
    mkdir $prof_dir &&
    eval ./$target -n 5000 &&
    rm -f *.o ./$target &&
    CXX=icpc MPICXX=mpiicpc APP="-prof-use -prof-dir$prof_dir" OPENMP=-qopenmp LAUNCH_FROM_BUILD_FISH=t make $target &&
    rm -rf $prof_dir

    return $?
}

function build_target_normal () {
    target="$1"

    echo "Building target $target..."
    CXX=icpc OPENMP=-qopenmp LAUNCH_FROM_BUILD_FISH=t make $target
    return $?
}

make clean &&
build_target_pgo serial &&
build_target_pgo openmp &&
build_target_pgo mpi &&
build_target_normal autograder

exit $?

