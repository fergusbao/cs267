#!/bin/bash

function build_target_ipo () {
    target="$1"
    prof_dir="./prof_$target"

    echo "Building target $target with IPO technic..."

    CXX=icpc APP="-prof-gen -prof-dir$prof_dir" OPENMP=-qopenmp LAUNCH_FROM_BUILD_FISH=t make $target &&
    rm -rf $prof_dir &&
    mkdir $prof_dir &&
    eval ./$target -n 5000 &&
    rm -f *.o ./$target &&
    CXX=icpc APP="-prof-use -ipo -prof-dir$prof_dir" OPENMP=-qopenmp LAUNCH_FROM_BUILD_FISH=t make $target &&
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
build_target_ipo serial &&
build_target_ipo openmp &&
build_target_normal autograder

exit $?

