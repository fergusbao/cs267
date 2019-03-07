#!/usr/bin/fish

set app $argv[1]

function build_target_pgo
    set target $argv[1]
    set prof_dir "./prof_$target"

    echo "Building target $target with PGO technic..."

    env CXX=icpc APP="-prof-gen -prof-dir$prof_dir $app" OPENMP=-qopenmp LAUNCH_FROM_BUILD_FISH=t make $target
    and rm -rf $prof_dir
    and mkdir $prof_dir
    and eval ./$target -n 5000
    and rm -f *.o ./$target
        env CXX=icpc APP="-prof-use -prof-dir$prof_dir $app" OPENMP=-qopenmp LAUNCH_FROM_BUILD_FISH=t make $target
    and rm -rf $prof_dir

    return $status
end

function build_target_normal
    set target $argv[1]

    echo "Building target $target..."
    env CXX=icpc APP="$app" OPENMP=-qopenmp LAUNCH_FROM_BUILD_FISH=t make $target
    return $status
end

make clean
and build_target_pgo serial
and build_target_pgo openmp
and build_target_pgo mpi
and build_target_normal autograder

exit $status

