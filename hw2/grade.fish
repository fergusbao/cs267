#!/usr/bin/fish

function do_grade
    set target $argv[1]
    echo 'Gradding target '"$target ..."
    set tmpFile (mktemp)
    
    for i in {100,500,1000,5000,10000,50000}
        eval ./$target -s $tmpFile -n $i
    end
    
    ./autograder -s $tmpFile -v $target
end

do_grade serial
do_grade openmp

