# CS267 hw2 part1

### IMPORTANT NOTE

My code MUST be built with `build.sh` or `build.fish` rather than bare `Makefile`, otherwise it will be 50% SLOWER!!!

### A plot in log-log scale that shows that your serial and parallel codes run in O(n) time and a description of the data structures that you used to achieve it.

- Serial version

![](https://s.gjw.moe/res/267-serial.png)

- OpenMP version

![](https://s.gjw.moe/res/267-omp.png)

### A description of the synchronization you used in the shared memory implementation.

Because my data structure is thread-safe, no synchronization is required. (Except the implicit join after `#pragma omp for` block).

### A description of the design choices that you tried and how did they affect the performance.

While profiling my previous OpenMP version, Intel Vtune shows that 2s is wasted because of imbalance workload between CPU cores. Then I added `#pragma omp parallel for schedule(dynamic, 1)` and the wasted time is reduced to 0.8s.

### Speedup plots that show how closely your OpenMP code approaches the idealized p-times speedup and a discussion on whether it is possible to do better.

The serial part of my OpenMP code is really slow. Detailed discussion is in the next section.

### Where does the time go? Consider breaking down the runtime into computation time, synchronization time and/or communication time. How do they scale with p?

I did some profiling with Intel Vtune Profiler and I can tell this question clearly. 

For `N=5000`, the parallel program costs 1.710s in total. 52.2% is serial time. Things will be much better if I can have it optimized. However I failed because there's no serial hotspots detected. 

![](https://s.gjw.moe/res/snap-0221-013626.png)

Parallel region took 0.817s in total, and 0.792s is cost on `r267::compute_forces`. Intel vtune calculated that 0.072s is the maximum time that could be saved if the OpenMP region is optimized to have no load imbalance assuming no runtime overhead (Parallel Region Time minus Region Ideal Time). Because effective CPU physical core utilization is 94.8% (including hyper-threading), so I think parallel region is good enough, and the biggest problem is either the serial region or the memory access problem.

Here's a screenshot which shows the CPU utilization.

![](https://s.gjw.moe/res/snap-0221-012948.png)

After profiling memory access performance, I found that 14.6% CPU time is wasted on waiting for L1, L2 or L3 cache and that's OK.


### A discussion on using OpenMP


