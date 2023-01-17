# LLVM opt (energy)

## Case Study Information

- software: LLVM opt 6.0.0
- cluster: zmiy (i5 only)
- benchmark/workload: all cpp files of TMV (https://github.com/rmjarvis/tmv) compiled with clang++ -S -emit-llvm
- configurations: 65536
- properties:
  - performance (run time)
  - energy (energy consumption)
  - cpu (cpu load)
- fixed time: 120 s = 2 min (static projection)
- notes:
  - 5 repetitions
  - < 10% relative standard deviation for properties performance and energy

## Measurements Information

- average relative standard deviations:
  - performance: 0.21 %
  - energy: 2.2 %
- relative standard deviations available in separate file
