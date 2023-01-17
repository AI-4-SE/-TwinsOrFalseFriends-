# Apache httpd (pervolution, energy)

Discretized results available in [Apache_pervolution_energy_bin](../Apache_pervolution_energy_bin).

## Case Study Information

- software: Apache httpd
- cluster: zmiy (i5 only)
- benchmark/workload:
  - ApacheBench, Version 2.3, Revision 1826891 (compiled from httpd-2.4.35 with apr-1.6.5 and apr-util-1.6.1)
  - 100000 requessts (1000 concurrent)
  - static HTML file (2.1 KB)
- configurations: 640
- revisions: 21
- properties:
  - performance (run time)
  - energy (energy consumption)
  - cpu (cpu load)
- fixed time: 180s = 3min (dynamic projection (30s))
- notes:
  - 5 repetitions
  - < 10% relative standard deviation for properties performance and energy
- configuration space constraints:
  - `not compression or not keepalive`
    - reason: application domain restriction

## Measurements Information

- average relative standard deviations:
  - performance: 0.96 %
  - benchmark-energy: 3.31 %
  - fixed-energy: 3.12 %
- relative standard deviations available in separate files
- discretized results available
