# nginx (energy)

## Case Study Information

- software: nginx 1.14.0
- cluster: zmiy (i5 only)
- benchmark/workload:
  - ApacheBench, Version 2.3, Revision 1826891 (compiled from httpd-2.4.35 with apr-1.6.5 and apr-util-1.6.1)
  - 100000 requessts (1000 concurrent)
  - static HTML file (2.1 KB)
- configurations: 4416
- properties:
  - performance (run time)
  - energy (energy consumption)
  - cpu (cpu load)
- fixed time: 600s = 10min (dynamic projection (60s))
- notes:
  - 5 repetitions
  - < 10% relative standard deviation for properties performance and energy
- configuration space constraints:
  - `not compression or not keepalive`
    - reason: application domain restriction
  - `tls or keepalive or (basicAuth and multiAccept) or (not compression and multiAccept)`
    - note: excludes the following sets of configurations:
      `not tls and not keepalive and not basicAuth and compression`, `not tls and not keepalive and not multiAccept`
    - reason: high deviations

## Measurements Information

- average relative standard deviations:
  - performance: 2.86 %
  - benchmark-energy: 4.44 %
  - fixed-energy: 2.68 %
- relative standard deviations available in separate file
