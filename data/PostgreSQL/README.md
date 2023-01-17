# PostgreSQL (pervolution, energy)

This case study uses the same configuration space and revisions as [PostgreSQL_pervolution](../PostgreSQL_pervolution),
but is measured on a different cluster and with energy consumption.

Discretized results available in [PostgreSQL_pervolution_energy_bin](../PostgreSQL_pervolution_energy_bin).

## Case Study Information

- software: PostgreSQL
- cluster: zmiy (i5 only)
- benchmark/workload: PolePosition 0.6.0 with custom configuration
- configurations: 864
- revisions: 22
- properties:
  - performance (run time)
  - energy consumption
  - cpu load
- notes:
  - 5 repetitions
  - < 10% relative standard deviation for properties performance and energy

## Measurements Information

- average relative standard deviations:
  - performance: 1.84 %
  - benchmark-energy: 4.14 %
  - fixed-energy: 3.51 %
- relative standard deviations available in separate files
