# VP8 (pervolution, energy)

Discretized results available in [VP8_pervolution_energy_bin](../VP8_pervolution_energy_bin).

## Case Study Information

- software: libvpx/vpxenc (codec VP8)
- cluster: zmiy (i5 only)
- benchmark/workload:
  - input: lossless version of the 'Sintel' trailer (y4m, 480p)
  - output: webm (VP8)
- configurations: 2736
- revisions: 15
- properties:
  - performance (run time)
  - energy consumption
  - output file size
  - cpu load
- notes:
  - 5 repetitions
  - < 10% relative standard deviation for properties performance and energy
  - in revisions >= v1.4.0, some configurations are invalid:  
    configurations with rtQuality and twoPass=1 are actually executed with twoPass=0  
    note that some options are dependent on twoPass

## Measurements Information

- average relative standard deviations:
  - performance: 1.50 %
  - energy: 4.25 %
- relative standard deviations available in separate files
- discretized results available
