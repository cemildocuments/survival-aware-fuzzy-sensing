# Data provenance and licenses

The software is released under the MIT license in `LICENSE`. Dataset rights remain with their original sources. The source archives are preserved byte for byte; the SHA-256 values and download URLs are in `data/download_manifest.json`.

| Dataset | Attribution and persistent identifier | Source license |
|---|---|---|
| Air Quality | Vito, S. (2008), UCI Machine Learning Repository, https://doi.org/10.24432/C59K5F | The current UCI record displays CC BY 4.0 and retains the original research-use notice. |
| Gas Turbine CO and NOx Emission Data Set | UCI Machine Learning Repository (2019), https://doi.org/10.24432/C5WC95 | CC BY 4.0 |
| Appliances Energy Prediction | Candanedo, L. (2017), UCI Machine Learning Repository, https://doi.org/10.24432/C5VC8G | CC BY 4.0 |

CC BY 4.0 terms: https://creativecommons.org/licenses/by/4.0/

The Air Quality source also states: “This dataset can be used exclusively for research purposes. Commercial purposes are fully excluded.” This research-use notice is retained with the source attribution. The experiments use the data for research.

Associated source publications:

- S. De Vito, E. Massera, M. Piga, L. Martinotto, and G. Di Francia, *On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario*, Sensors and Actuators B: Chemical 129(2), 750–757 (2008), https://doi.org/10.1016/j.snb.2007.09.060.
- H. Kaya, P. Tüfekci, and E. Uzun, *Predicting CO and NOx emissions from gas turbines: novel data and a benchmark PEMS*, Turkish Journal of Electrical Engineering and Computer Sciences 27(6), 4783–4796 (2019), https://doi.org/10.3906/elk-1807-87.
- L. M. Candanedo, V. Feldheim, and D. Deramaix, *Data driven prediction models of energy use of appliances in a low-energy house*, Energy and Buildings 140, 81–97 (2017), https://doi.org/10.1016/j.enbuild.2017.01.083.

The processing pipeline filters missing values, defines sensor groups, standardizes predictors using fitting data, and injects artificial group outages and biases. The injected failures are experimental scenarios, not recorded equipment faults. Original observations are kept separate across the three datasets.
