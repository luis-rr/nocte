# nocte
=======

Analysis of neuronal oscillations, cycles and time events in electrophysiological recordings.
This code developed over the analysis of LFP and spike recordings in _Pogona vitticeps_
during sleep. _nocte_ means _at night_ in latin.

Early versions were used in the publications:

> Central pattern generator control of a vertebrate ultradian sleep rhythm \
> Fenk L. A., Riquelme J.L., and Laurent G. \
> Nature (2024) \
> https://doi.org/10.1038/s41586-024-08162-w \
> \
> Original code: \
> https://gitlab.mpcdf.mpg.de/mpibr/laur/sleep-cpg

> Interhemispheric competition during sleep \
> Fenk, L.A., Riquelme J.L., and Laurent G. \
> Nature (2023) \
> https://doi.org/10.1038/s41586-023-05827-w \
> 
> Original code: \
> https://gitlab.mpcdf.mpg.de/mpibr/laur/inter-hemispheric-rem

Project Organization
--------------------
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    ├── pyproject.toml           <- Makes module installable with pip install -e .
    │    
    ├── nocte                    <- Package encapsulating all code in this project.
    │    │
    │    ├── extract.py          <- Main code to process raw data.
    │    ├── io_*.py             <- Code to load data recorded from Neuronexus and Neuropixel probes.
    │    ├── traces.py           <- Data container for collecitons of time-series data.
    │    ├── timeslice.py        <- Data container for sets of time windows.
    │    │
    │    └── analysis            <- Contains less general code for specific analysis 
    │        ├── prc.py          <- Code for beta phase analysis.
    │        ├── stim.py         <- Code to analyse light pulse stimulation.
    │        ├── entrainment.py  <- Code to analyse trains of pulses.
    │        └── sleep.py        <- Code to extract beta power and detect Sharp Wave Ripples.
    │    
    └── notebooks                <- Example Jupyter notebooks.

--------

You can install editable package as

    pip install -e .

Data is processed using `nocte/extract.py` which orchestrates the analysis code in `nocte/analysis`.
See notebooks in `notebooks` for examples of visualizing the processed results.
Refer to the original publications for data availability, detailed methods, and previous versions of the package
(called "ihrem").
