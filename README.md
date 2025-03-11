# nocte

Analysis of neuronal oscillations, cycles and time events in electrophysiological recordings.

_nocte_ means _at night_ in latin.

This codebase evolved from the analysis used in several publications (see [background](#background))
and is currently in a state of refactoring and cleanup.

## Usage

You can install the package in editable mode with

    pip install -e .

Data is processed using `nocte/extract.py` which orchestrates the analysis code in `nocte/analysis`.
See notebooks in `notebooks` for examples of visualizing the processed results.
Refer to the original publications for data availability, detailed methods, and the repository 
of previous versions of the package (see [background](#background)).

# Style and conventions

The code makes heavy use of pandas `Series` and `DataFrames` to handle data.
The main data containers extend `DataFrameWrapper`.

All methods that start with `is_*` return boolean masks (one entry per row). All methods 
that start with `are_*` return a bool performing a check on the entire object.

Constructors `__init__` implement minimal logic. Classes provide class methods that start with
`build_*` or `from_*` to create objects in different ways. To load and store data from different containers
there are `load_*` and `store_*` methods. 

All time values, unless otherwise specified, are assumed to be in milliseconds.

# Organization

The main general data containers are: 

- `Traces`: contains time series and associated metadata.
- `Stack` (deprecated, please use `Traces`).
- `Win`: a fancy tuple indicating start and stop times.
- `Windows`: A collection of `Win` in the shape of a `DataFrame`.
- `Events`: An extension of `Windows` that provides methods to handle events with a certain duration.

Logic-specific data containers are:
- `*Loader`: Classes to load traces from different probe types (neuropixel, neuralynx). See `io_*` files.
- `SharpNegativeEvents`: Class to handle bilateral sharp negative deflections of LFP. In the process of being absorbed into `Events`.
- `ScrollablePlot`: Lightweight class to scroll through LFP using an interactive matplotlib plot.
- `Registry`: Class to handle experimental metadata as a `DataFrame` with each experiment as a row.
- `SpikeTrains`: Class to handle spike trains keeping metadata about spikes and associated units. In the process of being simplified.

The general layout of the project is:

    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    ├── pyproject.toml           <- Makes module installable with pip install -e .
    │    
    ├── nocte                    <- Package encapsulating all code in this project.
    │    │
    │    ├── extract.py          <- Main code to process raw data.
    │    ├── traces.py           <- Data container for collections of time-series data.
    │    ├── timeslice.py        <- Data container for sets of time windows.
    │    │
    │    ├── io                  <- Submodule with code to load data recorded from Neuronexus and Neuropixel probes.
    │    │
    │    └── analysis            <- Contains less general code for specific analysis 
    │        ├── prc.py          <- Code for beta phase analysis.
    │        ├── stim.py         <- Code to analyse light pulse stimulation.
    │        ├── entrainment.py  <- Code to analyse trains of pulses.
    │        └── sleep.py        <- Code to extract beta power and detect Sharp Wave Ripples.
    │    
    └── notebooks                <- Example Jupyter notebooks.


## Background

This code developed over the analysis of LFP and spike recordings in _Pogona vitticeps_
during sleep. In previous versions, the package called was called `ihrem`.

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

