# nocte v1

## The v1 refactor

`nocte v1` is a ground-up redesign of the research library previously released as `nocte v0.1`. The v0.1 codebase supported several published electrophysiology projects and accumulated years of practical analysis code, but its architecture grew organically around active research. v1 distills the abstractions and workflows that proved useful into a smaller, explicit, stable library.

The refactor preserves the core strengths of v0.1: metadata-aware experimental objects, stable provenance, temporal selection and extraction, lightweight loaders, efficient signal processing, HDF5 persistence, and pragmatic plotting. It removes historical implementation machinery such as `Stack`, xarray-centered workflows, scrolling support, duplicated matching/extraction logic, and project-specific infrastructure from the core.

`nocte v0.1` remains frozen for reproducibility of published analyses. v1 does not preserve v0.1 APIs. During v1 development, backwards compatibility and deprecation machinery are deliberately ignored; public APIs and architectural assumptions may change directly until the design stabilizes.

`nocte v1.0` is the first intentionally designed stable API of the library.

## Purpose and design philosophy

`nocte` is a lightweight toolkit for organizing, selecting, extracting, aligning, processing, serializing, and visualizing experimental time-series and event data. It is designed primarily for electrophysiology and related neuroscience workflows.

The central abstraction is an **indexed collection of items**. Each collection combines collection-wide semantic state and a pandas table describing its items, and may also contain a type-specific payload where the represented items require separate storage. Pandas handles metadata and labelled relations; compact positional storage handles numerical work.

The codebase follows a few general rules:

* Keep the codebase small, explicit, and easy to understand.
* Prefer simple implementations over clever abstractions.
* Avoid unnecessary indirection and framework-like machinery.
* Extend existing concepts instead of creating parallel ways of doing the same thing.
* Add dependencies only when they provide substantial value.
* Prioritize correctness and interpretability over abstraction or convenience.
* Fail loudly on invalid states instead of adding defensive recovery logic.
* Keep public state inspectable and unsurprising.
* Use explicit attributes for structural state that transformations must maintain.
* Prefer composition and generic functions over deep inheritance.

## Repository and package architecture

The repository uses a standard `src` layout:

```text
nocte/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .pre-commit-config.yaml
│
├── src/
│   └── nocte/
│
├── tests/
├── docs/
│   └── design.md
└── examples/
    └── notebooks/
```

The package is organized by responsibility:

```text
src/nocte/
├── core/
│   ├── collection.py
│   ├── hdf.py
│   ├── grouping.py
│   ├── frames.py
│   ├── traces.py
│   ├── stored.py
│   ├── windows.py
│   ├── events.py
│   ├── spikes.py
│   ├── registry.py
│   └── extract.py
│
├── loaders/
│   ├── common.py
│   ├── neuralynx.py
│   └── neuropixels.py
│
├── analysis/
│   ├── spectral.py
│   ├── xcorr.py
│   ├── cycles.py
│   ├── waves.py
│   └── sne.py
│
└── plot/
    ├── axes.py
    ├── grid.py
    ├── windows.py
    └── traces.py
```

The dependency direction is static and one-way:

* `core` defines data semantics and common collection behavior.
* `loaders` provides format-specific access to externally stored data.
* `extract` coordinates loading, chunking, and computation on large recordings.
* `analysis` consumes core objects and implements scientific operations.
* `plot` consumes core objects and analysis results.
* `core`, `loaders`, and `analysis` do not depend on `plot`.
* Project-specific repositories depend on `nocte`; `nocte` does not depend on project code.

## The collection model

Every major data container follows the same general model:

1. **Metadata** — a `pd.DataFrame` with one row per primary item.
2. **Data** — optional type-specific data associated with those items.
3. **Global** — semantic values that apply to the collection as a whole.

For example:

```text
Traces
├── meta
│   └── one row per trace
├── data
│   └── dense numerical signals
└── global
    ├── sampling information
    └── temporal origin
```

Most of `meta` management is handled by a base `Collection` class. Where a separate `data` payload is required, most of `data` and `global` are isolated in collection-specific "payload" classes. These are dumb by design, and their only responsibility is access, copy and construct data. Each specific collection object (e.g. `Traces`) then inherits `Collection` and owns its concrete payload object where appropriate (e.g. `TracesData`), and exposes a public API of scientifically convenient functions (e.g. `mean`). Payload classes are read-only but metadata might change. When a collection performs an operation on its payload the result is always a copy.

Some collections do not require a separate physical payload. Their metadata may itself contain the complete authoritative representation of their items, with individual items exposed through lightweight views.

### IDs and indices

*Metadata and payload, where a separate payload exists, are aligned positionally*. Payload objects are indexed by position (`int`), while metadata and thus the public API of collections is based on arbitrary int-based pd.Index. `obj.meta` contains exactly one row per item and `obj.meta.index[i]` is the stable public identity of item `i`. A pd.Series respecting this index is the primary exchange container for per-item operations in collections.

Collection indices are unique, non-null, single-level integer indices with an optional meaningful name such as `event_id`, `trace_id`, or `unit_id`.

One-to-one transformations preserve indices. Selection never resets them. One-to-many and many-to-many operations create new output identities and retain source identities explicitly in metadata.

Label lookup happens at the API boundary. Heavy operations use integer positions. `Collection` contains the shared machinery for index validation, label-to-position conversion, masks, metadata selection, grouping, and related operations.

### Shared collection API

Concrete classes implement `_get_pos` and `_sel_pos` because item access and selection may be type-specific.

Shared operations include:

* `sel*(...)` — select items from metadata or index
* `get()` and `items()` - access individual items
* grouping - split the collection by some metadata property.
* concat - join multiple compatible, same-type collections.
* label/position conversion.
* temporal operations where meaningful.
* matching collections via metadata where meaningful.
* simple serialization where defined.

Meaningful type-specific behavior remains explicit in each class.

Selection acts on collection items through metadata and preserves item identities.

```text
Traces       → select animals, channels, hemispheres
Windows      → select conditions or event classes
Spikes       → select units by cell metadata
StoredTraces → select recordings or channels
Registry     → select experiments or recordings
```

Grouping is implemented via `Grouping`: grouping a collection produces a homogeneous `Grouping` object containing collections of the original specific type. This provides a simple iterable representation of groups that can subsequently be mapped, reduced, or concatenated where appropriate.

Collections also provide ways to aggregate multiple collections of the same type, verifying compatible global properties.

### Temporal operations

* `crop(...)` restricts temporal payload without changing the primary items represented by a collection. Collection-wide temporal state is updated together with the payload.
* Align and shift. Temporal alignment changes coordinates while preserving identity when the underlying items remain the same entities.
* Extract. Extraction composes matching with type-specific crop and alignment operations. It carries provenance through the operation and collates homogeneous outputs. A one-to-one extraction can preserve the source event/window index. Many-to-many extraction creates a new output index and stores the source identities in metadata.

### Matching

Matching relates collections through stable indices and metadata. Shared matching machinery lives outside individual container implementations and converts relations to compact positional indices for processing. 

### Conversion

Certain objects might be used to construct others. For example one might define analysis windows relative to a set of events. When such conversions happen and remain one-to-one, identity is always preserved for provenance and metadata is carried over.

### Serialization

HDF5 is the standard format for serialized `nocte` analysis objects and intermediates. Serialization stores:

* collection-wide semantic state;
* item metadata and stable indices;
* compact payload data;
* format/version information.

HDF serialization is collection-specific and shares a common public API. It is implemented by the base collection subclass `HDFCollection`. Each serializable collection knows the representation of its own payload and implements its corresponding abstract methods directly.

## Core classes


### `Grouping`

`Grouping[T]` is an indexed collection whose payload items are of a single homogeneous collection type.

It provides a lightweight bridge between grouping and homogeneous collection operations:

```text
metadata describing entries
+
homogeneous data objects
```

Typical uses include:

* materialized groups;
* coarse matched subsets;
* map/apply workflows;
* homogeneous intermediate results;
* results that are not yet or cannot be concatenated.

In particular, grouping a specific collection produces a `Grouping` containing collections of that same type, for example `Grouping[Traces]` or `Grouping[Windows]`. Such groups can be iterated, transformed, reduced, or concatenated where the contained type supports it.

### `Frames`

Frames is a collection of standard pandas DataFrames with associated metadata.


### `Win` and `Windows`

`Windows` is an indexed collection of temporal windows. Its temporal geometry is separate from general item metadata and includes `start`, `ref`, and `stop`.

Key semantics:

* windows may overlap;
* `ref` is some user-defined explicit scientifically meaningful reference.

`Windows` represents arbitrary temporal observations, not merely non-overlapping recording support.

`Win` is the small scalar temporal interval type.

### `Events`

`Events` is the event-oriented temporal collection and shares the same temporal vocabulary as `Windows`.

Events carry stable event identities and event metadata, and convert naturally to windows for extraction and alignment workflows.

### `Spikes`

`Spikes` represents a collection of multiple spike trains, presumably aligned to individual units.

Unit properties such as cell type, depth, and quality are ordinary item metadata. `sel(...)` therefore selects units directly. Cropping restricts spike times and updates the collection temporal extent. Binning and counting operate across units and can produce `Traces`.

Spikes in individual trains can be converted to event-like data when spike-level operations are required.

### `Traces`

`Traces` represents a collection of materialized continuous sampled signals.

Collection state carries the sampling and temporal information required to interpret the signal array.

`Traces` provides the core operations needed for selection, temporal crop, extraction, alignment, concatenation, and efficient numerical analysis. Dense numerical storage remains array-backed rather than requiring dataframe-backed numerical storage. It owns sampling-related responsibilities such as interpolations.

Light-weight standard functions (e.g. `mean`) are provided as chainable methods (e.g. `traces.zscore().mean()`). More advanced analysis methods (e.g. cross-correlations and spectral processing) live as dedicated modules in `analysis`.

### `StoredTraces`

`StoredTraces` represents a collection of continuous sampled signals backed by data stored outside memory. These different sources may represent different channels of a multi-probe, different files recorded simultaneously, etc. Loading all or part of a `StoredTraces` produces materialized `Traces`.

The boundary is explicit:

```text
StoredTraces = externally backed / potentially very large signals
Traces       = materialized signal data
```

`StoredTraces` uses the normal `Collection` metadata and index machinery to describe and select its streams or channels. Its payload is a single loader object that provides access to the underlying data, exactly as `Traces` is a collection of individual traces backed by one array payload. The loader payload only needs to satisfy a small `Protocol` defined in core, next to `StoredTraces`. This is what keeps `StoredTraces` decoupled from any specific loader implementation: `loaders` imports and implements the core protocol, `core` never imports `loaders`.


### `Registry`

`Registry` is a `Collection` of experimental entities such as experiments or recordings. It stores experiment- and recording-level metadata and supports the common indexed-collection selection model.

The registry metadata table is the authoritative representation of its items and does not require an artificial separate payload. Individual items are exposed as lightweight `RegistryEntry` view objects matching rows of the registry, allowing per-entry functionality.

`Registry` is a general-purpose, project-agnostic container for experiment- and recording-level metadata; the specific columns, paths, and conventions it holds are project-specific, but the container itself stays in core. It serves as the primary exchange mechanism between experimental protocol, experimental logs, and analysis pipeline code.

## Large-experiment processing

`extract.py` implements reusable infrastructure for processing recordings that do not fit comfortably into one in-memory operation.

Its job is orchestration:

```text
StoredTraces / DataLoader(s)
    ↓
chunk plan
    ↓
load chunk
    ↓
materialized Traces
    ↓
analysis/user function
    ↓
result
    ↓
yield / collate / reduce / store
```

Chunk definitions distinguish the data that must be **read** from the region whose output is **valid**. This provides explicit halo/context for filters, Hilbert transforms, xcorr, and other operations with boundary effects.

The processing engine is agnostic to result type. Chunk functions can return core containers, NumPy arrays, pandas objects, scalars, or project-specific result objects. Results can be iterated, concatenated, reduced incrementally, or serialized.

## Analysis

`nocte.analysis` contains reusable scientific analysis built on the core data model.

The main modules are:

* `spectral.py` — spectra, filtering/transforms, and related signal operations;
* `xcorr.py` — continuous cross-correlation and rolling/paired variants;
* `cycles.py` — cycle and phase-oriented analysis;
* `waves.py` — generic wave/event detection;
* `sne.py` — reusable SNE extraction and analysis.

Analysis code follows these boundaries:

* core data semantics stay in `core`;
* analysis contains scientific algorithms, not project configuration;
* standard numerical work delegates to NumPy, SciPy, or focused Numba kernels;
* animal-specific paths, publication-specific schemas, and strongly project-bound workflows live downstream;
* analysis contains no plotting code or Matplotlib imports.


## Data loading

A `DataLoader` is a lightweight, format-specific object that provides access to data stored outside memory. A loader is not itself a `Collection`. It is a low-level object that knows how to read samples from its underlying storage format.

`DataLoader` acts as a payload class to `StoredTraces` and is concerned only with providing access to the underlying stored samples. The `Protocol` it must satisfy is defined in core, next to `StoredTraces`; `loaders/common.py` only provides shared implementation helpers that concrete format-specific loaders use to satisfy that protocol.

Format-specific loaders implement this interface directly:

* `neuralynx.py` provides direct Neuralynx access.
* `neuropixels.py` provides lightweight Neuropixels access.

Loaders expose raw/large data without forcing a heavyweight external object model or duplicating collection semantics.


## Plotting

`nocte.plot` is an optional Matplotlib-based subpackage. Installing or importing core `nocte` and `nocte.analysis` does not require Matplotlib.

The plotting package contains:

* `axes.py` — axes, ticks, scale bars, annotations, and general Matplotlib helpers;
* `grid.py` — the reusable `Grid`/`Cell` plotting abstraction;
* `windows.py` — temporal window visualization;
* `traces.py` — trace-oriented visualization.

Plotting code provides mechanics rather than enforcing a house style:

* Matplotlib keyword arguments remain overridable.
* Importing `nocte` does not modify global `rcParams`.
* Generic plotting functions do not encode project-specific colors or publication conventions.
* Project-specific figures remain downstream.

## Performance and data quality

`nocte` is designed for long recordings, many channels or units, millions of events, repeated extraction, and chunked processing.

Performance rules:

* numerical payloads and temporal geometry use compact array storage;
* pandas is used for metadata, not dense numerical signals;
* labels are converted to integer positions once before heavy processing;
* fine-grained operations avoid Python object-per-event designs;
* sampled data do not materialize large time vectors unnecessarily;
* large sources are processed through loaders and chunks rather than universal lazy containers;
* hot numerical paths are benchmarked before introducing optimization complexity;
* Numba is used where a focused compiled kernel provides clear value.

Missing-data behavior is explicit for every analysis routine that can encounter it. Algorithms never silently bridge missing regions or recover from invalid states in scientifically ambiguous ways. Tests cover missing data, boundary conditions, and other failure modes relevant to each routine.

## Coding style

The implementation favors readable, idiomatic Python and static modular design.

* Prefer single quotes.
* Use blank lines to separate logical steps.
* Keep functions focused and reasonably small.
* Avoid deeply nested conditionals; move distinct logic into focused functions.
* Prefer module imports (`import x.y`) over symbol imports (`from x.y import z`).
* Use type annotations when they clarify interfaces; avoid complicated typing machinery.
* Use light OO: methods live on focused data containers, while shared algorithms remain modular.
* Data transformations return updated objects rather than mutating containers in place.
* Avoid runtime polymorphism and framework-style dispatch unless it solves a concrete problem.
* Avoid over-engineering.
* Keep public APIs flat and discoverable.
* Preserve direct, convenient access to pandas metadata.

Every transformation has clear semantics for:

* item identity and provenance;
* collection-wide state;
* metadata;
* ordering;
* copies versus views when relevant;
* missing data.

## Testing and development workflow

Core functionality has deterministic, lightweight tests that run quickly enough to execute routinely.

* tests prioritize readability and scientific correctness
* tests should be light and fast whenever possible
* test known solutions and edge cases
* test mathematical routines against analytical results when available
* test collection invariants and provenance explicitly
* test missing-data and boundary behavior
* test chunked processing against equivalent whole-data operations
* test installed-package behavior, not only source-tree imports
