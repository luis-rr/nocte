# nocte v1

## 1. The v1 refactor

`nocte v1` is a ground-up redesign of the research library previously released as `nocte v0.1`. The v0.1 codebase supported several published electrophysiology projects and accumulated years of practical analysis code, but its architecture grew organically around active research. v1 distills the abstractions and workflows that proved useful into a smaller, explicit, stable library.

The refactor preserves the core strengths of v0.1: metadata-aware experimental objects, stable provenance, temporal selection and extraction, lightweight loaders, efficient signal processing, HDF5 persistence, and pragmatic plotting. It removes historical implementation machinery such as `Stack`, xarray-centered workflows, scrolling support, duplicated matching/extraction logic, and project-specific infrastructure from the core.

`nocte v0.1` remains frozen for reproducibility of published analyses. v1 does not preserve accidental v0.1 APIs. During v1 development, backwards compatibility and deprecation machinery are deliberately ignored; public APIs and architectural assumptions may change directly until the design stabilizes.

`nocte v1.0` is the first intentionally designed stable API of the library.

## 2. Purpose and design philosophy

`nocte` is a lightweight toolkit for organizing, selecting, extracting, aligning, processing, serializing, and visualizing experimental time-series and event data. It is designed primarily for electrophysiology and related neuroscience workflows.

The central abstraction is an **indexed collection of items**. Each collection combines collection-wide semantic state, a pandas table describing its items, and a type-specific payload containing the data. Pandas handles metadata and labelled relations; compact positional storage handles numerical work.

The codebase follows a few general rules:

- Keep the codebase small, explicit, and easy to understand.
- Prefer simple implementations over clever abstractions.
- Avoid unnecessary indirection and framework-like machinery.
- Extend existing concepts instead of creating parallel ways of doing the same thing.
- Add dependencies only when they provide substantial value.
- Prioritize correctness and interpretability over abstraction or convenience.
- Fail loudly on invalid states instead of adding defensive recovery logic.
- Keep public state inspectable and unsurprising.
- Use explicit attributes for structural state that transformations must maintain.
- Prefer composition and generic functions over deep inheritance.
- Inspect relevant existing code before making broad changes.
- Do not silently change public APIs or architectural assumptions.

## 3. Repository and package architecture

The repository uses a standard `src` layout:

```text
nocte/
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
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
│   ├── collections.py
│   ├── datadict.py
│   ├── traces.py
│   ├── windows.py
│   ├── events.py
│   ├── spikes.py
│   ├── registry.py
│   └── extract.py
│
├── io/
│   ├── loaders.py
│   ├── neuralynx.py
│   ├── neuropixels.py
│   └── hdf.py
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

- `core` defines data semantics and common collection behavior.
- `io` loads and serializes core objects.
- `analysis` consumes core objects and implements scientific operations.
- `plot` consumes core objects and analysis results.
- `core`, `io`, and `analysis` do not depend on `plot`.
- Project-specific repositories depend on `nocte`; `nocte` does not depend on project code.


## 4. The collection model

Every major data container follows the same three-level model:

1. **Collection state** — semantic values that apply to the collection as a whole.
2. **Item metadata** — a `pd.DataFrame` with one row per primary item.
3. **Payload** — the type-specific data associated with those items.

For example:

```text
Traces
├── collection state
│   ├── sampling information
│   └── temporal origin
├── meta
│   └── one row per trace
└── data
    └── dense numerical signals
```

Structural state such as sampling information or the temporal extent of a spike collection is represented by explicit attributes. Optional descriptive annotations can live in an `attrs` mapping, but correctness does not depend on opaque dictionary entries.

### 4.1 Item identity

For every collection:

- `len(obj)` is the number of primary items.
- `obj.meta` contains exactly one row per item.
- `obj.meta.iloc[i]` describes payload item `i`.
- `obj.meta.index[i]` is the stable public identity of item `i`.
- metadata and payload are aligned positionally.
- the external API works with stable labelled identities.

Collection indices are unique, non-null, single-level integer indices with an optional meaningful name such as `event_id`, `trace_id`, or `unit_id`.

One-to-one transformations preserve indices. Selection never resets them. One-to-many and many-to-many operations create new output identities and retain source identities explicitly in metadata.

### 4.2 Positional internals

Label lookup happens at the API boundary. Heavy operations use integer positions:

```text
item IDs
   ↓
pandas Index lookup
   ↓
integer positions
   ↓
NumPy / SciPy / Numba work
```

`core/collections.py` contains the shared machinery for index validation, label-to-position conversion, masks, metadata selection, grouping, and related operations.


## 5. Shared collection API

The common collection protocol is intentionally small. A collection exposes:

- `meta`;
- `index`;
- `__len__`;
- an internal positional item-selection primitive such as `_take_pos`.

Concrete classes implement `_take_pos` because payload selection is type-specific.

A stateless collection mixin exposes the common user-facing API and delegates its implementation to `core/collections.py`. It contains no constructor, state, reconstruction logic, or assumptions about payload layout.

Shared operations include:

- `sel(...)` — select items from metadata;
- `sel_mask(...)` — select items with a boolean mask;
- explicit selection by item index;
- grouping;
- label/position conversion.

Meaningful type-specific behavior remains explicit in each class. A few short implementations are preferable to generic reconstruction machinery.


## 6. Common operations

### Selection

Selection acts on collection items through metadata and preserves item identities.

```text
Traces  → select animals, channels, hemispheres
Windows → select conditions or event classes
Spikes  → select units by cell metadata
Loaders → select recordings or channels
```

### Grouping

Grouping follows pandas `groupby` semantics. It is a derived view over item metadata, not a permanent third metadata layer.

Grouped collections support group access, group-wise operations, and conversion to `DataDict` when materialized grouped objects are useful.

### Crop

`crop(...)` restricts temporal payload without changing the primary items represented by a collection. `Traces`, `Windows`, `Events`, and `Spikes` expose crop semantics appropriate to their payloads. Collection-wide temporal state is updated together with the payload.

### Align and shift

Temporal alignment changes coordinates while preserving identity when the underlying items remain the same entities. `start`, `ref`, and `stop` are first-class temporal concepts where relevant.

### Match

Matching relates collections through stable indices and metadata. Shared matching machinery lives outside individual container implementations and converts relations to compact positional indices for processing.

### Extract

Extraction composes matching with type-specific crop and alignment operations. It carries provenance through the operation and collates homogeneous outputs.

A one-to-one extraction can preserve the source event/window index. Many-to-many extraction creates a new output index and stores the source identities in metadata.

### Concatenation

Concatenation is implemented by each homogeneous container because compatibility rules depend on payload semantics.

Distinct forms of concatenation remain distinct when they mean different things; for example, adding more trace items and concatenating trace data through time are separate operations.


## 7. Core classes

### `Win` and `Windows`

`Win` is the small scalar temporal interval type.

`Windows` is an indexed collection of temporal windows. Its temporal geometry is separate from general item metadata and includes `start`, `ref`, and `stop`.

Key semantics:

- windows may overlap;
- `ref` is an explicit scientifically meaningful reference, not an inferred midpoint;
- temporal geometry supports crop, shift, and alignment;
- conversions from other temporal objects preserve identity when they remain one-to-one.

`Windows` represents arbitrary temporal observations, not merely non-overlapping recording support.

### `Events`

`Events` is the event-oriented temporal collection and shares the same temporal vocabulary as `Windows`.

Events carry stable event identities and event metadata, and convert naturally to windows for extraction and alignment workflows. Event and window code shares common temporal logic rather than duplicating it.

### `Spikes`

`Spikes` is organized around **units/spike trains as primary items**, not individual spike timestamps.

```text
collection state
    temporal extent

meta
    one row per unit

payload
    one variable-length spike-time sequence per unit
```

Unit properties such as cell type, depth, and quality are ordinary item metadata. `sel(...)` therefore selects units directly. Cropping restricts spike times and updates the collection temporal extent. Binning and counting operate across units and can produce `Traces`.

Individual spikes can be exposed as event-like data when spike-level operations are required.

### `Traces`

`Traces` represents a collection of continuous sampled signals.

Dense signal payload is stored separately from pandas metadata. Collection state carries the sampling and temporal information required to interpret the signal array.

`Traces` provides the core operations needed for selection, temporal crop, extraction, alignment, concatenation, and efficient numerical analysis. Large recordings remain compatible with memory-efficient array access rather than requiring dataframe-backed numerical storage.

### `DataDict`

`DataDict` is an indexed collection whose payload items are arbitrary Python objects.

It provides a lightweight bridge between grouping/matching and homogeneous collection operations:

```text
metadata describing entries
+
arbitrary data objects
```

Typical uses include:

- materialized groups;
- coarse matched subsets;
- heterogeneous intermediate results;
- map/apply workflows;
- results that are not yet or cannot be concatenated.

Fine-grained matches involving millions of events remain compact positional relations rather than millions of Python objects.

### `DataLoader` and loader collections

A `DataLoader` represents potentially large data that have not yet been materialized.

The boundary is explicit:

```text
DataLoader = lazy / potentially very large source
Traces     = materialized signal data
```

Collections of loaders use the same metadata/index conventions as other indexed collections.

### `Registry`

`Registry` stores experiment- and recording-level metadata and supports the common indexed-collection selection model.

Machine-specific paths, experiment-specific conventions, and project configuration do not live in the registry core.


## 8. Large-experiment extraction and processing

`core/extract.py` implements reusable infrastructure for processing recordings that do not fit comfortably into one in-memory operation.

Its job is orchestration:

```text
DataLoader(s)
    ↓
chunk plan
    ↓
load chunk
    ↓
materialized data
    ↓
analysis/user function
    ↓
result
    ↓
yield / collate / reduce / store
```

Chunk definitions distinguish the data that must be **read** from the region whose output is **valid**. This provides explicit halo/context for filters, Hilbert transforms, xcorr, and other operations with boundary effects.

The processing engine is agnostic to result type. Chunk functions can return core containers, NumPy arrays, pandas objects, scalars, or project-specific result objects. Results can be iterated, concatenated, reduced incrementally, or serialized.


## 9. Analysis

`nocte.analysis` contains reusable scientific analysis built on the core data model.

The main modules are:

- `spectral.py` — spectra, filtering/transforms, and related signal operations;
- `xcorr.py` — continuous cross-correlation and rolling/paired variants;
- `cycles.py` — cycle and phase-oriented analysis;
- `waves.py` — generic wave/event detection;
- `sne.py` — reusable SNE extraction and analysis.

Analysis code follows these boundaries:

- core data semantics stay in `core`;
- analysis contains scientific algorithms, not project configuration;
- standard numerical work delegates to NumPy, SciPy, or focused Numba kernels;
- animal-specific paths, publication-specific schemas, and strongly project-bound workflows live downstream;
- analysis contains no plotting code or Matplotlib imports.

## 10. I/O and serialization

`nocte.io` keeps data access lightweight and explicit.

- `loaders.py` defines common loader behavior.
- `neuralynx.py` provides direct Neuralynx access.
- `neuropixels.py` provides lightweight Neuropixels access.
- `hdf.py` owns HDF5 serialization.

Loaders expose raw/large data without forcing a heavyweight external object model.

HDF5 is the standard format for serialized `nocte` analysis objects and intermediates. Serialization stores:

- collection-wide semantic state;
- item metadata and stable indices;
- compact payload data;
- format/version information.

HDF mechanics are centralized in `io/hdf.py`. Container convenience methods delegate to that layer rather than duplicating storage code.


## 11. Plotting

`nocte.plot` is an optional Matplotlib-based subpackage. Installing or importing core `nocte` and `nocte.analysis` does not require Matplotlib.

The plotting package contains:

- `axes.py` — axes, ticks, scale bars, annotations, and general Matplotlib helpers;
- `grid.py` — the reusable `Grid`/`Cell` plotting abstraction;
- `windows.py` — temporal window visualization;
- `traces.py` — trace-oriented visualization.

Plotting code provides mechanics rather than enforcing a house style:

- Matplotlib keyword arguments remain overridable.
- Importing `nocte` does not modify global `rcParams`.
- Generic plotting functions do not encode project-specific colors or publication conventions.
- Project-specific figures remain downstream.


## 12. Performance and data quality

`nocte` is designed for long recordings, many channels or units, millions of events, repeated extraction, and chunked processing.

Performance rules:

- numerical payloads and temporal geometry use compact array storage;
- pandas is used for metadata, not dense numerical signals;
- labels are converted to integer positions once before heavy processing;
- fine-grained operations avoid Python object-per-event designs;
- sampled data do not materialize large time vectors unnecessarily;
- large sources are processed through loaders and chunks rather than universal lazy containers;
- hot numerical paths are benchmarked before introducing optimization complexity;
- Numba is used where a focused compiled kernel provides clear value.

Missing-data behavior is explicit for every analysis routine that can encounter it. Algorithms never silently bridge missing regions or recover from invalid states in scientifically ambiguous ways. Tests cover missing data, boundary conditions, and other failure modes relevant to each routine.

## 13. Coding style

The implementation favors readable, idiomatic Python and static modular design.

- Prefer single quotes.
- Use blank lines to separate logical steps.
- Keep functions focused and reasonably small.
- Avoid deeply nested conditionals; move distinct logic into focused functions.
- Prefer module imports (`import x.y`) over symbol imports (`from x.y import z`).
- Use type annotations when they clarify interfaces; avoid complicated typing machinery.
- Use light OO: methods live on focused data containers, while shared algorithms remain modular.
- Data transformations return updated objects rather than mutating containers in place.
- Avoid runtime polymorphism and framework-style dispatch unless it solves a concrete problem.
- Avoid over-engineering.
- Keep public APIs flat and discoverable.
- Preserve direct, convenient access to pandas metadata.

Every transformation has clear semantics for:

- item identity and provenance;
- collection-wide state;
- metadata;
- ordering;
- copies versus views when relevant;
- missing data.

## 14. Testing and development workflow

Core functionality has deterministic, lightweight tests that run quickly enough to execute routinely.

Tests prioritize readability and scientific correctness:

- test known solutions and edge cases;
- test mathematical routines against analytical results when available;
- test collection invariants and provenance explicitly;
- test missing-data and boundary behavior;
- test chunked processing against equivalent whole-data operations;
- test installed-package behavior, not only source-tree imports.

Universal collection tests cover:

- `len(obj) == len(obj.meta)`;
- valid unique integer item indices;
- positional alignment between metadata and payload;
- stable identities after selection and one-to-one transformations;
- correct collection state after crop/alignment;
- consistent metadata and mask selection.

An important end-to-end provenance test follows the common workflow:

```text
Events
  ↓ select
Events with original IDs
  ↓ convert
Windows with the same IDs
  ↓ extract
processed data linked back to the original event IDs
```

The pre-commit workflow runs Ruff formatting/checks and the fast pytest suite. CI builds and installs the package in a clean environment before running tests, and verifies the plotting extra separately.
