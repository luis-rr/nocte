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

The source tree distinguishes public API from implementation structure. Public namespaces are exposed explicitly through package `__init__.py` files, which act as static API manifests. Implementation modules and packages are private by default and use underscore-prefixed paths; their contents need not repeat that privacy marker individually. Public symbols are re-exported from stable, shallow semantic namespaces rather than exposing their physical implementation paths.

The layout is roughly:

```text
src/nocte/
│
├── __init__.py
├── py.typed
│
├── _core/
│   ├── __init__.py
│   ├── collection.py
│   ├── grouping.py
│   ├── matches.py
│   ├── hdf.py
│   ├── sampling.py
│   ├── num.py
│   └── time.py
│
├── _collections/
│   ├── __init__.py
│   ├── traces.py
│   ├── windows.py
│   ├── events.py
│   ├── trains.py
│   ├── frames.py
│   ├── registry.py
│   └── registry_builder.py
│
├── xcorr/
│   ├── __init__.py
│   ├── _xcorr.py
│   └── _core.py
│
├── spec/
│   ├── __init__.py
│   ├── _spec.py
│   └── _core.py
│
├── waves/
│   └── ...
├── loaders/
│   └── ...
└── plot/
    └── ...
```

Within `core`, `collection.py`, `grouping.py`, `matching.py`, and `hdf.py` form the generic collection substrate. The remaining modules define the main scientific collections and reusable processing infrastructure built on it.

The dependency direction is static and one-way:

* `core` defines data semantics, collection machinery, and the main data containers.
* `loaders` provides format-specific access to externally stored data.
* `core.extract` coordinates loading, chunking, and computation on large recordings.
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

Most of `meta` management is handled by a base `Collection` class. Where a separate `data` payload is required, most of `data` and `global` are isolated in collection-specific payload classes. These are dumb by design, and their only responsibility is access, copy, and data construction. Each specific collection object (e.g. `Traces`) inherits `Collection`, owns its concrete payload where appropriate, and exposes a public API of scientifically convenient functions. Payload classes are read-only but metadata might change. When a collection performs an operation on its payload the result is always a copy.

Some collections do not require a separate physical payload. Their metadata may itself contain the complete authoritative representation of their items, with individual items exposed through lightweight views.

### IDs and indices

*Metadata and payload, where a separate payload exists, are aligned positionally.* Payload objects are indexed by position (`int`), while metadata and thus the public API of collections is based on arbitrary int-based `pd.Index`. `obj.meta` contains exactly one row per item and `obj.meta.index[i]` is the stable public identity of item `i`. A `pd.Series` respecting this index is the primary exchange container for per-item operations in collections.

Collection indices are unique, non-null, single-level integer indices. Their non-null name identifies the collection's item identity namespace, as described below.

One-to-one transformations preserve indices. Selection never resets them. One-to-many and many-to-many operations create new output identities and retain source identities explicitly in metadata.

Label lookup happens at the API boundary. Heavy operations use integer positions. `Collection` contains the shared machinery for index validation, label-to-position conversion, masks, metadata selection, grouping, and related operations.

### Shared collection API

Concrete classes implement `_get_pos` and `_sel_pos` because item access and selection may be type-specific.

Shared operations include:

* `sel*(...)` — select items from metadata or index;
* `get()` and `items()` — access individual items;
* `groupby(...)` — split the collection by metadata;
* `rename(...)` — rename the item identity namespace;
* label/position conversion and common metadata operations.

Meaningful type-specific behavior remains explicit in each class. Temporal operations, concatenation, numerical transformations, and other domain-specific behavior live where their semantics are known.

Selection acts on collection items through metadata and preserves item identities.

```text
Traces       → select animals, channels, hemispheres
Windows      → select conditions or event classes
Trains       → select units by cell metadata
StoredTraces → select recordings or channels
Registry     → select experiments or recordings
```

### Naming

Collections have a **name** identifying the singular semantic entity represented by one item. It is the identity namespace used when IDs are propagated as provenance.

* `Collection.name` is exactly `meta.index.name`.
* Concrete types provide defaults such as `trace`, `win`, `event`, `train`, `frame`, and `match`.
* `rename(name)` changes this namespace without changing item identities.
* User-defined names such as `beta`, `trial`, or `analysis_win` make downstream provenance more informative.
* Analysis functions may give the collections they produce meaningful default names such as `xcorr` or `power`.
* Provenance columns use these names directly. No `_id` suffix.
* If two related collections share a name, `left_{name}` and `right_{name}` prefixes disambiguate their identities where required.

### Grouping

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

Grouping a collection produces a homogeneous grouping containing collections of the same specific type. Where useful, concrete collections provide specialized groupings such as `TracesGrouping` that add operations meaningful for that contained type. Groups can be iterated, transformed, reduced, or concatenated where appropriate.

Specialized groupings expose a small, useful subset of their contained collection's API with compatible names and semantics. For example:

* `TracesGrouping.shift_time(...)`
* `TracesGrouping.resample(...)`
* `TracesGrouping.concat()`

Less common transformations remain available through `Grouping.map()`.

### Matching

`Matches` is a `Collection` representing an ordered relation between items of two collections. One match is one item with its own unique identity; its payload stores the corresponding left and right source IDs, while `meta` stores optional pair-specific metadata.

Common constructors include:

* `Matches.from_meta(left, right, by=...)` — metadata equality;
* `Matches.from_product(left, right)` — Cartesian relation;
* `Matches.from_identity(collection)` — one-to-one self relation;
* `Matches.from_combinations(collection)` — unique unordered self-pairs;
* `Matches.from_pairs(...)` — explicit pairs.

Source IDs may repeat freely; `Matches.index` remains unique. Matching policy lives on `Matches`, not on `Collection`, so the same relation machinery can be reused by extraction, lookup, cross-correlation, autocorrelation, and other pairwise analyses.

Relation-wide construction state, such as metadata columns used for matching, belongs to collection-wide state rather than being repeated as pair metadata.

Where one result is produced per match, the match identity should be preserved where practical. Source identities are carried as provenance using their collection names, e.g. `beta` and `analysis_win`, or `left_beta` and `right_beta`.

A relation can be materialized by grouping on either side. The resulting `Grouping` preserves the index and metadata of the side being grouped by and contains the corresponding subsets or derived objects from the other side.

For example, extracting traces by windows produces a `TracesGrouping` with one group per window:

```text
Matches[trace, win]
        ↓ group by win
TracesGrouping
    outer identity/meta = win
    contained items     = matched, cropped traces
```

Derived inner items may use match identities, keeping them globally unique even when the same source item appears in multiple groups.

### Serialization

`HDFCollection` is the base collection subclass implementing serialization. HDF5 is the standard format for serialized `nocte` analysis objects and intermediates. Serialization stores:

* collection-wide semantic state;
* item metadata and stable indices;
* compact payload data;
* format/version information.

HDF serialization is collection-specific and shares a common public API. Each serializable collection knows the representation of its own payload and implements its corresponding abstract methods directly.

## Main collections

### Frames

`Frames` is a collection of standard pandas DataFrames with associated metadata.

### Win and Windows

`Windows` is an indexed collection of duration-bearing temporal observations. Its payload contains `start`, `ref`, and `stop`, while `meta` contains one row of descriptive metadata per window.

Windows may overlap. `ref` is an explicit scientifically meaningful anchor and need not be the midpoint.

`Windows` is the single interval abstraction for arbitrary duration-bearing observations: analysis windows, stimulation pulses, behavioral states, oscillatory episodes, detected waves, and similar phenomena. Domain objects such as ripples or stimulation epochs are represented as `Windows` with appropriate metadata rather than additional interval container classes.

`Win` is the small scalar temporal interval type.

### Events

`Events` represents point-like temporal occurrences. Its payload contains one finite timestamp per event; `meta` contains one row per occurrence. Event identity is independent of timestamp value, so duplicate timestamps are valid and remain distinct events.

Examples include stimulus onsets, detected peaks, behavioral transitions, camera triggers, and individual spikes. Each event has exactly one structural temporal value, `time`; duration is not part of the representation.

The payload is a small immutable `_EventsData` backed by a contiguous one-dimensional floating-point array of milliseconds. `time` is reserved from metadata. There is no scalar `Event` class: `get(event)` may return the timestamp directly while descriptive information remains in `meta`.

`intervals()` computes successive intervals in chronological order. More complicated conditions should normally be expressed through selection or grouping rather than a growing collection-specific `by=...` API.

Simple point-process transforms such as binned counts and Gaussian-smoothed rates belong here; more elaborate event analysis belongs in `analysis`.

### Trains

`Trains` is an indexed collection of related timestamp sequences. In electrophysiology a train commonly corresponds to a sorted unit and contains its spikes, but the abstraction is intentionally generic.

```text
global
    support: Win

meta
    one row per train

payload
    one variable-length timestamp array per train
```

The in-memory payload is a small ragged container with one one-dimensional NumPy array per train. Each array is finite, monotonically non-decreasing, interpreted in milliseconds, and contained within collection support. Duplicate timestamps are valid.

`support: Win` is explicit collection-wide state defining the period during which the entire collection was observed. It is required because empty trains are valid and rates or binned representations cannot be inferred correctly from timestamp extrema alone.

Selection operates on trains and leaves support unchanged. `extract_win(win)` restricts every train to the requested region, preserves every train identity including trains that become silent, and updates support. `shift_time(...)` shifts both timestamps and support.

`Trains` is intentionally generic rather than spike-specific. Spike-sorting import, waveform processing, unit-quality metrics, and format-specific adapters such as Kilosort live outside `core/trains.py` and construct ordinary `Trains` plus metadata.

### Traces

`Traces` represents a collection of materialized continuous sampled signals.

Collection state carries the sampling and temporal information required to interpret the signal array. Dense numerical storage remains array-backed rather than requiring dataframe-backed numerical storage.

`Traces` provides the core operations needed for selection, temporal crop, time shifting, resampling, extraction, concatenation, and efficient numerical analysis. It owns sampling-related responsibilities such as interpolation.

Light-weight standard functions (e.g. `mean`) are provided as chainable methods (e.g. `traces.zscore().mean()`). More advanced analysis methods such as cross-correlations and spectral processing live as dedicated modules in `analysis`.

### Registry

`Registry` is the canonical description of a project's experimental entities and their associated resources. It is a `Collection`; each item represents whatever unit the project chooses to register, such as an experiment, recording, session, trial, or similar entity, and `Registry.name` defines that identity namespace. Its metadata table contains the cleaned project-level attributes of those items and follows the normal `Collection` identity and selection model.

Registry use has two stages:

1. **Ingestion** is project-specific: human-maintained spreadsheets and other records are interpreted, cleaned, validated, and normalized, typically through the thin `RegistryBuilder`.
2. **Runtime use** begins with the resulting clean `Registry`, which can be serialized and reloaded independently of the original source layout and conventions.

Each Registry item may own tagged resources of three kinds:

* `WindowsGrouping` for interval-valued temporal resources.
* `EventsGrouping` for point-valued temporal resources.
* `SourcesGrouping` for filesystem resources.

`Source` is a small Registry-specific descriptor of a concrete filesystem resource, such as a recording directory, video, tracking output, or derived data file. `Sources` is its lightweight serializable collection, and `SourcesGrouping` gives filesystem resources the same grouped structure as temporal resources. Sources describe data locations and any small interpretation metadata, but never live loaders, memmaps, file handles, or other runtime I/O state.

Every outer resource group is linked to one Registry item and carries a canonical `tag`. Tags provide a lightweight project-defined namespace, for example `'lights'`, `'injections'`, `('video', 'cam0')`, or `('tracking', 'dlc')`, while the contained `Windows`, `Events`, or `Sources` preserve their own item-level semantics.

A `RegistryEntry` is the per-item view over the resources linked to one Registry identity. Registry selection also restricts these grouped resources, so any Registry subset remains a self-contained description of the selected entities.

Projects define the entity granularity, metadata columns, tags, and ingestion rules. Core only enforces collection identity, clean metadata, resource linkage, selection, validation, and serialization.

## Managing time

`nocte` distinguishes three fundamental arrangements of discrete temporal data:

```text
Events
    one item = one point occurrence

Windows
    one item = one interval occurrence

Trains
    one item = one ordered collection of point occurrences
```

These are complementary representations, not an inheritance hierarchy. `Events` and `Windows` differ by temporal geometry; `Trains` differs by primary item identity. All three use the normal collection infrastructure.

### Temporal conventions

Time-bearing collections follow a common public temporal language:

* physical time is represented as floating-point milliseconds
* intervals are half-open, `[start, stop)`
* structural temporal values are authoritative payload data, not ordinary descriptive metadata

Time representations belong to collection-specific payloads. Public item-aligned values expose collection identities through pandas objects where appropriate; numerical kernels operate on NumPy arrays and positional indices.

Relations between temporal collections use shared matching machinery where possible. For example, event containment, relative position, and classification by window metadata are genuine relations; overlapping windows must not silently collapse a one-to-many relation into one classification.

Continuous-value lookup belongs to `Traces`; sampling phase or amplitude at event times should use trace lookup against `events.time`. Point-process operations on `Trains` may internally reuse `Events` or use optimized grouped implementations where efficiency warrants it.

### Temporal conversions

Conversions follow the general collection identity rules.

**`Events → Windows`** adds extent one-to-one:

```python
windows = events.to_windows(win)
```

Each event time becomes the window `ref`; requested relative start and stop define the interval. Event indices and metadata are preserved.

**`Windows → Events`** selects one point from each interval:

```python
events = windows.to_events(at='ref')
events = windows.to_events(at='mid')
events = windows.to_events(at='start')
events = windows.to_events(at='stop')
events = windows.to_events(at=0.25)
```

The window index and metadata are preserved and interval geometry is discarded. `'ref'` is the natural default.

**`Trains → Events`** flattens trains one-to-many. The resulting `Events` receives new identities and records the source train identity using the train collection name. Train metadata is not duplicated over potentially millions of event rows by default.

**`Events → Trains`** is an explicit change of representation and primary identity:

```python
trains = Trains.from_events(
    events,
    by='unit',
    support=support,
)
```

New train identities are created from the grouping keys. Arbitrary event-level metadata is not automatically aggregated into train metadata.

**`Trains → Traces`** through binning preserves primary item identity and train metadata. Silent trains remain present as zero-valued traces.

`Grouping` remains distinct from these conversions. `Grouping[Events]` is appropriate when grouping is temporary or individual event identity and metadata remain important; `Trains` is appropriate when each timestamp sequence is itself the primary scientific item. Likewise, related duration-bearing observations remain `Grouping[Windows]` unless a future concrete use case requires another representation.

```text
                    add extent
       Events ─────────────────────► Windows
          ▲                            │
          │ flatten                    │ choose ref/start/
                               │ stop/mid/quantile
       Trains                          ▼
                             Events
          │ bin / smooth
          ▼
        Traces

temporary grouping:
Events / Windows / Trains ──► Grouping[...]
```

The model keeps point occurrences, interval occurrences, collections of points, temporary groupings, and sampled representations distinct rather than forcing them into one temporal type.

### Temporal extraction

Temporal collections share four extraction methods:

```python
source.extract_win(win, ...)
source.extract_matched(wins, matches, ...)
source.extract_all(wins, ...)
source.extract_by(wins, by=..., ...)
```

`extract_win()` is the scalar primitive. The others differ only in how the source-to-window relation is constructed: explicit `Matches`, all-to-all, or metadata equality.

Multi-window extraction is implemented through specialized grouping:

```text
source + Windows
       ↓
     Matches
       ↓
specialized Grouping
       ↓
extract each group against its Window
       ↓
concat()
       ↓
source collection type
```

The grouping preserves match identities, source provenance, and outer window metadata while type-specific extraction is performed independently per window. `concat()` then flattens compatible groups back into the source collection type and broadcasts grouping provenance into metadata.

`align=None` is supported only by `extract_win()` and preserves absolute time. Multi-window extraction aligns groups before concatenation. Items with no surviving observations are dropped by default where the type has a valid empty representation; `drop=False` preserves them. `Events` has no empty-item representation.

The type-specific restriction is simple: `Windows` intersects intervals, `Events` selects contained timestamps, `Trains` restricts timestamps and observation support, and `Traces` interpolates onto an explicit sampling grid.

Concatenation never silently repairs incompatible collection-wide state. For example, `TracesGrouping` requires compatible sampling grids and `TrainsGrouping` requires a common realized support.

## Large-experiment processing

`core/extract.py` implements reusable infrastructure for processing recordings that do not fit comfortably into one in-memory operation. This is distinct from the collection-level temporal `extract()` operation described above.

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

The main collections own representation-preserving, broadly useful operations such as selection, temporal restriction, shifting, simple counts and rates, inter-event intervals, binning, conversion, and serialization. Higher-level scientific algorithms belong in `analysis`.

The main modules are:

* `spectral.py` — spectra, filtering/transforms, and related signal operations;
* `xcorr.py` — continuous cross-correlation and rolling/paired variants;
* `cycles.py` — cycle and phase-oriented analysis;
* `waves.py` — generic wave/event detection;
* `sne.py` — reusable SNE extraction and analysis.

Examples of analysis-level operations include auto- and cross-correlograms, peri-event histograms and averages, burst detection, surrogate generation, tuning curves, decoding, statistical comparisons, trial-selectivity metrics, and waveform or unit-quality analysis.

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

`nocte` is designed for long recordings, many channels or units, millions of events or spikes, repeated matching/extraction, and chunked processing.

Performance rules:

* numerical payloads and temporal geometry use compact array storage;
* pandas is used for metadata, not dense numerical signals or timestamp payloads;
* `Events` stores timestamps as one contiguous numeric array;
* `Trains` stores one numeric array per train in memory, keeping metadata proportional to train count rather than timestamp count;
* labels are converted to integer positions once before heavy processing;
* fine-grained operations avoid Python object-per-event designs;
* sampled data do not materialize large time vectors unnecessarily;
* large sources are processed through loaders and chunks rather than universal lazy containers;
* hot numerical paths are benchmarked before introducing optimization complexity;
* Numba is used where a focused compiled kernel provides clear value.

For HDF5, `Events` stores a flat times array. `Trains` stores support plus flat concatenated times and train offsets, reconstructing the simple tuple-of-arrays representation in memory without object arrays in storage.

Missing-data behavior is explicit for every analysis routine that can encounter it. Algorithms never silently bridge missing regions or recover from invalid states in scientifically ambiguous ways. Tests cover missing data, boundary conditions, and other failure modes relevant to each routine.

## Development

### Coding style

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
* Preserve direct, convenient access to pandas metadata.

Every transformation has clear semantics for:

* item identity and provenance;
* collection-wide state;
* metadata;
* ordering;
* copies versus views when relevant;
* missing data.

### Packaging and public API

* The public API should be self-documenting, shallow, semantically grouped, and easy to discover.
* Publicness is explicit. Public packages define their supported surface deliberately, typically through `__all__`.
* `__init__.py` files are API manifests, not implementation modules.
* Prefer private implementation modules and packages, marked by underscore-prefixed paths. Their contents do not all need redundant underscore prefixes.
* Expose public symbols from stable semantic namespaces, not implementation paths. Avoid organizational nesting that adds no meaning.
* Internal code imports definitions from their owning modules, not back through the public `nocte` facade.
* Public signatures should use clean public vocabulary and avoid leaking implementation paths.
* Use ordinary Python imports. Avoid lazy imports, dynamic exports, automatic discovery, generated namespaces, and import-time side effects.
* Typing is part of the API. Use inline annotations, ship `py.typed`, and avoid patterns that hinder static analysis by tools such as Pyright and Pylance.
* A leading underscore marks unsupported implementation detail, not inaccessible state.
* Prefer absolute imports throughout the package, including internal imports.
* Internal code imports definitions from their owning modules, not back through the public `nocte` facade.
* Prefer direct symbol imports for types and symbols that form part of the public vocabulary, especially in public signatures. Import private implementation dependencies as modules and access their contents through the module namespace, keeping their origin and private status explicit.

### Testing

Core functionality has deterministic, lightweight tests that run quickly enough to execute routinely.

* tests prioritize readability and scientific correctness
* tests should be light and fast whenever possible
* test known solutions and edge cases
* test mathematical routines against analytical results when available
* test collection invariants and provenance explicitly
* test missing-data and boundary behavior
* test chunked processing against equivalent whole-data operations
* test installed-package behavior, not only source-tree imports
