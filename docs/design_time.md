# Temporal collections: `Events`, `Windows`, and `Trains`

## Purpose

`nocte` distinguishes three fundamental arrangements of discrete temporal data:

```text
Events
    one item = one point occurrence

Windows
    one item = one interval occurrence

Trains
    one item = one ordered collection of point occurrences
```

These are complementary representations, not an inheritance hierarchy.

`Events` and `Windows` differ by **temporal geometry**: an event has one time; a window has interval extent.

`Trains` differs by **item identity**: the primary item is the train itself rather than any individual timestamp. In electrophysiology a train will commonly correspond to a unit and contain its spikes, but the abstraction is intentionally generic enough for other collections of related point occurrences.

All three inherit only from the normal collection infrastructure.

---

## Core semantics

### `Events`

`Events` is a collection of point-like temporal occurrences.

```text
meta
    one row per event

payload
    one time per event
```

An event has exactly one structural temporal value: `time`.

Duration is not part of `Events`. An occurrence with meaningful temporal extent is represented by `Windows`.

Examples include stimulus onsets, detected peaks, behavioral transitions, camera triggers, and individual spikes after flattening a `Trains` collection.

Event identity is independent of timestamp value: duplicate timestamps are valid and remain distinct events.

### `Windows`

`Windows` is the interval counterpart to `Events`.

```text
meta
    one row per window

payload
    start
    ref
    stop
```

It continues to represent arbitrary duration-bearing observations, including analysis windows, stimulation pulses, behavioral states, oscillatory episodes, detected waves, and similar phenomena.

`Windows` remains the single interval abstraction. Domain objects such as ripples or stimulation epochs are represented as `Windows` with appropriate metadata rather than through additional temporal container classes.

### `Trains`

`Trains` is an indexed collection of related timestamp sequences.

```text
global
    support: Win

meta
    one row per train

payload
    one variable-length timestamp array per train
```

For sorted electrophysiological data:

```text
train_id ≈ unit
payload item ≈ spike times for that unit
meta ≈ depth, quality, cell type, probe, ...
```

The name `Trains` reflects the actual collection identity: selecting a `Trains` collection selects trains, not individual spikes.

`Trains` is not spike-specific. Spike sorting import, waveform processing, unit-quality metrics, and other spike-domain operations should not define the abstraction itself.

---

## Structural data and metadata

Temporal structure must remain separate from descriptive metadata.

`Events.time`, `Windows.start/ref/stop`, and the timestamp sequences contained by `Trains` are authoritative numerical payloads. They are not ordinary metadata columns.

Metadata describes the represented items:

```text
Events.meta     → one row per occurrence
Windows.meta    → one row per interval
Trains.meta     → one row per train
```

This distinction is especially important for `Trains`: unit-level metadata must not be repeated once for every spike.

Public item-aligned values should follow the normal collection convention and expose identities through pandas objects where appropriate. Numerical kernels should operate directly on NumPy arrays and integer positions.

---

## `Events` implementation

`Events` should be rewritten around a small immutable `_EventsData` payload containing a contiguous one-dimensional `float` array of milliseconds.

Conceptually:

```python
class _EventsData:
    times: np.ndarray  # shape (n_events,)


class Events(HDFCollection[float]):
    _data: _EventsData
    meta: pd.DataFrame
```

The payload must contain finite values and have exactly one value per metadata row. `time` should be reserved from `meta`.

There is no need for a scalar `Event` class. `get(event_id)` can return the timestamp directly; descriptive information remains available through `meta`.

The minimum event-oriented API should include:

```text
time
sort_time()
shift(...)
crop(...)
intervals()
count_bins(...)
rate_gaussian(...)
to_windows(...)
```

`crop(win)` performs temporal selection: events outside the half-open interval are removed, while identities and metadata of retained events are preserved.

`intervals()` computes successive inter-event intervals in chronological order. More complicated grouping conditions should normally be expressed by first grouping or selecting the collection rather than growing a large collection-specific `by=...` API.

Simple event-density operations such as binned counts or Gaussian-smoothed rate are appropriate here because they are direct transforms of a point process. More elaborate event analyses belong in `analysis`.

Relations with `Windows` should use shared temporal matching machinery where possible. Common use cases include determining which state/window contains an event, obtaining its relative position within a window, and classifying events by interval metadata. Overlapping windows must be treated as a genuine one-to-many relation rather than silently forcing one classification.

Continuous-value lookup should not be reimplemented in `Events` when the corresponding operation naturally belongs to `Traces`; for example, sampling phase or amplitude at event times should use trace lookup against `events.time`.

---

## `Trains` implementation

`Trains` must be independent of `Events`.

Its in-memory payload should be a small ragged container containing one sorted NumPy array per train:

```python
class _TrainsData:
    times: tuple[np.ndarray, ...]
```

With hundreds of trains and potentially millions of total timestamps, one NumPy array per train gives a useful balance between simplicity and efficiency: there are only hundreds of Python objects and never one Python object per spike.

Each timestamp array must be:

* one-dimensional;
* finite;
* monotonically non-decreasing;
* interpreted in milliseconds;
* contained within the collection support.

Duplicate timestamps may exist.

`support: Win` is explicit collection-wide state and defines the period during which the entire trains collection was observed. It is required because an empty train is still a valid train, and rates or binned representations cannot be interpreted correctly from timestamp extrema alone.

Empty trains are valid and must be retained.

Selection operates on trains:

```python
trains.sel(cell_type='pyramidal')
```

and therefore selects both the corresponding metadata rows and ragged payload entries while leaving `support` unchanged.

`crop(win)` is different from `Events.crop`: it restricts every train to the requested temporal region, preserves every train identity including trains that become silent, and updates `support` to the cropped interval.

A scalar `shift(...)` shifts both timestamps and support.

The minimum train-oriented API should include:

```text
counts()
rates()
intervals()
crop(...)
shift(...)
count_in(...)
rate_in(...)
bin_counts(...)
rate_gaussian(...)
to_events()
```

note that each of these may either:
- be implemented through iterative converstion of each train to an Events collection and leveraging that code, or
- as a optimized already grouped operation
the choice will depend predominantly on efficiency.


`counts()` returns one total event count per train.

`rates()` returns the corresponding mean event rate using the explicit support duration.

`intervals()` returns the successive within-train intervals without promoting them to another temporal collection.

`count_in(windows)` produces the usual dense window × train count table. `rate_in(windows)` performs the corresponding duration normalization. Operations should reject observation windows extending outside known support unless clipping is explicitly requested; unobserved time must not silently become zero activity.

`bin_counts(bin_ms, ...)` converts the ragged point processes into regularly sampled `Traces`. Bins are half-open and trace sample positions correspond to bin centers.

`rate_gaussian(...)` may provide the standard lightweight kernel estimate of instantaneous event rate and return `Traces`. More elaborate point-process estimation belongs in `analysis`.

Spike-specific file import should live outside `core/trains.py`. A Kilosort adapter, for example, constructs ordinary `Trains` plus unit metadata rather than making `Trains` know about Kilosort.

---

## Conversion rules

Conversions follow the general collection identity rules.

### `Events → Windows`

Adding extent to point occurrences is a one-to-one transformation.

```python
windows = events.to_windows(win)
```

For each event:

```text
Windows.ref   = Events.time
Windows.start = requested relative start
Windows.stop  = requested relative stop
```

The event index and metadata are preserved exactly.

### `Windows → Events`

Selecting one meaningful point from every interval is also one-to-one:

```python
events = windows.to_events(at='ref')
events = windows.to_events(at='mid')
events = windows.to_events(at='start')
events = windows.to_events(at='stop')
events = windows.to_events(at=0.25)
```

The window index and metadata are preserved. Interval geometry is intentionally discarded.

`'ref'` is the natural default because it already represents the scientifically meaningful temporal anchor.

### `Trains → Events`

Flattening trains into individual occurrences is one-to-many.

```python
events = trains.to_events()
```

The resulting `Events` receives new event identities. Every event explicitly records the identity of its source train, normally as `train_id`.

Train metadata is **not** duplicated over millions of event rows by default. It remains available from the source `Trains.meta` and can be joined explicitly when needed.

This representation is intended for analyses in which individual spikes or occurrences become the primary observations.

### `Events → Trains`

Constructing trains from events is an explicit grouping/conversion operation:

```python
trains = Trains.from_events(
    events,
    by='unit',
    support=support,
)
```

or an equivalent convenience API.

This creates new train identities from the grouping keys.

Event-level metadata cannot generally become train-level metadata and must not be automatically aggregated. Train metadata may be supplied explicitly or derived deliberately by the caller.

The conversion is therefore intentionally lossy with respect to arbitrary event-level metadata.

### `Trains → Traces`

Binning preserves primary item identity:

```text
train 17 → trace 17
train 23 → trace 23
...
```

The output `Traces` preserves train indices and train metadata. Silent trains remain present as zero-valued traces.

This is the standard bridge from point-process representation into the existing continuous/sampled analysis stack.

---

## Relationship with `Grouping`

`Grouping` remains the generic mechanism for grouping collections and must not be replaced by temporal special cases.

```text
Events grouped by metadata  → Grouping[Events]
Windows grouped by metadata → Grouping[Windows]
Trains grouped by metadata  → Grouping[Trains]
```

A `Grouping[Events]` and a `Trains` collection are not equivalent.

Use `Grouping[Events]` when the grouping is temporary or when individual events and their metadata remain important.

Use `Trains` when each group is itself the primary scientific item and its members are fundamentally just a variable-length sequence of timestamps requiring point-process operations.

Consequently:

```python
events.group('unit')
```

should retain ordinary grouping semantics, while:

```python
Trains.from_events(events, by='unit', support=...)
```

is the explicit change of representation and primary identity.

Groups of duration-bearing occurrences do not require a fourth temporal type. A stimulation train composed of pulses, a burst of ripple windows, or another collection of related intervals is naturally represented as:

```text
Grouping[Windows]
```

A dedicated "train of windows" container should only be introduced if a future concrete use case requires representation or operations that `Grouping[Windows]` cannot provide cleanly.

---

## Responsibility boundary

The temporal containers should own representation-preserving, broadly useful operations:

```text
selection
temporal restriction
shifting
simple counts
simple rates
inter-event intervals
binning
conversion
serialization
```

Higher-level scientific algorithms belong in `analysis`, including for example:

```text
auto- and cross-correlograms
peri-event histograms and averages
burst detection
surrogate generation
tuning curves
decoding
statistical comparisons
trial-selectivity metrics
waveform and unit-quality analysis
```

The distinction is whether the operation is a basic manipulation or representation of the temporal data itself, rather than whether it happens to be commonly used in neuroscience.

---

## Storage and performance

The design must remain practical for millions of events or spikes.

`Events` stores timestamps as one contiguous numeric array rather than in pandas.

`Trains` stores only one numeric array per train rather than a long DataFrame or Python object per timestamp.

Train-level metadata stays proportional to the number of trains, not the number of timestamps.

Heavy operations resolve collection labels to positions once and then work with arrays.

For HDF5:

```text
Events
    metadata
    flat times array

Trains
    metadata
    support
    flat concatenated times array
    train offsets
```

The flattened HDF representation avoids object arrays and variable-length Python structures while allowing the simple tuple-of-arrays representation to be reconstructed in memory.

---

## Resulting temporal model

```text
                    duration
       Events ─────────────────────► Windows
          ▲                            │
          │                            │ choose ref/start/
 flatten  │                            │ stop/mid/quantile
          │                            ▼
       Trains                       Events
          │
          │ bin / smooth
          ▼
        Traces


temporary grouping:

Events  ──► Grouping[Events]
Windows ──► Grouping[Windows]
Trains  ──► Grouping[Trains]
```

The resulting model keeps the core small:

* `Events` owns point occurrences;
* `Windows` owns interval occurrences;
* `Trains` owns collections of related point occurrences;
* `Grouping` handles generic higher-order grouping;
* `Traces` owns regularly sampled numerical representations.

No class needs to pretend that a spike, a spike train, a temporal interval, and a temporal point are the same kind of item.
