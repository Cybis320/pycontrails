# Parallel Fleet Processing

This feature adds efficient parallel processing support for processing multiple flights in pycontrails models.

## Motivation

When processing many flights individually in parallel workers, each worker must load the full meteorological dataset into memory, leading to excessive memory usage. This feature introduces **chunked parallel processing**, where flights are grouped into chunks and each chunk is processed as a Fleet, significantly reducing memory overhead while maintaining parallelism.

## Memory Comparison

| Approach | Met Data Loads | Memory Usage | Speed |
|----------|----------------|--------------|-------|
| Sequential Fleet | 1× | Lowest | Slowest |
| **Parallel Chunks** ✨ | N_chunks × | Medium | Fast |
| Parallel Individual | N_flights × | Highest | Fast |

## Features

- ✅ **Automatic chunking**: Flights are split into optimal chunks
- ✅ **Memory efficient**: Each chunk loads met data once (not per flight)
- ✅ **Configurable**: Control chunk size and number of workers
- ✅ **Compatible**: Works with DryAdvection and CoCiP models
- ✅ **No breaking changes**: Parallel processing is opt-in

## Usage

### Basic Example

```python
from pycontrails.models.dry_advection import DryAdvection

# Create model with parallel processing enabled
model = DryAdvection(
    met=met,
    parallel=True,  # Enable parallel processing
    n_jobs=-1,      # Use all CPU cores
)

# Process list of flights in parallel
flights = [flight1, flight2, flight3, ...]
results = model.eval(flights)  # Returns list[Flight]
```

### Advanced Configuration

```python
# Custom chunk size for memory control
model = DryAdvection(
    met=met,
    parallel=True,
    n_jobs=4,                # Use 4 workers
    parallel_chunk_size=10,  # 10 flights per chunk
)

results = model.eval(flights)
```

### With CoCiP

```python
from pycontrails.models.cocip import Cocip

model = Cocip(
    met=met,
    rad=rad,
    parallel=True,
    n_jobs=-1,
)

results = model.eval(flights)
```

## Parameters

### `parallel: bool = False`
Enable parallel processing. When `True`, flights are processed in parallel chunks.

### `n_jobs: int = -1`
Number of parallel workers:
- `-1`: Use all available CPU cores
- `> 0`: Use specified number of workers

### `parallel_chunk_size: int | None = None`
Number of flights per chunk:
- `None`: Split flights evenly across workers (default)
- `> 0`: Fixed number of flights per chunk

## How It Works

1. **Input**: User provides `Sequence[Flight]` with `parallel=True`
2. **Chunking**: Flights are split into N chunks
3. **Fleet Conversion**: Each chunk is converted to a Fleet
4. **Parallel Processing**: Chunks processed in parallel workers
5. **Met Efficiency**: Each worker loads met data once per chunk
6. **Output**: Results are combined into `list[Flight]`

## Performance Tips

### Chunk Size Selection

```python
# Small chunks: Lower memory per worker, more overhead
parallel_chunk_size=5

# Large chunks: Higher memory per worker, less overhead
parallel_chunk_size=50

# Auto (recommended): Evenly distributed across workers
parallel_chunk_size=None
```

### When to Use Parallel Processing

✅ **Good use cases:**
- Processing many flights (>10)
- Flights have overlapping regions
- Multi-core machine available
- Memory is limited

❌ **Not recommended:**
- Few flights (<5)
- Single-core machine
- Unlimited memory available

## Dependencies

Parallel processing requires `joblib`:

```bash
pip install joblib
```

## Implementation Details

- Added 3 new parameters to `ModelParams`: `parallel`, `n_jobs`, `parallel_chunk_size`
- Added 3 new methods to `Model` base class:
  - `_split_flights_into_chunks()`: Chunk splitting logic
  - `_eval_chunk()`: Per-chunk evaluation wrapper
  - `eval_parallel()`: Main parallel processing orchestration
- Updated `DryAdvection.eval()` and `Cocip.eval()` to support parallel mode
- Modified `Model._get_source()` to preserve flight lists when `parallel=True`

## Version

Added in version 0.54.12

## Example Benchmark

```python
import time

# Sequential processing
start = time.time()
model_seq = DryAdvection(met=met, parallel=False)
results_seq = model_seq.eval(flights)
time_seq = time.time() - start

# Parallel processing
start = time.time()
model_par = DryAdvection(met=met, parallel=True, n_jobs=-1)
results_par = model_par.eval(flights)
time_par = time.time() - start

print(f"Sequential: {time_seq:.1f}s")
print(f"Parallel: {time_par:.1f}s")
print(f"Speedup: {time_seq/time_par:.1f}x")
```

## See Also

- [Fleet documentation](docs/fleet.md)
- [DryAdvection documentation](docs/dry_advection.md)
- [CoCiP documentation](docs/cocip.md)
