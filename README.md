# presence

A thread-safe library for efficient temporal presence tracking at microsecond resolution.

## Python Usage

```python
from presence import PresenceTracker

# Create a new tracker (default 30 FPS)
tracker = PresenceTracker()

# Track some detections
detections = [(0, 0.9)]  # (class_id, confidence)
frame_time = 33_333  # microseconds (about 30 FPS)

# Update multiple frames
for i in range(1, 10):
    t = frame_time * i
    tracker.update_batch(t, detections, confidence_threshold=0.5)

# Query all presence timestamps
timestamps = tracker.list_presence_timestamps(
    class_id=0, 
    start_time=0, 
    end_time=frame_time * 1000
)

# Save/load state
tracker.save_to_file("tracker_state.bin")
tracker.load_from_file("tracker_state.bin")
```

## Quickrun

The following command should initialize a virtual environment, install the package, and run the example script:

```bash
uv run --reinstall --directory bindings/python examples/readme.py
# Presence timestamps: [2, 4, 6, 8]
```


## Overview

`presence` provides a bit-packed temporal tracking system that efficiently stores and queries boolean states over time. Each 64-bit block can store 64 distinct temporal states, making it extremely memory efficient:

| Time Resolution | Memory Usage | Number of frames indexed over |
| --------------- | ------------ | ----------------------------- |
| 1 second        | 8 bytes      | 30 frames                     |
| 1 minute        | 480 bytes    | 1,800 frames                  |
| 1 hour          | 28.8 KB      | 108,000 frames                |
| 1 day           | 691.2 KB     | 2,592,000 frames              |
| 1 week          | 4.8 MB       | 18,144,000 frames             |
| 1 month         | 21 MB        | 78,893,168 frames             |
| 1 year          | 252 MB       | 946,080,000 frames            |

The bit-packing approach allows for instant access to any temporal state through bitwise operations, making both reads and writes extremely fast.

## Key Features

- **Ultra-Compact Storage**: Uses bit-packing to store 64 states per 8 bytes
- **Fast Access**: O(1) writes, efficient bitwise operations for reads
- **Thread Safety**: All operations are protected by RwLocks for concurrent access
- **Configurable Resolution**: Adjustable frames per second and time resolution
- **Persistence**: Save and load state from files
- **Memory Efficient**: Automatic cleanup of old data
- **Batch Operations**: Update multiple detections efficiently
- **Python Bindings**: Simple Python API for easy integration

## Rust Usage

```rust
use presence::PresenceTracker;

// Create tracker with default config (30 FPS)
let tracker = PresenceTracker::new();

// Update batch of detections with confidence threshold
let current_time = 1_000_000; // microseconds
let detections = vec![(class_id, confidence_score)];
tracker.update_batch(current_time, &detections, 0.5)?;

// Query presence ratio for a time range
let ratio = tracker.query_presence(class_id, start_us, end_us)?;

// Get specific presence timestamps 
let times = tracker.list_presence_times(class_id, start_us, end_us)?;
```

## Configuration

```rust
pub struct Config {
    /// Frames per second for temporal resolution
    pub fps: f32,
    /// Time resolution in microseconds
    pub time_resolution_us: u64,
    /// Number of bits per presence block
    pub bits_per_block: u32,
}

// Default configuration:
// - 30 FPS
// - ~33.33μs time resolution
// - 64 bits per block
```

## Implementation Details

The library uses a linked chain of `PresenceBlock` structures to store temporal data:

- Each block contains a 64-bit atomic bitmap for presence states
- Bits represent presence/absence at specific timestamps
- Blocks are chained only when needed, minimizing memory allocation
- Query cache improves performance for repeated time range queries
- Memory is allocated dynamically and can be cleaned up through `cleanup_old_data()`

## Notes

We can calculate the memory usage for a specific time range using the following formula:

```python
def memory_usage(time_resolution: int) -> int:
    return 8 * (time_resolution // 1_000_000)

# make the table
for res in [1, 60, 3600, 86400, 604800, 2_629_746, 31_556_952]:
    microseconds = res * 1_000_000
    frames = microseconds // 33_333
    print(f"{res} seconds: {memory_usage(microseconds)} bytes ({frames} frames)")
```
