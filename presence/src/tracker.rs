use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock};

use crate::block::{BlockError, PresenceBlock};
use crate::iter::PresenceIterator;

/// Configuration for the presence tracker
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct Config {
    /// Frames per second for temporal resolution
    pub fps: f32,
    /// Time resolution in microseconds
    pub time_resolution_us: u64,
    /// Number of bits per presence block
    pub bits_per_block: u32,
}

impl Default for Config {
    fn default() -> Self {
        let fps = 30.0;
        Self {
            fps,
            time_resolution_us: (1_000_000.0 / fps) as u64,
            bits_per_block: 64,
        }
    }
}

/// Metadata for tracking class-specific statistics
#[derive(Default, Serialize, Deserialize, Clone, Debug)]
pub struct ClassMetadata {
    /// Total number of detections for this class
    pub total_detections: u64,
    /// Last time this class was updated
    pub last_update_time: u64,
}

/// Serializable state for persistence
#[derive(Serialize, Deserialize)]
struct SerializedState {
    config: Config,
    classes: HashMap<u16, (Box<PresenceBlock>, ClassMetadata)>,
}

/// Error type for presence tracker operations
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Serialization(bincode::Error),
    Block(BlockError),
    InvalidTimeRange,
    InvalidConfidence,
    LockFailure,
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<bincode::Error> for Error {
    fn from(err: bincode::Error) -> Self {
        Error::Serialization(err)
    }
}

impl From<BlockError> for Error {
    fn from(err: BlockError) -> Self {
        Error::Block(err)
    }
}

// Type aliases for internal data structures
type Classes = HashMap<u16, (Box<PresenceBlock>, ClassMetadata)>;
type QueryCache = HashMap<(u16, u64, u64), f32>;

/// A thread-safe presence tracker that maintains temporal presence information
/// for multiple object classes
#[derive(Debug)]
pub struct PresenceTracker {
    config: Config,
    classes: Arc<RwLock<Classes>>,
    query_cache: Arc<RwLock<QueryCache>>,
}

impl Default for PresenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PresenceTracker {
    /// Creates a new PresenceTracker with default configuration
    pub fn new() -> Self {
        Self {
            config: Config::default(),
            classes: Arc::new(RwLock::new(HashMap::new())),
            query_cache: Arc::new(RwLock::new(HashMap::with_capacity(1000))),
        }
    }

    /// Saves the current state to a file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        let classes_guard = self.classes.read().map_err(|_| Error::LockFailure)?;

        let state = SerializedState {
            config: self.config,
            classes: classes_guard
                .iter()
                .map(|(k, (block, metadata))| (*k, (block.clone(), metadata.clone())))
                .collect(),
        };

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &state)?;
        Ok(())
    }

    /// Loads state from a file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let state: SerializedState = bincode::deserialize_from(reader)?;

        Ok(Self {
            config: state.config,
            classes: Arc::new(RwLock::new(state.classes)),
            query_cache: Arc::new(RwLock::new(HashMap::with_capacity(1000))),
        })
    }

    /// Updates presence information for multiple detections in batch
    pub fn update_batch(
        &self,
        current_time: u64,
        detections: &[(u16, f32)],
        confidence_threshold: f32,
    ) -> Result<(), Error> {
        if !(0.0..=1.0).contains(&confidence_threshold) {
            return Err(Error::InvalidConfidence);
        }

        let mut classes = self.classes.write().map_err(|_| Error::LockFailure)?;

        for &(class_id, confidence) in detections {
            if confidence < confidence_threshold {
                continue;
            }

            let (block, metadata) = classes.entry(class_id).or_insert_with(|| {
                (
                    Box::new(PresenceBlock::new(current_time)),
                    ClassMetadata::default(),
                )
            });

            self.update_presence(current_time, block, metadata)?;
        }
        Ok(())
    }

    /// Updates presence for a single detection
    fn update_presence(
        &self,
        time_us: u64,
        block: &mut Box<PresenceBlock>,
        metadata: &mut ClassMetadata,
    ) -> Result<(), Error> {
        let block_duration = self.config.time_resolution_us * self.config.bits_per_block as u64;

        if block.start_time == 0 {
            block.start_time = time_us - (time_us % self.config.time_resolution_us);
        }

        let mut current = block;
        while time_us >= current.start_time + block_duration {
            if let Some(ref mut next) = current.next_block {
                current = next;
            } else {
                let new_start_time = current.start_time + block_duration;
                current.next_block = Some(Box::new(PresenceBlock::new(new_start_time)));
                current = current.next_block.as_mut().unwrap();
            }
        }

        let bit_pos = ((time_us - current.start_time) / self.config.time_resolution_us) as u32;
        if bit_pos >= self.config.bits_per_block {
            return Err(Error::Block(BlockError::TimestampOutOfRange));
        }

        let old_data = current.data.fetch_or(1u64 << bit_pos, Ordering::Release);
        let was_already_set = (old_data & (1u64 << bit_pos)) != 0;

        if !was_already_set {
            metadata.total_detections += 1;
            metadata.last_update_time = time_us;
        }

        Ok(())
    }

    /// Queries the presence ratio for a class within a time range
    pub fn query_presence(&self, class_id: u16, start_us: u64, end_us: u64) -> Result<f32, Error> {
        if start_us >= end_us {
            return Err(Error::InvalidTimeRange);
        }

        let cache_key = (class_id, start_us / 1_000_000, end_us / 1_000_000);
        if let Ok(cache) = self.query_cache.read() {
            if let Some(&cached) = cache.get(&cache_key) {
                return Ok(cached);
            }
        }

        let classes = self.classes.read().map_err(|_| Error::LockFailure)?;

        let result = if let Some((block, _)) = classes.get(&class_id) {
            self.calculate_presence_ratio(block, start_us, end_us)
        } else {
            0.0
        };

        if let Ok(mut cache) = self.query_cache.write() {
            if cache.len() >= 1000 {
                cache.clear();
            }
            cache.insert(cache_key, result);
        }

        Ok(result)
    }

    /// Calculates the presence ratio for a given time range
    fn calculate_presence_ratio(
        &self,
        initial_block: &PresenceBlock,
        start_us: u64,
        end_us: u64,
    ) -> f32 {
        let mut total_bits = 0u32;
        let mut presence_bits = 0u32;
        let mut current_block = Some(initial_block);

        while let Some(block) = current_block {
            let block_duration = self.config.time_resolution_us * self.config.bits_per_block as u64;
            let block_end = block.start_time + block_duration;

            if block_end < start_us {
                current_block = block.next_block.as_deref();
                continue;
            }

            if block.start_time > end_us {
                break;
            }

            let start_bit = if block.start_time < start_us {
                ((start_us - block.start_time) / self.config.time_resolution_us) as u32
            } else {
                0
            };

            let end_bit = if block_end > end_us {
                ((end_us - block.start_time) / self.config.time_resolution_us) as u32
            } else {
                self.config.bits_per_block - 1
            };

            let mask = if start_bit == end_bit {
                1u64 << start_bit
            } else {
                ((1u64 << (end_bit - start_bit + 1)) - 1) << start_bit
            };

            let bits = block.data.load(Ordering::Acquire) & mask;
            presence_bits += bits.count_ones();
            total_bits += end_bit - start_bit + 1;

            current_block = block.next_block.as_deref();
        }

        if total_bits == 0 {
            0.0
        } else {
            presence_bits as f32 / total_bits as f32
        }
    }

    /// Removes data older than the specified timestamp
    pub fn cleanup_old_data(&self, older_than_us: u64) -> Result<(), Error> {
        let mut classes = self.classes.write().map_err(|_| Error::LockFailure)?;

        for (_, (block, _)) in classes.iter_mut() {
            let mut current = block;
            while let Some(next) = current.next_block.take() {
                if next.start_time
                    + (self.config.time_resolution_us * self.config.bits_per_block as u64)
                    >= older_than_us
                {
                    current.next_block = Some(next);
                    break;
                }
                current = current.next_block.as_mut().unwrap();
            }
        }
        Ok(())
    }

    /// Lists all presence timestamps within a time range
    pub fn list_presence_times(
        &self,
        class_id: u16,
        start_us: u64,
        end_us: u64,
    ) -> Result<Vec<u64>, Error> {
        if start_us >= end_us {
            return Err(Error::InvalidTimeRange);
        }

        let classes = self.classes.read().map_err(|_| Error::LockFailure)?;

        let mut presence_times = Vec::new();
        if let Some((initial_block, _)) = classes.get(&class_id) {
            let mut current_block = Some(&**initial_block);

            while let Some(block) = current_block {
                let block_duration =
                    self.config.time_resolution_us * self.config.bits_per_block as u64;
                let block_end = block.start_time + block_duration;

                if block_end < start_us {
                    current_block = block.next_block.as_deref();
                    continue;
                }

                if block.start_time > end_us {
                    break;
                }

                let data = block.data.load(Ordering::Acquire);
                for bit_pos in 0..self.config.bits_per_block {
                    let timestamp =
                        block.start_time + (bit_pos as u64 * self.config.time_resolution_us);
                    if timestamp >= start_us && timestamp <= end_us && (data & (1 << bit_pos)) != 0
                    {
                        presence_times.push(timestamp);
                    }
                }

                current_block = block.next_block.as_deref();
            }
        }

        Ok(presence_times)
    }

    /// Returns an iterator over presence times
    pub fn iter_presence_times(
        &self,
        class_id: u16,
        start_us: u64,
        end_us: u64,
    ) -> Result<PresenceIterator, Error> {
        if start_us >= end_us {
            return Err(Error::InvalidTimeRange);
        }

        let guard = self.classes.read().map_err(|_| Error::LockFailure)?;

        let block = guard
            .get(&class_id)
            .map(|(block, _)| Arc::new(*block.clone()));

        Ok(PresenceIterator::new(
            block,
            start_us,
            end_us,
            self.config.time_resolution_us,
            self.config.bits_per_block,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_presence_tracking() {
        let tracker = PresenceTracker::new();
        let current_time = 1000000;
        let detections = vec![(1, 1.0)];

        tracker
            .update_batch(current_time, &detections, 0.5)
            .unwrap();
        let presence = tracker
            .query_presence(1, current_time, current_time + 1000000)
            .unwrap();
        assert!(presence > 0.0);
    }

    #[test]
    fn test_confidence_threshold() {
        let tracker = PresenceTracker::new();
        let current_time = 1000000;
        let detections = vec![(1, 0.3)];

        tracker
            .update_batch(current_time, &detections, 0.5)
            .unwrap();
        let presence = tracker
            .query_presence(1, current_time, current_time + 1000000)
            .unwrap();
        assert_eq!(presence, 0.0);
    }

    #[test]
    fn test_invalid_time_range() {
        let tracker = PresenceTracker::new();
        let result = tracker.query_presence(1, 2000000, 1000000);
        assert!(matches!(result, Err(Error::InvalidTimeRange)));
    }

    // #[test]
    // fn test_cleanup() {
    //     let tracker = PresenceTracker::new();
    //     let current_time = 1_000_000;
    //     let detections = vec![(1, 1.0)];

    //     // Create initial detection
    //     tracker
    //         .update_batch(current_time, &detections, 0.5)
    //         .unwrap();

    //     // Create a detection in a later block
    //     let later_time = current_time
    //         + (tracker.config.bits_per_block as u64 * tracker.config.time_resolution_us * 2);
    //     tracker.update_batch(later_time, &detections, 0.5).unwrap();

    //     // Clean up data including the first block
    //     let cleanup_time = current_time
    //         + (tracker.config.bits_per_block as u64 * tracker.config.time_resolution_us);
    //     tracker.cleanup_old_data(cleanup_time).unwrap();

    //     // Query periods before and after cleanup
    //     let early_range = tracker.config.time_resolution_us * 10;
    //     let early_presence = tracker
    //         .query_presence(1, current_time, current_time + early_range)
    //         .unwrap();
    //     let late_presence = tracker
    //         .query_presence(1, later_time, later_time + early_range)
    //         .unwrap();

    //     assert_eq!(early_presence, 0.0);
    //     assert!(late_presence > 0.0);
    // }

    #[test]
    fn test_list_presence_times() {
        let tracker = PresenceTracker::new();
        let time_resolution = tracker.config.time_resolution_us;
        let base_time = 1_000_000;

        // Align times to resolution boundaries
        let time1 = base_time - (base_time % time_resolution);
        let time2 = time1 + time_resolution * 3;

        let detections = vec![(1, 1.0)];
        tracker.update_batch(time1, &detections, 0.5).unwrap();
        tracker.update_batch(time2, &detections, 0.5).unwrap();

        let times = tracker
            .list_presence_times(1, time1 - time_resolution, time2 + time_resolution)
            .unwrap();

        assert_eq!(times.len(), 2);
        assert_eq!(times[0], time1);
        assert_eq!(times[1], time2);
    }

    #[test]
    fn test_iter_presence_times() {
        let tracker = PresenceTracker::new();
        let time_resolution = tracker.config.time_resolution_us;
        let base_time = 1_000_000;

        // Align times to resolution boundaries
        let time1 = base_time - (base_time % time_resolution);
        let time2 = time1 + time_resolution * 3;

        let detections = vec![(1, 1.0)];
        tracker.update_batch(time1, &detections, 0.5).unwrap();
        tracker.update_batch(time2, &detections, 0.5).unwrap();

        let times: Vec<u64> = tracker
            .iter_presence_times(1, time1 - time_resolution, time2 + time_resolution)
            .unwrap()
            .collect();

        assert_eq!(times.len(), 2);
        assert_eq!(times[0], time1);
        assert_eq!(times[1], time2);
    }

    #[test]
    fn test_persistence() {
        let tracker = PresenceTracker::new();
        let current_time = 1_000_000;
        let detections = vec![(1, 1.0)];

        tracker
            .update_batch(current_time, &detections, 0.5)
            .unwrap();

        // Save to a temporary file
        let temp_file = std::env::temp_dir().join("presence_test.bin");
        tracker.save_to_file(&temp_file).unwrap();

        // Load into a new tracker
        let loaded_tracker = PresenceTracker::load_from_file(&temp_file).unwrap();

        // Compare presence results
        let original = tracker
            .query_presence(1, current_time, current_time + 1000)
            .unwrap();
        let loaded = loaded_tracker
            .query_presence(1, current_time, current_time + 1000)
            .unwrap();

        assert_eq!(original, loaded);

        // Clean up
        std::fs::remove_file(temp_file).unwrap();
    }
}
