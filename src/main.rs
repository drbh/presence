use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

#[repr(C, align(64))]
#[derive(Serialize, Deserialize)]
struct PresenceBlock {
    start_time: u64,

    // Each AtomicU64 represents 64 consecutive time windows.
    // At 30fps, each bit represents ~33.33ms (1/30th of a second),
    // so one u64 can store about 2.13 seconds of presence data (64 * 33.33ms).

    // We're using bits as a super compact representation (64 time windows in 8 bytes)
    #[serde(with = "atomic_u64")]
    data: AtomicU64,
    next_block: Option<Box<PresenceBlock>>,
}

impl PresenceBlock {
    pub fn debug_info(&self) -> String {
        format!(
            "Start time: {}, Data: {:064b}",
            self.start_time,
            self.data.load(Ordering::Relaxed)
        )
    }
}

// Custom Clone implementation for PresenceBlock
impl Clone for PresenceBlock {
    fn clone(&self) -> Self {
        Self {
            start_time: self.start_time,
            data: AtomicU64::new(self.data.load(Ordering::Relaxed)),
            next_block: self.next_block.clone(),
        }
    }
}

mod atomic_u64 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::sync::atomic::AtomicU64;

    pub fn serialize<S>(atomic: &AtomicU64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        atomic
            .load(std::sync::atomic::Ordering::Relaxed)
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<AtomicU64, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(AtomicU64::new(u64::deserialize(deserializer)?))
    }
}

impl PresenceBlock {
    fn new(start_time: u64) -> Self {
        Self {
            start_time,
            data: AtomicU64::new(0),
            next_block: None,
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
struct Config {
    fps: f32,
    time_resolution_us: u64,
    bits_per_block: u32,
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

#[derive(Default, Serialize, Deserialize, Clone)]
struct ClassMetadata {
    total_detections: u64,
    last_update_time: u64,
}

#[derive(Serialize, Deserialize)]
struct SerializedState {
    config: Config,
    classes: HashMap<u16, (Box<PresenceBlock>, ClassMetadata)>,
}

pub struct PresenceTracker {
    config: Config,
    classes: Arc<RwLock<HashMap<u16, (Box<PresenceBlock>, ClassMetadata)>>>,
    query_cache: Arc<RwLock<HashMap<(u16, u64, u64), f32>>>,
}

impl PresenceTracker {
    pub fn new() -> Self {
        Self {
            config: Config::default(),
            classes: Arc::new(RwLock::new(HashMap::new())),
            query_cache: Arc::new(RwLock::new(HashMap::with_capacity(1000))),
        }
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let classes_guard = self.classes.read().unwrap();
        let state = SerializedState {
            config: self.config,
            classes: classes_guard
                .iter()
                .map(|(k, (block, metadata))| (*k, (block.clone(), metadata.clone())))
                .collect(),
        };

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &state)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let state: SerializedState = bincode::deserialize_from(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        Ok(Self {
            config: state.config,
            classes: Arc::new(RwLock::new(state.classes)),
            query_cache: Arc::new(RwLock::new(HashMap::with_capacity(1000))),
        })
    }

    pub fn update_batch(
        &self,
        current_time: u64,
        detections: &[(u16, f32)],
        confidence_threshold: f32,
    ) {
        // let current_time = SystemTime::now()
        //     .duration_since(UNIX_EPOCH)
        //     .unwrap()
        //     .as_micros() as u64;

        let mut classes = self.classes.write().unwrap();

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

            self.update_presence(current_time, block, metadata);
        }
    }

    // // Fix the update_presence method to maintain block links correctly
    // fn update_presence(
    //     &self,
    //     time_us: u64,
    //     block: &mut Box<PresenceBlock>,
    //     metadata: &mut ClassMetadata,
    // ) {
    //     let block_duration = self.config.time_resolution_us * self.config.bits_per_block as u64;

    //     // When we detect presence at a timestamp:
    //     //
    //     // Calculate which of the 64 bits corresponds to that time
    //     // Set that bit to 1 using bitwise operations
    //     //
    //     // For example:
    //     // Time windows: [0ms, 33ms, 66ms, 99ms, ...]
    //     // Data bits:    [1,   0,   1,   0,   ...] = presence detected at 0ms and 66ms
    //     if time_us >= block.start_time + block_duration {
    //         // Find the appropriate block or create new ones as needed
    //         let mut current = block;
    //         while time_us >= current.start_time + block_duration {
    //             if let Some(ref mut next) = current.next_block {
    //                 current = next;
    //             } else {
    //                 let new_start_time = current.start_time + block_duration;
    //                 current.next_block = Some(Box::new(PresenceBlock::new(new_start_time)));
    //                 current = current.next_block.as_mut().unwrap();
    //             }
    //         }

    //         // Now we're in the correct block for the current time
    //         let bit_pos = ((time_us - current.start_time) / self.config.time_resolution_us) as u32;
    //         current.data.fetch_or(1 << bit_pos, Ordering::Release);
    //     } else {
    //         // Time is within the current block
    //         let bit_pos = ((time_us - block.start_time) / self.config.time_resolution_us) as u32;
    //         block.data.fetch_or(1 << bit_pos, Ordering::Release);
    //     }

    //     metadata.total_detections += 1;
    //     metadata.last_update_time = time_us;
    // }

    // fn update_presence(
    //     &self,
    //     time_us: u64,
    //     block: &mut Box<PresenceBlock>,
    //     metadata: &mut ClassMetadata,
    // ) {
    //     let block_duration = self.config.time_resolution_us * self.config.bits_per_block as u64;

    //     // Find or create the correct block for this timestamp
    //     let mut current = block;
    //     while time_us >= current.start_time + block_duration {
    //         if let Some(ref mut next) = current.next_block {
    //             current = next;
    //         } else {
    //             let new_start_time = current.start_time + block_duration;
    //             current.next_block = Some(Box::new(PresenceBlock::new(new_start_time)));
    //             current = current.next_block.as_mut().unwrap();
    //         }
    //     }

    //     // Calculate the bit position within the block
    //     let relative_time = time_us - current.start_time;
    //     let bit_pos = (relative_time / self.config.time_resolution_us) as u32;

    //     // Ensure bit_pos is within bounds
    //     if bit_pos < self.config.bits_per_block {
    //         current.data.fetch_or(1u64 << bit_pos, Ordering::Release);

    //         // Update metadata
    //         metadata.total_detections += 1;
    //         metadata.last_update_time = time_us;
    //     }
    // }

    // fn update_presence(
    //     &self,
    //     time_us: u64,
    //     block: &mut Box<PresenceBlock>,
    //     metadata: &mut ClassMetadata,
    // ) {
    //     let block_duration = self.config.time_resolution_us * self.config.bits_per_block as u64;

    //     // Find or create the correct block for this timestamp
    //     let mut current = block;

    //     // If this is the first update, align the start time
    //     if current.start_time == 0 {
    //         current.start_time = time_us - (time_us % self.config.time_resolution_us);
    //     }

    //     // Navigate to the correct block
    //     while time_us >= current.start_time + block_duration {
    //         if let Some(ref mut next) = current.next_block {
    //             current = next;
    //         } else {
    //             let new_start_time = current.start_time + block_duration;
    //             current.next_block = Some(Box::new(PresenceBlock::new(new_start_time)));
    //             current = current.next_block.as_mut().unwrap();
    //         }
    //     }

    //     // Calculate the bit position within the block
    //     let relative_time = time_us - current.start_time;
    //     let bit_pos = (relative_time / self.config.time_resolution_us) as u32;

    //     // Add debug output for the first few updates
    //     if metadata.total_detections < 5 {
    //         println!(
    //             "Setting bit {} in block starting at {} for time {}",
    //             bit_pos, current.start_time, time_us
    //         );
    //         println!(
    //             "Block data before: {:064b}",
    //             current.data.load(Ordering::Relaxed)
    //         );
    //     }

    //     // Set the bit
    //     if bit_pos < self.config.bits_per_block {
    //         let old_data = current.data.fetch_or(1u64 << bit_pos, Ordering::Release);

    //         // Debug output for first few updates
    //         if metadata.total_detections < 5 {
    //             println!(
    //                 "Block data after:  {:064b}",
    //                 current.data.load(Ordering::Relaxed)
    //             );
    //             println!(
    //                 "Bit was {}set",
    //                 if old_data & (1u64 << bit_pos) != 0 {
    //                     "already "
    //                 } else {
    //                     ""
    //                 }
    //             );
    //         }

    //         metadata.total_detections += 1;
    //         metadata.last_update_time = time_us;
    //     }
    // }

    fn update_presence(
        &self,
        time_us: u64,
        block: &mut Box<PresenceBlock>,
        metadata: &mut ClassMetadata,
    ) {
        let block_duration = self.config.time_resolution_us * self.config.bits_per_block as u64;

        // If this is the first update, align the block's start time to our time grid
        if block.start_time == 0 {
            block.start_time = time_us - (time_us % self.config.time_resolution_us);
        }

        // Find the correct block for this timestamp
        let mut current = block;
        while time_us >= current.start_time + block_duration {
            if let Some(ref mut next) = current.next_block {
                current = next;
            } else {
                // Create new block aligned to the time grid
                let new_start_time = current.start_time + block_duration;
                current.next_block = Some(Box::new(PresenceBlock::new(new_start_time)));
                current = current.next_block.as_mut().unwrap();
            }
        }

        // Calculate relative position within the current block
        let bit_pos = ((time_us - current.start_time) / self.config.time_resolution_us) as u32;

        if bit_pos >= self.config.bits_per_block {
            // This shouldn't happen if our block management is correct
            println!(
                "Warning: Invalid bit position {} for time {} in block starting at {}",
                bit_pos, time_us, current.start_time
            );
            return;
        }

        // Set the bit and update metadata
        let old_data = current.data.fetch_or(1u64 << bit_pos, Ordering::Release);
        let was_already_set = (old_data & (1u64 << bit_pos)) != 0;

        // Only increment total detections if this was a new detection
        if !was_already_set {
            metadata.total_detections += 1;
            metadata.last_update_time = time_us;
        }
    }

    pub fn query_presence(&self, class_id: u16, start_us: u64, end_us: u64) -> f32 {
        let cache_key = (class_id, start_us / 1_000_000, end_us / 1_000_000);
        if let Some(&cached) = self.query_cache.read().unwrap().get(&cache_key) {
            println!("Cache hit!");
            return cached;
        }

        let classes = self.classes.read().unwrap();
        let result = if let Some((block, _)) = classes.get(&class_id) {
            self.calculate_presence_ratio(&*block, start_us, end_us)
        } else {
            0.0
        };

        let mut cache = self.query_cache.write().unwrap();
        if cache.len() >= 1000 {
            cache.clear();
        }
        cache.insert(cache_key, result);

        result
    }

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
                current_block = block.next_block.as_ref().map(|b| &**b);
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

            // To find presence in a time range:
            //
            // Create a bit mask for the time window
            // AND it with our data
            // Count the 1s (which CPU can do very efficiently with a single instruction)
            let mask = if start_bit == end_bit {
                1u64 << start_bit
            } else {
                ((1u64 << (end_bit - start_bit + 1)) - 1) << start_bit
            };

            let bits = block.data.load(Ordering::Acquire) & mask;
            presence_bits += bits.count_ones();
            total_bits += end_bit - start_bit + 1;

            current_block = block.next_block.as_ref().map(|b| &**b);
        }

        if total_bits == 0 {
            0.0
        } else {
            presence_bits as f32 / total_bits as f32
        }
    }

    pub fn cleanup_old_data(&self, older_than_us: u64) {
        let mut classes = self.classes.write().unwrap();
        for (_, (block, _)) in classes.iter_mut() {
            while let Some(next) = block.next_block.take() {
                if next.start_time
                    + (self.config.time_resolution_us * self.config.bits_per_block as u64)
                    >= older_than_us
                {
                    block.next_block = Some(next);
                    break;
                }
            }
        }
    }

    /// Returns a vector of microsecond timestamps where presence was detected
    /// in the given time range [start_us, end_us]
    pub fn list_presence_times(&self, class_id: u16, start_us: u64, end_us: u64) -> Vec<u64> {
        let classes = self.classes.read().unwrap();

        if let Some((initial_block, _)) = classes.get(&class_id) {
            let mut presence_times = Vec::new();
            let mut current_block = Some(&**initial_block);

            while let Some(block) = current_block {
                let block_duration =
                    self.config.time_resolution_us * self.config.bits_per_block as u64;
                let block_end = block.start_time + block_duration;

                // Skip blocks before our range
                if block_end < start_us {
                    current_block = block.next_block.as_ref().map(|b| &**b);
                    continue;
                }

                // Stop if we're past our range
                if block.start_time > end_us {
                    break;
                }

                // Get the data and find set bits
                let data = block.data.load(Ordering::Acquire);

                // For each bit in the block
                for bit_pos in 0..self.config.bits_per_block {
                    // Calculate the exact timestamp for this bit
                    let timestamp =
                        block.start_time + (bit_pos as u64 * self.config.time_resolution_us);

                    // Check if it's in our range and the bit is set
                    if timestamp >= start_us && timestamp <= end_us && (data & (1 << bit_pos)) != 0
                    {
                        presence_times.push(timestamp);
                    }
                }

                current_block = block.next_block.as_ref().map(|b| &**b);
            }

            presence_times
        } else {
            Vec::new()
        }
    }

    /// Returns an iterator over presence times
    pub fn iter_presence_times(
        &self,
        class_id: u16,
        start_us: u64,
        end_us: u64,
    ) -> PresenceIterator {
        let guard = self.classes.read().unwrap();
        let block = guard
            .get(&class_id)
            .map(|(block, _)| Arc::new(*block.clone()));

        PresenceIterator {
            block,
            current_bit: 0,
            start_us,
            end_us,
            time_resolution_us: self.config.time_resolution_us,
            bits_per_block: self.config.bits_per_block,
        }
    }
}

// pub struct PresenceIterator {
//     block: Option<Arc<PresenceBlock>>,
//     current_bit: u32,
//     start_us: u64,
//     end_us: u64,
//     time_resolution_us: u64,
//     bits_per_block: u32,
// }

// impl Iterator for PresenceIterator {
//     type Item = u64;

//     fn next(&mut self) -> Option<Self::Item> {
//         while let Some(block) = &self.block {
//             while self.current_bit < self.bits_per_block {
//                 let timestamp =
//                     block.start_time + (self.current_bit as u64 * self.time_resolution_us);
//                 let bit_pos = self.current_bit;
//                 self.current_bit += 1;

//                 if timestamp > self.end_us {
//                     return None;
//                 }

//                 if timestamp < self.start_us {
//                     continue;
//                 }

//                 let data = block.data.load(std::sync::atomic::Ordering::Acquire);
//                 if (data & (1 << bit_pos)) != 0 {
//                     return Some(timestamp);
//                 }
//             }

//             // Move to next block if it exists
//             self.block = block.next_block.as_ref().map(|b| Arc::new(*b.clone()));
//             self.current_bit = 0;
//         }

//         None
//     }
// }

// pub struct PresenceIterator {
//     block: Option<Arc<PresenceBlock>>,
//     current_bit: u32,
//     start_us: u64,
//     end_us: u64,
//     time_resolution_us: u64,
//     bits_per_block: u32,
// }

// impl Iterator for PresenceIterator {
//     type Item = u64;

//     fn next(&mut self) -> Option<Self::Item> {
//         loop {
//             if let Some(block) = &self.block {
//                 if self.current_bit >= self.bits_per_block {
//                     // Move to next block
//                     self.block = block.next_block.as_ref().map(|b| Arc::new(*b.clone()));
//                     self.current_bit = 0;
//                     continue;
//                 }

//                 let timestamp =
//                     block.start_time + (self.current_bit as u64 * self.time_resolution_us);
//                 let bit_pos = self.current_bit;
//                 self.current_bit += 1;

//                 if timestamp > self.end_us {
//                     return None;
//                 }

//                 if timestamp < self.start_us {
//                     continue;
//                 }

//                 let data = block.data.load(std::sync::atomic::Ordering::Acquire);
//                 if (data & (1 << bit_pos)) != 0 {
//                     return Some(timestamp);
//                 }
//             } else {
//                 return None;
//             }
//         }
//     }
// }

pub struct PresenceIterator {
    block: Option<Arc<PresenceBlock>>,
    current_bit: u32,
    start_us: u64,
    end_us: u64,
    time_resolution_us: u64,
    bits_per_block: u32,
}

// impl Iterator for PresenceIterator {
//     type Item = u64;

//     fn next(&mut self) -> Option<Self::Item> {
//         while let Some(block) = &self.block {
//             let block_duration = self.time_resolution_us * self.bits_per_block as u64;
//             let block_end = block.start_time + block_duration;

//             // Skip blocks entirely before our range
//             if block_end < self.start_us {
//                 self.block = block.next_block.as_ref().map(|b| Arc::new(*b.clone()));
//                 self.current_bit = 0;
//                 continue;
//             }

//             // Find the next set bit in the current block
//             while self.current_bit < self.bits_per_block {
//                 let timestamp =
//                     block.start_time + (self.current_bit as u64 * self.time_resolution_us);
//                 let bit_pos = self.current_bit;
//                 self.current_bit += 1;

//                 // Check if we've gone past the end time
//                 if timestamp > self.end_us {
//                     return None;
//                 }

//                 // Skip timestamps before our start time
//                 if timestamp < self.start_us {
//                     continue;
//                 }

//                 let data = block.data.load(std::sync::atomic::Ordering::Acquire);
//                 if (data & (1 << bit_pos)) != 0 {
//                     return Some(timestamp);
//                 }
//             }

//             // Move to the next block when we've exhausted all bits
//             if self.current_bit >= self.bits_per_block {
//                 self.block = block.next_block.as_ref().map(|b| Arc::new(*b.clone()));
//                 self.current_bit = 0;
//             }
//         }

//         None
//     }
// }

impl Iterator for PresenceIterator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(block) = &self.block {
            // Skip blocks entirely before our range
            if block.start_time + (self.bits_per_block as u64 * self.time_resolution_us)
                < self.start_us
            {
                self.block = block.next_block.as_ref().map(|b| Arc::new(*b.clone()));
                self.current_bit = 0;
                continue;
            }

            // Process bits in the current block
            while self.current_bit < self.bits_per_block {
                let timestamp =
                    block.start_time + (self.current_bit as u64 * self.time_resolution_us);
                let bit_pos = self.current_bit;
                self.current_bit += 1;

                // Check time bounds
                if timestamp > self.end_us {
                    return None;
                }
                if timestamp < self.start_us {
                    continue;
                }

                // Check if this bit is set
                let data = block.data.load(std::sync::atomic::Ordering::Acquire);
                if (data & (1u64 << bit_pos)) != 0 {
                    return Some(timestamp);
                }
            }

            // Move to next block
            self.block = block.next_block.as_ref().map(|b| Arc::new(*b.clone()));
            self.current_bit = 0;
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    use tempfile::NamedTempFile;

    use std::collections::HashSet;

    #[test]
    fn test_serialization() {
        let tracker = PresenceTracker::new();

        // Add some data
        let detections = vec![(0u16, 1.0f32)];
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        tracker.update_batch(current_time, &detections, 0.5);

        // Create temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Save state
        tracker.save_to_file(path).unwrap();

        // Load state into new tracker
        let loaded_tracker = PresenceTracker::load_from_file(path).unwrap();

        // Verify data
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let original = tracker.query_presence(0, now - 1_000_000, now);
        let loaded = loaded_tracker.query_presence(0, now - 1_000_000, now);

        assert!((original - loaded).abs() < f32::EPSILON);
    }

    #[test]
    fn test_iterator_basics() {
        let tracker = PresenceTracker::new();

        // Create a simple detection pattern
        let detections = vec![(0u16, 1.0f32)];
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        tracker.update_batch(current_time, &detections, 0.5);

        // Get current time
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        // Try to iterate over a small time window
        let window_start = now - 1_000_000; // 1 second ago
        let window_end = now + 1_000_000; // 1 second from now

        println!("Testing iterator from {} to {}", window_start, window_end);

        let mut count = 0;
        for time in tracker.iter_presence_times(0, window_start, window_end) {
            println!("Found presence at: {}", time);
            count += 1;
        }

        println!("Found {} presence times", count);
        assert!(count > 0, "Should find at least one presence time");
    }

    use rand::Rng;

    #[test]
    fn benchmark_presence_tracking() {
        use std::time::Instant;

        let tracker = PresenceTracker::new();
        let mut rng = rand::thread_rng();

        // Test parameters
        let fps = 30;
        let frame_duration_us = 1_000_000 / fps;
        // let test_duration_seconds = 60 * 60 * 24; // 24 hours
        let test_duration_seconds = 60 * 60; // 24 hours
        let total_frames = fps * test_duration_seconds;
        let num_classes = 5;
        let detection_probability = 0.05;

        // Get base timestamp and ensure it aligns with our time resolution
        let base_time = {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64;
            now - (now % frame_duration_us) // Align to frame boundaries
        };

        println!("Base time: {}", base_time);

        // Ground truth storage
        let mut ground_truth = HashMap::new();
        for class_id in 0..num_classes {
            ground_truth.insert(class_id as u16, HashSet::new());
        }

        let start_time = Instant::now();
        let mut total_detections = 0;

        // let mut frame_time = base_time;

        // Process frames
        for frame in 0..total_frames {
            let frame_time = base_time + (frame as u64 * frame_duration_us);
            let mut frame_detections = Vec::new();

            // Generate detections for this frame
            for class_id in 0..num_classes {
                if rng.gen::<f32>() < detection_probability {
                    frame_detections.push((class_id as u16, 1.0));
                    ground_truth
                        .get_mut(&(class_id as u16))
                        .unwrap()
                        .insert(frame_time);
                    total_detections += 1;
                }
            }

            // // Debug output for first few frames
            // if frame < 5 {
            //     println!(
            //         "Frame {}: {} detections at time {}",
            //         frame,
            //         frame_detections.len(),
            //         frame_time
            //     );
            // }

            if !frame_detections.is_empty() {
                // println!("Frame {:?}", frame_detections);
                // let current_time = SystemTime::now()
                //     .duration_since(UNIX_EPOCH)
                //     .unwrap()
                //     .as_micros() as u64;

                // frame_time = base_time + (frame as u64 * frame_duration_us);

                tracker.update_batch(frame_time, &frame_detections, 0.5);
            }

            // Progress update
            let progress = ((frame * 100) / total_frames) as u32;
            // if progress % 10 == 0 && frame > 0 {
            //     println!(
            //         "Progress: {}% - {} detections processed in {:?}",
            //         progress,
            //         total_detections,
            //         start_time.elapsed()
            //     );
            // }
        }

        println!("ground_truth:\n{:#?}", ground_truth);

        let mut did_error = false;

        // In the benchmark test, just before verification:
        println!("\nVerifying data integrity...");
        for class_id in 0..num_classes {
            let class_id = class_id as u16;
            let true_times = ground_truth.get(&class_id).unwrap();

            println!(
                "\nClass {}: Expecting {} detections",
                class_id,
                true_times.len()
            );

            // Print block information for this class
            if let Some((block, metadata)) = tracker.classes.read().unwrap().get(&class_id) {
                let mut current = Some(block);
                let mut block_num = 0;
                while let Some(b) = current {
                    println!("\nBlock {}:", block_num);
                    println!("Start time: {}", b.start_time);
                    println!("Data: {:064b}", b.data.load(Ordering::Relaxed));
                    println!("Bit count: {}", b.data.load(Ordering::Relaxed).count_ones());
                    current = b.next_block.as_ref().map(|next| next);
                    block_num += 1;
                }
            }

            let start_time = base_time;
            let end_time = base_time + (total_frames as u64 * frame_duration_us);
            println!("\nQuerying from {} to {}", start_time, end_time);

            let detected_times: HashSet<u64> = tracker
                .iter_presence_times(class_id, start_time, end_time)
                .collect();

            println!("Found {} detections", detected_times.len());

            // If there's a mismatch, print detailed debugging info
            if detected_times.len() != true_times.len() {
                println!("\nMismatch details:");
                println!("First 5 ground truth times:");
                for &time in true_times.iter().take(5) {
                    let offset = time.saturating_sub(base_time);
                    println!(
                        "  {} ({} ms from base, frame {})",
                        time,
                        offset / 1000,
                        offset / frame_duration_us
                    );
                }
                println!("\nFirst 5 detected times:");
                for &time in detected_times.iter().take(5) {
                    let offset = time.saturating_sub(base_time);
                    println!(
                        "  {} ({} ms from base, frame {})",
                        time,
                        offset / 1000,
                        offset / frame_duration_us
                    );
                }

                // Check for specific mismatches
                println!("\nFirst 5 missing detections:");
                for &time in true_times.difference(&detected_times).take(5) {
                    let offset = time.saturating_sub(base_time);
                    println!(
                        "  {} ({} ms from base, frame {})",
                        time,
                        offset / 1000,
                        offset / frame_duration_us
                    );
                    did_error = true;
                }
            }
        }

        assert!(!did_error, "Presence tracking verification failed");
    }
}

fn main() {
    // let tracker = PresenceTracker::new();

    // tracker.save_to_file("tracker_state.bin").unwrap();
    // let detections = vec![
    //     (0u16, 0.9f32),  // person with 90% confidence
    //     (2u16, 0.85f32), // car with 85% confidence
    // ];

    // tracker.update_batch(&detections, 0.5);

    // Load state
    let loaded_tracker = PresenceTracker::load_from_file("tracker_state.bin").unwrap();

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;

    // let presence = loaded_tracker.query_presence(0, now - 1_000_000, now);
    // let presence = loaded_tracker.query_presence(0, 0, now);
    // println!("Person presence in last second: {:.2}", presence);

    // iterate over presence times for class 0
    for time in loaded_tracker.iter_presence_times(0, 0, now) {
        println!("Presence at: {}", time);
    }

    // add a new detection
    let detections = vec![(0u16, 1.0f32)];
    let current_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;
    loaded_tracker.update_batch(current_time, &detections, 0.5);

    for time in loaded_tracker.iter_presence_times(0, 0, now) {
        println!("After Add | Presence at: {}", time);
    }

    // Save state
    loaded_tracker.save_to_file("tracker_state.bin").unwrap();
}
