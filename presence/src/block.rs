use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Constants for presence tracking configuration
pub const WINDOWS_PER_BLOCK: u64 = 64;
pub const WINDOW_DURATION_MS: u64 = 33; // Approximately 30fps

/// Represents a block of presence data that efficiently stores temporal presence information
/// using a bitmap representation. Each block can store up to 64 time windows, with each window
/// typically representing 33ms (configured for 30fps).
///
/// # Memory Layout
/// The struct is aligned to 64 bytes to prevent false sharing in concurrent scenarios.
/// Each bit in the `data` field represents a single time window where:
/// - 1 indicates presence
/// - 0 indicates absence
///
/// # Example
/// ```rust
/// use presence::PresenceBlock;
///
/// let block = PresenceBlock::new(1000); // Start at timestamp 1000
/// block.mark_present(1033); // Mark presence at timestamp 1033 (window index 1)
/// ```
#[repr(C, align(64))]
#[derive(Serialize, Deserialize, Debug)]
pub struct PresenceBlock {
    /// The starting timestamp of this block in milliseconds
    pub start_time: u64,

    /// Bitmap representing 64 consecutive time windows
    #[serde(with = "atomic_u64")]
    pub data: AtomicU64,

    /// Optional link to the next block for chaining
    pub next_block: Option<Box<PresenceBlock>>,
}

impl PresenceBlock {
    /// Creates a new empty presence block starting at the specified timestamp
    ///
    /// # Arguments
    /// * `start_time` - The starting timestamp in milliseconds
    ///
    /// # Returns
    /// A new `PresenceBlock` instance with all windows initialized to absent
    pub fn new(start_time: u64) -> Self {
        Self {
            start_time,
            data: AtomicU64::new(0),
            next_block: None,
        }
    }

    /// Calculates the window index for a given timestamp
    ///
    /// # Arguments
    /// * `timestamp` - The timestamp to convert to a window index
    ///
    /// # Returns
    /// * `Some(index)` if the timestamp falls within this block
    /// * `None` if the timestamp is outside this block's range
    pub fn get_window_index(&self, timestamp: u64) -> Option<u32> {
        if timestamp < self.start_time {
            return None;
        }

        let window = (timestamp - self.start_time) / WINDOW_DURATION_MS;
        if window >= WINDOWS_PER_BLOCK {
            return None;
        }

        Some(window as u32)
    }

    /// Marks presence at the specified timestamp
    ///
    /// # Arguments
    /// * `timestamp` - The timestamp to mark as present
    ///
    /// # Returns
    /// * `Ok(())` if the presence was marked successfully
    /// * `Err(BlockError)` if the timestamp is outside this block's range
    pub fn mark_present(&self, timestamp: u64) -> Result<(), BlockError> {
        let window = self
            .get_window_index(timestamp)
            .ok_or(BlockError::TimestampOutOfRange)?;

        let mask = 1u64 << window;
        self.data.fetch_or(mask, Ordering::Release);
        Ok(())
    }

    /// Checks if presence was marked at the specified timestamp
    ///
    /// # Arguments
    /// * `timestamp` - The timestamp to check for presence
    ///
    /// # Returns
    /// * `Ok(bool)` indicating presence status
    /// * `Err(BlockError)` if the timestamp is outside this block's range
    pub fn was_present(&self, timestamp: u64) -> Result<bool, BlockError> {
        let window = self
            .get_window_index(timestamp)
            .ok_or(BlockError::TimestampOutOfRange)?;

        let mask = 1u64 << window;
        Ok((self.data.load(Ordering::Acquire) & mask) != 0)
    }

    /// Returns the duration this block covers
    pub fn duration(&self) -> Duration {
        Duration::from_millis(WINDOWS_PER_BLOCK * WINDOW_DURATION_MS)
    }

    /// Returns a debug representation of the block's data
    pub fn debug_info(&self) -> String {
        format!(
            "Start time: {}, Data: {:064b}",
            self.start_time,
            self.data.load(Ordering::Relaxed)
        )
    }

    /// Sets the next block in the chain
    pub fn set_next(&mut self, next: PresenceBlock) {
        self.next_block = Some(Box::new(next));
    }

    /// Get a reference to the next block in the chain
    pub fn next(&self) -> Option<&PresenceBlock> {
        self.next_block.as_deref()
    }
}

/// Custom Clone implementation for PresenceBlock
impl Clone for PresenceBlock {
    fn clone(&self) -> Self {
        Self {
            start_time: self.start_time,
            data: AtomicU64::new(self.data.load(Ordering::Relaxed)),
            next_block: self.next_block.clone(),
        }
    }
}

/// Error types for PresenceBlock operations
#[derive(Debug, PartialEq)]
pub enum BlockError {
    TimestampOutOfRange,
}

/// Serialization helpers for AtomicU64
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_block() {
        let block = PresenceBlock::new(1000);
        assert_eq!(block.start_time, 1000);
        assert_eq!(block.data.load(Ordering::Relaxed), 0);
        assert!(block.next_block.is_none());
    }

    #[test]
    fn test_mark_and_check_presence() {
        let block = PresenceBlock::new(1000);

        // Mark presence at different timestamps
        block.mark_present(1033).unwrap(); // Window 1
        block.mark_present(1066).unwrap(); // Window 2

        // Check presence
        assert!(block.was_present(1033).unwrap());
        assert!(block.was_present(1066).unwrap());
        assert!(!block.was_present(1099).unwrap());
    }

    #[test]
    fn test_out_of_range() {
        let block = PresenceBlock::new(1000);

        // Test timestamp before block start
        assert_eq!(
            block.mark_present(999).unwrap_err(),
            BlockError::TimestampOutOfRange
        );

        // Test timestamp after block end
        let end_time = 1000 + (WINDOWS_PER_BLOCK * WINDOW_DURATION_MS);
        assert_eq!(
            block.mark_present(end_time).unwrap_err(),
            BlockError::TimestampOutOfRange
        );
    }

    #[test]
    fn test_clone() {
        let block = PresenceBlock::new(1000);
        block.mark_present(1033).unwrap();

        let cloned = block.clone();
        assert_eq!(cloned.start_time, block.start_time);
        assert_eq!(
            cloned.data.load(Ordering::Relaxed),
            block.data.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn test_duration() {
        let block = PresenceBlock::new(1000);
        assert_eq!(
            block.duration(),
            Duration::from_millis(WINDOWS_PER_BLOCK * WINDOW_DURATION_MS)
        );
    }
}
