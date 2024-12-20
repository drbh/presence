use crate::block::PresenceBlock;
use std::sync::Arc;

/// Iterator over presence timestamps within a specified time range.
///
/// This iterator traverses a linked list of `PresenceBlock`s and yields timestamps
/// where presence was detected. It efficiently skips blocks and bits that are outside
/// the specified time range.
///
/// # Example
/// ```rust
/// use std::sync::Arc;
/// use presence::{PresenceBlock, PresenceIterator};
///
/// let block = Arc::new(PresenceBlock::new(1000));
/// block.mark_present(1033).unwrap();
///
/// let iterator = PresenceIterator::new(
///     Some(block),
///     1000,    // start time
///     2000,    // end time
///     33,      // time resolution
///     64,      // bits per block
/// );
///
/// for timestamp in iterator {
///     println!("Present at: {}", timestamp);
/// }
/// ```
#[derive(Debug)]
pub struct PresenceIterator {
    /// Current block being processed
    pub block: Option<Arc<PresenceBlock>>,
    /// Current bit position within the block
    pub current_bit: u32,
    /// Start time of the range to iterate over (microseconds)
    pub start_us: u64,
    /// End time of the range to iterate over (microseconds)
    pub end_us: u64,
    /// Time resolution (microseconds per bit)
    pub time_resolution_us: u64,
    /// Number of bits in each block
    pub bits_per_block: u32,
}

impl PresenceIterator {
    /// Creates a new PresenceIterator.
    ///
    /// # Arguments
    /// * `initial_block` - First block to start iteration from
    /// * `start_us` - Start time in microseconds
    /// * `end_us` - End time in microseconds
    /// * `time_resolution_us` - Time resolution in microseconds
    /// * `bits_per_block` - Number of bits per block
    ///
    /// # Returns
    /// A new PresenceIterator instance
    pub fn new(
        initial_block: Option<Arc<PresenceBlock>>,
        start_us: u64,
        end_us: u64,
        time_resolution_us: u64,
        bits_per_block: u32,
    ) -> Self {
        Self {
            block: initial_block,
            current_bit: 0,
            start_us,
            end_us,
            time_resolution_us,
            bits_per_block,
        }
    }

    /// Calculates the timestamp for a given block and bit position
    #[inline]
    fn calculate_timestamp(&self, block: &PresenceBlock, bit_pos: u32) -> u64 {
        block.start_time + (bit_pos as u64 * self.time_resolution_us)
    }

    /// Checks if a block ends before the start of our range
    #[inline]
    fn block_precedes_range(&self, block: &PresenceBlock) -> bool {
        block.start_time + (self.bits_per_block as u64 * self.time_resolution_us) < self.start_us
    }

    /// Advances to the next block, handling the Arc cloning
    #[inline]
    fn advance_to_next_block(&self, block: &PresenceBlock) -> Option<Arc<PresenceBlock>> {
        block.next_block.as_ref().map(|b| Arc::new(*b.clone()))
    }
}

impl Iterator for PresenceIterator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(block) = &self.block {
            // Skip blocks entirely before our range
            if self.block_precedes_range(block) {
                self.block = self.advance_to_next_block(block);
                self.current_bit = 0;
                continue;
            }

            // Process bits in the current block
            while self.current_bit < self.bits_per_block {
                let timestamp = self.calculate_timestamp(block, self.current_bit);
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
            self.block = self.advance_to_next_block(block);
            self.current_bit = 0;
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_block(start_time: u64) -> Arc<PresenceBlock> {
        Arc::new(PresenceBlock::new(start_time))
    }

    #[test]
    fn test_empty_iterator() {
        let iterator = PresenceIterator::new(None, 0, 1000, 33, 64);
        assert_eq!(iterator.collect::<Vec<_>>(), vec![]);
    }

    #[test]
    fn test_single_presence() {
        let block = create_test_block(1000);
        block.mark_present(1033).unwrap();

        let iterator = PresenceIterator::new(Some(block), 1000, 2000, 33, 64);

        assert_eq!(iterator.collect::<Vec<_>>(), vec![1033]);
    }

    #[test]
    fn test_multiple_blocks() {
        let mut block1 = PresenceBlock::new(1000);
        let block2 = PresenceBlock::new(3000);

        block1.mark_present(1033).unwrap();
        block1.set_next(block2);

        // Now get a reference to block2 through block1 to mark presence
        if let Some(next_block) = block1.next() {
            next_block.mark_present(3033).unwrap();
        }

        let block1 = Arc::new(block1);
        let iterator = PresenceIterator::new(Some(block1), 1000, 4000, 33, 64);

        assert_eq!(iterator.collect::<Vec<_>>(), vec![1033, 3033]);
    }

    #[test]
    fn test_out_of_range() {
        let block = create_test_block(1000);
        block.mark_present(1033).unwrap();
        block.mark_present(1066).unwrap();

        // Test range before presence
        let iterator = PresenceIterator::new(Some(block.clone()), 0, 500, 33, 64);
        assert_eq!(iterator.collect::<Vec<_>>(), vec![]);

        // Test range after presence
        let iterator = PresenceIterator::new(Some(block.clone()), 2000, 3000, 33, 64);
        assert_eq!(iterator.collect::<Vec<_>>(), vec![]);

        // Test partial range
        let iterator = PresenceIterator::new(Some(block), 1050, 1100, 33, 64);
        assert_eq!(iterator.collect::<Vec<_>>(), vec![1066]);
    }

    #[test]
    fn test_time_resolution() {
        let block = create_test_block(1000);
        block.mark_present(1100).unwrap();

        // Test with different time resolutions
        let iterator = PresenceIterator::new(
            Some(block),
            1000,
            2000,
            50, // different time resolution
            64,
        );

        let timestamps: Vec<_> = iterator.collect();

        // With a time resolution of 50μs, the timestamp 1100 should be rounded
        // to the nearest multiple of 50 after the block start time
        assert_eq!(timestamps, vec![1150]);
    }
}
