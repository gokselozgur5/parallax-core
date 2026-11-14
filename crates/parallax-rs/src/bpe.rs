//! Byte-pair encoding merge table binary search implementation.

/// Status codes for tokenizer operations.
pub const STATUS_NOT_FOUND: i32 = -5;

/// An individual merge rule mapping (left, right) token IDs to rank and new_id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct MergeEntry {
    pub left: u32,
    pub right: u32,
    pub rank: u32,
    pub new_id: u32,
}

impl MergeEntry {
    /// Constructs a new MergeEntry.
    #[inline]
    pub const fn new(left: u32, right: u32, rank: u32, new_id: u32) -> Self {
        Self { left, right, rank, new_id }
    }

    /// Comparison key for sorting and binary searching: (left, right).
    #[inline]
    pub const fn key(&self) -> (u32, u32) {
        (self.left, self.right)
    }
}

/// Searches for a pair in a slice of MergeEntry sorted by (left, right).
pub fn find_merge(table: &[MergeEntry], left: u32, right: u32) -> Option<&MergeEntry> {
    let target = (left, right);
    table.binary_search_by(|entry| entry.key().cmp(&target)).ok().map(|idx| &table[idx])
}
