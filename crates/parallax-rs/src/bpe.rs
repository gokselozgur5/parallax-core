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
