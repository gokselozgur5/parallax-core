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
        Self {
            left,
            right,
            rank,
            new_id,
        }
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
    table
        .binary_search_by(|entry| entry.key().cmp(&target))
        .ok()
        .map(|idx| &table[idx])
}

/// Validates if a table slice is strictly sorted by (left, right) with no duplicates.
pub fn is_table_sorted(table: &[MergeEntry]) -> bool {
    if table.len() <= 1 {
        return true;
    }
    table.windows(2).all(|w| w[0].key() < w[1].key())
}

/// Finds the best (lowest rank) applicable merge across adjacent pairs in tokens.
pub fn find_best_merge(table: &[MergeEntry], tokens: &[u32]) -> Option<(usize, MergeEntry)> {
    if tokens.len() < 2 {
        return None;
    }
    let mut best: Option<(usize, MergeEntry)> = None;
    for i in 0..tokens.len() - 1 {
        if let Some(entry) = find_merge(table, tokens[i], tokens[i + 1]) {
            match best {
                None => best = Some((i, *entry)),
                Some((_, current_best)) if entry.rank < current_best.rank => {
                    best = Some((i, *entry));
                }
                _ => {}
            }
        }
    }
    best
}

/// Encodes raw tokens iteratively using binary search lookups on the merge table.
pub fn bpe_encode(table: &[MergeEntry], tokens: &mut Vec<u32>) {
    while tokens.len() >= 2 {
        let Some((idx, entry)) = find_best_merge(table, tokens) else {
            break;
        };
        tokens[idx] = entry.new_id;
        tokens.remove(idx + 1);
    }
}
