//! Storage memory telemetry and eviction statistics.

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StorageTelemetry {
    pub resident_bytes: usize,
    pub mapped_bytes: usize,
    pub eviction_count: u64,
    pub evicted_bytes: u64,
}

impl StorageTelemetry {
    pub const fn new() -> Self {
        Self {
            resident_bytes: 0,
            mapped_bytes: 0,
            eviction_count: 0,
            evicted_bytes: 0,
        }
    }
}
