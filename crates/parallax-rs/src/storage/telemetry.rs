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

impl StorageTelemetry {
    pub fn record_eviction(&mut self, bytes: u64) {
        self.eviction_count = self.eviction_count.saturating_add(1);
        self.evicted_bytes = self.evicted_bytes.saturating_add(bytes);
    }
}
