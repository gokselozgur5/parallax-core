use parallax_rs::storage::StorageTelemetry;

#[test]
fn test_storage_telemetry_defaults() {
    let t = StorageTelemetry::new();
    assert_eq!(t.resident_bytes, 0);
    assert_eq!(t.mapped_bytes, 0);
    assert_eq!(t.eviction_count, 0);
    assert_eq!(t.evicted_bytes, 0);
}

#[test]
fn test_storage_telemetry_eviction_and_reset() {
    let mut t = StorageTelemetry::new();
    t.update_memory_usage(1024, 4096);
    assert_eq!(t.resident_bytes, 1024);
    assert_eq!(t.mapped_bytes, 4096);

    t.record_eviction(512);
    assert_eq!(t.eviction_count, 1);
    assert_eq!(t.evicted_bytes, 512);

    t.reset();
    assert_eq!(t.resident_bytes, 0);
    assert_eq!(t.eviction_count, 0);
}
