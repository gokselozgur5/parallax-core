use parallax_rs::storage::StorageTelemetry;

#[test]
fn test_storage_telemetry_defaults() {
    let t = StorageTelemetry::new();
    assert_eq!(t.resident_bytes, 0);
    assert_eq!(t.mapped_bytes, 0);
    assert_eq!(t.eviction_count, 0);
    assert_eq!(t.evicted_bytes, 0);
}
