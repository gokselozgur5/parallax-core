use parallax_rs::bpe::*;

#[test]
fn test_find_merge_binary_search() {
    let mut table = vec![
        MergeEntry::new(10, 20, 0, 100),
        MergeEntry::new(1, 2, 2, 102),
        MergeEntry::new(5, 5, 1, 101),
    ];
    table.sort_by_key(|e| e.key());
    assert!(is_table_sorted(&table));

    let found = find_merge(&table, 5, 5);
    assert_eq!(found, Some(&MergeEntry::new(5, 5, 1, 101)));
    assert_eq!(find_merge(&table, 99, 99), None);
}
