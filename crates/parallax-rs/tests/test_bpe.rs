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

#[test]
fn test_find_best_merge_and_encode() {
    let mut table = vec![
        MergeEntry::new(1, 2, 10, 20),
        MergeEntry::new(2, 3, 5, 30),
        MergeEntry::new(20, 3, 2, 40),
    ];
    table.sort_by_key(|e| e.key());

    let mut tokens = vec![1, 2, 3];
    bpe_encode(&table, &mut tokens);
    assert_eq!(tokens, vec![1, 30]);
}

#[test]
fn test_is_table_sorted() {
    let unsorted = vec![MergeEntry::new(5, 5, 0, 100), MergeEntry::new(1, 2, 1, 101)];
    assert!(!is_table_sorted(&unsorted));

    let duplicate = vec![MergeEntry::new(1, 2, 0, 100), MergeEntry::new(1, 2, 1, 101)];
    assert!(!is_table_sorted(&duplicate));
}
