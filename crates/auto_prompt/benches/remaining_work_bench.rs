//! G2 gate for the modelless remaining-work classifier (benchmark 011).
//!
//! Measures the per-message cost that would be paid on every agent stop, so it
//! can be compared against the keyword scan it would join and the local-tier
//! LLM call it would avoid (200-500 ms, see `local_mlx`).

use auto_prompt::remaining_work::indicates_remaining_work;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

/// Representative of what benchmark 011 measured: ~46 words, unfinished work
/// stated without any trigger keyword.
const TYPICAL: &str = "The API endpoints for the product catalog are functional, though the \
    integration with the Redis cache layer is incomplete. I managed to update routes/products.py, \
    but the caching decorator is currently missing from the get_product function. The latency \
    benefits won't be visible until that logic is applied.";

/// A long stop-message, to show how cost scales with input length.
const LONG: &str = "I have finished the migration of the authentication subsystem across every \
    service in src/services/, including the token refresh path, the expiry handling, and the \
    error branches that previously went unhandled. Each of those now has a dedicated regression \
    test in tests/auth.rs and the whole suite passes. I also updated docs/auth.md to describe \
    the new refresh semantics, regenerated the OpenAPI schema, and verified the staging \
    deployment still boots cleanly against the updated middleware chain. No further work is \
    outstanding on this task.";

fn bench(c: &mut Criterion) {
    // First call trains the centroids; keep that out of the measured loop.
    let _ = indicates_remaining_work(TYPICAL);

    c.bench_function("remaining_work/typical_46_words", |b| {
        b.iter(|| indicates_remaining_work(black_box(TYPICAL)))
    });
    c.bench_function("remaining_work/long_92_words", |b| {
        b.iter(|| indicates_remaining_work(black_box(LONG)))
    });
    c.bench_function("remaining_work/short_abstains", |b| {
        b.iter(|| indicates_remaining_work(black_box("All done.")))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
