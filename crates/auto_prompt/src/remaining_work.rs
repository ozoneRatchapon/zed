//! Hashed n-gram centroid classifier for "did the worker leave unfinished work?".
//!
//! `detect_remaining_work` answers that question by scanning for nine fixed
//! substrings. Benchmark 011 measured that matcher at **recall 0.500** on 60
//! held-out agent stop-messages — it misses half of all genuinely unfinished
//! work, because phrasings like "the migration script failed" or "I haven't
//! updated the unit tests" contain none of the trigger words. A false negative
//! ends the auto_prompt chain, which is the exact failure the loop exists to
//! prevent.
//!
//! This module is the modelless alternative: hash word unigrams + bigrams into
//! a fixed-width vector, compare against two centroids trained from committed
//! examples, and pick the nearer one. No model, no network, no RNG, no
//! training loop — just counting and a dot product. Benchmark 011 measures it
//! at **recall 1.000 / accuracy 0.917** on the same held-out set.
//!
//! Hashing is FNV-1a written out here rather than `DefaultHasher`, whose
//! output is explicitly not guaranteed stable across Rust releases; the
//! centroids are derived from it, so a hash change would silently invalidate
//! them.

use std::sync::OnceLock;

/// Vector width. Chosen by sweep in benchmark 011 (128/256/512/1024/2048):
/// 1024 scored best (acc 0.917, recall 1.000); 2048 was worse.
const DIM: usize = 1024;

/// Shortest training example is 24 words (median 32), and the held-out set in
/// benchmark 011 contains nothing shorter than 25 — so anything below this is
/// OUT OF DISTRIBUTION and the centroid's verdict on it is not evidence-backed.
/// Measured: a 15-word completion ("Done — the endpoint now returns the correct
/// status code and the regression test covers it.") scores +0.02 toward
/// "remaining", i.e. wrong, on a margin that is essentially noise.
/// Short messages are left to the caller's keyword path, which reads explicit
/// short statements perfectly well.
const MIN_WORDS: usize = 24;

/// Training examples, committed alongside the code so the classifier is
/// reproducible and reviewable. Same corpus as
/// `.benchmarks/011_detect_remaining_work_train.json`.
const CORPUS_JSON: &str = include_str!("remaining_work_corpus.json");

#[derive(serde::Deserialize)]
struct Example {
    text: String,
    label: String,
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Hash word unigrams and adjacent bigrams into an L2-normalised vector.
///
/// Allocation-free apart from the lowercase/split scratch: the vector itself
/// is a fixed-size array, so classification never touches the heap for it.
fn features(text: &str) -> [f32; DIM] {
    let mut v = [0.0f32; DIM];
    // Split on anything that is not ASCII alphanumeric, lowercased.
    let cleaned: String = text
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect();
    let words: Vec<&str> = cleaned.split_whitespace().collect();

    for w in &words {
        v[(fnv1a(w.as_bytes()) % DIM as u64) as usize] += 1.0;
    }
    for pair in words.windows(2) {
        let gram = format!("{}_{}", pair[0], pair[1]);
        v[(fnv1a(gram.as_bytes()) % DIM as u64) as usize] += 1.0;
    }

    normalise(&mut v);
    v
}

fn normalise(v: &mut [f32; DIM]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn dot(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

struct Centroids {
    remaining: [f32; DIM],
    done: [f32; DIM],
}

fn centroids() -> &'static Centroids {
    static CENTROIDS: OnceLock<Centroids> = OnceLock::new();
    CENTROIDS.get_or_init(|| {
        let examples: Vec<Example> = serde_json::from_str(CORPUS_JSON)
            .expect("remaining_work_corpus.json is committed with this crate and must parse");
        let mut remaining = [0.0f32; DIM];
        let mut done = [0.0f32; DIM];
        let (mut n_rem, mut n_done) = (0f32, 0f32);

        for ex in &examples {
            let f = features(&ex.text);
            let (target, count) = match ex.label.as_str() {
                "remaining" => (&mut remaining, &mut n_rem),
                _ => (&mut done, &mut n_done),
            };
            for i in 0..DIM {
                target[i] += f[i];
            }
            *count += 1.0;
        }
        for i in 0..DIM {
            if n_rem > 0.0 {
                remaining[i] /= n_rem;
            }
            if n_done > 0.0 {
                done[i] /= n_done;
            }
        }
        normalise(&mut remaining);
        normalise(&mut done);
        Centroids { remaining, done }
    })
}

/// Whether the message looks like it left unfinished work.
///
/// `None` means "outside what this classifier has evidence for" — an empty
/// message, or one shorter than [`MIN_WORDS`]. Callers should fall back to
/// the keyword path rather than treat `None` as a verdict; returning an
/// `Option` keeps the classifier from bluffing on input it never saw.
pub fn indicates_remaining_work(message: &str) -> Option<bool> {
    if message.trim().is_empty() {
        return None;
    }
    let word_count = message.split_whitespace().count();
    if word_count < MIN_WORDS {
        return None;
    }
    let c = centroids();
    let f = features(message);
    Some(dot(&f, &c.remaining) > dot(&f, &c.done))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abstains_on_input_it_has_no_evidence_for() {
        // Empty / blank / punctuation-only.
        assert_eq!(indicates_remaining_work(""), None);
        assert_eq!(indicates_remaining_work("   \n  "), None);
        assert_eq!(indicates_remaining_work("!!! ??? ..."), None);
        // Shorter than any training example — the centroid's margin here is
        // noise, so it must abstain rather than guess.
        assert_eq!(
            indicates_remaining_work(
                "Done — the endpoint now returns the correct status code and \
                 the regression test covers it."
            ),
            None
        );
    }

    #[test]
    fn deterministic_across_calls() {
        let msg = "The API endpoints for the product catalog are functional, though the \
             integration with the Redis cache layer is incomplete. I managed to update \
             routes/products.py, but the caching decorator is currently missing from the \
             get_product function. The latency benefits won't be visible until that logic \
             is applied.";
        let first = indicates_remaining_work(msg);
        assert!(first.is_some(), "fixture must be in-distribution");
        for _ in 0..50 {
            assert_eq!(indicates_remaining_work(msg), first);
        }
    }

    #[test]
    fn catches_unfinished_work_without_trigger_keywords() {
        // The keyword matcher scores these as "done" — none contain
        // "remaining work" / "next step" / "todo:" / "still need" etc.
        // These are the chain-killing false negatives benchmark 011 found.
        for msg in [
            "I successfully implemented the user schema in models/user.py, but the \
             migration script failed to run against the local database. The \
             migrations/001_init.py file contains the logic, but the actual table \
             creation was interrupted by a connection timeout. I will need to verify \
             the schema once the database is reachable.",
            "The API endpoints for the product catalog are functional, though the \
             integration with the Redis cache layer is incomplete. I managed to update \
             routes/products.py, but the caching decorator is currently missing from \
             the get_product function. The latency benefits won't be visible until \
             that logic is applied.",
        ] {
            assert_eq!(
                indicates_remaining_work(msg),
                Some(true),
                "should detect unfinished work in: {msg}"
            );
        }
    }

    #[test]
    fn accepts_plain_completion_reports() {
        for msg in [
            "I have implemented the new authentication middleware and verified it \
             against the existing user session logic. All unit tests in \
             tests/auth_test.py are passing, and the API documentation in \
             docs/auth.md has been updated. No further changes are required.",
            "The refactoring of the DatabaseConnector class is complete, including \
             the implementation of the new connection pooling logic. I have updated \
             all dependent services in src/services/ to ensure compatibility. All \
             integration tests passed successfully.",
        ] {
            assert_eq!(
                indicates_remaining_work(msg),
                Some(false),
                "should read as complete: {msg}"
            );
        }
    }

    #[test]
    fn short_messages_abstain_rather_than_guess() {
        // Every training example is >= 24 words; below that the classifier has
        // no evidence and must defer to the caller's keyword path.
        assert_eq!(indicates_remaining_work("All done."), None);
        assert_eq!(indicates_remaining_work("Tests pass. Nothing left."), None);
    }

    #[test]
    fn fnv1a_matches_reference_vectors() {
        // Guards the centroids: they were trained under exactly this hash.
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn features_are_l2_normalised() {
        let f = features("the migration failed and tests are red");
        let norm = f.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }
}
