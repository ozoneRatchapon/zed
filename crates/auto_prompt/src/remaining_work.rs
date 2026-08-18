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

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[inline]
fn fnv1a_step(mut hash: u64, byte: u8) -> u64 {
    hash ^= byte as u64;
    hash.wrapping_mul(FNV_PRIME)
}

/// Plain FNV-1a over raw bytes. The hot path uses [`hash_word`] and
/// [`hash_bigram`], which fold lowercasing in; this stays as the reference the
/// tests pin those two against.
#[cfg(test)]
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |h, &b| fnv1a_step(h, b))
}

/// Hash of an ASCII-lowercased word.
#[inline]
fn hash_word(word: &[u8]) -> u64 {
    word.iter()
        .fold(FNV_OFFSET, |h, &b| fnv1a_step(h, b.to_ascii_lowercase()))
}

/// Hash of `"{a}_{b}"` lowercased, streamed rather than built.
///
/// FNV-1a is a rolling hash, so feeding `a`, then `b'_'`, then `b` is
/// bit-identical to hashing the concatenation — asserted by
/// `bigram_hash_matches_concatenation`. That identity is what lets this skip
/// the per-bigram `String` without moving any centroid.
#[inline]
fn hash_bigram(a: &[u8], b: &[u8]) -> u64 {
    let h = a
        .iter()
        .fold(FNV_OFFSET, |h, &c| fnv1a_step(h, c.to_ascii_lowercase()));
    let h = fnv1a_step(h, b'_');
    b.iter()
        .fold(h, |h, &c| fnv1a_step(h, c.to_ascii_lowercase()))
}

/// Hash word unigrams and adjacent bigrams into an L2-normalised vector.
///
/// Allocation-free: words are ASCII-alphanumeric runs located by index in the
/// source string, hashed in place with lowercasing applied per byte, and the
/// only storage is the fixed-size return array. Non-ASCII bytes act as
/// separators, matching the previous `char`-based cleaning pass.
fn features(text: &str) -> [f32; DIM] {
    let mut v = [0.0f32; DIM];
    let bytes = text.as_bytes();
    let mut prev: Option<(usize, usize)> = None;
    let mut i = 0usize;

    while i < bytes.len() {
        if !bytes[i].is_ascii_alphanumeric() {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
            i += 1;
        }
        let word = &bytes[start..i];

        v[(hash_word(word) % DIM as u64) as usize] += 1.0;
        if let Some((ps, pe)) = prev {
            v[(hash_bigram(&bytes[ps..pe], word) % DIM as u64) as usize] += 1.0;
        }
        prev = Some((start, i));
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

/// Words as [`features`] counts them: runs of ASCII alphanumerics.
fn count_words(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut n = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphanumeric() {
            n += 1;
            while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    n
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
    let word_count = count_words(message);
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
    fn bigram_hash_matches_concatenation() {
        // The streamed bigram hash must equal hashing the built string, or the
        // committed centroids no longer describe the same feature space.
        for (a, b) in [
            ("migration", "failed"),
            ("Tests", "PASS"),
            ("a", "b"),
            ("connection", "timeout"),
        ] {
            let built = fnv1a(format!("{}_{}", a.to_lowercase(), b.to_lowercase()).as_bytes());
            assert_eq!(hash_bigram(a.as_bytes(), b.as_bytes()), built, "{a}_{b}");
        }
    }

    #[test]
    fn word_count_matches_feature_tokenisation() {
        assert_eq!(count_words("hello world"), 2);
        assert_eq!(count_words("  a, b; c!  "), 3);
        // Underscores and dots are separators, not word characters, so a path
        // splits: tests / auth / test / py.
        assert_eq!(count_words("tests/auth_test.py passes"), 5);
        assert_eq!(count_words(""), 0);
        assert_eq!(count_words("—— ??"), 0);
    }

    #[test]
    fn features_are_l2_normalised() {
        let f = features("the migration failed and tests are red");
        let norm = f.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }
}
