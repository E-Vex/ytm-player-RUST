//! Native cores for `src/ytm_player/services/queue.py` — the `QueueCore`
//! bridge class (shuffle/repeat state machine).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PySet, PyTuple};

// ---------------------------------------------------------------------------
// Playback queue engine (mirrors ytm_player/services/queue.py)
// ---------------------------------------------------------------------------

/// Repeat-mode codes shared with the Python `QueueManager` facade:
/// 0 = off, 1 = all, 2 = one — the `RepeatMode` member order.
const REPEAT_OFF: u8 = 0;

const REPEAT_ALL: u8 = 1;

const REPEAT_ONE: u8 = 2;

/// Native playback-order engine backing `QueueManager`.
///
/// Owns the queue's ordering state — the track list, one entry id per
/// occurrence, the shuffle order, the current position, the repeat/shuffle
/// flags and the removed-current bookkeeping — and mirrors the pure-Python
/// `_QueueCore` fallback operation for operation, including the exact order
/// in which random draws happen (the parity tests rely on both).
///
/// Randomness always goes through the Python `random` module (the default
/// `rng`) rather than an in-crate generator, so `random.seed(...)` and
/// monkeypatched `random.shuffle` / `random.randint` / `random.randrange`
/// behave exactly like the fallback.
///
/// `QueueManager` holds its `threading.Lock` around every call; nothing here
/// blocks. Tracks are stored as owned references to the caller's dicts and
/// handed back unchanged — the core never inspects or copies them (except
/// reading `video_id` for radio dedup, mirroring the fallback).
#[pyclass]
pub(crate) struct QueueCore {
    /// Track objects in insertion order.
    tracks: Vec<Py<PyAny>>,
    /// One internal id per queue occurrence, parallel to `tracks`.
    entry_ids: Vec<i64>,
    next_entry_id: i64,
    generation: i64,
    /// A removed audible occurrence can finish outside the queue; keep its
    /// successor by occurrence, never pretend that successor is playing.
    removed_current: bool,
    removed_next_id: Option<i64>,
    current_index: isize,
    repeat: u8,
    shuffle: bool,
    /// When shuffling, maps playback positions to indices in `tracks`.
    shuffle_order: Vec<usize>,
    shuffle_position: isize,
    /// Random source: the Python `random` module. Held as the module object
    /// so attribute patches (`random.shuffle = ...`) apply here too.
    rng: Py<PyAny>,
}

impl QueueCore {
    /// Current index into `tracks`, resolving the shuffle indirection.
    /// (The pre-port `_real_index` helper — does NOT consult
    /// `removed_current`; callers that must guard it.)
    fn current_real_index(&self) -> isize {
        if self.shuffle
            && self.shuffle_position >= 0
            && (self.shuffle_position as usize) < self.shuffle_order.len()
        {
            self.shuffle_order[self.shuffle_position as usize] as isize
        } else {
            self.current_index
        }
    }

    /// Mint `count` fresh entry ids.
    fn new_entry_ids(&mut self, count: usize) -> Vec<i64> {
        let start = self.next_entry_id + 1;
        self.next_entry_id += count as i64;
        (start..start + count as i64).collect()
    }

    /// Visible position of the removed occurrence's successor, or the end.
    ///
    /// Mirrors `_removed_next_position`, including its unguarded
    /// `list.index` semantics: a shuffle order that no longer contains the
    /// successor's real index raises `ValueError` exactly like Python.
    fn removed_next_position(&self) -> PyResult<usize> {
        if let Some(id) = self.removed_next_id {
            if let Some(real) = self.entry_ids.iter().position(|&e| e == id) {
                if self.shuffle {
                    return match self.shuffle_order.iter().position(|&r| r == real) {
                        Some(pos) => Ok(pos),
                        None => Err(PyValueError::new_err(format!("{real} is not in list"))),
                    };
                }
                return Ok(real);
            }
        }
        Ok(self.tracks.len())
    }

    /// Rebuild the shuffle order, optionally keeping the current track first.
    fn rebuild_shuffle(&mut self, py: Python, keep_current: bool) -> PyResult<()> {
        let mut indices: Vec<usize> = (0..self.tracks.len()).collect();
        let current_real = self.current_real_index();

        if keep_current
            && !self.removed_current
            && current_real >= 0
            && (current_real as usize) < self.tracks.len()
        {
            let current_real = current_real as usize;
            if let Some(pos) = indices.iter().position(|&i| i == current_real) {
                indices.remove(pos);
            }
            self.rng_shuffle(py, &mut indices)?;
            self.shuffle_order = Vec::with_capacity(self.tracks.len());
            self.shuffle_order.push(current_real);
            self.shuffle_order.extend_from_slice(&indices);
            self.shuffle_position = 0;
        } else {
            self.rng_shuffle(py, &mut indices)?;
            self.shuffle_order = indices;
            self.shuffle_position = -1;
        }
        Ok(())
    }

    /// `random.shuffle` over a Rust vec, routed through the Python rng so
    /// seeded sequences and monkeypatches match the fallback exactly.
    fn rng_shuffle(&self, py: Python, values: &mut Vec<usize>) -> PyResult<()> {
        let list = PyList::new(py, values.iter().copied())?;
        self.rng.bind(py).call_method1("shuffle", (list.clone(),))?;
        values.clear();
        for item in list.iter() {
            values.push(item.extract::<usize>()?);
        }
        Ok(())
    }

    /// `random.randint(a, b)` — both ends inclusive, like Python.
    fn rng_randint(&self, py: Python, a: i64, b: i64) -> PyResult<i64> {
        self.rng.bind(py).call_method1("randint", (a, b))?.extract()
    }

    /// `random.randrange(n)` — the one-argument form.
    fn rng_randrange(&self, py: Python, n: i64) -> PyResult<i64> {
        self.rng.bind(py).call_method1("randrange", (n,))?.extract()
    }

    /// `list.insert` semantics for a `Vec`: out-of-range positions append.
    fn vec_insert<T>(v: &mut Vec<T>, pos: usize, item: T) {
        let pos = pos.min(v.len());
        v.insert(pos, item);
    }

    /// The track at real index `real`, or `None` when out of range.
    fn track_at(&self, py: Python, real: isize) -> Option<Py<PyAny>> {
        if real >= 0 && (real as usize) < self.tracks.len() {
            Some(self.tracks[real as usize].clone_ref(py))
        } else {
            None
        }
    }

    /// Add a track without acquiring anything (mirrors `_add_unlocked`).
    fn add_unlocked(
        &mut self,
        py: Python,
        track: Py<PyAny>,
        position: Option<isize>,
    ) -> PyResult<()> {
        let entry_id = self.new_entry_ids(1)[0];
        if self.removed_current && self.removed_next_id.is_none() {
            self.removed_next_id = Some(entry_id);
        }
        let new_idx: usize;
        let append = match position {
            None => true,
            Some(pos) => pos >= self.tracks.len() as isize,
        };
        if append {
            self.tracks.push(track);
            self.entry_ids.push(entry_id);
            new_idx = self.tracks.len() - 1;
        } else {
            let pos = position.unwrap().max(0) as usize;
            self.tracks.insert(pos, track);
            self.entry_ids.insert(pos, entry_id);
            new_idx = pos;
            // Adjust the current index if we inserted before it.
            if !self.shuffle && pos as isize <= self.current_index {
                self.current_index += 1;
            }
        }

        if self.shuffle {
            // Insert the new track at a random future position in shuffle order.
            let insert_at: isize = if self.shuffle_position >= 0 {
                self.shuffle_position + 1
            } else {
                0
            };
            // Shift existing shuffle indices that are >= new_idx.
            for i in self.shuffle_order.iter_mut() {
                if *i >= new_idx {
                    *i += 1;
                }
            }
            let len_order = self.shuffle_order.len();
            let future_pos: usize = if (insert_at as usize) < len_order {
                self.rng_randint(py, insert_at as i64 + 1, len_order as i64)? as usize
            } else {
                len_order
            };
            Self::vec_insert(&mut self.shuffle_order, future_pos, new_idx);
        }
        Ok(())
    }

    /// Insert at the removed audible row's boundary (mirrors
    /// `_add_next_removed`). Callers guarantee at least one track.
    fn add_next_removed(&mut self, tracks: Vec<Py<PyAny>>) -> PyResult<()> {
        let position = self.removed_next_position()?;
        let ids = self.new_entry_ids(tracks.len());
        let count = tracks.len();
        if self.shuffle {
            let start = self.tracks.len();
            self.tracks.extend(tracks);
            self.entry_ids.extend(ids.iter().copied());
            for (offset, new_idx) in (start..start + count).enumerate() {
                Self::vec_insert(&mut self.shuffle_order, position + offset, new_idx);
            }
        } else {
            for (offset, t) in tracks.into_iter().enumerate() {
                Self::vec_insert(&mut self.tracks, position + offset, t);
            }
            for (offset, id) in ids.iter().copied().enumerate() {
                Self::vec_insert(&mut self.entry_ids, position + offset, id);
            }
        }
        self.removed_next_id = Some(ids[0]);
        Ok(())
    }

    /// Append tracks + update the shuffle order (mirrors
    /// `_add_multiple_unlocked`).
    fn add_multiple_unlocked(&mut self, py: Python, tracks: Vec<Py<PyAny>>) -> PyResult<()> {
        let was_empty = self.tracks.is_empty();
        let start_idx = self.tracks.len();
        let count = tracks.len();
        self.tracks.extend(tracks);
        let ids = self.new_entry_ids(count);
        self.entry_ids.extend(ids);
        if count > 0 && self.removed_current && self.removed_next_id.is_none() {
            self.removed_next_id = Some(self.entry_ids[start_idx]);
        }
        let mut new_indices: Vec<usize> = (start_idx..start_idx + count).collect();

        if self.shuffle {
            if was_empty {
                // Fresh queue after clear() — full shuffle build instead of
                // piecemeal insertion into an empty list.
                self.rebuild_shuffle(py, false)?;
            } else {
                // Insert new indices at random future positions.
                let insert_after: isize = if self.shuffle_position >= 0 {
                    self.shuffle_position + 1
                } else {
                    0
                };
                self.rng_shuffle(py, &mut new_indices)?;
                for new_idx in new_indices {
                    // len() is re-read per iteration, like the fallback.
                    let pos =
                        self.rng_randint(py, insert_after as i64, self.shuffle_order.len() as i64)?
                            as usize;
                    Self::vec_insert(&mut self.shuffle_order, pos, new_idx);
                }
            }
        }
        Ok(())
    }

    /// Remove the entry at visible/playback `index` (mirrors
    /// `_remove_unlocked`).
    fn remove_unlocked(&mut self, index: isize) {
        if !(index >= 0 && index < self.tracks.len() as isize) {
            return;
        }
        let index = index as usize;
        let order_len = if self.shuffle {
            self.shuffle_order.len()
        } else {
            self.tracks.len()
        };
        // `order[index]` — the shuffle order when shuffling, identity
        // otherwise. In bounds by the outer check because the shuffle order
        // is a permutation of the track indices.
        let real_idx = if self.shuffle {
            self.shuffle_order[index]
        } else {
            index
        };
        let current_pos = if self.shuffle {
            self.shuffle_position
        } else {
            self.current_index
        };
        let is_current = (!self.removed_current && index as isize == current_pos)
            || (self.removed_current && self.removed_next_id == Some(self.entry_ids[real_idx]));
        if is_current {
            self.removed_current = true;
            self.removed_next_id = if index + 1 < order_len {
                let next_real = if self.shuffle {
                    self.shuffle_order[index + 1]
                } else {
                    index + 1
                };
                Some(self.entry_ids[next_real])
            } else {
                None
            };
        }

        if self.shuffle {
            if index >= self.shuffle_order.len() {
                return;
            }
            let real_idx = self.shuffle_order[index];
            self.shuffle_order.remove(index);
            // Shift indices that pointed beyond the removed track.
            for i in self.shuffle_order.iter_mut() {
                if *i > real_idx {
                    *i -= 1;
                }
            }
            self.tracks.remove(real_idx);
            self.entry_ids.remove(real_idx);
            if (index as isize) < self.shuffle_position {
                self.shuffle_position -= 1;
            } else if index as isize == self.shuffle_position {
                // Current track removed; clamp position (mirrors the
                // non-shuffle branch: current() lands on the next track).
                if self.shuffle_position >= self.shuffle_order.len() as isize {
                    self.shuffle_position = self.shuffle_order.len() as isize - 1;
                }
            }
        } else {
            self.tracks.remove(index);
            self.entry_ids.remove(index);
            if (index as isize) < self.current_index {
                self.current_index -= 1;
            } else if index as isize == self.current_index {
                // Current track removed; clamp index.
                if self.current_index >= self.tracks.len() as isize {
                    self.current_index = self.tracks.len() as isize - 1;
                }
            }
        }
    }
}

#[pymethods]
impl QueueCore {
    #[new]
    fn new(py: Python) -> PyResult<Self> {
        Ok(Self {
            tracks: Vec::new(),
            entry_ids: Vec::new(),
            next_entry_id: 0,
            generation: 0,
            removed_current: false,
            removed_next_id: None,
            current_index: -1,
            repeat: REPEAT_OFF,
            shuffle: false,
            shuffle_order: Vec::new(),
            shuffle_position: -1,
            rng: py.import("random")?.into_any().unbind(),
        })
    }

    // -- Queue manipulation ---------------------------------------------

    /// Add a track to the queue. `None` position = append to end.
    #[pyo3(signature = (track, position=None))]
    fn add(&mut self, py: Python, track: Py<PyAny>, position: Option<isize>) -> PyResult<()> {
        self.add_unlocked(py, track, position)
    }

    /// Insert a track immediately after the currently playing track.
    fn add_next(&mut self, py: Python, track: Py<PyAny>) -> PyResult<()> {
        if self.removed_current {
            self.add_next_removed(vec![track])?;
            return Ok(());
        }
        if self.shuffle {
            // Insert into tracks and put it next in shuffle order.
            self.tracks.push(track);
            let ids = self.new_entry_ids(1);
            self.entry_ids.extend(ids);
            let new_idx = self.tracks.len() - 1;
            let insert_pos = (self.shuffle_position + 1).max(0) as usize;
            Self::vec_insert(&mut self.shuffle_order, insert_pos, new_idx);
        } else {
            let insert_pos = if self.current_index >= 0 {
                self.current_index + 1
            } else {
                0
            };
            self.add_unlocked(py, track, Some(insert_pos))?;
        }
        Ok(())
    }

    /// Insert *tracks* immediately after the current track, preserving order.
    fn add_next_multiple(&mut self, tracks: Vec<Py<PyAny>>) -> PyResult<()> {
        if tracks.is_empty() {
            return Ok(());
        }
        if self.removed_current {
            self.add_next_removed(tracks)?;
            return Ok(());
        }
        let entry_ids = self.new_entry_ids(tracks.len());
        if self.shuffle {
            let start_idx = self.tracks.len();
            let count = tracks.len();
            self.tracks.extend(tracks);
            self.entry_ids.extend(entry_ids);
            let insert_pos = (self.shuffle_position + 1).max(0) as usize;
            for (offset, new_idx) in (start_idx..start_idx + count).enumerate() {
                Self::vec_insert(&mut self.shuffle_order, insert_pos + offset, new_idx);
            }
        } else {
            let insert_pos = if self.current_index >= 0 {
                self.current_index + 1
            } else {
                0
            };
            let insert_pos = (insert_pos.clamp(0, self.tracks.len() as isize)) as usize;
            for (offset, t) in tracks.into_iter().enumerate() {
                Self::vec_insert(&mut self.tracks, insert_pos + offset, t);
            }
            for (offset, id) in entry_ids.into_iter().enumerate() {
                Self::vec_insert(&mut self.entry_ids, insert_pos + offset, id);
            }
        }
        Ok(())
    }

    /// Append multiple tracks to the end of the queue.
    fn add_multiple(&mut self, py: Python, tracks: Vec<Py<PyAny>>) -> PyResult<()> {
        if tracks.is_empty() {
            return Ok(());
        }
        self.add_multiple_unlocked(py, tracks)
    }

    /// Append the tracks whose video_id is not already queued (radio dedup).
    /// Returns the filtered list; the caller decides what to do with it.
    fn filter_not_in_queue(
        &self,
        py: Python,
        candidates: Vec<Py<PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let existing: Vec<Py<PyAny>> = self
            .tracks
            .iter()
            .map(|t| {
                t.bind(py)
                    .call_method1("get", ("video_id",))
                    .map(|v| v.unbind())
            })
            .collect::<PyResult<Vec<Py<PyAny>>>>()?;
        let set = PySet::new(py, existing)?;
        let mut out: Vec<Py<PyAny>> = Vec::new();
        for t in candidates {
            let vid = t.bind(py).call_method1("get", ("video_id",))?;
            if !set.contains(vid)? {
                out.push(t);
            }
        }
        Ok(out)
    }

    /// The fused radio append — `QueueManager.set_radio_tracks`'s
    /// filter-then-add orchestration folded into one core operation:
    /// the `filter_not_in_queue` dedup (the video_id set of the queued
    /// tracks, the candidates read through their own `get`) and, when
    /// anything survives it, the `add_multiple` append — under the one
    /// lock section the facade already holds, with no Python list
    /// crossing the boundary in between. Returns how many tracks were
    /// appended; 0 leaves the queue untouched (no entry ids minted, no
    /// shuffle rebuild, no removed-current bookkeeping), matching the
    /// reference's `if new_tracks:` gate. Duplicates WITHIN *candidates*
    /// pass exactly like the reference's two-step compose — the dedup
    /// only consults the tracks already queued when the filter ran.
    fn add_radio_tracks(&mut self, py: Python, candidates: Vec<Py<PyAny>>) -> PyResult<usize> {
        let existing: Vec<Py<PyAny>> = self
            .tracks
            .iter()
            .map(|t| {
                t.bind(py)
                    .call_method1("get", ("video_id",))
                    .map(|v| v.unbind())
            })
            .collect::<PyResult<Vec<Py<PyAny>>>>()?;
        let set = PySet::new(py, existing)?;
        let mut fresh: Vec<Py<PyAny>> = Vec::new();
        for t in candidates {
            let vid = t.bind(py).call_method1("get", ("video_id",))?;
            if !set.contains(vid)? {
                fresh.push(t);
            }
        }
        let added = fresh.len();
        if added > 0 {
            self.add_multiple_unlocked(py, fresh)?;
        }
        Ok(added)
    }

    /// Remove the track at the given index (in visible/playback order).
    fn remove(&mut self, index: isize) {
        self.remove_unlocked(index);
    }

    /// Remove the one occurrence *entry_id* names. False when no entry
    /// carries that id any more — never falls back to another occurrence.
    fn remove_entry(&mut self, entry_id: i64) -> bool {
        let real_idx = match self.entry_ids.iter().position(|&e| e == entry_id) {
            Some(r) => r,
            None => return false,
        };
        let index = if self.shuffle {
            match self.shuffle_order.iter().position(|&r| r == real_idx) {
                Some(p) => p as isize,
                None => return false,
            }
        } else {
            real_idx as isize
        };
        self.remove_unlocked(index);
        true
    }

    /// Remove all tracks from the queue (bumps the generation).
    fn clear(&mut self) {
        self.generation += 1;
        self.removed_current = false;
        self.removed_next_id = None;
        self.tracks.clear();
        self.entry_ids.clear();
        self.current_index = -1;
        self.shuffle_order.clear();
        self.shuffle_position = -1;
    }

    /// Move a track from one position to another in the visible order.
    #[pyo3(name = "move")]
    fn move_(&mut self, from_idx: isize, to_idx: isize) {
        if from_idx == to_idx {
            return;
        }
        if !(from_idx >= 0 && from_idx < self.tracks.len() as isize) {
            return;
        }
        let to_idx = to_idx.max(0).min(self.tracks.len() as isize - 1);

        if self.shuffle {
            // Move within shuffle order.
            if from_idx >= self.shuffle_order.len() as isize
                || to_idx >= self.shuffle_order.len() as isize
            {
                return;
            }
            let item = self.shuffle_order.remove(from_idx as usize);
            Self::vec_insert(&mut self.shuffle_order, to_idx as usize, item);
            if from_idx == self.shuffle_position {
                self.shuffle_position = to_idx;
            } else if from_idx < self.shuffle_position && to_idx >= self.shuffle_position {
                self.shuffle_position -= 1;
            } else if to_idx <= self.shuffle_position && self.shuffle_position < from_idx {
                self.shuffle_position += 1;
            }
        } else {
            let track = self.tracks.remove(from_idx as usize);
            self.tracks.insert(to_idx as usize, track);
            let id = self.entry_ids.remove(from_idx as usize);
            self.entry_ids.insert(to_idx as usize, id);
            if from_idx == self.current_index {
                self.current_index = to_idx;
            } else if from_idx < self.current_index && to_idx >= self.current_index {
                self.current_index -= 1;
            } else if to_idx <= self.current_index && self.current_index < from_idx {
                self.current_index += 1;
            }
        }
    }

    // -- Playback navigation ---------------------------------------------

    /// Return the currently selected track, or None.
    fn current(&self, py: Python) -> Option<Py<PyAny>> {
        if self.removed_current {
            return None;
        }
        self.track_at(py, self.current_real_index())
    }

    /// Advance to the next track and return it (repeat/shuffle aware).
    fn next_track(&mut self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if self.tracks.is_empty() {
            return Ok(None);
        }

        if self.removed_current {
            let mut pos = self.removed_next_position()? as isize;
            if pos == self.tracks.len() as isize {
                if self.repeat != REPEAT_ALL {
                    return Ok(None);
                }
                pos = 0;
            }
            self.removed_current = false;
            self.removed_next_id = None;
            if self.shuffle {
                self.shuffle_position = pos;
            } else {
                self.current_index = pos;
            }
            return Ok(self.track_at(py, self.current_real_index()));
        }

        if self.repeat == REPEAT_ONE {
            return Ok(self.track_at(py, self.current_real_index()));
        }

        if self.shuffle {
            let next_pos = self.shuffle_position + 1;
            if next_pos >= self.shuffle_order.len() as isize {
                if self.repeat == REPEAT_ALL {
                    self.rebuild_shuffle(py, false)?;
                    self.shuffle_position = 0;
                } else {
                    return Ok(None);
                }
            } else {
                self.shuffle_position = next_pos;
            }
        } else {
            let next_idx = self.current_index + 1;
            if next_idx >= self.tracks.len() as isize {
                if self.repeat == REPEAT_ALL {
                    self.current_index = 0;
                } else {
                    return Ok(None);
                }
            } else {
                self.current_index = next_idx;
            }
        }

        Ok(self.track_at(py, self.current_real_index()))
    }

    /// Go back to the previous track and return it (repeat/shuffle aware).
    fn previous_track(&mut self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if self.tracks.is_empty() {
            return Ok(None);
        }

        if self.removed_current {
            let mut pos = self.removed_next_position()? as isize - 1;
            if pos < 0 {
                if self.repeat != REPEAT_ALL {
                    return Ok(None);
                }
                pos = self.tracks.len() as isize - 1;
            }
            self.removed_current = false;
            self.removed_next_id = None;
            if self.shuffle {
                self.shuffle_position = pos;
            } else {
                self.current_index = pos;
            }
            return Ok(self.track_at(py, self.current_real_index()));
        }

        if self.repeat == REPEAT_ONE {
            return Ok(self.track_at(py, self.current_real_index()));
        }

        if self.shuffle {
            let prev_pos = self.shuffle_position - 1;
            if prev_pos < 0 {
                if self.repeat == REPEAT_ALL {
                    self.shuffle_position = self.shuffle_order.len() as isize - 1;
                } else {
                    return Ok(None);
                }
            } else {
                self.shuffle_position = prev_pos;
            }
        } else {
            let prev_idx = self.current_index - 1;
            if prev_idx < 0 {
                if self.repeat == REPEAT_ALL {
                    self.current_index = self.tracks.len() as isize - 1;
                } else {
                    return Ok(None);
                }
            } else {
                self.current_index = prev_idx;
            }
        }

        Ok(self.track_at(py, self.current_real_index()))
    }

    /// Return the next track WITHOUT advancing the position (prefetch).
    fn peek_next(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if self.tracks.is_empty() {
            return Ok(None);
        }

        if self.removed_current {
            let mut pos = self.removed_next_position()? as isize;
            if pos == self.tracks.len() as isize {
                if self.repeat != REPEAT_ALL {
                    return Ok(None);
                }
                pos = 0;
            }
            let real = if self.shuffle {
                self.shuffle_order[pos as usize] as isize
            } else {
                pos
            };
            return Ok(self.track_at(py, real));
        }

        if self.repeat == REPEAT_ONE {
            return Ok(self.track_at(py, self.current_real_index()));
        }

        if self.shuffle {
            let next_pos = self.shuffle_position + 1;
            if next_pos >= self.shuffle_order.len() as isize {
                // End of the shuffle order. Even under RepeatMode.ALL we
                // can't predict the next reshuffle — nothing to prefetch.
                return Ok(None);
            }
            let real_idx = self.shuffle_order[next_pos as usize] as isize;
            return Ok(self.track_at(py, real_idx));
        }

        let next_idx = self.current_index + 1;
        if next_idx >= self.tracks.len() as isize {
            if self.repeat == REPEAT_ALL {
                return Ok(self.track_at(py, 0));
            }
            return Ok(None);
        }
        Ok(self.track_at(py, next_idx))
    }

    // -- Repeat / Shuffle -------------------------------------------------

    /// Set the repeat mode from a `RepeatMode` code (0 off, 1 all, 2 one).
    fn set_repeat(&mut self, code: u8) {
        self.repeat = code;
    }

    /// Cycle through repeat modes: OFF -> ALL -> ONE -> OFF. Returns the
    /// new code.
    fn cycle_repeat(&mut self) -> u8 {
        self.repeat = match self.repeat {
            REPEAT_OFF => REPEAT_ALL,
            REPEAT_ALL => REPEAT_ONE,
            _ => REPEAT_OFF,
        };
        self.repeat
    }

    /// Toggle shuffle mode on or off.
    fn toggle_shuffle(&mut self, py: Python) -> PyResult<()> {
        // Resolve the playing occurrence under the mode being left. Once
        // the flag flips, current_real_index() reads the other mode's
        // pointer, and current_index is stale after next/previous under
        // shuffle.
        let real = self.current_real_index();
        self.shuffle = !self.shuffle;
        if self.shuffle {
            self.rebuild_shuffle(py, true)?;
        } else {
            // Exiting shuffle: the real index becomes the current position.
            self.current_index = if real >= 0 && (real as usize) < self.tracks.len() {
                real
            } else {
                -1
            };
            self.shuffle_order.clear();
            self.shuffle_position = -1;
        }
        Ok(())
    }

    // -- Random / Radio ---------------------------------------------------

    /// Pick and jump to a random track from the queue.
    fn play_random(&mut self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if self.tracks.is_empty() {
            return Ok(None);
        }
        self.removed_current = false;
        self.removed_next_id = None;
        let idx = self.rng_randrange(py, self.tracks.len() as i64)? as usize;
        if self.shuffle {
            // Find or set position in shuffle order.
            self.shuffle_position = match self.shuffle_order.iter().position(|&r| r == idx) {
                Some(pos) => pos as isize,
                None => 0,
            };
        } else {
            self.current_index = idx as isize;
        }
        Ok(self.track_at(py, self.current_real_index()))
    }

    // -- Jumps -------------------------------------------------------------

    /// Resolve and select an occurrence atomically, including under shuffle.
    fn jump_to_entry(&mut self, py: Python, entry_id: i64) -> PyResult<Option<Py<PyAny>>> {
        let real = match self.entry_ids.iter().position(|&e| e == entry_id) {
            Some(r) => r,
            None => return Ok(None),
        };
        self.removed_current = false;
        self.removed_next_id = None;
        self.current_index = real as isize;
        if self.shuffle {
            // Mirrors the fallback's unguarded `list.index`: a real index
            // missing from the shuffle order raises ValueError.
            let pos = match self.shuffle_order.iter().position(|&r| r == real) {
                Some(p) => p,
                None => return Err(PyValueError::new_err(format!("{real} is not in list"))),
            };
            self.shuffle_position = pos as isize;
        }
        Ok(self.track_at(py, real as isize))
    }

    /// Jump to a specific index in the visible order and return the track.
    fn jump_to(&mut self, py: Python, index: isize) -> Option<Py<PyAny>> {
        if !(index >= 0 && index < self.tracks.len() as isize) {
            return None;
        }
        self.removed_current = false;
        self.removed_next_id = None;
        if self.shuffle {
            if !(index >= 0 && index < self.shuffle_order.len() as isize) {
                return None;
            }
            self.shuffle_position = index;
            // Keep current_index in sync as a safety fallback.
            self.current_index = self.shuffle_order[index as usize] as isize;
        } else {
            self.current_index = index;
        }
        self.track_at(py, self.current_real_index())
    }

    /// Jump to a track by its index in the internal track list.
    fn jump_to_real(&mut self, py: Python, real_index: isize) -> Option<Py<PyAny>> {
        if !(real_index >= 0 && real_index < self.tracks.len() as isize) {
            return None;
        }
        self.removed_current = false;
        self.removed_next_id = None;
        if self.shuffle {
            match self
                .shuffle_order
                .iter()
                .position(|&r| r == real_index as usize)
            {
                Some(pos) => self.shuffle_position = pos as isize,
                None => {
                    // Track not in shuffle order — insert it right after the
                    // current position so playback can continue from here.
                    let insert_pos = (self.shuffle_position + 1).max(0) as usize;
                    Self::vec_insert(&mut self.shuffle_order, insert_pos, real_index as usize);
                    self.shuffle_position = insert_pos as isize;
                }
            }
        } else {
            self.current_index = real_index;
        }
        self.track_at(py, self.current_real_index())
    }

    // -- State accessors ---------------------------------------------------

    /// Lifetime of this queue; ordinary navigation/insertion keeps it.
    fn generation(&self) -> i64 {
        self.generation
    }

    /// Index of the currently playing track in the visible order.
    fn current_index(&self) -> isize {
        if self.removed_current {
            return -1;
        }
        if self.shuffle {
            return self.shuffle_position;
        }
        self.current_index
    }

    fn length(&self) -> usize {
        self.tracks.len()
    }

    fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    fn repeat_code(&self) -> u8 {
        self.repeat
    }

    fn shuffle_enabled(&self) -> bool {
        self.shuffle
    }

    /// Current index into `tracks` regardless of shuffle mode.
    fn real_index(&self) -> isize {
        if self.removed_current {
            return -1;
        }
        self.current_real_index()
    }

    /// Number of tracks remaining after the current position.
    fn remaining_tracks(&self) -> PyResult<i64> {
        if self.removed_current {
            return Ok(self.tracks.len() as i64 - self.removed_next_position()? as i64);
        }
        if self.shuffle {
            return Ok(self.shuffle_order.len() as i64 - self.shuffle_position as i64 - 1);
        }
        Ok(self.tracks.len() as i64 - self.current_index as i64 - 1)
    }

    /// Tracks in the current playback order (a Python tuple, like the
    /// fallback's property).
    fn tracks(&self, py: Python) -> PyResult<Py<PyAny>> {
        let items: Vec<Py<PyAny>> = if self.shuffle {
            self.shuffle_order
                .iter()
                .map(|&i| self.tracks[i].clone_ref(py))
                .collect()
        } else {
            self.tracks.iter().map(|t| t.clone_ref(py)).collect()
        };
        Ok(PyTuple::new(py, items)?.into_any().unbind())
    }

    /// `(entry_id, track)` pairs in the current playback order.
    fn entries(&self, py: Python) -> PyResult<Py<PyAny>> {
        let mut items: Vec<Py<PyAny>> = Vec::with_capacity(self.entry_ids.len());
        if self.shuffle {
            for &i in &self.shuffle_order {
                let pair = (self.entry_ids[i], self.tracks[i].clone_ref(py))
                    .into_pyobject(py)?
                    .into_any()
                    .unbind();
                items.push(pair);
            }
        } else {
            for i in 0..self.entry_ids.len() {
                let pair = (self.entry_ids[i], self.tracks[i].clone_ref(py))
                    .into_pyobject(py)?
                    .into_any()
                    .unbind();
                items.push(pair);
            }
        }
        Ok(PyTuple::new(py, items)?.into_any().unbind())
    }

    // -- Test seam ---------------------------------------------------------

    /// Overwrite the shuffle order/position wholesale. No validation —
    /// exists so tests can pin a deterministic playback order on both the
    /// native and the pure-Python core.
    fn force_shuffle_state(&mut self, order: Vec<usize>, position: isize) {
        self.shuffle = true;
        self.shuffle_order = order;
        self.shuffle_position = position;
    }
}
