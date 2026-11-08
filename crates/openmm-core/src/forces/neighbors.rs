use super::pme::reciprocal_box;
use std::collections::HashMap;
use std::sync::Mutex;

const EMPTY: u32 = u32::MAX;

#[derive(Default)]
pub(crate) struct NeighborCells(Mutex<CellStorage>);

impl Clone for NeighborCells {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[derive(Default)]
struct CellStorage {
    heads: Vec<u32>,
    sparse_heads: HashMap<[i64; 3], u32>,
    next: Vec<u32>,
}

struct Geometry {
    dimensions: [i64; 3],
    origin: [f64; 3],
    reciprocal: Option<[[f64; 3]; 3]>,
    inverse_cutoff: f64,
}

impl Geometry {
    fn new(positions: &[f64], cutoff: f64, box_vectors: Option<&[[f64; 3]; 3]>) -> Self {
        if let Some(vectors) = box_vectors {
            let reciprocal = reciprocal_box(vectors).0;
            let dimensions = reciprocal.map(|axis| {
                let norm = axis.iter().map(|v| v * v).sum::<f64>().sqrt();
                (1.0 / (cutoff * norm)).floor().clamp(1.0, 1e9) as i64
            });
            Self {
                dimensions,
                origin: [0.0; 3],
                reciprocal: Some(reciprocal),
                inverse_cutoff: cutoff.recip(),
            }
        } else {
            let mut minimum = [f64::INFINITY; 3];
            let mut maximum = [f64::NEG_INFINITY; 3];
            for position in positions.chunks_exact(3) {
                for axis in 0..3 {
                    minimum[axis] = minimum[axis].min(position[axis]);
                    maximum[axis] = maximum[axis].max(position[axis]);
                }
            }
            let dimensions = std::array::from_fn(|axis| {
                ((maximum[axis] - minimum[axis]) / cutoff).floor() as i64 + 1
            });
            Self {
                dimensions,
                origin: minimum,
                reciprocal: None,
                inverse_cutoff: cutoff.recip(),
            }
        }
    }

    fn key(&self, position: &[f64]) -> [i64; 3] {
        if let Some(reciprocal) = self.reciprocal {
            std::array::from_fn(|axis| {
                let fraction = (0..3)
                    .map(|d| position[d] * reciprocal[axis][d])
                    .sum::<f64>();
                (((fraction - fraction.floor()) * self.dimensions[axis] as f64).floor() as i64)
                    .min(self.dimensions[axis] - 1)
            })
        } else {
            std::array::from_fn(|axis| {
                ((position[axis] - self.origin[axis]) * self.inverse_cutoff).floor() as i64
            })
        }
    }

    fn offset(&self, key: [i64; 3]) -> usize {
        ((key[0] * self.dimensions[1] + key[1]) * self.dimensions[2] + key[2]) as usize
    }

    fn offsets(&self, axis: usize) -> &'static [i64] {
        match (self.reciprocal.is_some(), self.dimensions[axis]) {
            (true, 1) => &[0],
            (true, 2) => &[0, 1],
            _ => &[-1, 0, 1],
        }
    }
}

impl NeighborCells {
    /// Enumerate candidate pairs without storing an atom-pair list or dense matrix.
    pub fn visit(
        &self,
        positions: &[f64],
        cutoff: f64,
        box_vectors: Option<&[[f64; 3]; 3]>,
        mut pair: impl FnMut(usize, usize),
    ) {
        let n = positions.len() / 3;
        assert!(
            n < u32::MAX as usize,
            "particle count exceeds cell-list index capacity"
        );
        assert!(
            positions.iter().all(|value| value.is_finite()),
            "particle positions must be finite"
        );
        if n < 2 {
            return;
        }
        let geometry = Geometry::new(positions, cutoff, box_vectors);
        let full_span = if box_vectors.is_some() { 3 } else { 2 };
        if geometry.dimensions.iter().all(|&size| size <= full_span) {
            for i in 0..n {
                for j in i + 1..n {
                    pair(i, j);
                }
            }
            return;
        }
        let dense_size = geometry
            .dimensions
            .iter()
            .try_fold(1_usize, |count, &d| count.checked_mul(d as usize))
            .filter(|&count| count <= 8 * n);
        let mut storage = self.0.lock().unwrap();
        storage.next.resize(n, EMPTY);
        if let Some(size) = dense_size {
            storage.heads.resize(size, EMPTY);
            storage.heads.fill(EMPTY);
            storage.sparse_heads.clear();
        } else {
            storage.sparse_heads.clear();
        }
        for (i, position) in positions.chunks_exact(3).enumerate() {
            let key = geometry.key(position);
            let previous = if dense_size.is_some() {
                let index = geometry.offset(key);
                let previous = storage.heads[index];
                storage.heads[index] = i as u32;
                previous
            } else {
                storage.sparse_heads.insert(key, i as u32).unwrap_or(EMPTY)
            };
            storage.next[i] = previous;
        }
        for (i, position) in positions.chunks_exact(3).enumerate() {
            let center = geometry.key(position);
            for &dx in geometry.offsets(0) {
                for &dy in geometry.offsets(1) {
                    for &dz in geometry.offsets(2) {
                        let mut key = [center[0] + dx, center[1] + dy, center[2] + dz];
                        if geometry.reciprocal.is_some() {
                            for (axis, value) in key.iter_mut().enumerate() {
                                *value = value.rem_euclid(geometry.dimensions[axis]);
                            }
                        } else if (0..3)
                            .any(|axis| key[axis] < 0 || key[axis] >= geometry.dimensions[axis])
                        {
                            continue;
                        }
                        let mut j = if dense_size.is_some() {
                            storage.heads[geometry.offset(key)]
                        } else {
                            storage.sparse_heads.get(&key).copied().unwrap_or(EMPTY)
                        };
                        while j != EMPTY {
                            if j as usize > i {
                                pair(i, j as usize);
                            }
                            j = storage.next[j as usize];
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::harmonic_bond::apply_minimum_image_ortho;
    use std::collections::BTreeSet;

    #[test]
    fn cell_candidates_cover_cutoff_pairs_without_duplicates() {
        for box_vectors in [
            None,
            Some([[2.0, 0.0, 0.0], [0.7, 2.0, 0.0], [-0.4, 0.5, 2.0]]),
            Some([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ] {
            for cutoff in [0.2, 0.7, 1.1] {
                let positions: Vec<f64> = (0..240)
                    .map(|i| (((i * 137 + 37) % 997) as f64) / 121.0 - 4.0)
                    .collect();
                let mut expected = BTreeSet::new();
                let distance = |i: usize, j: usize| {
                    let mut delta = std::array::from_fn(|axis| {
                        positions[3 * j + axis] - positions[3 * i + axis]
                    });
                    if let Some(vectors) = &box_vectors {
                        apply_minimum_image_ortho(&mut delta, vectors);
                    }
                    delta.iter().map(|x| x * x).sum::<f64>()
                };
                for i in 0..80 {
                    for j in i + 1..80 {
                        if distance(i, j) < cutoff * cutoff {
                            expected.insert((i, j));
                        }
                    }
                }
                let mut actual = BTreeSet::new();
                NeighborCells::default().visit(&positions, cutoff, box_vectors.as_ref(), |i, j| {
                    if distance(i, j) < cutoff * cutoff {
                        assert!(actual.insert((i, j)), "duplicate pair");
                    }
                });
                assert_eq!(actual, expected, "cutoff={cutoff}, box={box_vectors:?}");
            }
        }
    }
}
