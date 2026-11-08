#[derive(Clone)]
pub(super) struct ParticleParameters {
    data: Vec<f64>,
    offsets: Vec<usize>,
}
impl Default for ParticleParameters {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            offsets: vec![0],
        }
    }
}
impl ParticleParameters {
    pub(super) fn len(&self) -> usize {
        self.offsets.len() - 1
    }
    pub(super) fn push(&mut self, values: Vec<f64>) {
        self.data.extend(values);
        self.offsets.push(self.data.len());
    }
    pub(super) fn set(&mut self, index: usize, values: Vec<f64>) {
        let start = self.offsets[index];
        let end = self.offsets[index + 1];
        if values.len() == end - start {
            self.data[start..end].copy_from_slice(&values);
        } else {
            let count = values.len();
            self.data.splice(start..end, values);
            for offset in &mut self.offsets[index + 1..] {
                *offset = *offset - (end - start) + count;
            }
        }
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = &[f64]> {
        self.offsets
            .windows(2)
            .map(|range| &self.data[range[0]..range[1]])
    }
    pub(super) fn copy_from(&mut self, source: &Self) {
        self.data.clone_from(&source.data);
        self.offsets.clone_from(&source.offsets);
    }
}
impl std::ops::Index<usize> for ParticleParameters {
    type Output = [f64];
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.offsets[index]..self.offsets[index + 1]]
    }
}
