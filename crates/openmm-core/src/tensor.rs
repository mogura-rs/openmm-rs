use burn::prelude::*;
use burn::tensor::{ElementConversion, TensorData};

/// Construct a floating tensor in the backend's configured precision.
pub fn from_floats<B: Backend, const D: usize>(
    data: impl Into<TensorData>,
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::from_data(data.into().convert::<B::FloatElem>(), device)
}

pub(crate) fn scalar<B: Backend>(value: Tensor<B, 1>) -> f64 {
    value.into_scalar().elem::<f64>()
}
