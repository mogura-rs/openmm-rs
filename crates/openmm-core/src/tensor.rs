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

pub(crate) fn to_f64_vec<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Vec<f64> {
    #[cfg(feature = "ndarray")]
    {
        use burn::backend::ndarray::NdArrayTensor;
        use std::any::Any;
        let primitive = tensor.clone().into_primitive().tensor();
        if let Some(primitive) = (&primitive as &dyn Any).downcast_ref::<NdArrayTensor>() {
            match primitive {
                NdArrayTensor::F64(storage) => {
                    let view = storage.view();
                    return view
                        .as_slice()
                        .map_or_else(|| view.iter().copied().collect(), <[f64]>::to_vec);
                }
                NdArrayTensor::F32(storage) => {
                    return storage
                        .view()
                        .iter()
                        .map(|&value| f64::from(value))
                        .collect()
                }
                _ => {}
            }
        }
    }
    tensor.to_data().convert::<f64>().to_vec::<f64>().unwrap()
}

pub(crate) fn from_f64_vec<B: Backend, const D: usize>(
    values: Vec<f64>,
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    from_floats(TensorData::new(values, shape), device)
}

pub(crate) fn with_ndarray_f64<B: Backend, const D: usize, R>(
    _tensor: &Tensor<B, D>,
    _read: impl FnOnce(&[f64]) -> R,
) -> Option<R> {
    #[cfg(feature = "ndarray")]
    {
        use burn::backend::ndarray::NdArrayTensor;
        use std::any::Any;
        let primitive = _tensor.clone().into_primitive().tensor();
        if let Some(NdArrayTensor::F64(storage)) =
            (&primitive as &dyn Any).downcast_ref::<NdArrayTensor>()
        {
            return storage.view().as_slice().map(_read);
        }
    }
    None
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn host_reads_preserve_precision_and_logical_order_without_mutating_shared_storage() {
        let values = vec![1.0 + 1e-10, 2.0, 3.0, 4.0, 5.0, 6.0];
        let double = from_f64_vec::<NdArray<f64>, 2>(values.clone(), [2, 3], &Default::default());
        let shared = double.clone();
        assert_eq!(to_f64_vec(&double), values);
        assert_eq!(to_f64_vec(&shared), values);
        assert_eq!(
            to_f64_vec(&double.swap_dims(0, 1)),
            vec![values[0], 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        let single = from_f64_vec::<NdArray<f32>, 2>(values, [2, 3], &Default::default());
        assert_eq!(to_f64_vec(&single), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            to_f64_vec(&single.swap_dims(0, 1)),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }
}
