use burn::prelude::*;
use burn::tensor::{ElementConversion, TensorData};

/// Construct a floating tensor in the backend's configured precision.
pub fn from_floats<B: Backend, const D: usize>(
    data: impl Into<TensorData>,
    device: &B::Device,
) -> Tensor<B, D> {
    let primitive =
        Tensor::<B, D>::from_data(data.into().convert::<B::FloatElem>(), device).into_primitive();
    #[cfg(feature = "ndarray")]
    let primitive = {
        use burn::backend::ndarray::NdArrayTensor;
        use burn::tensor::TensorPrimitive;
        use std::any::Any;
        let mut primitive = primitive;
        if let TensorPrimitive::Float(raw) = &mut primitive {
            if let Some(array) = (raw as &mut dyn Any).downcast_mut::<NdArrayTensor>() {
                // Native TensorData bytes clone their allocation; ArcArray clones share it.
                match array {
                    NdArrayTensor::F64(storage) if storage.is_borrowed() => {
                        *array = storage.view().to_owned().into_shared().into();
                    }
                    NdArrayTensor::F32(storage) if storage.is_borrowed() => {
                        *array = storage.view().to_owned().into_shared().into();
                    }
                    _ => {}
                }
            }
        }
        primitive
    };
    Tensor::from_primitive(primitive)
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
    #[cfg(feature = "ndarray")]
    {
        use burn::backend::ndarray::NdArrayTensor;
        use burn::tensor::TensorPrimitive;
        use std::any::{Any, TypeId};
        if TypeId::of::<B::FloatTensorPrimitive>() == TypeId::of::<NdArrayTensor>()
            && TypeId::of::<B::FloatElem>() == TypeId::of::<f64>()
        {
            let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), values)
                .expect("tensor data length must match its shape")
                .into_shared();
            let primitive: Box<dyn Any> = Box::new(NdArrayTensor::from(array));
            let primitive = *primitive.downcast::<B::FloatTensorPrimitive>().unwrap();
            return Tensor::from_primitive(TensorPrimitive::Float(primitive));
        }
    }
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

pub(crate) fn with_mut_ndarray_f64<B: Backend, const D: usize, R>(
    _tensor: &mut Tensor<B, D>,
    _modify: impl FnOnce(&mut [f64]) -> R,
) -> Option<R> {
    #[cfg(feature = "ndarray")]
    {
        use burn::backend::ndarray::NdArrayTensor;
        use burn::tensor::TensorPrimitive;
        use std::any::{Any, TypeId};
        if TypeId::of::<B::FloatTensorPrimitive>() != TypeId::of::<NdArrayTensor>()
            || TypeId::of::<B::FloatElem>() != TypeId::of::<f64>()
        {
            return None;
        }
        let mut result = None;
        _tensor.inplace(|tensor| {
            let mut primitive = tensor.into_primitive();
            if let TensorPrimitive::Float(raw) = &mut primitive {
                if let Some(NdArrayTensor::F64(storage)) =
                    (raw as &mut dyn Any).downcast_mut::<NdArrayTensor>()
                {
                    let array = storage.ensure_owned();
                    if !array.is_standard_layout() {
                        *array = array.as_standard_layout().into_owned().into_shared();
                    }
                    result = Some(_modify(array.as_slice_mut().unwrap()));
                }
            }
            Tensor::from_primitive(primitive)
        });
        result
    }
    #[cfg(not(feature = "ndarray"))]
    None
}

pub(crate) fn zeros<B: Backend, const D: usize>(
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    #[cfg(feature = "ndarray")]
    {
        use burn::backend::ndarray::NdArrayTensor;
        use std::any::TypeId;
        if TypeId::of::<B::FloatTensorPrimitive>() == TypeId::of::<NdArrayTensor>()
            && TypeId::of::<B::FloatElem>() == TypeId::of::<f64>()
        {
            return from_f64_vec(vec![0.0; shape.iter().product()], shape, device);
        }
    }
    Tensor::zeros(shape, device)
}

pub(crate) fn with_f64_data<B: Backend, const D: usize, R>(
    tensor: &Tensor<B, D>,
    read: impl FnOnce(&[f64]) -> R,
) -> R {
    let mut read = Some(read);
    with_ndarray_f64(tensor, |data| read.take().unwrap()(data))
        .unwrap_or_else(|| read.take().unwrap()(&to_f64_vec(tensor)))
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

    #[test]
    fn host_writes_reuse_unique_storage_and_preserve_shared_and_strided_values() {
        let mut values = from_f64_vec::<NdArray<f64>, 2>(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            &Default::default(),
        );
        let address = with_ndarray_f64(&values, |v| v.as_ptr() as usize).unwrap();
        assert_eq!(with_mut_ndarray_f64(&mut values, |v| v[0] += 1.0), Some(()));
        assert_eq!(
            with_ndarray_f64(&values, |v| v.as_ptr() as usize),
            Some(address)
        );
        let snapshot = values.clone();
        with_mut_ndarray_f64(&mut values, |v| v[1] = 7.0).unwrap();
        assert_eq!(to_f64_vec(&snapshot), [2.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_ne!(
            with_ndarray_f64(&values, |v| v.as_ptr() as usize),
            Some(address)
        );
        let mut strided = values.swap_dims(0, 1);
        with_mut_ndarray_f64(&mut strided, |v| v[2] = 8.0).unwrap();
        assert_eq!(strided.dims(), [3, 2]);
        assert_eq!(to_f64_vec(&strided), [2.0, 4.0, 8.0, 5.0, 3.0, 6.0]);
        let mut single = from_f64_vec::<NdArray<f32>, 2>(vec![1.0; 6], [2, 3], &Default::default());
        assert!(with_mut_ndarray_f64(&mut single, |_| panic!(
            "single precision must use its fallback"
        ))
        .is_none());
        assert_eq!(to_f64_vec(&single), [1.0; 6]);
    }
}
