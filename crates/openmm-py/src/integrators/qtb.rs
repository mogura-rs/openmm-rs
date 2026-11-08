use super::*;
use openmm_core::QTBIntegrator;

integrator_class!(PyQTBIntegrator, "QTBIntegrator", QTBIntegrator, {
    #[new]
    fn new(temperature: f64, friction: f64, step_size: f64) -> PyResult<Self> {
        nonnegative(temperature, "temperature")?;
        nonnegative(friction, "friction")?;
        positive(step_size, "step size")?;
        Ok(Self::from_core(QTBIntegrator::new(
            temperature,
            friction,
            step_size,
        )))
    }
    fn get_temperature(&self) -> f64 {
        self.inner.lock().unwrap().temperature()
    }
    fn set_temperature(&self, value: f64) -> PyResult<()> {
        nonnegative(value, "temperature")?;
        self.inner.lock().unwrap().set_temperature(value);
        Ok(())
    }
    fn get_friction(&self) -> f64 {
        self.inner.lock().unwrap().friction()
    }
    fn set_friction(&self, value: f64) -> PyResult<()> {
        nonnegative(value, "friction")?;
        self.inner.lock().unwrap().set_friction(value);
        Ok(())
    }
    fn get_segment_length(&self) -> f64 {
        self.inner.lock().unwrap().segment_length()
    }
    fn set_segment_length(&self, value: f64) -> PyResult<()> {
        positive(value, "segment length")?;
        self.inner.lock().unwrap().set_segment_length(value);
        Ok(())
    }
    fn get_cutoff_frequency(&self) -> f64 {
        self.inner.lock().unwrap().cutoff_frequency()
    }
    fn set_cutoff_frequency(&self, value: f64) -> PyResult<()> {
        positive(value, "cutoff frequency")?;
        self.inner.lock().unwrap().set_cutoff_frequency(value);
        Ok(())
    }
    fn get_default_adaptation_rate(&self) -> f64 {
        self.inner.lock().unwrap().default_adaptation_rate()
    }
    fn set_default_adaptation_rate(&self, value: f64) -> PyResult<()> {
        nonnegative(value, "adaptation rate")?;
        self.inner
            .lock()
            .unwrap()
            .set_default_adaptation_rate(value);
        Ok(())
    }
    fn get_random_number_seed(&self) -> i32 {
        self.inner.lock().unwrap().random_number_seed()
    }
    fn set_random_number_seed(&self, value: i32) {
        self.inner.lock().unwrap().set_random_number_seed(value);
    }
    fn get_particle_types(&self) -> std::collections::BTreeMap<i32, i32> {
        self.inner.lock().unwrap().particle_types().clone()
    }
    fn set_particle_type(&self, index: i32, value: i32) {
        self.inner.lock().unwrap().set_particle_type(index, value);
    }
    fn get_type_adaptation_rates(&self) -> std::collections::BTreeMap<i32, f64> {
        self.inner.lock().unwrap().type_adaptation_rates().clone()
    }
    fn set_type_adaptation_rate(&self, kind: i32, value: f64) -> PyResult<()> {
        nonnegative(value, "adaptation rate")?;
        self.inner
            .lock()
            .unwrap()
            .set_type_adaptation_rate(kind, value);
        Ok(())
    }
    fn get_adapted_friction(&self, index: usize) -> PyResult<Vec<f64>> {
        if self.link.0.borrow().context.upgrade().is_none() {
            return Err(err("this Integrator is not bound to a Context"));
        }
        self.inner
            .lock()
            .unwrap()
            .adapted_friction(index)
            .map_err(|e| err(&e.to_string()))
    }
});
