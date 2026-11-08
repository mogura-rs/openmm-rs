use super::*;
use openmm_core::DPDIntegrator;

integrator_class!(PyDPDIntegrator, "DPDIntegrator", DPDIntegrator, {
    #[new]
    fn new(
        temperature: f64,
        default_friction: f64,
        default_cutoff: f64,
        step_size: f64,
    ) -> PyResult<Self> {
        nonnegative(temperature, "temperature")?;
        nonnegative(default_friction, "default friction")?;
        positive(default_cutoff, "default cutoff")?;
        positive(step_size, "step size")?;
        Ok(Self::from_core(DPDIntegrator::new(
            temperature,
            default_friction,
            default_cutoff,
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
    fn get_default_friction(&self) -> f64 {
        self.inner.lock().unwrap().default_friction()
    }
    fn set_default_friction(&self, value: f64) -> PyResult<()> {
        nonnegative(value, "default friction")?;
        self.inner.lock().unwrap().set_default_friction(value);
        Ok(())
    }
    fn get_default_cutoff(&self) -> f64 {
        self.inner.lock().unwrap().default_cutoff()
    }
    fn set_default_cutoff(&self, value: f64) -> PyResult<()> {
        positive(value, "default cutoff")?;
        self.inner.lock().unwrap().set_default_cutoff(value);
        Ok(())
    }
    fn get_random_number_seed(&self) -> i32 {
        self.inner.lock().unwrap().random_number_seed()
    }
    fn set_random_number_seed(&self, value: i32) {
        self.inner.lock().unwrap().set_random_number_seed(value);
    }
    fn get_particle_type(&self, index: i32) -> i32 {
        self.inner.lock().unwrap().particle_type(index)
    }
    fn set_particle_type(&self, index: i32, value: i32) {
        self.inner.lock().unwrap().set_particle_type(index, value);
    }
    fn get_particle_types(&self) -> std::collections::BTreeMap<i32, i32> {
        self.inner.lock().unwrap().particle_types().clone()
    }
    fn get_num_type_pairs(&self) -> usize {
        self.inner.lock().unwrap().num_type_pairs()
    }
    fn add_type_pair(&self, a: i32, b: i32, friction: f64, cutoff: f64) -> usize {
        self.inner
            .lock()
            .unwrap()
            .add_type_pair(a, b, friction, cutoff)
    }
    fn get_type_pair_parameters(&self, index: usize) -> PyResult<(i32, i32, f64, f64)> {
        self.inner
            .lock()
            .unwrap()
            .type_pair_parameters(index)
            .map_err(|e| err(&e.to_string()))
    }
    fn set_type_pair_parameters(
        &self,
        index: usize,
        a: i32,
        b: i32,
        friction: f64,
        cutoff: f64,
    ) -> PyResult<()> {
        self.inner
            .lock()
            .unwrap()
            .set_type_pair_parameters(index, a, b, friction, cutoff)
            .map_err(|e| err(&e.to_string()))
    }
});
