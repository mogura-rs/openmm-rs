use super::*;
use openmm_core::{NoseHooverChain, NoseHooverIntegrator};

integrator_class!(
    PyNoseHooverIntegrator,
    "NoseHooverIntegrator",
    NoseHooverIntegrator,
    {
        #[new]
        #[pyo3(signature = (temperature=None, collision_frequency=None, step_size=None, chain_length=3, num_mts=3, num_yoshida_suzuki=7))]
        fn new(
            temperature: Option<f64>,
            collision_frequency: Option<f64>,
            step_size: Option<f64>,
            chain_length: usize,
            num_mts: usize,
            num_yoshida_suzuki: usize,
        ) -> PyResult<Self> {
            match (temperature, collision_frequency, step_size) {
                (Some(dt), None, None) | (None, None, Some(dt)) => {
                    positive(dt, "step size")?;
                    Ok(Self::from_core(NoseHooverIntegrator::new(dt)))
                }
                (Some(t), Some(f), Some(dt)) => {
                    positive(dt, "step size")?;
                    Ok(Self::from_core(
                        NoseHooverIntegrator::thermostated(
                            t,
                            f,
                            dt,
                            chain_length,
                            num_mts,
                            num_yoshida_suzuki,
                        )
                        .map_err(|e| err(&e.to_string()))?,
                    ))
                }
                _ => Err(err(
                    "specify a step size alone, or temperature, collision frequency and step size",
                )),
            }
        }
        #[pyo3(signature = (temperature, collision_frequency, chain_length=3, num_mts=3, num_yoshida_suzuki=7))]
        fn add_thermostat(
            &self,
            temperature: f64,
            collision_frequency: f64,
            chain_length: usize,
            num_mts: usize,
            num_yoshida_suzuki: usize,
        ) -> PyResult<usize> {
            self.inner
                .lock()
                .unwrap()
                .add_thermostat(
                    temperature,
                    collision_frequency,
                    chain_length,
                    num_mts,
                    num_yoshida_suzuki,
                )
                .map_err(|e| err(&e.to_string()))
        }
        #[allow(clippy::too_many_arguments)]
        #[pyo3(signature = (particles, pairs, temperature, collision_frequency, relative_temperature, relative_collision_frequency, chain_length=3, num_mts=3, num_yoshida_suzuki=7))]
        fn add_subsystem_thermostat(
            &self,
            particles: Vec<usize>,
            pairs: Vec<(usize, usize)>,
            temperature: f64,
            collision_frequency: f64,
            relative_temperature: f64,
            relative_collision_frequency: f64,
            chain_length: usize,
            num_mts: usize,
            num_yoshida_suzuki: usize,
        ) -> PyResult<usize> {
            self.inner
                .lock()
                .unwrap()
                .add_subsystem_thermostat(
                    particles,
                    pairs,
                    temperature,
                    collision_frequency,
                    relative_temperature,
                    relative_collision_frequency,
                    chain_length,
                    num_mts,
                    num_yoshida_suzuki,
                )
                .map_err(|e| err(&e.to_string()))
        }
        fn get_num_thermostats(&self) -> usize {
            self.inner.lock().unwrap().num_thermostats()
        }
        fn has_subsystem_thermostats(&self) -> bool {
            self.inner.lock().unwrap().has_subsystem_thermostats()
        }
        #[pyo3(signature = (chain_id=0))]
        fn get_thermostat(&self, chain_id: usize) -> PyResult<PyNoseHooverChain> {
            if chain_id >= self.get_num_thermostats() {
                return Err(err("thermostat index out of range"));
            }
            Ok(PyNoseHooverChain {
                owner: Some(self.inner.clone()),
                index: chain_id,
                standalone: None,
            })
        }
        fn get_maximum_pair_distance(&self) -> f64 {
            self.inner.lock().unwrap().maximum_pair_distance()
        }
        fn set_maximum_pair_distance(&self, value: f64) -> PyResult<()> {
            nonnegative(value, "maximum pair distance")?;
            self.inner.lock().unwrap().set_maximum_pair_distance(value);
            Ok(())
        }
        fn compute_heat_bath_energy(&self) -> f64 {
            self.inner.lock().unwrap().compute_heat_bath_energy()
        }
        fn get_all_thermostated_individual_particles(&self) -> Vec<usize> {
            self.inner
                .lock()
                .unwrap()
                .all_thermostated_individual_particles()
                .to_vec()
        }
        fn get_all_thermostated_pairs(&self) -> Vec<(usize, usize, f64)> {
            self.inner.lock().unwrap().all_thermostated_pairs().to_vec()
        }
        #[pyo3(signature = (chain_id=0))]
        fn get_temperature(&self, chain_id: usize) -> PyResult<f64> {
            Ok(self.get_thermostat(chain_id)?.read().temperature)
        }
        #[pyo3(signature = (value, chain_id=0))]
        fn set_temperature(&self, value: f64, chain_id: usize) -> PyResult<()> {
            self.get_thermostat(chain_id)?
                .update(|c| c.temperature = value)
        }
        #[pyo3(signature = (chain_id=0))]
        fn get_relative_temperature(&self, chain_id: usize) -> PyResult<f64> {
            Ok(self.get_thermostat(chain_id)?.read().relative_temperature)
        }
        #[pyo3(signature = (value, chain_id=0))]
        fn set_relative_temperature(&self, value: f64, chain_id: usize) -> PyResult<()> {
            self.get_thermostat(chain_id)?
                .update(|c| c.relative_temperature = value)
        }
        #[pyo3(signature = (chain_id=0))]
        fn get_collision_frequency(&self, chain_id: usize) -> PyResult<f64> {
            Ok(self.get_thermostat(chain_id)?.read().collision_frequency)
        }
        #[pyo3(signature = (value, chain_id=0))]
        fn set_collision_frequency(&self, value: f64, chain_id: usize) -> PyResult<()> {
            self.get_thermostat(chain_id)?
                .update(|c| c.collision_frequency = value)
        }
        #[pyo3(signature = (chain_id=0))]
        fn get_relative_collision_frequency(&self, chain_id: usize) -> PyResult<f64> {
            Ok(self
                .get_thermostat(chain_id)?
                .read()
                .relative_collision_frequency)
        }
        #[pyo3(signature = (value, chain_id=0))]
        fn set_relative_collision_frequency(&self, value: f64, chain_id: usize) -> PyResult<()> {
            self.get_thermostat(chain_id)?
                .update(|c| c.relative_collision_frequency = value)
        }
    }
);

#[pyclass(unsendable, name = "NoseHooverChain")]
pub(crate) struct PyNoseHooverChain {
    owner: Option<Arc<Mutex<NoseHooverIntegrator>>>,
    index: usize,
    standalone: Option<RefCell<NoseHooverChain>>,
}
impl PyNoseHooverChain {
    fn read(&self) -> NoseHooverChain {
        if let Some(owner) = &self.owner {
            owner.lock().unwrap().thermostat(self.index).clone()
        } else {
            self.standalone.as_ref().unwrap().borrow().clone()
        }
    }
    fn update(&self, change: impl FnOnce(&mut NoseHooverChain)) -> PyResult<()> {
        if let Some(owner) = &self.owner {
            let mut owner = owner.lock().unwrap();
            let mut chain = owner.thermostat(self.index).clone();
            change(&mut chain);
            owner
                .set_thermostat(self.index, chain)
                .map_err(|e| err(&e.to_string()))
        } else {
            let mut chain = self.read();
            change(&mut chain);
            chain.validate().map_err(|e| err(&e.to_string()))?;
            *self.standalone.as_ref().unwrap().borrow_mut() = chain;
            Ok(())
        }
    }
}
#[pymethods]
impl PyNoseHooverChain {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        temperature: f64,
        relative_temperature: f64,
        collision_frequency: f64,
        relative_collision_frequency: f64,
        num_degrees_of_freedom: i32,
        chain_length: usize,
        num_mts: usize,
        num_yoshida_suzuki: usize,
        chain_id: usize,
        atoms: Vec<usize>,
        pairs: Vec<(usize, usize)>,
    ) -> PyResult<Self> {
        let chain = NoseHooverChain {
            temperature,
            relative_temperature,
            collision_frequency,
            relative_collision_frequency,
            num_degrees_of_freedom,
            chain_length,
            num_mts,
            num_ys: num_yoshida_suzuki,
            chain_id,
            atoms,
            pairs,
        };
        chain.validate().map_err(|e| err(&e.to_string()))?;
        Ok(Self {
            owner: None,
            index: 0,
            standalone: Some(RefCell::new(chain)),
        })
    }
    fn uses_periodic_boundary_conditions(&self) -> bool {
        false
    }
    fn get_yoshida_suzuki_weights(&self) -> Vec<f64> {
        self.read().yoshida_suzuki_weights().to_vec()
    }
    fn get_temperature(&self) -> f64 {
        self.read().temperature
    }
    fn set_temperature(&self, value: f64) -> PyResult<()> {
        self.update(|c| c.temperature = value)
    }
    fn get_relative_temperature(&self) -> f64 {
        self.read().relative_temperature
    }
    fn set_relative_temperature(&self, value: f64) -> PyResult<()> {
        self.update(|c| c.relative_temperature = value)
    }
    fn get_collision_frequency(&self) -> f64 {
        self.read().collision_frequency
    }
    fn set_collision_frequency(&self, value: f64) -> PyResult<()> {
        self.update(|c| c.collision_frequency = value)
    }
    fn get_relative_collision_frequency(&self) -> f64 {
        self.read().relative_collision_frequency
    }
    fn set_relative_collision_frequency(&self, value: f64) -> PyResult<()> {
        self.update(|c| c.relative_collision_frequency = value)
    }
    fn get_num_degrees_of_freedom(&self) -> i32 {
        self.read().num_degrees_of_freedom
    }
    fn set_num_degrees_of_freedom(&self, value: i32) -> PyResult<()> {
        self.update(|c| c.num_degrees_of_freedom = value)
    }
    fn get_chain_length(&self) -> usize {
        self.read().chain_length
    }
    fn get_num_multi_time_steps(&self) -> usize {
        self.read().num_mts
    }
    fn get_num_yoshida_suzuki_time_steps(&self) -> usize {
        self.read().num_ys
    }
    fn get_chain_id(&self) -> usize {
        self.read().chain_id
    }
    fn get_thermostated_atoms(&self) -> Vec<usize> {
        self.read().atoms
    }
    fn set_thermostated_atoms(&self, value: Vec<usize>) -> PyResult<()> {
        self.update(|c| c.atoms = value)
    }
    fn get_thermostated_pairs(&self) -> Vec<(usize, usize)> {
        self.read().pairs
    }
    fn set_thermostated_pairs(&self, value: Vec<(usize, usize)>) -> PyResult<()> {
        self.update(|c| c.pairs = value)
    }
}
