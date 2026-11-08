use burn::{backend::NdArray, tensor::TensorData};
use openmm_core::{
    tensor::from_floats, BrownianIntegrator, CompoundIntegrator, Context, Integrator,
    LangevinMiddleIntegrator, System, VariableLangevinIntegrator, VariableVerletIntegrator,
    VerletIntegrator,
};
use std::alloc::{GlobalAlloc, Layout, System as SystemAllocator};
use std::sync::atomic::{AtomicUsize, Ordering};

const PARTICLES: usize = 4096;
const DENSE_BYTES: usize = 3 * PARTICLES * std::mem::size_of::<f64>();
static LARGE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { SystemAllocator.alloc(layout) };
        if !pointer.is_null() && layout.size() >= DENSE_BYTES {
            LARGE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { SystemAllocator.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn stochastic_cpu_steps_reuse_state_without_dense_temporaries() {
    type B = NdArray<f64>;
    let device = Default::default();
    let mass = from_floats::<B, 1>(TensorData::new(vec![1.; PARTICLES], [PARTICLES]), &device);
    let forces = from_floats::<B, 2>(
        TensorData::new(vec![0.; 3 * PARTICLES], [PARTICLES, 3]),
        &device,
    );
    let mut position = from_floats::<B, 2>(
        TensorData::new(vec![0.; 3 * PARTICLES], [PARTICLES, 3]),
        &device,
    );
    let mut velocity = from_floats::<B, 2>(
        TensorData::new(vec![0.; 3 * PARTICLES], [PARTICLES, 3]),
        &device,
    );
    let mut variable_verlet = VariableVerletIntegrator::new(1e-5);
    variable_verlet.set_maximum_step_size(0.001);
    let mut variable_langevin = VariableLangevinIntegrator::new(300., 1., 1e-5);
    variable_langevin.set_maximum_step_size(0.001);
    let integrators: [Box<dyn Integrator<B>>; 4] = [
        Box::new(variable_verlet),
        Box::new(variable_langevin),
        Box::new(LangevinMiddleIntegrator::new(300., 1., 0.001)),
        Box::new(BrownianIntegrator::new(300., 1., 0.001)),
    ];
    for integrator in integrators {
        integrator.step(&mut position, &mut velocity, &forces, &mass);
        let before = LARGE_ALLOCATIONS.load(Ordering::Relaxed);
        for _ in 0..10 {
            integrator.step(&mut position, &mut velocity, &forces, &mass);
        }
        assert_eq!(
            LARGE_ALLOCATIONS.load(Ordering::Relaxed) - before,
            0,
            "unconstrained CPU integration must reuse dense storage"
        );
        assert_eq!(position.dims(), [PARTICLES, 3]);
    }
    let mut system = System::<B>::new();
    for _ in 0..PARTICLES {
        system.add_particle(1.0);
    }
    let mut variable_verlet = VariableVerletIntegrator::new(1e-5);
    variable_verlet.set_maximum_step_size(0.001);
    let mut variable_langevin = VariableLangevinIntegrator::new(300., 1., 1e-5);
    variable_langevin.set_maximum_step_size(0.001);
    let mut compound = CompoundIntegrator::<B>::new();
    compound.add_integrator(Box::new(VerletIntegrator::new(0.001)));
    let integrators: [Box<dyn Integrator<B>>; 6] = [
        Box::new(compound),
        Box::new(variable_verlet),
        Box::new(variable_langevin),
        Box::new(VerletIntegrator::new(0.001)),
        Box::new(LangevinMiddleIntegrator::new(300., 1., 0.001)),
        Box::new(BrownianIntegrator::new(300., 1., 0.001)),
    ];
    for integrator in integrators {
        let mut context = Context::new(system.clone(), integrator, Default::default());
        context.step(1);
        let before = LARGE_ALLOCATIONS.load(Ordering::Relaxed);
        context.step(10);
        assert_eq!(
            LARGE_ALLOCATIONS.load(Ordering::Relaxed) - before,
            0,
            "Context must reuse its empty force buffer"
        );
    }
}
