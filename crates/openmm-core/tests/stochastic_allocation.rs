use burn::{backend::NdArray, tensor::TensorData};
use openmm_core::{
    tensor::from_floats, BrownianIntegrator, CompoundIntegrator, Context, CustomIntegrator,
    DPDIntegrator, Integrator, LangevinMiddleIntegrator, NoseHooverIntegrator, QTBIntegrator,
    System, VariableLangevinIntegrator, VariableVerletIntegrator, VerletIntegrator,
};
use std::alloc::{GlobalAlloc, Layout, System as SystemAllocator};
use std::sync::atomic::{AtomicUsize, Ordering};

const PARTICLES: usize = 4096;
const DENSE_BYTES: usize = 3 * PARTICLES * std::mem::size_of::<f64>();
static LARGE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static MAX_ALLOCATION: AtomicUsize = AtomicUsize::new(0);

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { SystemAllocator.alloc(layout) };
        if !pointer.is_null() {
            MAX_ALLOCATION.fetch_max(layout.size(), Ordering::Relaxed);
        }
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
    let mut qtb = QTBIntegrator::new(300., 1., 0.001);
    qtb.set_segment_length(0.003);
    let mut custom = CustomIntegrator::new(0.001);
    custom.add_per_dof_variable("q", 0.0).unwrap();
    custom.add_global_variable("total", 0.0).unwrap();
    custom.add_compute_per_dof("v", "v+dt*f/m").unwrap();
    custom.add_compute_per_dof("x", "x+dt*v").unwrap();
    custom.add_compute_per_dof("q", "cross(x,v)").unwrap();
    custom.add_compute_sum("total", "dot(q,q)").unwrap();
    let integrators: [Box<dyn Integrator<B>>; 10] = [
        Box::new(custom),
        Box::new(qtb),
        Box::new(DPDIntegrator::new(300., 1., 0.35, 0.001)),
        Box::new(NoseHooverIntegrator::thermostated(300., 1., 0.001, 3, 3, 7).unwrap()),
        Box::new(compound),
        Box::new(variable_verlet),
        Box::new(variable_langevin),
        Box::new(VerletIntegrator::new(0.001)),
        Box::new(LangevinMiddleIntegrator::new(300., 1., 0.001)),
        Box::new(BrownianIntegrator::new(300., 1., 0.001)),
    ];
    for integrator in integrators {
        let mut context = Context::new(system.clone(), integrator, Default::default());
        context.set_positions(from_floats(
            TensorData::new(
                (0..PARTICLES)
                    .flat_map(|i| {
                        [
                            0.2 * (i % 16) as f64,
                            0.2 * ((i / 16) % 16) as f64,
                            0.2 * (i / 256) as f64,
                        ]
                    })
                    .collect::<Vec<_>>(),
                [PARTICLES, 3],
            ),
            &device,
        ));
        context.set_velocities(from_floats(
            TensorData::new(vec![1.; 3 * PARTICLES], [PARTICLES, 3]),
            &device,
        ));
        context.step(1);
        let before = LARGE_ALLOCATIONS.load(Ordering::Relaxed);
        context.step(10);
        assert_eq!(
            LARGE_ALLOCATIONS.load(Ordering::Relaxed) - before,
            0,
            "Context must reuse its empty force buffer"
        );
    }
    let mut read_only = CustomIntegrator::new(0.001);
    read_only.add_per_dof_variable("q", 0.0).unwrap();
    read_only.add_global_variable("total", 0.0).unwrap();
    read_only.add_compute_per_dof("q", "x+v").unwrap();
    read_only.add_compute_sum("total", "q*q").unwrap();
    let mut context = Context::new(system.clone(), Box::new(read_only), Default::default());
    context.step(1);
    let snapshot = context.get_state(openmm_core::StateFlags {
        positions: true,
        velocities: true,
        ..Default::default()
    });
    let before = LARGE_ALLOCATIONS.load(Ordering::Relaxed);
    context.step(10);
    assert_eq!(
        LARGE_ALLOCATIONS.load(Ordering::Relaxed) - before,
        0,
        "reading shared particle state must not trigger copy-on-write"
    );
    assert!(snapshot.positions.is_some() && snapshot.velocities.is_some());
    MAX_ALLOCATION.store(0, Ordering::Relaxed);
    let mut typed = DPDIntegrator::new(300., 1., 1., 0.001);
    for i in 0..PARTICLES {
        typed.set_particle_type(i as i32, i as i32);
    }
    typed.add_type_pair(0, (PARTICLES - 1) as i32, 2., 0.8);
    let typed_context = Context::new(system, Box::new(typed), Default::default());
    assert!(
        MAX_ALLOCATION.load(Ordering::Relaxed) <= 256 * PARTICLES,
        "DPD initialization must not allocate a quadratic table over distinct types"
    );
    assert_eq!(
        typed_context
            .get_state(openmm_core::StateFlags::none())
            .step_count,
        0
    );
}
