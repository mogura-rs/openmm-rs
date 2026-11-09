use burn::backend::NdArray;
use openmm_core::{
    tensor::from_floats, ATMForce, ATMTransformation, CustomManyParticleForce, Force,
};
use std::alloc::{GlobalAlloc, Layout, System as SystemAllocator};
use std::sync::atomic::{AtomicUsize, Ordering};
const PARTICLES: usize = 4096;
const FORCE_BYTES: usize = 24 * PARTICLES;
static DENSE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static MAX_ALLOCATION: AtomicUsize = AtomicUsize::new(0);
struct TrackingAllocator;
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { SystemAllocator.alloc(layout) };
        if !p.is_null() {
            MAX_ALLOCATION.fetch_max(layout.size(), Ordering::Relaxed);
            if layout.size() >= FORCE_BYTES {
                DENSE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            }
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { SystemAllocator.dealloc(p, layout) };
    }
}
#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn atm_reuses_two_state_forces_and_one_transformed_position_buffer() {
    type B = NdArray<f64>;
    let coordinates: Vec<_> = (0..PARTICLES)
        .flat_map(|i| [0.01 * i as f64, 0.2, 0.3])
        .collect();
    let positions =
        from_floats::<B, 1>(coordinates.as_slice(), &Default::default()).reshape([PARTICLES, 3]);
    let mut child = CustomManyParticleForce::new(1, "q1*(x1*x1+2*y1*y1+3*z1*z1)").unwrap();
    child.add_per_particle_parameter("q");
    let mut force = ATMForce::<B>::new("0.4*u0+0.6*u1").unwrap();
    for i in 0..PARTICLES {
        child.add_particle(vec![0.1 + 0.01 * (i % 5) as f64], 0);
        force.add_particle(if i % 2 == 0 {
            ATMTransformation::fixed([0.1, 0.2, 0.3], [-0.1, 0.1, 0.])
        } else {
            ATMTransformation::particle_offset(
                (i + 1) % PARTICLES,
                (i + 2) % PARTICLES,
                Some([(i + 2) % PARTICLES, (i + 3) % PARTICLES]),
            )
        });
    }
    force.add_force(Box::new(child));
    force.initialize(PARTICLES, None);
    MAX_ALLOCATION.store(0, Ordering::Relaxed);
    drop(force.calc_forces_and_energy(&positions, None));
    assert!(
        MAX_ALLOCATION.load(Ordering::Relaxed) <= 64 * PARTICLES,
        "ATM workspace storage must remain linear"
    );
    let before = DENSE_ALLOCATIONS.load(Ordering::Relaxed);
    let (expected, energy) = force.calc_forces_and_energy(&positions, None);
    assert_eq!(
        DENSE_ALLOCATIONS.load(Ordering::Relaxed) - before,
        1,
        "only the returned force buffer may allocate"
    );
    assert!(energy.is_finite());
    let expected = expected.to_data().to_vec::<f64>().unwrap();
    let mut output = vec![0.; 3 * PARTICLES];
    let before = DENSE_ALLOCATIONS.load(Ordering::Relaxed);
    force
        .prepare_host_evaluation(&positions, None)
        .unwrap()
        .accumulate(0.3, &mut output);
    assert_eq!(
        DENSE_ALLOCATIONS.load(Ordering::Relaxed) - before,
        0,
        "host accumulation must reuse both states and transformed coordinates"
    );
    for (got, value) in output.into_iter().zip(expected) {
        assert!((got - 0.3 * value).abs() < 1e-12);
    }
}
