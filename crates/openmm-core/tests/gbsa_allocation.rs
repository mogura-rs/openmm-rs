use burn::backend::NdArray;
use openmm_core::{tensor::from_floats, Force, GBSAOBCForce, GBSAOBCMethod};
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
fn gbsa_reuses_linear_scratch_and_transfers_force_output() {
    type B = NdArray<f64>;
    let coordinates: Vec<_> = (0..PARTICLES)
        .flat_map(|i| {
            [
                0.3 * (i % 16) as f64,
                0.3 * ((i / 16) % 16) as f64,
                0.3 * (i / 256) as f64,
            ]
        })
        .collect();
    let positions =
        from_floats::<B, 1>(coordinates.as_slice(), &Default::default()).reshape([PARTICLES, 3]);
    let vectors = [[4.8, 0., 0.], [0., 4.8, 0.], [0., 0., 4.8]];
    let mut force = GBSAOBCForce::new();
    force.set_nonbonded_method(GBSAOBCMethod::CutoffPeriodic);
    force.set_cutoff_distance(0.5);
    for i in 0..PARTICLES {
        force.add_particle(if i % 2 == 0 { 0.3 } else { -0.3 }, 0.15, 0.8);
    }
    <GBSAOBCForce as Force<B>>::initialize(&mut force, PARTICLES, Some(&vectors));
    MAX_ALLOCATION.store(0, Ordering::Relaxed);
    drop(force.calc_forces_and_energy(&positions, Some(&vectors)));
    assert!(
        MAX_ALLOCATION.load(Ordering::Relaxed) <= 32 * PARTICLES,
        "GBSA must not allocate a pair matrix"
    );
    let before = DENSE_ALLOCATIONS.load(Ordering::Relaxed);
    let (f, energy) = force.calc_forces_and_energy(&positions, Some(&vectors));
    assert_eq!(
        DENSE_ALLOCATIONS.load(Ordering::Relaxed) - before,
        1,
        "only the returned force buffer may allocate"
    );
    assert!(energy.is_finite());
    assert_eq!(f.dims(), [PARTICLES, 3]);
    let mut output = vec![0.; 3 * PARTICLES];
    force
        .prepare_host_evaluation(&positions, Some(&vectors))
        .unwrap()
        .accumulate(1., &mut output);
    let before = DENSE_ALLOCATIONS.load(Ordering::Relaxed);
    let prepared = force
        .prepare_host_evaluation(&positions, Some(&vectors))
        .unwrap();
    assert_eq!(prepared.energy, energy);
    prepared.accumulate(0.5, &mut output);
    assert_eq!(
        DENSE_ALLOCATIONS.load(Ordering::Relaxed) - before,
        0,
        "host force accumulation must reuse dense scratch"
    );
}
