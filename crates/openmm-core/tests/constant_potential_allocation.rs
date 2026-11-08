use burn::backend::NdArray;
use openmm_core::{tensor::from_floats, ConstantPotentialForce, ConstantPotentialMethod, Force};
use std::alloc::{GlobalAlloc, Layout, System as SystemAllocator};
use std::sync::atomic::{AtomicUsize, Ordering};
const PARTICLES: usize = 512;
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
fn charge_solvers_reuse_workspace_and_pack_the_symmetric_matrix() {
    type B = NdArray<f64>;
    let vectors = [[3.4, 0., 0.], [0., 3.4, 0.], [0., 0., 3.4]];
    for method in [ConstantPotentialMethod::CG, ConstantPotentialMethod::Matrix] {
        let n = if method == ConstantPotentialMethod::CG {
            PARTICLES
        } else {
            256
        };
        let coordinates: Vec<_> = (0..n)
            .flat_map(|i| {
                [
                    0.1 + 0.35 * (i % 8) as f64 + 0.002 * (i as f64).sin(),
                    0.1 + 0.35 * ((i / 8) % 8) as f64 + 0.002 * (0.7 * i as f64).cos(),
                    0.1 + 0.35 * (i / 64) as f64,
                ]
            })
            .collect();
        let positions =
            from_floats::<B, 1>(coordinates.as_slice(), &Default::default()).reshape([n, 3]);
        let mut force = ConstantPotentialForce::new();
        for _ in 0..n {
            force.add_particle(0.);
        }
        force.add_electrode((0..n).step_by(2), 0.3, 0.06, 0.01);
        force.add_electrode((1..n).step_by(2), -0.3, 0.06, 0.01);
        force.set_constant_potential_method(method);
        force.set_cutoff_distance(0.55);
        force.set_pme_parameters(2.4, [8, 8, 8]);
        force.set_cg_error_tolerance(1e-8);
        force.set_use_charge_constraint(true);
        <ConstantPotentialForce as Force<B>>::initialize_masses(&mut force, &vec![0.; n]);
        <ConstantPotentialForce as Force<B>>::initialize(&mut force, n, Some(&vectors));
        MAX_ALLOCATION.store(0, Ordering::Relaxed);
        let mut output = vec![0.; 3 * n];
        force
            .prepare_host_evaluation(&positions, Some(&vectors))
            .unwrap()
            .accumulate(1., &mut output);
        let largest = MAX_ALLOCATION.load(Ordering::Relaxed);
        if method == ConstantPotentialMethod::Matrix {
            assert!(
                largest <= 8 * n * (n + 1) / 2,
                "matrix storage must be packed: {largest}"
            );
        } else {
            assert!(
                largest < 4 * n * n,
                "CG must not allocate a dense electrode matrix: {largest}"
            );
        }
        let before = DENSE_ALLOCATIONS.load(Ordering::Relaxed);
        let (_, energy) = force.calc_forces_and_energy(&positions, Some(&vectors));
        let expected = usize::from(24 * n >= FORCE_BYTES);
        assert_eq!(DENSE_ALLOCATIONS.load(Ordering::Relaxed) - before, expected);
        assert!(energy.is_finite());
        let before = DENSE_ALLOCATIONS.load(Ordering::Relaxed);
        output.fill(0.);
        force
            .prepare_host_evaluation(&positions, Some(&vectors))
            .unwrap()
            .accumulate(1., &mut output);
        assert_eq!(DENSE_ALLOCATIONS.load(Ordering::Relaxed) - before, 0);
    }
}
