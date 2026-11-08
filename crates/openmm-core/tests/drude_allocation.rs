use burn::backend::NdArray;
use openmm_core::{tensor::from_floats, DrudeForce, DrudeParticle, Force};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
const N: usize = 8192;
static LARGE: AtomicUsize = AtomicUsize::new(0);
struct Allocator;
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() && layout.size() >= 24 * N {
            LARGE.fetch_add(1, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
#[test]
fn drude_host_accumulation_reuses_the_force_buffer() {
    type B = NdArray<f64>;
    let mut force = DrudeForce::new();
    let mut points = Vec::with_capacity(3 * N);
    for i in 0..N / 2 {
        force
            .add_particle(DrudeParticle {
                particle: 2 * i + 1,
                parent: 2 * i,
                axis2: None,
                axis3: None,
                axis4: None,
                charge: -0.1,
                polarizability: 0.001,
                aniso12: 1.,
                aniso34: 1.,
            })
            .unwrap();
        points.extend_from_slice(&[0.3 * i as f64, 0., 0., 0.3 * i as f64 + 0.01, 0.02, -0.01]);
        if i > 0 {
            force.add_screened_pair(i - 1, i, 1.7);
        }
    }
    <DrudeForce as Force<B>>::initialize(&mut force, N, None);
    let points = from_floats::<B, 1>(points.as_slice(), &Default::default()).reshape([N, 3]);
    let mut output = vec![0.; 3 * N];
    force
        .prepare_host_evaluation(&points, None)
        .unwrap()
        .accumulate(1., &mut output);
    let before = LARGE.load(Ordering::Relaxed);
    output.fill(0.);
    force
        .prepare_host_evaluation(&points, None)
        .unwrap()
        .accumulate(1., &mut output);
    assert_eq!(LARGE.load(Ordering::Relaxed) - before, 0);
    let before = LARGE.load(Ordering::Relaxed);
    let (values, energy) = force.calc_forces_and_energy(&points, None);
    assert_eq!(LARGE.load(Ordering::Relaxed) - before, 1);
    assert!(energy.is_finite());
    assert_eq!(values.into_data().to_vec::<f64>().unwrap(), output);
}
