use burn::{backend::NdArray, prelude::*};
use openmm_core::{CustomNonbondedForce, Force};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct TrackingAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe {
            System.dealloc(pointer, layout);
        }
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn repeated_custom_nonbonded_evaluations_do_not_retain_pair_names() {
    type B = NdArray;
    let mut force = CustomNonbondedForce::new("q1*q2/r").unwrap();
    force.add_per_particle_parameter("q");
    let mut positions = Vec::new();
    for i in 0..16 {
        force.add_particle(vec![0.1]);
        positions.extend_from_slice(&[0.2 * i as f32, 0.0, 0.0]);
    }
    let positions =
        Tensor::<B, 1>::from_floats(positions.as_slice(), &Default::default()).reshape([16, 3]);
    for _ in 0..5 {
        drop(Force::<B>::calc_forces_and_energy(&force, &positions, None));
    }
    let before = LIVE.load(Ordering::Relaxed);
    for _ in 0..100 {
        drop(Force::<B>::calc_forces_and_energy(&force, &positions, None));
    }
    let retained = LIVE.load(Ordering::Relaxed).saturating_sub(before);
    assert!(
        retained < 4096,
        "retained {retained} bytes after repeated force evaluation"
    );
}
