use burn::backend::NdArray;
use burn::tensor::TensorData;
use openmm_core::{
    tensor::from_floats, Context, CustomCVForce, Force, RGForce, StateFlags, System,
    VerletIntegrator,
};
use std::alloc::{GlobalAlloc, Layout, System as SystemAllocator};
use std::sync::atomic::{AtomicUsize, Ordering};

const PARTICLES: usize = 4096;
const FORCE_BYTES: usize = 3 * PARTICLES * std::mem::size_of::<f64>();
static LARGE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { SystemAllocator.alloc(layout) };
        if !pointer.is_null() && layout.size() >= FORCE_BYTES {
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
fn rg_collective_variables_do_not_allocate_a_dense_force_buffer_per_variable() {
    type B = NdArray<f64>;
    let positions = from_floats::<B, 2>(
        TensorData::new(
            (0..3 * PARTICLES)
                .map(|i| (i as f64 * 0.1).sin())
                .collect::<Vec<_>>(),
            [PARTICLES, 3],
        ),
        &Default::default(),
    );
    let names: Vec<_> = (0..32).map(|i| format!("v{i}^2")).collect();
    let mut force = CustomCVForce::new(&names.join("+")).unwrap();
    for i in 0..32 {
        force.add_collective_variable(
            &format!("v{i}"),
            Box::new(RGForce::new((i..PARTICLES).step_by(32).collect())),
        );
    }
    Force::<B>::initialize(&mut force, PARTICLES, None);
    for bytes in [0, 16 * 1024 * 1024, usize::MAX] {
        force.set_force_cache_bytes(bytes);
        drop(force.calc_forces_and_energy(&positions, None));
        let before = LARGE_ALLOCATIONS.load(Ordering::Relaxed);
        let (forces, energy) = force.calc_forces_and_energy(&positions, None);
        let allocations = LARGE_ALLOCATIONS.load(Ordering::Relaxed) - before;
        assert_eq!(forces.dims(), [PARTICLES, 3]);
        assert!(energy.is_finite());
        assert!(allocations <= 4, "allocated {allocations} dense buffers");
    }
    let mut system = System::<B>::new();
    for _ in 0..PARTICLES {
        system.add_particle(1.0);
    }
    system.add_force(Box::new(force));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.set_positions(positions);
    let before = LARGE_ALLOCATIONS.load(Ordering::Relaxed);
    let state = context.get_state(StateFlags {
        positions: true,
        velocities: true,
        ..StateFlags::none()
    });
    assert_eq!(
        LARGE_ALLOCATIONS.load(Ordering::Relaxed) - before,
        0,
        "state snapshots must share initial CPU storage"
    );
    assert_eq!(state.velocities.unwrap().dims(), [PARTICLES, 3]);
    let before = LARGE_ALLOCATIONS.load(Ordering::Relaxed);
    let state = context.get_state(StateFlags {
        forces: true,
        energy: true,
        ..StateFlags::none()
    });
    let allocations = LARGE_ALLOCATIONS.load(Ordering::Relaxed) - before;
    assert!(
        allocations <= 2,
        "Context evaluation allocated {allocations} dense buffers"
    );
    assert!(state.kinetic_energy.unwrap().is_finite());
}
