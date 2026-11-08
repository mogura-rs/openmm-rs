use burn::{backend::NdArray, tensor::TensorData};
use openmm_core::{
    Context, Force, HarmonicBondForce, MonteCarloAnisotropicBarostat, MonteCarloBarostat,
    StateFlags, System, VerletIntegrator,
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
fn volume_moves_reuse_dense_storage_and_preserve_snapshots() {
    for (rigid, anisotropic) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut system = System::<NdArray<f64>>::new();
        for _ in 0..PARTICLES {
            system.add_particle(1.0);
        }
        let mut bonds = HarmonicBondForce::new();
        for atom in (0..PARTICLES).step_by(2) {
            bonds.add_bond(atom, atom + 1, 1.0, 0.0);
        }
        bonds.set_uses_periodic_boundary_conditions(true);
        system.add_force(Box::new(bonds));
        let mut barostat = MonteCarloBarostat::new(1.0, 300.0, 1);
        barostat.set_scale_molecules_as_rigid(rigid);
        barostat.set_random_number_seed(47);
        if anisotropic {
            let mut barostat = MonteCarloAnisotropicBarostat::new([1.0; 3], 300.0, [true; 3], 1);
            barostat.set_scale_molecules_as_rigid(rigid);
            barostat.set_random_number_seed(47);
            system.add_force(Box::new(barostat));
        } else {
            system.add_force(Box::new(barostat));
        }
        let mut integrator = VerletIntegrator::new(0.001);
        integrator.set_integration_force_groups(0);
        let mut context = Context::new(system, Box::new(integrator), Default::default());
        let values: Vec<_> = (0..PARTICLES)
            .flat_map(|atom| [atom as f64 * 0.0001, 0.1, 0.2])
            .collect();
        context.set_positions(openmm_core::tensor::from_floats(
            TensorData::new(values, [PARTICLES, 3]),
            &Default::default(),
        ));
        context.step(5);
        let before = LARGE_ALLOCATIONS.load(Ordering::Relaxed);
        context.step(20);
        assert_eq!(
            LARGE_ALLOCATIONS.load(Ordering::Relaxed) - before,
            0,
            "volume proposals must reuse dense position buffers"
        );
        let snapshot = context.get_state(StateFlags {
            positions: true,
            ..StateFlags::none()
        });
        let expected = snapshot
            .positions
            .as_ref()
            .unwrap()
            .to_data()
            .to_vec::<f64>()
            .unwrap();
        let before = LARGE_ALLOCATIONS.load(Ordering::Relaxed);
        context.step(20);
        assert!(
            LARGE_ALLOCATIONS.load(Ordering::Relaxed) - before <= 1,
            "only the retained snapshot may require a copy-on-write allocation"
        );
        assert_eq!(
            snapshot
                .positions
                .unwrap()
                .to_data()
                .to_vec::<f64>()
                .unwrap(),
            expected
        );
        assert_ne!(
            context
                .get_state(StateFlags {
                    positions: true,
                    ..StateFlags::none()
                })
                .positions
                .unwrap()
                .to_data()
                .to_vec::<f64>()
                .unwrap(),
            expected
        );
    }
    let mut bonds = HarmonicBondForce::new();
    for atom in (0..PARTICLES).step_by(2) {
        bonds.add_bond(atom, atom + 1, 0.09, 10.0);
    }
    let coordinates: Vec<_> = (0..PARTICLES)
        .flat_map(|atom| [atom as f64 * 0.1, 0.0, 0.0])
        .collect();
    let positions = openmm_core::tensor::from_floats::<NdArray<f64>, 2>(
        TensorData::new(coordinates, [PARTICLES, 3]),
        &Default::default(),
    );
    let before = LARGE_ALLOCATIONS.load(Ordering::Relaxed);
    let (forces, energy) = bonds.calc_forces_and_energy(&positions, None);
    assert_eq!(
        LARGE_ALLOCATIONS.load(Ordering::Relaxed) - before,
        1,
        "bond forces must borrow coordinates and allocate only the output array"
    );
    assert_eq!(forces.dims(), [PARTICLES, 3]);
    assert!((energy - 0.0005 * (PARTICLES / 2) as f64).abs() < 1e-10);
}
