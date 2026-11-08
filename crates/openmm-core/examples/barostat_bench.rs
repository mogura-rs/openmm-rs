use burn::{backend::NdArray, tensor::TensorData};
use openmm_core::{
    tensor::from_floats, Context, HarmonicBondForce, MonteCarloAnisotropicBarostat,
    MonteCarloBarostat, StateFlags, System, VerletIntegrator,
};
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(4096);
    let iterations: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let rigid = args.get(3).is_none_or(|s| s == "rigid");
    let kind = args.get(4).map(String::as_str).unwrap_or("isotropic");
    assert!(matches!(kind, "isotropic" | "anisotropic"));
    assert!(n >= 2 && n.is_multiple_of(2) && iterations > 0);
    let scaled = if rigid { n / 2 } else { n };
    let kt = 1.380649e-23 * 6.02214076e23 / 1000.0 * 300.0;
    let thermal_volume = kt / (20.0 * 0.0602214076);
    let expected_volume = (scaled + 1) as f64 * thermal_volume;
    let volume_stddev = ((scaled + 1) as f64).sqrt() * thermal_volume;
    let length = expected_volume.cbrt();
    let mut system = System::<NdArray<f64>>::new();
    for _ in 0..n {
        system.add_particle(1.0);
    }
    system.set_default_periodic_box_vectors([
        [length, 0.0, 0.0],
        [0.0, length, 0.0],
        [0.0, 0.0, length],
    ]);
    let mut bond = HarmonicBondForce::new();
    for i in (0..n).step_by(2) {
        bond.add_bond(i, i + 1, 0.02, 0.0);
    }
    bond.set_uses_periodic_boundary_conditions(true);
    system.add_force(Box::new(bond));
    if kind == "anisotropic" {
        let mut barostat =
            MonteCarloAnisotropicBarostat::new([20.0; 3], 300.0, [true, false, false], 1);
        barostat.set_scale_molecules_as_rigid(rigid);
        barostat.set_random_number_seed(3721);
        system.add_force(Box::new(barostat));
    } else {
        let mut barostat = MonteCarloBarostat::new(20.0, 300.0, 1);
        barostat.set_scale_molecules_as_rigid(rigid);
        barostat.set_random_number_seed(3721);
        system.add_force(Box::new(barostat));
    }
    let mut integrator = VerletIntegrator::new(0.001);
    integrator.set_integration_force_groups(0);
    let mut context = Context::new(system, Box::new(integrator), Default::default());
    let side = ((n / 2) as f64).cbrt().ceil() as usize;
    let mut positions = Vec::with_capacity(3 * n);
    for i in 0..n {
        let cell = i / 2;
        positions.extend([
            (cell % side) as f64 * 0.1 + (i % 2) as f64 * 0.02,
            ((cell / side) % side) as f64 * 0.1,
            (cell / (side * side)) as f64 * 0.1,
        ]);
    }
    context.set_positions(from_floats(
        TensorData::new(positions, [n, 3]),
        &Default::default(),
    ));
    context.step(100);
    let start = Instant::now();
    context.step(iterations);
    let seconds = start.elapsed().as_secs_f64();
    let kinetic = black_box(context.get_state(StateFlags {
        energy: true,
        ..StateFlags::none()
    }))
    .kinetic_energy
    .unwrap();
    let vectors = black_box(context.periodic_box_vectors().unwrap());
    let volume = vectors[0][0] * vectors[1][1] * vectors[2][2];
    println!("{{\"implementation\":\"rust\",\"precision\":\"f64\",\"api\":\"Context.step\",\"barostat\":true,\"barostat_kind\":\"{kind}\",\"rigid\":{rigid},\"particles\":{n},\"iterations\":{iterations},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":0,\"kinetic_energy\":{kinetic},\"expected_kinetic_energy\":0,\"kinetic_energy_stddev\":0,\"volume\":{volume},\"expected_volume\":{expected_volume},\"volume_stddev\":{volume_stddev}}}",iterations as f64/seconds);
}
