use burn::backend::NdArray;
use openmm_core::{
    tensor::from_floats, ConstantPotentialForce, ConstantPotentialMethod, Context, StateFlags,
    System, VerletIntegrator,
};
use std::{hint::black_box, time::Instant};
fn main() {
    type B = NdArray<f64>;
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args[1].parse().unwrap();
    let iterations: usize = args[2].parse().unwrap();
    let solver: usize = args[3].parse().unwrap();
    let grid = ((n as f64).cbrt().ceil() as usize).max(5);
    let side = 0.32 * grid as f64;
    let mesh = ((6. * side).ceil() as usize).next_power_of_two().max(16);
    let k = n / 4;
    let setup_start = Instant::now();
    let mut system = System::<B>::new();
    system.set_default_periodic_box_vectors([[side, 0., 0.], [0., side, 0.], [0., 0., side]]);
    let mut force = ConstantPotentialForce::new();
    force.set_constant_potential_method(if solver == 2 {
        ConstantPotentialMethod::Matrix
    } else {
        ConstantPotentialMethod::CG
    });
    force.set_use_preconditioner(solver == 1);
    force.set_cg_error_tolerance(1e-8);
    force.set_cutoff_distance(0.8);
    force.set_pme_parameters(3., [mesh; 3]);
    force.set_use_charge_constraint(true);
    force.set_charge_constraint_target(0.25);
    force.set_external_field([0.1, -0.05, 0.08]);
    force.add_electrode((0..k).step_by(2), 2., 0.08, 0.005);
    force.add_electrode((1..k).step_by(2), -2., 0.10, 0.005);
    let mut base = Vec::with_capacity(3 * n);
    for i in 0..n {
        system.add_particle(if i < k { 0. } else { 12. });
        force.add_particle(if i < k {
            0.
        } else if i % 2 == 0 {
            0.02
        } else {
            -0.02
        });
        base.extend_from_slice(&[
            0.05 + 0.32 * (i % grid) as f64 + 0.015 * (1.7 * i as f64).sin(),
            0.05 + 0.32 * ((i / grid) % grid) as f64 + 0.017 * (0.9 * i as f64).cos(),
            0.05 + 0.32 * (i / (grid * grid)) as f64 + 0.013 * (1.3 * i as f64).sin(),
        ]);
    }
    system.add_force(Box::new(force));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context
        .set_positions(from_floats::<B, 1>(base.as_slice(), &Default::default()).reshape([n, 3]));
    let flags = StateFlags {
        forces: true,
        energy: true,
        ..StateFlags::none()
    };
    let first_start = Instant::now();
    black_box(context.get_state(flags));
    let first_seconds = first_start.elapsed().as_secs_f64();
    let setup_seconds = setup_start.elapsed().as_secs_f64();
    let mut points = base.clone();
    let start = Instant::now();
    for iteration in 0..iterations {
        for i in k..n {
            points[3 * i] =
                base[3 * i] + 0.002 * (0.37 * (iteration + 1) as f64 + 0.11 * i as f64).sin();
            points[3 * i + 1] =
                base[3 * i + 1] + 0.001 * (0.23 * (iteration + 1) as f64 + 0.07 * i as f64).cos();
        }
        context.set_positions(
            from_floats::<B, 1>(points.as_slice(), &Default::default()).reshape([n, 3]),
        );
        black_box(context.get_state(flags));
    }
    let seconds = start.elapsed().as_secs_f64();
    let state = context.get_state(flags);
    let norm = state
        .forces
        .unwrap()
        .to_data()
        .to_vec::<f64>()
        .unwrap()
        .iter()
        .map(|f| f * f)
        .sum::<f64>();
    let charges = context.constant_potential_charges(&[0]);
    let charge_norm = charges.iter().map(|q| q * q).sum::<f64>();
    let total_charge = charges.iter().sum::<f64>();
    println!("{{\"implementation\":\"rust\",\"precision\":\"f64\",\"api\":\"Context.setPositions/getState\",\"force\":\"constant_potential\",\"solver\":{solver},\"particles\":{n},\"electrode_particles\":{k},\"mesh\":{mesh},\"iterations\":{iterations},\"setup_seconds\":{setup_seconds},\"first_evaluation_seconds\":{first_seconds},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":{},\"force_squared_norm\":{norm},\"charge_squared_norm\":{charge_norm},\"total_charge\":{total_charge}}}",iterations as f64/seconds,state.potential_energy.unwrap());
}
