use burn::backend::NdArray;
use openmm_core::{
    tensor::from_floats, Context, Force, OrientationRestraintForce, RGForce, RMSDForce, StateFlags,
    System, VerletIntegrator,
};
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n = args
        .get(1)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(512);
    let iterations = args
        .get(2)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(1000);
    let kind = args.get(3).map(String::as_str).unwrap_or("rmsd");
    let api = args.get(4).map(String::as_str).unwrap_or("context");
    assert!(matches!(api, "context" | "kernel" | "forces"));
    assert!(n >= 4 && iterations > 0);
    let mut reference = Vec::with_capacity(n);
    let mut positions = Vec::with_capacity(3 * n);
    let (s, c) = 0.7_f64.sin_cos();
    let mut system = System::<NdArray<f64>>::new();
    for i in 0..n {
        let i = i as f64;
        let p = [
            2.0 * (0.17 * i).sin(),
            (0.31 * i).cos(),
            0.7 * (0.73 * i).sin(),
        ];
        reference.push(p);
        positions.extend_from_slice(&[
            c * p[0] - s * p[1] + 0.3,
            s * p[0] + c * p[1] - 0.4,
            p[2] + 0.03 * (0.11 * i).cos(),
        ]);
        system.add_particle(1.0);
    }
    let force: Box<dyn Force<NdArray<f64>>> = match kind {
        "rmsd" => Box::new(RMSDForce::new(reference, vec![])),
        "orientation" => Box::new(OrientationRestraintForce::new(1.7, reference, vec![])),
        "rg" => Box::new(RGForce::new(vec![])),
        _ => panic!("unknown collective force"),
    };
    system.add_force(force);
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.set_positions(
        from_floats::<NdArray<f64>, 1>(positions.as_slice(), &Default::default()).reshape([n, 3]),
    );
    let flags = StateFlags {
        forces: true,
        energy: api != "forces",
        ..StateFlags::none()
    };
    let evaluate = || {
        if api == "kernel" {
            let state = context.get_state(StateFlags {
                positions: true,
                ..StateFlags::none()
            });
            black_box(
                context.system().forces()[0]
                    .calc_forces_and_energy(state.positions.as_ref().unwrap(), None),
            )
            .1
        } else {
            black_box(context.get_state(flags))
                .potential_energy
                .unwrap_or(0.0)
        }
    };
    evaluate();
    let start = Instant::now();
    let mut energy = 0.0;
    for _ in 0..iterations {
        energy += evaluate();
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("{{\"implementation\":\"rust\",\"precision\":\"f64\",\"api\":\"{api}\",\"force\":\"{kind}\",\"particles\":{n},\"iterations\":{iterations},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":{}}}",iterations as f64/seconds,energy/iterations as f64);
}
