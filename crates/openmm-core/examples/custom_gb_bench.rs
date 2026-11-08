use burn::backend::NdArray;
use openmm_core::{
    tensor::from_floats, Context, CustomGBForce, CustomGBMethod, GBComputationType as C,
    StateFlags, System, VerletIntegrator,
};
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args[1].parse().unwrap();
    let iterations: usize = args[2].parse().unwrap();
    let method: u32 = args[3].parse().unwrap();
    let kind: u32 = args[4].parse().unwrap();
    let grid = (n as f64).cbrt().ceil() as usize;
    let side = 0.3 * grid as f64;
    let mut system = System::<NdArray<f64>>::new();
    system.set_default_periodic_box_vectors([[side, 0., 0.], [0., side, 0.], [0., 0., side]]);
    let mut gb = model(kind);
    gb.set_nonbonded_method(match method {
        0 => CustomGBMethod::NoCutoff,
        1 => CustomGBMethod::CutoffNonPeriodic,
        2 => CustomGBMethod::CutoffPeriodic,
        _ => panic!("invalid GBSA method"),
    });
    gb.set_cutoff_distance(0.8);
    let mut points = Vec::with_capacity(3 * n);
    for i in 0..n {
        system.add_particle(1.);
        gb.add_particle(if kind == 0 {
            vec![
                0.1 * ((i % 7) as f64 - 3.),
                0.11 + 0.02 * (i % 5) as f64,
                0.65 + 0.1 * (i % 4) as f64,
            ]
        } else {
            vec![0.01 * (i % 5) as f64]
        });
        points.extend_from_slice(&[
            0.3 * (i % grid) as f64,
            0.3 * ((i / grid) % grid) as f64,
            0.3 * (i / (grid * grid)) as f64,
        ]);
    }
    system.add_force(Box::new(gb));
    let mut context = Context::new(
        system,
        Box::new(VerletIntegrator::new(0.001)),
        Default::default(),
    );
    context.set_positions(
        from_floats::<NdArray<f64>, 1>(points.as_slice(), &Default::default()).reshape([n, 3]),
    );
    let flags = StateFlags {
        forces: true,
        energy: true,
        ..StateFlags::none()
    };
    black_box(context.get_state(flags));
    let start = Instant::now();
    for _ in 0..iterations {
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
    let derivative_state = context.get_state(StateFlags {
        parameter_derivatives: true,
        ..StateFlags::none()
    });
    let derivative_norm = derivative_state
        .energy_parameter_derivatives
        .unwrap()
        .values()
        .map(|v| v * v)
        .sum::<f64>();
    println!("{{\"implementation\":\"rust\",\"precision\":\"f64\",\"api\":\"Context.getState\",\"force\":\"custom_gb\",\"method\":{method},\"model\":{kind},\"derivative_squared_norm\":{derivative_norm},\"particles\":{n},\"iterations\":{iterations},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":{},\"force_squared_norm\":{norm}}}",iterations as f64/seconds,state.potential_energy.unwrap());
}

fn model(kind: u32) -> CustomGBForce {
    let mut f = CustomGBForce::new();
    if kind == 0 {
        for name in ["q", "radius", "scale"] {
            f.add_per_particle_parameter(name);
        }
        f.add_global_parameter("solvent", 78.3);
        f.add_global_parameter("solute", 1.);
        f.add_energy_parameter_derivative("solvent");
        f.add_energy_parameter_derivative("solute");
        f.add_computed_value("I","step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);U=r+sr2;C=2*(1/or1-1/L)*step(sr2-r-or1);L=max(or1,D);D=abs(r-sr2);sr2=scale2*or2;or1=radius1-0.009;or2=radius2-0.009",C::ParticlePairNoExclusions).unwrap();
        f.add_computed_value(
            "B",
            "1/(1/or-tanh(psi-0.8*psi^2+4.85*psi^3)/radius);psi=I*or;or=radius-0.009",
            C::SingleParticle,
        )
        .unwrap();
        f.add_energy_term("28.391955111572114*(radius+0.14)^2*(radius/B)^6-0.5*138.93545764438198*(1/solute-1/solvent)*q^2/B",C::SingleParticle).unwrap();
        f.add_energy_term(
            "-138.93545764438198*(1/solute-1/solvent)*q1*q2/sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))",
            C::ParticlePair,
        )
        .unwrap();
    } else {
        f.add_per_particle_parameter("p");
        let mut names = Vec::new();
        for i in 0..24 {
            let name = format!("g{i}");
            f.add_global_parameter(&name, 0.01 * (i + 1) as f64);
            f.add_energy_parameter_derivative(&name);
            names.push(name);
        }
        f.add_computed_value(
            "a",
            &format!("({})*exp(-r)*p2", names.join("+")),
            C::ParticlePair,
        )
        .unwrap();
        for (name, expr) in [
            ("b", "1+0.1*a*a"),
            ("c", "log(b)+g0*a"),
            ("d", "c+0.01*c*c+g1*b"),
            ("e", "sqrt(1+d*d)+g2*a"),
            ("f", "0.5*e+0.03*b"),
        ] {
            f.add_computed_value(name, expr, C::SingleParticle).unwrap();
        }
        f.add_energy_term("0.01*f*f", C::SingleParticle).unwrap();
        f.add_energy_term("0.001*(c1+c2)*exp(-r)", C::ParticlePair)
            .unwrap();
    }
    f
}
