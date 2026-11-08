use burn::backend::NdArray;
use openmm_core::{
    tensor::from_floats, Context, GayBerneForce, GayBerneMethod, GayBerneParticle, StateFlags,
    System, VerletIntegrator,
};
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args[1].parse().unwrap();
    let iterations: usize = args[2].parse().unwrap();
    let method: usize = args[3].parse().unwrap();
    let model: usize = args[4].parse().unwrap();
    let stride = model + 1;
    assert_eq!(n % stride, 0);
    let grid = ((n / stride) as f64).cbrt().ceil() as usize;
    let side = 0.45 * grid as f64;
    let mut system = System::<NdArray<f64>>::new();
    system.set_default_periodic_box_vectors([[side, 0., 0.], [0., side, 0.], [0., 0., side]]);
    let mut gb = GayBerneForce::new();
    gb.set_nonbonded_method(
        [
            GayBerneMethod::NoCutoff,
            GayBerneMethod::CutoffNonPeriodic,
            GayBerneMethod::CutoffPeriodic,
        ][method],
    );
    gb.set_cutoff_distance(0.9);
    gb.set_switching_distance(0.7);
    gb.set_use_switching_function(method != 0);
    let mut points = Vec::with_capacity(3 * n);
    for i in 0..n / stride {
        let center = [
            0.45 * (i % grid) as f64,
            0.45 * ((i / grid) % grid) as f64,
            0.45 * (i / (grid * grid)) as f64,
        ];
        system.add_particle(1.);
        points.extend_from_slice(&center);
        gb.add_particle(GayBerneParticle {
            sigma: 0.26 + 0.005 * (i % 3) as f64,
            epsilon: 0.7 + 0.1 * (i % 5) as f64,
            xparticle: (model == 1).then_some(stride * i + 1),
            yparticle: None,
            widths: if model == 0 {
                [0.25 + 0.01 * (i % 4) as f64; 3]
            } else {
                [0.32, 0.24, 0.24]
            },
            scales: [1.; 3],
        })
        .unwrap();
        if model == 1 {
            system.add_particle(1.);
            for axis in 0..3 {
                points.push(center[axis] + [0.08, 0.02, 0.03][axis]);
            }
            gb.add_particle(GayBerneParticle {
                sigma: 0.1,
                epsilon: 0.,
                xparticle: None,
                yparticle: None,
                widths: [0.1; 3],
                scales: [1.; 3],
            })
            .unwrap();
        }
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
    println!("{{\"implementation\":\"rust\",\"precision\":\"f64\",\"api\":\"Context.getState\",\"force\":\"gay_berne\",\"model\":{model},\"method\":{method},\"particles\":{n},\"iterations\":{iterations},\"seconds\":{seconds},\"evaluations_per_second\":{},\"energy\":{},\"force_squared_norm\":{norm}}}",iterations as f64/seconds,state.potential_energy.unwrap());
}
