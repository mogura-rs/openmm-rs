use burn::{backend::NdArray, prelude::*};
use openmm_core::{
    Context, CustomExternalForce, LocalEnergyMinimizer, MinimizationProgress, MinimizationReporter,
    MinimizationTermination, StateFlags, System, VerletIntegrator,
};

fn context(mass: f64) -> Context<NdArray> {
    let mut system = System::new();
    system.add_particle(mass);
    let mut force = CustomExternalForce::new("(x-2)^2+y*y+z*z").unwrap();
    force.add_particle(0, vec![]);
    system.add_force(Box::new(force));
    let mut context =
        system.create_context(Box::new(VerletIntegrator::new(0.001)), Default::default());
    context.set_positions(Tensor::from_floats([[10.0, 0.0, 0.0]], &Default::default()));
    context.set_velocities(Tensor::from_floats([[0.4, 0.0, 0.0]], &Default::default()));
    context.set_time(3.0);
    context.set_step_count(7);
    context
}

#[test]
fn minimization_converges_without_changing_velocities_or_clock() {
    let mut context = context(1.0);
    let result = LocalEnergyMinimizer::minimize(&mut context, 1e-5, 100).unwrap();
    assert_eq!(result.termination, MinimizationTermination::Converged);
    assert!(result.energy < 1e-10);
    let state = context.get_state(StateFlags {
        positions: true,
        velocities: true,
        ..Default::default()
    });
    assert!((state.positions.unwrap().to_data().to_vec::<f32>().unwrap()[0] - 2.0).abs() < 1e-5);
    assert_eq!(
        state.velocities.unwrap().to_data().to_vec::<f32>().unwrap(),
        vec![0.4, 0.0, 0.0]
    );
    assert_eq!(state.time, 3.0);
    assert_eq!(state.step_count, 7);
}

#[test]
fn massless_coordinates_are_fixed_during_minimization() {
    let mut context = context(0.0);
    let result = LocalEnergyMinimizer::minimize(&mut context, 1e-5, 100).unwrap();
    assert_eq!(result.termination, MinimizationTermination::Converged);
    assert_eq!(result.energy, 64.0);
    assert_eq!(result.iterations, 0);
}

#[test]
fn iteration_limits_and_reporter_stops_remain_distinct() {
    let mut limited = context(1.0);
    let result = LocalEnergyMinimizer::minimize(&mut limited, 1e-5, 1).unwrap();
    assert_eq!(result.termination, MinimizationTermination::IterationLimit);
    assert_eq!(result.iterations, 1);
    struct Reporter {
        calls: usize,
    }
    impl MinimizationReporter for Reporter {
        fn report(
            &mut self,
            iteration: usize,
            positions: &[f64],
            gradient: &[f64],
            progress: &MinimizationProgress,
        ) -> bool {
            assert_eq!(iteration, 0);
            assert_eq!(positions.len(), 3);
            assert_eq!(gradient.len(), 3);
            assert!(progress.system_energy < 64.0);
            self.calls += 1;
            true
        }
    }
    let mut stopped = context(1.0);
    let mut reporter = Reporter { calls: 0 };
    let result =
        LocalEnergyMinimizer::minimize_with_reporter(&mut stopped, 1e-5, 100, Some(&mut reporter))
            .unwrap();
    assert_eq!(result.termination, MinimizationTermination::ReporterStopped);
    assert_eq!(reporter.calls, 1);
}
