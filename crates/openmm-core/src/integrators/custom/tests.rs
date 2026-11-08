use super::*;
use crate::tensor::{from_floats, to_f64_vec};
use crate::{Context, CustomExternalForce, StateFlags, System};
use approx::assert_relative_eq;
use burn::backend::NdArray;
type B = NdArray<f64>;

#[test]
fn custom_control_flow_and_massless_particle_rules() {
    let mut system = System::<B>::new();
    system.add_particle(2.0);
    system.add_particle(0.0);
    let mut integrator = CustomIntegrator::new(0.1);
    integrator.add_global_variable("count", 0.0).unwrap();
    integrator.add_per_dof_variable("q", 2.0).unwrap();
    integrator.add_compute_global("count", "0").unwrap();
    integrator.begin_while_block("count < 3").unwrap();
    integrator.begin_if_block("count != 1").unwrap();
    integrator.add_compute_per_dof("x", "x+dt*v").unwrap();
    integrator.end_block().unwrap();
    integrator.add_compute_global("count", "count+1").unwrap();
    integrator.end_block().unwrap();
    integrator
        .add_compute_per_dof("q", "cross(x,v)+vector(1,2,3)")
        .unwrap();
    integrator
        .set_kinetic_energy_expression("m*v*v/2+q")
        .unwrap();
    let mut context = Context::new(system, Box::new(integrator), Default::default());
    context.set_positions(from_floats(
        [[1., 2., 3.], [4., 5., 6.]],
        &Default::default(),
    ));
    context.set_velocities(from_floats(
        [[0.1, 0.2, 0.3], [1., 1., 1.]],
        &Default::default(),
    ));
    context.try_step(1).unwrap();
    let state = context.get_state(StateFlags {
        positions: true,
        velocities: true,
        energy: true,
        ..Default::default()
    });
    let p = to_f64_vec(state.positions.as_ref().unwrap());
    for (a, b) in p.iter().zip([1.02, 2.04, 3.06, 4., 5., 6.]) {
        assert_relative_eq!(*a, b, epsilon = 1e-14);
    }
    assert_relative_eq!(state.kinetic_energy.unwrap(), 6.14, epsilon = 1e-13);
    assert_eq!(state.time, 0.1);
}

#[test]
fn custom_parameters_energy_derivatives_and_force_invalidation() {
    let mut system = System::<B>::new();
    system.add_particle(2.0);
    let mut force = CustomExternalForce::new("lambda*x*x").unwrap();
    force.add_global_parameter("lambda", 2.0);
    force.add_energy_parameter_derivative("lambda");
    force.add_particle(0, vec![]);
    force.set_force_group(3);
    system.add_force(Box::new(force));
    let mut integrator = CustomIntegrator::new(0.0);
    integrator.add_global_variable("derivative", 0.0).unwrap();
    integrator.add_per_dof_variable("old_force", 0.0).unwrap();
    integrator.add_compute_per_dof("old_force", "f3").unwrap();
    integrator.add_compute_global("lambda", "3").unwrap();
    integrator
        .add_compute_global("derivative", "deriv(energy3,lambda)")
        .unwrap();
    integrator.add_compute_per_dof("v", "old_force+f3").unwrap();
    integrator
        .add_compute_global("lambda", "derivative")
        .unwrap();
    let mut context = Context::new(system, Box::new(integrator), Default::default());
    context.set_positions(from_floats([[2., 0., 0.]], &Default::default()));
    context.try_step(1).unwrap();
    assert_eq!(context.get_parameter("lambda"), 4.0);
    let v = to_f64_vec(
        context
            .get_state(StateFlags {
                velocities: true,
                ..Default::default()
            })
            .velocities
            .as_ref()
            .unwrap(),
    );
    assert_eq!(v, [-20., 0., 0.]);
    assert_eq!(context.get_time(), 0.0);
    assert_eq!(context.get_step_count(), 1);
}

#[test]
fn custom_requires_explicit_context_update_and_ke_uses_requested_groups() {
    for update in [false, true] {
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        system.add_particle(1.0);
        system.add_force(Box::new(crate::CMMotionRemover::new(1)));
        let mut force = CustomExternalForce::new("x").unwrap();
        force.add_particle(0, vec![]);
        force.set_force_group(5);
        system.add_force(Box::new(force));
        let mut integrator = CustomIntegrator::new(0.1);
        integrator
            .set_kinetic_energy_expression("m*(v+f)^2/2")
            .unwrap();
        if update {
            integrator.add_update_context_state().unwrap();
        }
        let mut context = Context::new(system, Box::new(integrator), Default::default());
        context.set_velocities(from_floats(
            [[1., 0., 0.], [1., 0., 0.]],
            &Default::default(),
        ));
        context.try_step(1).unwrap();
        let flags = StateFlags {
            energy: true,
            ..Default::default()
        };
        assert_relative_eq!(
            context.get_state_groups(flags, 0).kinetic_energy.unwrap(),
            if update { 0.0 } else { 1.0 },
            epsilon = 1e-14
        );
        assert_relative_eq!(
            context.get_state_groups(flags, 32).kinetic_energy.unwrap(),
            0.5,
            epsilon = 1e-14
        );
    }
}

#[test]
fn custom_errors_are_reported_before_execution() {
    for (kind, variable, expression) in [
        (ComputationType::BlockEnd, "", ""),
        (ComputationType::IfBlockStart, "", "1"),
        (ComputationType::ComputePerDof, "x", "bad_name"),
        (ComputationType::ComputePerDof, "q", "x"),
        (ComputationType::ComputePerDof, "v", "f0+f1"),
        (ComputationType::ComputePerDof, "v", "f0+energy0"),
        (ComputationType::ComputeGlobal, "dt", "deriv(x,y)"),
    ] {
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        let mut integrator = CustomIntegrator::new(0.1);
        integrator
            .add_computation(kind, variable, expression)
            .unwrap();
        let mut context = Context::new(system, Box::new(integrator), Default::default());
        assert!(context.try_step(1).is_err());
        assert_eq!(context.get_step_count(), 0);
    }
}

#[test]
fn custom_single_precision_host_fallback_preserves_frozen_particles() {
    let mut system = System::<NdArray<f32>>::new();
    system.add_particle(1.0);
    system.add_particle(0.0);
    let mut integrator = CustomIntegrator::new(0.01);
    integrator.add_compute_per_dof("x", "x+dt*v").unwrap();
    let mut context = Context::new(system, Box::new(integrator), Default::default());
    context.set_positions(from_floats(
        [[1., 2., 3.], [4., 5., 6.]],
        &Default::default(),
    ));
    context.set_velocities(from_floats(
        [[0.1, 0.2, 0.3], [1., 1., 1.]],
        &Default::default(),
    ));
    context.try_step(20).unwrap();
    let p = to_f64_vec(
        context
            .get_state(StateFlags {
                positions: true,
                ..Default::default()
            })
            .positions
            .as_ref()
            .unwrap(),
    );
    for (a, b) in p.iter().zip([1.02, 2.04, 3.06, 4., 5., 6.]) {
        assert_relative_eq!(*a, b, epsilon = 3e-6);
    }
}

#[derive(Clone)]
struct CountingForce {
    force: CustomExternalForce,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}
impl crate::Force<B> for CountingForce {
    fn calc_forces_and_energy(
        &self,
        p: &Tensor<B, 2>,
        box_vectors: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.force.calc_forces_and_energy(p, box_vectors)
    }
    fn global_parameters(&self) -> std::collections::HashMap<String, f64> {
        <CustomExternalForce as crate::Force<B>>::global_parameters(&self.force)
    }
    fn set_global_parameter(&mut self, name: &str, value: f64) -> bool {
        <CustomExternalForce as crate::Force<B>>::set_global_parameter(&mut self.force, name, value)
    }
}

#[test]
fn custom_force_caches_survive_steps_and_invalidate_after_external_changes() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = Arc::new(AtomicUsize::new(0));
    let mut system = System::<B>::new();
    system.add_particle(1.0);
    let mut force = CustomExternalForce::new("lambda*x*x").unwrap();
    force.add_global_parameter("lambda", 1.0);
    force.add_particle(0, vec![]);
    system.add_force(Box::new(CountingForce {
        force,
        calls: calls.clone(),
    }));
    system.add_force(Box::new(crate::CMMotionRemover::new(1)));
    let mut it = CustomIntegrator::new(0.001);
    it.add_update_context_state().unwrap();
    it.add_compute_per_dof("v", "v+0.5*dt*f/m").unwrap();
    it.add_compute_per_dof("x", "x+dt*v").unwrap();
    it.add_compute_per_dof("v", "v+0.5*dt*f/m").unwrap();
    let mut context = Context::new(system, Box::new(it), Default::default());
    context.set_positions(from_floats([[1., 0., 0.]], &Default::default()));
    context.step(3);
    assert_eq!(calls.load(Ordering::Relaxed), 4);
    context.set_parameter("lambda", 2.0);
    context.step(1);
    assert_eq!(calls.load(Ordering::Relaxed), 6);
    context.set_positions(from_floats([[2., 0., 0.]], &Default::default()));
    context.step(1);
    assert_eq!(calls.load(Ordering::Relaxed), 8);
    context.set_periodic_box_vectors([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]]);
    context.step(1);
    assert_eq!(calls.load(Ordering::Relaxed), 10);
    let state = context.get_state(StateFlags {
        positions: true,
        energy: true,
        ..Default::default()
    });
    assert_eq!(calls.load(Ordering::Relaxed), 11);
    let p = to_f64_vec(state.positions.as_ref().unwrap());
    assert_relative_eq!(
        state.potential_energy.unwrap(),
        2.0 * p[0] * p[0],
        epsilon = 1e-14
    );
    context.step(1);
    assert_eq!(calls.load(Ordering::Relaxed), 12);
    context.set_state(&state);
    context.step(1);
    assert_eq!(calls.load(Ordering::Relaxed), 14);
}

#[test]
fn custom_explicit_box_updates_invalidate_energy_caches() {
    let mut system = System::<B>::new();
    system.add_particle(1.0);
    system.set_default_periodic_box_vectors([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]]);
    system.add_force(Box::new(crate::CustomVolumeForce::new("0.5*v").unwrap()));
    let mut barostat = crate::MonteCarloBarostat::new(1.0, 0.0, 1);
    barostat.set_random_number_seed(94);
    system.add_force(Box::new(barostat));
    let mut it = CustomIntegrator::new(0.001);
    it.add_global_variable("before", 0.0).unwrap();
    it.add_global_variable("after", 0.0).unwrap();
    it.add_compute_global("before", "energy").unwrap();
    it.add_update_context_state().unwrap();
    it.add_compute_global("after", "energy").unwrap();
    let data = it.data.clone();
    let mut context = Context::new(system, Box::new(it), Default::default());
    let mut accepted = 0;
    for _ in 0..30 {
        context.step(1);
        let data = data.lock().unwrap();
        let before = data.variables.globals[0].1;
        let after = data.variables.globals[1].1;
        let actual = context.calc_forces_and_energy().1;
        assert_relative_eq!(after, actual, epsilon = 1e-13);
        if before != after {
            accepted += 1;
        }
    }
    assert!(accepted > 0);
}

#[derive(Clone)]
struct UpdateStepProbe(Arc<Mutex<Vec<f64>>>);
impl crate::Force<B> for UpdateStepProbe {
    fn calc_forces_and_energy(
        &self,
        p: &Tensor<B, 2>,
        _: Option<&[[f64; 3]; 3]>,
    ) -> (Tensor<B, 2>, f64) {
        (crate::tensor::zeros(p.dims(), &p.device()), 0.0)
    }
    fn update_context_velocities(&mut self, _: &mut Tensor<B, 2>, _: &[f64], dt: f64, _: u64) {
        self.0.lock().unwrap().push(dt);
    }
}

#[test]
fn custom_clock_and_state_update_duration_follow_live_step_size() {
    for inactive in [false, true] {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut system = System::<B>::new();
        system.add_particle(1.0);
        system.add_force(Box::new(UpdateStepProbe(seen.clone())));
        let mut custom = CustomIntegrator::new(0.001);
        custom.add_compute_global("dt", "0.003").unwrap();
        custom.add_update_context_state().unwrap();
        custom.add_compute_per_dof("x", "x+dt").unwrap();
        let mut root = crate::CompoundIntegrator::<B>::new();
        root.add_integrator(Box::new(crate::VerletIntegrator::new(0.005)));
        root.add_integrator(Box::new(custom));
        if !inactive {
            root.set_current_integrator(1);
        }
        let mut context = Context::new(system, Box::new(root), Default::default());
        context.try_step_component(&[1], 1).unwrap();
        assert_eq!(context.get_time(), 0.003);
        assert_eq!(
            *seen.lock().unwrap(),
            [if inactive { 0.005 } else { 0.003 }]
        );
        let state = context.get_state(StateFlags {
            positions: true,
            ..Default::default()
        });
        assert_eq!(to_f64_vec(state.positions.as_ref().unwrap()), [0.003; 3]);
    }
}
