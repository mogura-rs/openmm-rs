use approx::assert_relative_eq;
use burn::{backend::NdArray, prelude::*};
use openmm_core::{
    tensor::from_floats, AndersenThermostat, CompoundIntegrator, Context, CustomExternalForce,
    Integrator, LangevinMiddleIntegrator, StateFlags, System, VariableVerletIntegrator,
    VerletIntegrator,
};

type B = NdArray<f64>;

fn vector<B: Backend>(values: [f64; 3]) -> Tensor<B, 2> {
    from_floats(
        TensorData::new(values.to_vec(), [1, 3]),
        &Default::default(),
    )
}

fn data<B: Backend>(value: &Tensor<B, 2>) -> Vec<f64> {
    value.to_data().convert::<f64>().to_vec().unwrap()
}

#[test]
fn settings_delegate_to_current_child_but_force_groups_propagate() {
    let mut compound = CompoundIntegrator::<B>::new();
    compound.add_integrator(Box::new(VerletIntegrator::new(0.001)));
    let mut nested = CompoundIntegrator::<B>::new();
    nested.add_integrator(Box::new(LangevinMiddleIntegrator::new(0., 0., 0.002)));
    nested.add_integrator(Box::new(VerletIntegrator::new(0.003)));
    compound.add_integrator(Box::new(nested));
    compound.set_current_integrator(1);
    compound.set_step_size(0.004);
    compound.set_constraint_tolerance(1e-9);
    compound.set_integration_force_groups(18);
    assert_eq!(compound.num_integrators(), 2);
    assert_eq!(compound.current_integrator(), 1);
    assert_eq!(compound.integrator(0).lock().unwrap().step_size(), 0.001);
    assert_eq!(compound.step_size(), 0.004);
    assert_eq!(compound.constraint_tolerance(), 1e-9);
    for path in [&[0][..], &[1, 0], &[1, 1]] {
        assert_eq!(
            compound
                .component(path)
                .unwrap()
                .lock()
                .unwrap()
                .integration_force_groups(),
            18
        );
    }
    compound
        .component(&[1, 1])
        .unwrap()
        .lock()
        .unwrap()
        .set_integration_force_groups(2);
    assert_eq!(compound.integration_force_groups(), 18);
    assert!(compound.component(&[2]).is_none());
    assert!(compound.component(&[0, 0]).is_none());
}

#[test]
fn direct_inactive_child_uses_its_force_mask_and_preserves_active_kinetic_energy() {
    let mut system = System::<B>::new();
    system.add_particle(1.);
    for (expression, group) in [("-3*x", 1), ("-7*y", 2)] {
        let mut force = CustomExternalForce::new(expression).unwrap();
        force.add_particle(0, vec![]);
        force.set_force_group(group);
        system.add_force(Box::new(force));
    }
    let mut a = VerletIntegrator::new(0.001);
    a.set_integration_force_groups(2);
    let mut b = LangevinMiddleIntegrator::new(0., 0., 0.002);
    b.set_integration_force_groups(4);
    let mut compound = CompoundIntegrator::<B>::new();
    compound.add_integrator(Box::new(a));
    compound.add_integrator(Box::new(b));
    let mut context = Context::new(system, Box::new(compound), Default::default());
    context.try_step_component(&[1], 1).unwrap();
    let state = context.get_state(StateFlags::all());
    assert_eq!(state.time, 0.002);
    assert_relative_eq!(
        data(state.positions.as_ref().unwrap())[1],
        0.000028,
        epsilon = 1e-15
    );
    assert_relative_eq!(
        data(state.velocities.as_ref().unwrap())[1],
        0.014,
        epsilon = 1e-15
    );
    assert_relative_eq!(state.kinetic_energy.unwrap(), 0.00015425, epsilon = 1e-15);
    assert!(context.try_step_component(&[9], 0).is_err());
}

#[test]
fn thermal_velocities_use_first_child_offset_without_rescaling_on_switch() {
    let mut system = System::<B>::new();
    system.add_particle(1.);
    let mut force = CustomExternalForce::new("-3*x-7*y").unwrap();
    force.add_particle(0, vec![]);
    system.add_force(Box::new(force));
    let mut compound = CompoundIntegrator::<B>::new();
    compound.add_integrator(Box::new(VerletIntegrator::new(0.001)));
    compound.add_integrator(Box::new(LangevinMiddleIntegrator::new(0., 0., 0.002)));
    compound.set_current_integrator(1);
    let mut context = Context::new(system, Box::new(compound), Default::default());
    context.set_velocities_to_temperature(0.);
    let state = context.get_state(StateFlags::all());
    assert_eq!(
        data(state.velocities.as_ref().unwrap()),
        vec![-0.0015, -0.0035, 0.]
    );
    assert_relative_eq!(state.kinetic_energy.unwrap(), 0.00000725, epsilon = 1e-15);
}

#[test]
fn context_updates_use_top_level_step_size_when_inactive_child_runs() {
    let mut system = System::<B>::new();
    system.add_particle(1.);
    system.add_force(Box::new(AndersenThermostat::new(0., 1e9)));
    let mut variable = VariableVerletIntegrator::new(1e-5);
    variable.set_maximum_step_size(0.001);
    let mut compound = CompoundIntegrator::<B>::new();
    compound.add_integrator(Box::new(variable));
    compound.add_integrator(Box::new(VerletIntegrator::new(0.001)));
    let mut context = Context::new(system, Box::new(compound), Default::default());
    context.set_velocities(vector([1., 2., 3.]));
    context.try_step_component(&[1], 1).unwrap();
    assert_eq!(
        data(
            context
                .get_state(StateFlags::all())
                .velocities
                .as_ref()
                .unwrap()
        ),
        vec![1., 2., 3.]
    );
    context.try_step_to_component(&[0], 0.003).unwrap();
    let state = context.get_state(StateFlags::all());
    assert_eq!(state.time, 0.003);
    assert_eq!(state.step_count, 3);
    assert_eq!(data(state.velocities.as_ref().unwrap()), vec![0.; 3]);
}

#[test]
fn nested_adaptive_child_keeps_its_history_and_advances_bounded_time() {
    let mut system = System::<B>::new();
    system.add_particle(1.);
    let mut variable = VariableVerletIntegrator::new(1e-5);
    variable.set_maximum_step_size(0.002);
    let mut nested = CompoundIntegrator::<B>::new();
    nested.add_integrator(Box::new(variable));
    let mut compound = CompoundIntegrator::<B>::new();
    compound.add_integrator(Box::new(VerletIntegrator::new(0.001)));
    compound.add_integrator(Box::new(nested));
    let mut context = Context::new(system, Box::new(compound), Default::default());
    context.set_velocities(vector([1., 0., 0.]));
    context.step(1);
    context.try_step_to_component(&[1, 0], 0.0047).unwrap();
    context.step(1);
    assert_relative_eq!(
        context.get_state(StateFlags::none()).time,
        0.0057,
        epsilon = 1e-15
    );
    assert_eq!(context.get_state(StateFlags::none()).step_count, 4);
    assert_relative_eq!(
        context
            .integrator()
            .component(&[1, 0])
            .unwrap()
            .lock()
            .unwrap()
            .step_size(),
        0.0017,
        epsilon = 1e-15
    );
    assert!(context.try_step_to(1.).is_err());
}

#[test]
fn compound_supports_single_precision_without_changing_backend() {
    type F = NdArray<f32>;
    let mut system = System::<F>::new();
    system.add_particle(1.);
    let mut compound = CompoundIntegrator::<F>::new();
    compound.add_integrator(Box::new(VerletIntegrator::new(0.001)));
    compound.add_integrator(Box::new(LangevinMiddleIntegrator::new(0., 0., 0.002)));
    let mut context = Context::new(system, Box::new(compound), Default::default());
    context.set_velocities(vector([1., 2., 3.]));
    context.step(10);
    context.integrator_mut().select_integrator(1).unwrap();
    assert_eq!(context.integrator().selected_integrator(), Some(1));
    assert!(context.integrator_mut().select_integrator(2).is_err());
    context.step(10);
    let state = context.get_state(StateFlags::all());
    assert_relative_eq!(state.time, 0.03, epsilon = 1e-15);
    assert_relative_eq!(
        data(state.positions.as_ref().unwrap())[0],
        0.03,
        epsilon = 1e-7
    );
}

#[test]
fn lfmiddle_thermal_shift_is_distinct_from_its_kinetic_energy_offset() {
    let mut system = System::<B>::new();
    system.add_particle(2.);
    let mut force = CustomExternalForce::new("-4*x").unwrap();
    force.add_particle(0, vec![]);
    system.add_force(Box::new(force));
    let mut context = Context::new(
        system,
        Box::new(LangevinMiddleIntegrator::new(300., 1., 0.002)),
        Default::default(),
    );
    context.set_velocities_to_temperature(0.);
    let state = context.get_state(StateFlags::all());
    assert_eq!(
        data(state.velocities.as_ref().unwrap()),
        vec![-0.002, 0., 0.]
    );
    assert_relative_eq!(state.kinetic_energy.unwrap(), 0.000004, epsilon = 1e-15);
}
