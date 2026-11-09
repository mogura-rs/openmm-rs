"""Compare ATM composition, coordinate transformations and captured updates with Reference."""
import argparse
import importlib.util
import json
import math
from pathlib import Path


def flatten(values):
    if isinstance(values, (list, tuple)):
        for value in values:
            yield from flatten(value)
    else:
        yield values


def points():
    return [(0.1, 0.2, 0.3), (0.55, 0.13, 0.24), (0.24, 0.68, 0.17),
            (0.64, 0.59, 0.72), (0.86, 0.35, 0.41), (0, 0, 0)]


def child(om, periodic):
    bond = om.CustomBondForce('g*r*r+0.1*sin(r)'); bond.add_global_parameter('g', 2)
    for a, b in [(0, 1), (2, 4), (3, 5)]: bond.add_bond(a, b, [])
    bond.set_force_group(11)
    nb = om.NonbondedForce()
    nb.set_nonbonded_method('CutoffPeriodic' if periodic else 'NoCutoff')
    nb.set_cutoff_distance(1.2); nb.set_use_dispersion_correction(False)
    nb.set_reaction_field_dielectric(1); nb.set_force_group(9)
    for i in range(6): nb.add_particle(0.02*(i % 3-1), 0.1+0.004*i, 0.015+0.001*i)
    nb.add_exception(0, 2, 0, 0.1, 0)
    cv = om.CustomCVForce('b+n+0.03*r')
    cv.add_collective_variable('b', bond); cv.add_collective_variable('n', nb)
    cv.add_collective_variable('r', om.RGForce([]))
    return cv


def transforms(om, force, mode, phase):
    for i in range(6):
        if mode == 0:
            force.setParticleParameters(i, (0.05*math.sin(i+phase), 0.02*i, -0.01*(i+phase)),
                                        (-0.01*i, 0.02*phase, 0.01*math.cos(i)))
        elif mode == 1:
            force.setParticleTransformation(i, om.ParticleOffsetDisplacement(
                (i+1+phase) % 5, (i+3) % 5, (1 if phase else 0) if i % 2 else -1, (2 if phase else 4) if i % 2 else -1))
        elif i < 2:
            force.setParticleTransformation(i, om.ParticleOffsetDisplacement(1-i, i, 1-i if phase else -1, i if phase else -1))
        else:
            force.setParticleParameters(i, (0, 0, 0), (0, 0, 0))


def model(om, kind, mode, periodic, nested):
    if kind < 2:
        force = om.ATMForce('u0+lambda*(u1-u0)' if kind == 0 else '0.7*u0+lambda*u1+0.003*u0*u1+0.1*g')
        force.addGlobalParameter('lambda', 0.4); force.addEnergyParameterDerivative('lambda')
    else:
        force = om.ATMForce(0.2, 0.7, 0.15, -0.3, 0.1, 30, 10 if kind == 2 else -1, 0.5, -1 if kind == 4 else 1)
        force.addEnergyParameterDerivative(om.ATMForce.Lambda2())
    force.addGlobalParameter('g', 1.7)
    for _ in range(6): force.addParticle()
    transforms(om, force, mode, 0)
    if nested:
        inner = om.ATMForce('0.6*u0+0.4*u1')
        for i in range(6): inner.addParticle((0.01*i, 0.02*math.sin(i), -0.01))
        inner.addForce(child(om, periodic)); inner.setForceGroup(12); force.addForce(inner)
    else:
        force.addForce(child(om, periodic))
    force.setForceGroup(5)
    return force


def create(om, row):
    system = om.System()
    for i in range(6): system.add_particle(0 if i >= 4 else 12)
    system.set_virtual_site(5, om.OutOfPlaneSite(0, 2, 4, 0.2, 0.3, 0.4))
    system.set_default_periodic_box_vectors((3.4, 0, 0), (0.4, 3.6, 0), (-0.2, 0.3, 3.8))
    force = model(om, row['kind'], row['mode'], row['periodic'], row['nested'])
    if row['cv']:
        cv = om.CustomCVForce('0.8*a+0.002*a*a'); cv.add_collective_variable('a', force)
        cv.set_force_group(5); system.add_force(cv)
    else:
        system.add_force(force)
    external = om.CustomExternalForce('0.01*g*x*x'); external.add_global_parameter('g', 1.7)
    for i in range(6): external.add_particle(i, [])
    external.set_force_group(2); system.add_force(external)
    integrator = om.VerletIntegrator(0.00001); context = om.Context(system, integrator)
    context.set_positions(points()); context.compute_virtual_sites()
    context.set_velocities([(0.01*math.sin(i), 0.02*math.cos(i), 0.01*i) if i < 4 else (0, 0, 0) for i in range(6)])
    return context, integrator, force


def api_checks(om):
    force = om.ATMForce(energy='u0+lambda*(u1-u0)')
    force.addGlobalParameter('lambda', 0.5); force.addEnergyParameterDerivative('lambda')
    force.addParticle(); force.addParticle(om.FixedDisplacement((1, 0, 0)))
    assert force.getParticleParameters(1) == ((1, 0, 0), (0, 0, 0))
    assert isinstance(force.getParticleTransformation(0), om.FixedDisplacement)
    swap = om.ParticleOffsetDisplacement(1, 0, 0, 1)
    assert (swap.getDestinationParticle1(), swap.getOriginParticle1(), swap.getDestinationParticle0(), swap.getOriginParticle0()) == (1, 0, 0, 1)
    bond = om.CustomBondForce('g*r*r'); bond.add_global_parameter('g', 2); bond.add_bond(0, 1, [])
    force.addForce(bond); assert force.getForce(0) is bond
    system = om.System(); system.add_particle(1); system.add_particle(1); system.add_force(force)
    force.setForceGroup(5); context = om.Context(system, om.VerletIntegrator(0.001))
    context.set_positions([(0, 0, 0), (1, 0, 0)])
    assert force.getPerturbationEnergy(context) == (8, 2, 5)
    assert context.get_state(parameter_derivatives=True).energy_parameter_derivatives['lambda'] == 6
    force.setForceGroup(6)
    assert context.get_state(energy=True, groups=32).potential_energy == 0
    assert context.get_state(energy=True, groups=64).potential_energy == 5
    bond.set_global_parameter_default_value(0, 9)
    force.setEnergyFunction('100*u1'); force.setParticleParameters(1, (2, 0, 0)); force.updateParametersInContext(context)
    assert force.getPerturbationEnergy(context) == (18, 2, 10)
    newer = om.Context(system, om.VerletIntegrator(0.001)); newer.set_positions([(0, 0, 0), (1, 0, 0)])
    assert newer.get_state(energy=True).potential_energy == 8100
    assert force.getPerturbationEnergy(context) == (18, 2, 10)
    for name in ['Lambda1', 'Lambda2', 'Alpha', 'Uh', 'W0', 'Umax', 'Ubcore', 'Acore', 'Direction']:
        assert getattr(om.ATMForce, name)() == name
    default = om.ATMForce(lambda1=0.5, lambda2=0.5, alpha=0, uh=0, w0=0, umax=0, ubcore=0, acore=0, direction=1)
    assert default.getNumGlobalParameters() == 9 and default.getGlobalParameterDefaultValue(0) == 0.5
    return {'constructors_and_transforms': True, 'live_group': True, 'transform_only_update': True, 'single_derivative': True}


def derivative_checks(om, rows):
    expected = {tuple(row['order']): row['derivatives'] for row in rows}
    single = {name: expected[(name,)][name] for name in ['lambda', 'k']}
    assert math.isclose(single['lambda'], 6, abs_tol=1e-12) and math.isclose(single['k'], 12, abs_tol=1e-12), single
    source_multiple = {}
    for order in [('lambda', 'k'), ('k', 'lambda')]:
        values = expected[order]
        assert values[order[0]] == 0 and math.isclose(values[order[1]], single[order[1]], abs_tol=1e-12), (order, values)
        source_multiple[','.join(order)] = values
        system = om.System(); system.add_particle(12); system.add_particle(12)
        force = om.ATMForce('u0+lambda*(u1-u0)+0.5*k*k*u1')
        force.addGlobalParameter('lambda', 0.5); force.addGlobalParameter('k', 1.5)
        force.addParticle(); force.addParticle(om.FixedDisplacement((1, 0, 0)))
        for name in order: force.addEnergyParameterDerivative(name)
        bond = om.CustomBondForce('g*r*r'); bond.add_global_parameter('g', 2); bond.add_bond(0, 1, [])
        force.addForce(bond); system.add_force(force)
        context = om.Context(system, om.VerletIntegrator(0.001)); context.set_positions([(0, 0, 0), (1, 0, 0)])
        actual = context.get_state(parameter_derivatives=True).energy_parameter_derivatives
        for name in order:
            assert math.isclose(actual[name], single[name], abs_tol=1e-12), (order, name, actual[name], single[name])
    assert len(rows) == 4
    return {'source_single_registration': single, 'source_multiple_registration': source_multiple,
            'rust_matches_single_registration_in_both_orders': True}


def stability_checks(om, rows):
    result = []
    for row in rows:
        mode, periodic = row['mode'], row['periodic']
        system = om.System(); p = points(); p[5] = (0.43, 0.48, 0.31)
        velocities = [(0.02*math.sin(1.7*i), 0.02*math.cos(0.9*i), 0.01*math.sin(0.3*i)) for i in range(6)]
        atm = om.ATMForce(0.2, 0.7, 0.15, -0.3, 0.1, 30, 0, 0.5, 1) if mode == 0 else om.ATMForce('0.3*u0+0.7*u1+0.003*u0*u1')
        child = om.CustomBondForce('g*r*r+0.1*sin(r)'); child.add_global_parameter('g', 0.7)
        child.set_uses_periodic_boundary_conditions(bool(periodic)); bonds = om.HarmonicBondForce()
        bonds.set_uses_periodic_boundary_conditions(bool(periodic))
        for i in range(6):
            system.add_particle(12); atm.addParticle()
            if mode == 0: atm.setParticleParameters(i, (0.05*math.sin(i), 0.02*i, -0.01*i), (-0.01*i, 0, 0.01*math.cos(i)))
            elif i < 2: atm.setParticleTransformation(i, om.ParticleOffsetDisplacement(1-i, i))
            for j in range(i):
                bonds.add_bond(i, j, math.dist(p[i], p[j]), 500); child.add_bond(i, j, [])
        atm.addForce(child); system.add_force(atm); system.add_force(bonds)
        system.set_default_periodic_box_vectors((3, 0, 0), (0.3, 3.2, 0), (-0.2, 0.1, 3.4))
        integrator = om.VerletIntegrator(0.0002); context = om.Context(system, integrator)
        context.set_positions(p); context.set_velocities(velocities)
        initial = context.get_state(energy=True); e0 = initial.potential_energy+initial.kinetic_energy
        assert math.isclose(e0, row['initial_energy'], abs_tol=1e-10)
        drift, short_errors = 0, {}
        for sample in range(250):
            integrator.step(100); state = context.get_state(energy=True)
            drift = max(drift, abs(state.potential_energy+state.kinetic_energy-e0))
            if sample == 4:
                state = context.get_state(positions=True, velocities=True)
                for name in ['positions', 'velocities']:
                    error = max(abs(a-b) for a, b in zip(flatten(getattr(state, name)), flatten(row['short_'+name]), strict=True))
                    assert error < 5e-9, (mode, periodic, name, error)
                    short_errors[name] = error
        assert max(drift, row['max_absolute_drift']) < 1e-5
        state = context.get_state(positions=True, velocities=True)
        final_errors = {name: max(abs(a-b) for a, b in zip(flatten(getattr(state, name)), flatten(row[name]), strict=True)) for name in ['positions', 'velocities']}
        assert max(final_errors.values()) < 1e-8, (mode, periodic, final_errors)
        initial_momentum = [12*sum(v[axis] for v in velocities) for axis in range(3)]
        momentum = {}
        for name, v in [('rust', state.velocities), ('reference', row['velocities'])]:
            momentum[name] = max(abs(12*sum(x[axis] for x in v)-initial_momentum[axis]) for axis in range(3))
            assert momentum[name] < 1e-8
        assert math.isclose(context.time, 5, abs_tol=1e-9)
        result.append({'mode': mode, 'periodic': periodic, 'steps': 25000, 'time_ps': context.time,
                       'max_absolute_energy_drift': drift, 'reference_max_absolute_energy_drift': row['max_absolute_drift'],
                       'point_one_ps_errors': short_errors, 'five_ps_errors': final_errors, 'momentum_errors': momentum})
    assert len(result) == 4
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--extension', type=Path, required=True); parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args(); spec = importlib.util.spec_from_file_location('openmm', args.extension)
    om = importlib.util.module_from_spec(spec); spec.loader.exec_module(om)
    rows = [json.loads(line) for line in args.reference.read_text().splitlines()]
    stability = [row for row in rows if row.get("stability")]
    derivative_probes = [row for row in rows if row.get("derivative_probe")]
    rows = [row for row in rows if not row.get("stability") and not row.get("derivative_probe")]
    maxima, checks, states, fd_checks, fd_max = {}, 0, 0, 0, 0
    for row in rows:
        if row.get('fd'):
            forces = context.get_state(forces=True, groups=32).forces
            for actual, expected in zip(flatten(forces[:5]), flatten(row['forces']), strict=True):
                fd_max = max(fd_max, abs(actual-expected)); fd_checks += 1
                assert math.isclose(actual, expected, abs_tol=2e-6, rel_tol=2e-7), (states, actual, expected)
            continue
        if row['groups'] == 32:
            phase = row['phase']
            if phase == 0: context, integrator, force = create(om, row)
            elif phase == 1:
                transforms(om, force, row['mode'], 1); force.setEnergyFunction('100*u1'); force.setGlobalParameterDefaultValue(0, 9)
                context.update_force_parameters_at_path([0, 0] if row['cv'] else [0], force)
            elif phase == 2:
                context.set_parameter('lambda' if row['kind'] < 2 else 'Lambda2', 0.9); context.set_parameter('g', 1.1)
                context.set_periodic_box_vectors((3.5, 0, 0), (-0.3, 3.7, 0), (0.1, -0.2, 3.9))
                p = points()
                if row['periodic']: p[3] = tuple(a+b for a, b in zip(p[3], (-0.3, 3.7, 0), strict=True))
                context.set_positions(p); context.compute_virtual_sites()
            else: integrator.step(3)
        state = context.get_state(positions=True, velocities=True, forces=True, energy=True, parameter_derivatives=True, groups=row['groups'])
        values = {'energy': state.potential_energy, 'kinetic': state.kinetic_energy, 'time': context.time,
                  'positions': state.positions, 'velocities': state.velocities, 'forces': state.forces,
                  'perturbation': context.get_atm_perturbation_energy([0, 0] if row['cv'] else [0])}
        for name, expected in row['derivatives'].items():
            key = 'derivative_'+name; values[key] = state.energy_parameter_derivatives.get(name, 0); row[key] = expected
        key = tuple(row[k] for k in ['kind', 'mode', 'periodic', 'nested', 'cv', 'phase', 'groups'])
        for name, value in values.items():
            for actual, expected in zip(flatten(value), flatten(row[name]), strict=True):
                maxima[name] = max(maxima.get(name, 0), abs(actual-expected)); checks += 1
                assert math.isclose(actual, expected, abs_tol=2e-8, rel_tol=2e-9), (key, name, actual, expected)
        states += 1
    assert states == 1440 and fd_checks == 1800, (states, fd_checks)
    print(json.dumps({'states': states, 'comparisons': checks, 'max_absolute_error': maxima,
                      'finite_difference': {'components': fd_checks, 'max_absolute_error': fd_max}, 'api': api_checks(om), 'derivatives': derivative_checks(om, derivative_probes),
                      'stability': stability_checks(om, stability)}, indent=2))


if __name__ == '__main__': main()
