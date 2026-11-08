"""Compare streamed many-particle expressions, filters and captured updates with Reference."""
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


def add_tables(om, force):
    force.addTabulatedFunction('c1', om.Continuous1DFunction([0.3, -0.1, 0.6, 0.2, 0.8], -2, 5))
    v2 = [0.03*i+0.01*math.sin(i) for i in range(9)]
    v3 = [0.05*i-0.01*math.cos(i) for i in range(8)]
    force.addTabulatedFunction('c2', om.Continuous2DFunction(3, 3, v2, -1, 2, -1, 2))
    force.addTabulatedFunction('c3', om.Continuous3DFunction(2, 2, 2, v3, -1, 5, -1, 5, -1, 2))
    force.addTabulatedFunction('d1', om.Discrete1DFunction([0.2, -0.1, 0.4]))
    force.addTabulatedFunction('d2', om.Discrete2DFunction(2, 2, [0.1, 0.3, -0.2, 0.5]))
    force.addTabulatedFunction('d3', om.Discrete3DFunction(2, 2, 2, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))


def configure(force, definition):
    force.setEnergyFunction(definition['energy'])
    force.setNonbondedMethod(definition['method'])
    force.setPermutationMode(definition['mode'])
    force.setCutoffDistance(definition['cutoff'])
    for i, (q, kind) in enumerate(definition['particles']):
        if i == force.getNumParticles():
            force.addParticle([q], kind)
        else:
            force.setParticleParameters(i, [q], kind)
    for i, types in enumerate(definition['filters']):
        force.setTypeFilter(i, set(types))
    for i, pair in enumerate(definition['exclusions']):
        if i == force.getNumExclusions():
            force.addExclusion(*pair)
        else:
            force.setExclusionParticles(i, *pair)


def create(om, row):
    system = om.System()
    for i in range(8):
        system.add_particle(0 if i >= 6 else 12)
    system.set_virtual_site(7, om.OutOfPlaneSite(0, 2, 4, 0.2, 0.3, 0.4))
    system.set_default_periodic_box_vectors(*map(tuple, row['box']))
    force = om.CustomManyParticleForce(row['count'], row['definition']['energy'])
    force.addPerParticleParameter('q'); force.addGlobalParameter('lambda', 0.7)
    add_tables(om, force); configure(force, row['definition'])
    if row['cv']:
        cv = om.CustomCVForce('0.8*m+0.002*m*m')
        cv.add_global_parameter('lambda', 0.7); cv.add_energy_parameter_derivative('lambda')
        cv.add_collective_variable('m', force); cv.set_force_group(5); system.add_force(cv)
    else:
        force.setForceGroup(5); system.add_force(force)
    external = om.CustomExternalForce('0.01*lambda*x*x')
    external.add_global_parameter('lambda', 0.7); external.set_force_group(2)
    for i in range(8): external.add_particle(i, [])
    system.add_force(external)
    integrator = om.VerletIntegrator(0.00001)
    context = om.Context(system, integrator)
    context.set_positions(list(map(tuple, row['positions'])))
    context.compute_virtual_sites()
    context.set_velocities([(0.01*math.sin(i+1), -0.02*math.cos(i), 0.015*math.sin(0.7*i)) if i < 6 else (0, 0, 0) for i in range(8)])
    return context, integrator, force


def api_checks(om, probes, boundary, periodic_updates):
    assert len(periodic_updates) == 1
    system = om.System(); force = om.CustomManyParticleForce(2, 'distance(p1,p2)+t(0)')
    for _ in range(2): system.add_particle(1); force.addParticle()
    table = om.Continuous1DFunction([0, 0], 0, 1); force.addTabulatedFunction('t', table)
    force.setNonbondedMethod(2); force.setCutoffDistance(0.9); system.add_force(force)
    system.set_default_periodic_box_vectors((3, 0, 0), (0, 3, 0), (0, 0, 3))
    context = om.Context(system, om.VerletIntegrator(0.001)); context.set_positions([(0, 0, 0), (2.8, 0, 0)])
    before = context.get_state(energy=True, forces=True)
    force.setNonbondedMethod(1); table.set_function_parameters([1, 1], 0, 1); force.updateParametersInContext(context)
    after = context.get_state(energy=True, forces=True)
    periodic_result = {'energies': [before.potential_energy, after.potential_energy], 'force_x': [before.forces[0][0], after.forces[0][0]]}
    for key, values in periodic_result.items():
        assert all(math.isclose(a, b, abs_tol=1e-13) for a, b in zip(values, periodic_updates[0][key], strict=True))
    f = om.CustomManyParticleForce(2, '1')
    assert f.getNumParticlesPerSet() == 2 and f.getEnergyFunction() == '1'
    assert (f.NoCutoff, f.CutoffNonPeriodic, f.CutoffPeriodic) == (0, 1, 2)
    assert (f.SinglePermutation, f.UniqueCentralParticle) == (0, 1)
    assert f.getPermutationMode() == f.SinglePermutation and f.getCutoffDistance() == 1
    assert f.getNonbondedMethod() == f.NoCutoff and not f.usesPeriodicBoundaryConditions()
    assert f.getNumParticles() == f.getNumExclusions() == f.getNumPerParticleParameters() == f.getNumGlobalParameters() == f.getNumTabulatedFunctions() == 0
    assert f.addParticle(type=-7) == 0 and f.getParticleParameters(0) == ([], -7)
    assert f.addParticle() == 1 and f.getParticleParameters(1) == ([], 0)
    f.setTypeFilter(0, {-7, 9}); assert f.getTypeFilter(0) == {-7, 9}
    f.setName('triplets'); f.setForceGroup(31)
    assert f.get_name() == 'triplets' and f.get_force_group() == 31
    f.addPerParticleParameter('a'); f.setPerParticleParameterName(0, 'b')
    assert f.getPerParticleParameterName(0) == 'b'
    f.addGlobalParameter('x', 2); f.setGlobalParameterName(0, 'y'); f.setGlobalParameterDefaultValue(0, 3)
    assert f.getGlobalParameterName(0) == 'y' and f.getGlobalParameterDefaultValue(0) == 3
    for call in [lambda: om.CustomManyParticleForce(0, '1'), lambda: f.getTypeFilter(2), lambda: f.setTypeFilter(2, {0}),
                 lambda: f.getParticleParameters(2), lambda: f.getExclusionParticles(0), lambda: f.setPermutationMode(2),
                 lambda: f.setNonbondedMethod(3), lambda: f.setForceGroup(32)]:
        try: call()
        except (IndexError, RuntimeError): pass
        else: raise AssertionError('invalid many-particle input was accepted')
    f = om.CustomManyParticleForce(3, '1')
    for _ in range(6): f.addParticle()
    f.createExclusionsFromBonds([(0, 1), (1, 2), (2, 3), (3, 4)], 2)
    assert {f.getExclusionParticles(i) for i in range(f.getNumExclusions())} == {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
    energies = []
    for row in probes:
        f = om.CustomManyParticleForce(row['count'], '1'); f.setPermutationMode(row['mode']); f.setTypeFilter(0, {1})
        system = om.System()
        for _ in range(3): system.add_particle(1); f.addParticle()
        system.add_force(f); context = om.Context(system, om.VerletIntegrator(0.001))
        context.set_positions([(0, 0, 0), (0.2, 0, 0), (0, 0.3, 0)])
        energy = context.get_state(energy=True).potential_energy
        assert energy == row['energy']
        energies.append({'count': row['count'], 'mode': row['mode'], 'energy': energy})
    boundary_energies = []
    for row in boundary:
        f = om.CustomManyParticleForce(2, '1'); f.setNonbondedMethod(f.CutoffNonPeriodic); f.setCutoffDistance(row['cutoff'])
        system = om.System()
        for _ in range(2): system.add_particle(1); f.addParticle()
        system.add_force(f); context = om.Context(system, om.VerletIntegrator(0.001)); context.set_positions([(0, 0, 0), (0.3, 0.2, 0)])
        energy = context.get_state(energy=True).potential_energy
        assert energy == row['energy']; boundary_energies.append(energy)
    assert boundary_energies == [0, 1]
    assert len(energies) == 6
    return {'metadata_and_defaults': True, 'type_keywords_and_sets': True, 'bond_exclusions': True,
            'reference_single_order_filter_behavior': energies, 'cutoff_rounding': boundary_energies, 'captured_periodic_update': periodic_result}


def stability_checks(om, rows):
    result = []
    for row in rows:
        mode, periodic = row['mode'], row['periodic']
        expression = ('lambda*(q1+q2+q3)*exp(-distance(p1,p2)-distance(p2,p3)-distance(p3,p1))' if mode == 0 else
                      'lambda*q1*(q2+q3)*(cos(angle(p2,p1,p3))+1/3)^2*exp(-distance(p1,p2)-distance(p1,p3))')
        system = om.System(); force = om.CustomManyParticleForce(3, expression)
        force.addPerParticleParameter('q'); force.addGlobalParameter('lambda', 0.7)
        force.setPermutationMode(mode); force.setNonbondedMethod(2 if periodic else 0); force.setCutoffDistance(0.9)
        positions, velocities = [], []
        for i in range(8):
            system.add_particle(12); force.addParticle([0.2+0.03*i])
            positions.append((0.14*(i % 2)+0.008*math.sin(1.1*i), 0.14*((i//2) % 2)+0.007*math.cos(0.9*i), 0.14*(i//4)+0.006*math.sin(1.3*i)))
            velocities.append((0.02*math.sin(1.7*i), 0.02*math.cos(0.9*i), 0.01*math.sin(0.3*i)))
        bonds = om.HarmonicBondForce()
        for i in range(8):
            for j in range(i+1, 8): bonds.add_bond(i, j, math.dist(positions[i], positions[j]), 1000)
        system.add_force(force); system.add_force(bonds)
        system.set_default_periodic_box_vectors((3, 0, 0), (0.4, 3.2, 0), (-0.2, 0.3, 3.4))
        integrator = om.VerletIntegrator(0.0002); context = om.Context(system, integrator)
        context.set_positions(positions); context.set_velocities(velocities)
        initial = context.get_state(energy=True); e0 = initial.potential_energy+initial.kinetic_energy
        assert math.isclose(e0, row['initial_energy'], abs_tol=1e-10, rel_tol=1e-12)
        drift, maximum_distance, short_errors = 0, 0, {}
        for sample in range(250):
            integrator.step(100)
            state = context.get_state(positions=True, energy=True)
            drift = max(drift, abs(state.potential_energy+state.kinetic_energy-e0))
            maximum_distance = max(maximum_distance, max(math.dist(state.positions[i], state.positions[j]) for i in range(8) for j in range(i+1, 8)))
            if sample == 4:
                short = context.get_state(positions=True, velocities=True)
                for key in ['positions', 'velocities']:
                    error = max(abs(a-b) for a, b in zip(flatten(getattr(short, key)), flatten(row['short_'+key]), strict=True))
                    assert error < 5e-9, (mode, periodic, key, error)
                    short_errors[key] = error
        assert max(drift, row['max_absolute_drift']) < 1e-5, (mode, periodic, drift, row['max_absolute_drift'])
        assert max(maximum_distance, row['max_distance']) < 0.6
        assert math.isclose(context.time, 5, abs_tol=1e-9)
        state = context.get_state(positions=True, velocities=True)
        initial_momentum = [12*sum(v[axis] for v in velocities) for axis in range(3)]
        momentum = {}
        for name, v in [('rust', state.velocities), ('reference', row['velocities'])]:
            momentum[name] = max(abs(12*sum(x[axis] for x in v)-initial_momentum[axis]) for axis in range(3))
            assert momentum[name] < 1e-8, (mode, periodic, name, momentum[name])
        errors = {key: max(abs(a-b) for a, b in zip(flatten(getattr(state, key)), flatten(row[key]), strict=True)) for key in ['positions', 'velocities']}
        result.append({'mode': mode, 'periodic': periodic, 'steps': 25000, 'time_ps': context.time,
                       'max_absolute_energy_drift': drift, 'reference_max_absolute_energy_drift': row['max_absolute_drift'],
                       'max_pair_distance': maximum_distance, 'point_one_ps_errors': short_errors,
                       'five_ps_errors': errors, 'momentum_errors': momentum})
    assert len(result) == 4
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('openmm', args.extension)
    om = importlib.util.module_from_spec(spec); spec.loader.exec_module(om)
    rows = [json.loads(line) for line in args.reference.read_text().splitlines()]
    stability = [row for row in rows if row.get('stability')]
    boundary = [row for row in rows if row.get('boundary_probe')]
    periodic_updates = [row for row in rows if row.get('periodic_update_probe')]
    probes = [row for row in rows if row.get('filter_probe')]
    rows = [row for row in rows if not row.get('filter_probe') and not row.get('stability') and not row.get('boundary_probe') and not row.get('periodic_update_probe')]
    maxima, checks, states, fd_checks, fd_max = {}, 0, 0, 0, 0
    for row in rows:
        if row.get('fd'):
            state = context.get_state(forces=True, groups=32)
            for actual, expected in zip(flatten(state.forces[:7]), flatten(row['forces']), strict=True):
                fd_max = max(fd_max, abs(actual-expected)); fd_checks += 1
                assert math.isclose(actual, expected, abs_tol=2e-6, rel_tol=2e-7), (states, actual, expected)
            continue
        phase = row['phase']
        if row['groups'] == 32:
            if phase == 0:
                context, integrator, force = create(om, row)
            elif phase == 1:
                configure(force, row['definition']); force.setGlobalParameterDefaultValue(0, 9)
                context.update_force_parameters_at_path([0, 0] if row['cv'] else [0], force)
                context.set_parameter('lambda', 1.1)
            elif phase == 2:
                definition = row['definition']
                force.getTabulatedFunction(0).set_function_parameters(definition['table_values'], *definition['table_bounds'])
                context.update_force_parameters_at_path([0, 0] if row['cv'] else [0], force)
                context.set_periodic_box_vectors(*map(tuple, row['box']))
                context.set_positions(list(map(tuple, row['positions'])))
                context.compute_virtual_sites()
            else:
                integrator.step(3)
        state = context.get_state(positions=True, velocities=True, forces=True, energy=True, parameter_derivatives=True, groups=row['groups'])
        actual = {'energy': state.potential_energy, 'kinetic': state.kinetic_energy, 'time': context.time,
                  'positions': state.positions, 'velocities': state.velocities, 'forces': state.forces,
                  'derivative': state.energy_parameter_derivatives.get('lambda', 0)}
        key = tuple(row[k] for k in ['count', 'initial_mode', 'initial_method', 'filter', 'cv', 'phase', 'groups'])
        for name, value in actual.items():
            for got, expected in zip(flatten(value), flatten(row[name]), strict=True):
                maxima[name] = max(maxima.get(name, 0), abs(got-expected)); checks += 1
                assert math.isclose(got, expected, abs_tol=2e-8, rel_tol=2e-9), (key, name, got, expected)
        states += 1
    assert states == 1728 and fd_checks == 3024, (states, fd_checks)
    print(json.dumps({'states': states, 'comparisons': checks, 'max_absolute_error': maxima,
                      'finite_difference': {'components': fd_checks, 'max_absolute_error': fd_max},
                      'api': api_checks(om, probes, boundary, periodic_updates), 'stability': stability_checks(om, stability)}, indent=2))


if __name__ == '__main__':
    main()
