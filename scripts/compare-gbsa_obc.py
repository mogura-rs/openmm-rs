"""Compare OBC-II solvation, nested CVs, updates and cutoff boundaries."""

import argparse
import importlib.util
import json
import math
from pathlib import Path


def create(om, row):
    system = om.System()
    n = len(row['parameters'])
    for i in range(n):
        system.add_particle(0 if i == n-1 and n > 2 else 2+i % 4)
    system.set_default_periodic_box_vectors(*map(tuple, row['box']))
    gb = om.GBSAOBCForce()
    for parameters in row['parameters']:
        gb.addParticle(*parameters)
    gb.setNonbondedMethod(row['method'])
    gb.setCutoffDistance(0.8)
    if row['settings'] == 1:
        gb.setSoluteDielectric(2.1)
        gb.setSolventDielectric(35)
        gb.setSurfaceAreaEnergy(0.75)
    if row['settings'] == 2:
        gb.setSurfaceAreaEnergy(0)
    if row['settings'] == 3:
        gb.setSolventDielectric(0)
    if row['cv']:
        cv = om.CustomCVForce('0.7*g+0.001*g*g')
        cv.add_collective_variable('g', gb)
        cv.set_force_group(5)
        system.add_force(cv)
    else:
        gb.setForceGroup(5)
        system.add_force(gb)
    external = om.CustomExternalForce('0.2*x*x+0.1*z')
    external.set_force_group(2)
    for i in range(n):
        external.add_particle(i, [])
    system.add_force(external)
    it = om.VerletIntegrator(0.0001)
    context = om.Context(system, it)
    context.set_positions(list(map(tuple, row['positions'])))
    context.set_velocities([(0.01, -0.02, 0.03)]*n)
    return context, gb, it


def api_checks(om):
    force = om.GBSAOBCForce()
    assert force.getNumParticles() == force.num_particles() == 0
    assert force.getNonbondedMethod() == force.NoCutoff == 0
    assert (force.CutoffNonPeriodic, force.CutoffPeriodic) == (1, 2)
    assert force.getCutoffDistance() == 1
    assert force.getSolventDielectric() == 78.3
    assert force.getSoluteDielectric() == 1
    assert force.getSurfaceAreaEnergy() == 2.25936
    assert force.getName() == 'GBSAOBCForce' and force.getForceGroup() == 0
    force.setName('implicit solvent')
    force.setForceGroup(31)
    assert force.get_name() == 'implicit solvent' and force.get_force_group() == 31
    assert force.addParticle(0.5, 0.15, 0.8) == 0
    assert force.getParticleParameters(0) == (0.5, 0.15, 0.8)
    for call in [lambda: force.setNonbondedMethod(3), lambda: force.setForceGroup(32),
                 lambda: force.getParticleParameters(1), lambda: force.setParticleParameters(1, 1, 1, 1)]:
        try:
            call()
        except (RuntimeError, IndexError):
            pass
        else:
            raise AssertionError('invalid input was accepted')
    system = om.System()
    system.add_particle(1)
    system.add_force(force)
    force.setParticleParameters(0, 0.4, 0.2, 0.9)
    first = om.Context(system, om.VerletIntegrator(0.001))
    first.set_positions([(0, 0, 0)])
    energy = first.get_state(energy=True).potential_energy
    force.setParticleParameters(0, 0.3, 0.25, 0.8)
    assert first.get_state(energy=True).potential_energy == energy
    second = om.Context(system, om.VerletIntegrator(0.001))
    second.set_positions([(0, 0, 0)])
    expected = second.get_state(energy=True).potential_energy
    force.updateParametersInContext(first)
    assert first.get_state(energy=True).potential_energy == expected != energy
    other = om.GBSAOBCForce()
    other.addParticle(0, 0.1, 1)
    try:
        other.updateParametersInContext(first)
    except RuntimeError:
        pass
    else:
        raise AssertionError('foreign force update was accepted')
    return {'definition_snapshots': True, 'particle_updates': True, 'metadata': True, 'invalid_indices': True}


def boundary_checks(om, rows):
    for row in rows:
        system = om.System()
        gb = om.GBSAOBCForce()
        for q in [-0.3, 0.4]:
            system.add_particle(1)
            gb.addParticle(q, 0.15, 0.8)
        system.set_default_periodic_box_vectors((2, 0, 0), (0, 2, 0), (0, 0, 2))
        gb.setNonbondedMethod(row['method'])
        gb.setCutoffDistance(0.8)
        system.add_force(gb)
        context = om.Context(system, om.VerletIntegrator(0.001))
        context.set_positions([(0, 0, 0), (row['distance'], 0, 0)])
        state = context.get_state(energy=True, forces=True)
        assert math.isclose(state.potential_energy, row['energy'], abs_tol=2e-10, rel_tol=2e-12)
        for got, expected in zip(flatten(state.forces), flatten(row['forces']), strict=True):
            assert math.isclose(got, expected, abs_tol=2e-10, rel_tol=2e-10), (row, got, expected)
    assert len(rows) == 6
    return len(rows)


def flatten(values):
    if isinstance(values, (list, tuple)):
        for value in values:
            yield from flatten(value)
    else:
        yield values


def small_radius_checks(om, rows):
    assert len(rows) == 2
    for row in rows:
        system = om.System()
        system.add_particle(1)
        gb = om.GBSAOBCForce()
        gb.addParticle(0.5, row['single_radius'], 1)
        system.add_force(gb)
        context = om.Context(system, om.VerletIntegrator(0.001))
        context.set_positions([(0, 0, 0)])
        state = context.get_state(energy=True, forces=True)
        assert math.isclose(state.potential_energy, row['energy'], abs_tol=1e-9, rel_tol=2e-12)
        assert list(flatten(state.forces)) == list(flatten(row['forces'])) == [0, 0, 0]
    return len(rows)


def stability_checks(om, rows):
    metrics = []
    assert len(rows) == 4
    controls = {row['settings']: row for row in rows if row['perturbation']}
    for row in [row for row in rows if not row['perturbation']]:
        system = om.System()
        gb, nb, bonds = om.GBSAOBCForce(), om.NonbondedForce(), om.HarmonicBondForce()
        if row['settings']:
            gb.setSoluteDielectric(2.1)
            gb.setSolventDielectric(35)
            gb.setSurfaceAreaEnergy(0)
        positions, velocities = [], []
        for i in range(16):
            system.add_particle(12)
            q = 0.1 if i % 2 else -0.1
            gb.addParticle(q, 0.15, 0.8)
            nb.add_particle(q, 0.25, 0.2)
            positions.append((0.32*(i % 4), 0.32*(i//4), 0.03*math.sin(i)))
            velocities.append((0.05*math.sin(1.7*i), 0.04*math.cos(0.9*i), 0.03*math.sin(0.3*i)))
        for i in range(16):
            for j in [i+1, i+4]:
                if j < 16 and (j == i+4 or i % 4 != 3):
                    distance = math.sqrt(sum((a-b)**2 for a, b in zip(positions[i], positions[j])))
                    bonds.add_bond(i, j, distance, 1000)
        for force in [gb, nb, bonds]:
            system.add_force(force)
        it = om.VerletIntegrator(0.0005)
        context = om.Context(system, it)
        context.set_positions(positions)
        context.set_velocities(velocities)
        initial = context.get_state(energy=True)
        e0 = initial.potential_energy+initial.kinetic_energy
        assert math.isclose(e0, row['initial_energy'], abs_tol=1e-10, rel_tol=1e-10)
        drift = 0
        short_errors = {}
        initial_momentum = [12*sum(v[axis] for v in velocities) for axis in range(3)]
        for sample in range(200):
            it.step(100)
            state = context.get_state(energy=True)
            drift = max(drift, abs((state.potential_energy+state.kinetic_energy-e0)/e0))
            if sample == 19:
                short = context.get_state(positions=True, velocities=True)
                for key in ['positions', 'velocities']:
                    error = max(abs(a-b) for a, b in zip(flatten(getattr(short, key)), flatten(row['short_'+key]), strict=True))
                    assert error < 5e-9, (key, error)
                    short_errors[key] = error
        state = context.get_state(energy=True, positions=True, velocities=True)
        final = state.potential_energy+state.kinetic_energy
        assert math.isclose(final, row['final_energy'], abs_tol=1e-8, rel_tol=1e-9)
        assert max(drift, row['max_relative_drift']) < 5e-5
        assert max(drift*abs(e0), row['max_relative_drift']*abs(row['initial_energy'])) < 1e-3
        assert math.isclose(context.time, 10, abs_tol=1e-9)
        errors, sensitivity, momentum_error = {}, {}, {}
        for key in ['positions', 'velocities']:
            errors[key] = max(abs(a-b) for a, b in zip(flatten(getattr(state, key)), flatten(row[key]), strict=True))
            sensitivity[key] = max(abs(a-b) for a, b in zip(flatten(row[key]), flatten(controls[row['settings']][key]), strict=True))
        for name, v in [('rust', state.velocities), ('reference', row['velocities'])]:
            momentum_error[name] = max(abs(12*sum(p[axis] for p in v)-initial_momentum[axis]) for axis in range(3))
            assert momentum_error[name] < 1e-9
        metrics.append({'settings': row['settings'], 'steps': 20000, 'time_ps': context.time,
                        'relative_energy_drift': drift, 'absolute_energy_drift': drift*abs(e0),
                        'reference_relative_energy_drift': row['max_relative_drift'],
                        'one_ps_errors': short_errors, 'ten_ps_errors': errors,
                        'reference_sensitivity_to_1e_12_nm': sensitivity, 'momentum_error': momentum_error})
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('openmm', args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    rows = [json.loads(line) for line in args.reference.read_text().splitlines()]
    boundaries = [row for row in rows if row.get('boundary')]
    stability = [row for row in rows if row.get('stability')]
    small_radii = [row for row in rows if 'single_radius' in row]
    rows = [row for row in rows if not row.get('boundary') and not row.get('stability') and 'single_radius' not in row]
    count = 0
    maximum = {}
    gradient_count, max_gradient = 0, 0
    for row in rows:
        phase = row['phase']
        if row['groups'] == 32:
            if phase == 0:
                context, gb, it = create(om, row)
            elif phase == 1:
                for i, parameters in enumerate(row['parameters']):
                    gb.setParticleParameters(i, *parameters)
                gb.setSoluteDielectric(7)
                gb.setSolventDielectric(11)
                gb.setSurfaceAreaEnergy(9)
                gb.setCutoffDistance(0.4)
                gb.setNonbondedMethod(gb.NoCutoff)
                if row['cv']:
                    context.update_force_parameters_at_path([0, 0], gb)
                else:
                    gb.updateParametersInContext(context)
            elif phase == 2:
                context.set_periodic_box_vectors(*map(tuple, row['box']))
                context.set_positions(list(map(tuple, row['positions'])))
            else:
                it.step(5)
        state = context.get_state(positions=True, velocities=True, forces=True, energy=True, groups=row['groups'])
        actual = {'energy': state.potential_energy, 'kinetic': state.kinetic_energy,
                  'time': context.time, 'positions': state.positions, 'velocities': state.velocities, 'forces': state.forces}
        for key, value in actual.items():
            for got, expected in zip(flatten(value), flatten(row[key]), strict=True):
                error = abs(got-expected)
                maximum[key] = max(maximum.get(key, 0), error)
                assert math.isclose(got, expected, abs_tol=5e-7, rel_tol=5e-9), (row['method'], row['geometry'], row['settings'], row['cv'], phase, row['groups'], key, got, expected)
                count += 1
        if phase == 0 and row['groups'] == 32 and row['geometry'] < 3:
            coordinates = [list(p) for p in state.positions]
            for i, force in enumerate(state.forces):
                for axis in range(3):
                    original = coordinates[i][axis]
                    coordinates[i][axis] = original+1e-6
                    context.set_positions(list(map(tuple, coordinates)))
                    plus = context.get_state(energy=True, groups=32).potential_energy
                    coordinates[i][axis] = original-1e-6
                    context.set_positions(list(map(tuple, coordinates)))
                    minus = context.get_state(energy=True, groups=32).potential_energy
                    coordinates[i][axis] = original
                    numerical = -(plus-minus)/2e-6
                    error = abs(force[axis]-numerical)
                    max_gradient = max(max_gradient, error)
                    assert error < 2e-5*(1+abs(force[axis])), (row['method'], row['geometry'], row['settings'], row['cv'], i, axis, force[axis], numerical)
                    gradient_count += 1
            context.set_positions(list(map(tuple, coordinates)))
    print(json.dumps({'states': len(rows), 'comparisons': count, 'max_absolute_error': maximum,
                      'finite_difference_components': gradient_count, 'max_gradient_error': max_gradient,
                      'boundaries': boundary_checks(om, boundaries), 'api': api_checks(om),
                      'stability': stability_checks(om, stability), 'small_radius_cases': small_radius_checks(om, small_radii)}, indent=2))


if __name__ == '__main__':
    main()
