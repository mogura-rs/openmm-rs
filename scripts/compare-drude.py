"""Compare anisotropic Drude springs, screened dipoles and captured definitions."""
import argparse
import importlib.util
import json
import math
from pathlib import Path


def flatten(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def configure(force, mode, update):
    for i in range(2):
        p2 = (4 if i == 0 else 6) if mode in (1, 3, 6, 7) else -1
        p3 = (5 if i == 0 else 4) if mode in (2, 3, 5, 6, 7) else -1
        p4 = 7 if mode in (2, 3, 4, 6, 7) else -1
        charge = (0 if mode == 6 and i == 0 else -.12 if i == 0 else .08)*(1.2 if update else 1)
        polar = (.001+.0004*i)*(1.1 if update else 1)
        a1 = (1.8 if mode == 7 else .7)+.03*i+(.04 if update else 0)
        a2 = (1.5 if mode == 7 else 1.2)-.02*i-(.05 if update else 0)
        args = (2*i+1, 2*i, p2, p3, p4, charge, polar, a1, a2)
        if update:
            force.setParticleParameters(i, *args)
        else:
            force.addParticle(*args)
    if update:
        force.setScreenedPairParameters(0, 0, 1, 2.1)
    else:
        force.addScreenedPair(0, 1, 1.7)


def create(om, row):
    system = om.System()
    for i in range(8):
        system.add_particle(0 if i >= 6 else 12 if i % 2 == 0 else .4)
    system.set_virtual_site(7, om.TwoParticleAverageSite(4, 6, .4, .6))
    system.set_default_periodic_box_vectors(*map(tuple, row['box']))
    force = om.DrudeForce(); configure(force, row['mode'], False)
    force.setUsesPeriodicBoundaryConditions(bool(row['periodic'])); force.setForceGroup(5)
    if row['cv']:
        outer = om.CustomCVForce('0.7*a+0.001*a*a'); outer.add_collective_variable('a', force); outer.set_force_group(5); system.add_force(outer)
    else:
        system.add_force(force)
    external = om.CustomExternalForce('.2*x*x+.1*z'); external.set_force_group(2)
    for i in range(8):
        external.add_particle(i, [])
    system.add_force(external)
    it = om.VerletIntegrator(.000001); context = om.Context(system, it)
    context.set_positions(list(map(tuple, row['positions']))); context.compute_virtual_sites(); context.set_velocities([(.01, -.02, .03)]*8)
    assert context.get_molecules() == [[0, 1], [2, 3], [4, 6, 7], [5]], context.get_molecules()
    return context, force, it


def api_checks(om, group_probe):
    force = om.DrudeForce()
    assert force.getName() == 'DrudeForce' and force.getForceGroup() == 0
    assert force.getNumParticles() == force.getNumScreenedPairs() == 0
    assert not force.usesPeriodicBoundaryConditions()
    force.setName('polarization'); force.setForceGroup(31)
    assert force.get_name() == 'polarization' and force.get_force_group() == 31
    parameters = (1, 0, -1, -1, -1, -.1, .001, -2., -3.)
    assert force.addParticle(*parameters) == 0
    assert tuple(force.getParticleParameters(0)) == parameters
    for call in [lambda: force.addParticle(3, 2, -1, -1, -1, -.1, 0, 1, 1),
                 lambda: force.addParticle(3, 2, 4, -1, -1, -.1, .001, 0, 1),
                 lambda: force.getParticleParameters(1), lambda: force.getScreenedPairParameters(0),
                 lambda: force.setForceGroup(32)]:
        try:
            call()
        except (RuntimeError, ValueError, IndexError):
            pass
        else:
            raise AssertionError('invalid parameters were accepted')
    system = om.System(); system.add_particle(12); system.add_particle(.4); system.add_force(force)
    first = om.Context(system, om.VerletIntegrator(.001)); first.set_positions([(0, 0, 0), (.01, .02, .03)])
    before = first.get_state(energy=True).potential_energy
    force.setParticleParameters(0, 1, 0, -1, -1, -1, -.2, .001, 1, 1)
    assert first.get_state(energy=True).potential_energy == before
    second = om.Context(system, om.VerletIntegrator(.001)); second.set_positions([(0, 0, 0), (.01, .02, .03)])
    expected = second.get_state(energy=True).potential_energy
    force.updateParametersInContext(first)
    assert first.get_state(energy=True).potential_energy == expected != before
    force.setForceGroup(3)
    assert first.get_state(energy=True, groups=1 << 31).potential_energy == 0
    assert first.get_state(energy=True, groups=8).potential_energy == expected
    system = om.System(); system.add_particle(12); system.add_particle(.4)
    force = om.DrudeForce(); force.addParticle(1, 0, -1, -1, -1, -.1, .001, 1, 1); system.add_force(force)
    it = om.CustomIntegrator(.001); it.addGlobalVariable('a', 0); it.addComputeGlobal('a', 'energy0')
    context = om.Context(system, it); positions = [(0, 0, 0), (.01, .02, .03)]; context.set_positions(positions)
    it.step(1); energies = [it.getGlobalVariable(0)]
    force.setForceGroup(1); it.step(1); energies.append(it.getGlobalVariable(0))
    context.set_positions(positions); it.step(1); energies.append(it.getGlobalVariable(0))
    assert group_probe is not None and group_probe[0] > 0 and group_probe[1:] == [0, 0]
    for actual, expected in zip(energies, group_probe, strict=True):
        assert abs(actual-expected) < 1e-12
    return {'defaults': True, 'indices': True, 'snapshots': True, 'numeric_updates': True, 'live_root_group': True, 'custom_integrator_group_cache': True, 'molecule_connectivity': True}


def stability_checks(om, rows):
    assert len(rows) == 4
    metrics = []
    for row in rows:
        periodic, anisotropic = bool(row['periodic']), bool(row['anisotropic'])
        system = om.System(); system.set_default_periodic_box_vectors((3, 0, 0), (.4, 3.2, 0), (-.3, .2, 3.4))
        positions = [(.1, .2, .3), (.11, .185, .31), (2.2, .45, .55), (2.19, .46, .53), (.45, .3, .7), (.6, .85, .4)]
        velocities = [(.01*math.sin(1.7*i), .012*math.cos(.9*i), .008*math.sin(.3*i)) for i in range(6)]
        masses = [.4 if i in (1, 3) else 12 for i in range(6)]
        for mass in masses:
            system.add_particle(mass)
        force = om.DrudeForce()
        force.addParticle(1, 0, 4 if anisotropic else -1, 4 if anisotropic else -1, 5 if anisotropic else -1, -.12, .001, .7, 1.2)
        force.addParticle(3, 2, 5 if anisotropic else -1, 0 if anisotropic else -1, 4 if anisotropic else -1, .08, .0014, .75, 1.15)
        force.addScreenedPair(0, 1, 1.7); force.setUsesPeriodicBoundaryConditions(periodic); system.add_force(force)
        bonds = om.HarmonicBondForce()
        for i in [0, 2, 4, 5]:
            for j in [0, 2, 4, 5]:
                if j > i:
                    bonds.add_bond(i, j, math.dist(positions[i], positions[j]), 500)
        system.add_force(bonds)
        it = om.VerletIntegrator(.0001); context = om.Context(system, it); context.set_positions(positions); context.set_velocities(velocities)
        state = context.get_state(energy=True); initial = state.potential_energy+state.kinetic_energy
        assert abs(initial-row['initial_energy']) < 1e-10
        drift = maximum_distance = 0; short_errors = {}
        for sample in range(500):
            it.step(100); state = context.get_state(energy=True, positions=True)
            drift = max(drift, abs(state.potential_energy+state.kinetic_energy-initial))
            maximum_distance = max(maximum_distance, *(math.dist(state.positions[i], state.positions[i+1]) for i in (0, 2)))
            if sample == 9:
                short = context.get_state(positions=True, velocities=True)
                for key in ['positions', 'velocities']:
                    error = max(abs(a-b) for a, b in zip(flatten(getattr(short, key)), flatten(row['short_'+key]), strict=True))
                    assert error < 1e-8, (key, error)
                    short_errors[key] = error
        state = context.get_state(positions=True, velocities=True)
        assert max(drift, row['max_absolute_drift']) < 1e-5, (drift, row['max_absolute_drift'])
        assert max(maximum_distance, row['max_drude_distance']) < .04
        assert math.isclose(context.time, 5, abs_tol=1e-9)
        errors, momentum_errors = {}, {}
        for key in ['positions', 'velocities']:
            error = max(abs(a-b) for a, b in zip(flatten(getattr(state, key)), flatten(row[key]), strict=True))
            assert error < 1e-6, (key, error)
            errors[key] = error
        initial_momentum = [sum(m*v[axis] for m, v in zip(masses, velocities)) for axis in range(3)]
        for name, velocities in [('rust', state.velocities), ('reference', row['velocities'])]:
            error = max(abs(sum(m*v[axis] for m, v in zip(masses, velocities))-initial_momentum[axis]) for axis in range(3))
            assert error < 1e-8, (name, error)
            momentum_errors[name] = error
        metrics.append({'periodic': periodic, 'anisotropic': anisotropic, 'steps': 50000, 'duration_ps': context.time,
                        'max_absolute_drift': drift, 'reference_max_absolute_drift': row['max_absolute_drift'],
                        'max_drude_distance': maximum_distance, 'short_errors': short_errors, 'final_errors': errors, 'momentum_errors': momentum_errors})
    return metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--extension', type=Path, required=True); parser.add_argument('--reference', type=Path, required=True); args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('openmm', args.extension); om = importlib.util.module_from_spec(spec); spec.loader.exec_module(om)
    maximum = {}; states = checks = fd_checks = 0; fd_error = 0; stability = []; group_probe = None
    for line in args.reference.read_text().splitlines():
        row = json.loads(line)
        if row.get('group_cache_probe'):
            group_probe = row['energies']
            continue
        if row.get('stability'):
            stability.append(row)
            continue
        if row.get('serialization_probe'):
            assert row['original_periodic'] and not row['copy_periodic'] and not row['inner_periodic']
            continue
        if row.get('fd'):
            state = context.get_state(forces=True, groups=32)
            for actual, expected in zip(flatten(state.forces[:7]), flatten(row['forces'][:7]), strict=True):
                fd_checks += 1; fd_error = max(fd_error, abs(actual-expected))
                assert math.isclose(actual, expected, abs_tol=2e-7, rel_tol=2e-8), (actual, expected)
            continue
        if row['groups'] == 32:
            phase = row['phase']
            if phase == 0:
                context, force, it = create(om, row)
            elif phase == 1:
                configure(force, row['mode'], True); force.setUsesPeriodicBoundaryConditions(not row['periodic'])
                if row['cv']:
                    context.update_force_parameters_at_path([0, 0], force)
                else:
                    force.setForceGroup(6); force.updateParametersInContext(context)
            elif phase == 2:
                context.set_periodic_box_vectors(*map(tuple, row['box'])); context.set_positions(list(map(tuple, row['positions']))); context.compute_virtual_sites()
            else:
                it.step(5)
        state = context.get_state(positions=True, velocities=True, forces=True, energy=True, groups=row['groups'])
        actual = {'energy': state.potential_energy, 'kinetic': state.kinetic_energy, 'time': context.time, 'positions': state.positions, 'velocities': state.velocities, 'forces': state.forces}
        for name, values in actual.items():
            for actual, expected in zip(flatten(values), flatten(row[name]), strict=True):
                checks += 1; maximum[name] = max(maximum.get(name, 0), abs(actual-expected))
                assert math.isclose(actual, expected, abs_tol=2e-8, rel_tol=2e-10), (row, name, actual, expected)
        states += 1
    assert states == 1024 and fd_checks == 1344, (states, fd_checks)
    print(json.dumps({'states': states, 'comparisons': checks, 'max_absolute_error': maximum, 'finite_difference': {'components': fd_checks, 'max_absolute_error': fd_error}, 'api': api_checks(om, group_probe), 'stability': stability_checks(om, stability)}, indent=2))


if __name__ == '__main__':
    main()
