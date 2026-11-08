"""Compare LCPO surface forces, parameter updates and conservative trajectories."""
import argparse
import importlib.util
import json
import math
from pathlib import Path


def flatten(values):
    if isinstance(values, (tuple, list)):
        for value in values:
            yield from flatten(value)
    else:
        yield values


def parameters(setting, i):
    radius = 0 if setting == 1 and i % 3 == 1 else 0.18+0.02*(i % 5)
    p = [radius, 0.7+0.05*(i % 4), -0.2+0.03*(i % 3), 0.08+0.01*(i % 3), 0.1+0.02*(i % 5)]
    if setting == 2 and i % 3 == 1:
        p[1:] = [0]*4
    return p


def create(om, row):
    n = len(row['positions'])
    system = om.System()
    force = om.LCPOForce()
    force.setSurfaceTension([1, 5, -2][row['setting']])
    force.setUsesPeriodicBoundaryConditions(bool(row['periodic']))
    for i in range(n):
        system.add_particle(0 if n == 8 and i in (0, 7) else 2+i % 4)
        force.addParticle(*parameters(row['setting'], i))
    if n == 8:
        system.set_virtual_site(7, om.TwoParticleAverageSite(2, 6, 0.4, 0.6))
    system.set_default_periodic_box_vectors(*map(tuple, row['box']))
    if row['cv']:
        cv = om.CustomCVForce('0.7*a+0.001*a*a')
        cv.add_collective_variable('a', force)
        cv.set_force_group(5)
        system.add_force(cv)
    else:
        force.setForceGroup(5)
        system.add_force(force)
    external = om.CustomExternalForce('0.2*x*x+0.1*z')
    external.set_force_group(2)
    for i in range(n):
        external.add_particle(i, [])
    system.add_force(external)
    it = om.VerletIntegrator(0.00001)
    context = om.Context(system, it)
    context.set_positions(list(map(tuple, row['positions'])))
    context.compute_virtual_sites()
    context.set_velocities([(0.01, -0.02, 0.03)]*n)
    return context, force, it


def api_checks(om):
    f = om.LCPOForce()
    assert f.getNumParticles() == f.num_particles() == 0
    assert f.getSurfaceTension() == 1 and not f.usesPeriodicBoundaryConditions()
    assert f.getName() == 'LCPOForce' and f.getForceGroup() == 0
    f.setName('surface'); f.setForceGroup(31)
    assert f.get_name() == 'surface' and f.get_force_group() == 31
    assert f.addParticle(0.2, 1, 2, 3, 4) == 0
    assert f.getParticleParameters(0) == (0.2, 1, 2, 3, 4)
    for call in [lambda: f.getParticleParameters(1), lambda: f.setParticleParameters(1, 1, 2, 3, 4, 5), lambda: f.setForceGroup(32)]:
        try:
            call()
        except (RuntimeError, IndexError):
            pass
        else:
            raise AssertionError('invalid LCPO index was accepted')
    system = om.System(); system.add_particle(1); system.add_force(f)
    f.setParticleParameters(0, 0.3, 1, 2, 3, 4)
    first = om.Context(system, om.VerletIntegrator(0.001)); first.set_positions([(0, 0, 0)])
    before = first.get_state(energy=True).potential_energy
    f.setSurfaceTension(2); f.setParticleParameters(0, 0.4, 1, 2, 3, 4)
    assert first.get_state(energy=True).potential_energy == before
    second = om.Context(system, om.VerletIntegrator(0.001)); second.set_positions([(0, 0, 0)])
    expected = second.get_state(energy=True).potential_energy
    f.updateParametersInContext(first)
    assert first.get_state(energy=True).potential_energy == expected != before
    return {'snapshots': True, 'particle_and_tension_updates': True, 'metadata': True, 'indices': True}


def stability_checks(om, rows):
    assert len(rows) == 2
    metrics = []
    for row in rows:
        system = om.System(); force = om.LCPOForce(); bonds = om.HarmonicBondForce()
        force.setSurfaceTension(5); force.setUsesPeriodicBoundaryConditions(bool(row['periodic']))
        system.set_default_periodic_box_vectors((3, 0, 0), (0.7, 3.2, 0), (-0.6, 0.4, 3.4))
        positions, velocities = [], []
        for i in range(8):
            system.add_particle(12); force.addParticle(0.3, 0.7, -0.2, 0.08, 0.1)
            positions.append((0.12*(i % 2), 0.12*((i//2) % 2), 0.12*(i//4)))
            velocities.append((0.02*math.sin(1.7*i), 0.02*math.cos(0.9*i), 0.01*math.sin(0.3*i)))
        for i in range(8):
            for j in range(i+1, 8):
                bonds.add_bond(i, j, math.dist(positions[i], positions[j]), 1000)
        system.add_force(force); system.add_force(bonds)
        it = om.VerletIntegrator(0.0002); context = om.Context(system, it)
        context.set_positions(positions); context.set_velocities(velocities)
        initial = context.get_state(energy=True); e0 = initial.potential_energy+initial.kinetic_energy
        drift, maximum_distance = 0, 0
        short_errors = {}
        initial_momentum = [12*sum(v[axis] for v in velocities) for axis in range(3)]
        for sample in range(250):
            it.step(100)
            state = context.get_state(energy=True, positions=True)
            drift = max(drift, abs(state.potential_energy+state.kinetic_energy-e0))
            maximum_distance = max(maximum_distance, max(math.dist(a, b) for a in state.positions for b in state.positions))
            if sample == 4:
                short = context.get_state(positions=True, velocities=True)
                for key in ['positions', 'velocities']:
                    error = max(abs(a-b) for a, b in zip(flatten(getattr(short, key)), flatten(row['short_'+key]), strict=True))
                    assert error < 5e-9, (key, error)
                    short_errors[key] = error
        state = context.get_state(positions=True, velocities=True)
        assert max(maximum_distance, row['max_distance']) < 0.55
        assert max(drift, row['max_absolute_drift']) < 1e-5, (drift, row['max_absolute_drift'])
        assert math.isclose(context.time, 5, abs_tol=1e-9)
        errors, momentum = {}, {}
        for key in ['positions', 'velocities']:
            errors[key] = max(abs(a-b) for a, b in zip(flatten(getattr(state, key)), flatten(row[key]), strict=True))
        for name, v in [('rust', state.velocities), ('reference', row['velocities'])]:
            momentum[name] = max(abs(12*sum(x[axis] for x in v)-initial_momentum[axis]) for axis in range(3))
            assert momentum[name] < 1e-8, (name, momentum[name])
        metrics.append({'periodic': row['periodic'], 'steps': 25000, 'time_ps': context.time,
                        'max_absolute_energy_drift': drift, 'reference_max_absolute_energy_drift': row['max_absolute_drift'],
                        'max_pair_distance': maximum_distance, 'point_one_ps_errors': short_errors,
                        'five_ps_errors': errors, 'momentum_errors': momentum})
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('openmm', args.extension)
    om = importlib.util.module_from_spec(spec); spec.loader.exec_module(om)
    rows = [json.loads(line) for line in args.reference.read_text().splitlines()]
    stability = [r for r in rows if r.get('stability')]
    rows = [r for r in rows if not r.get('stability')]
    checks, states, fd_checks, fd_error = 0, 0, 0, 0
    maximum = {}
    for row in rows:
        if row.get('fd'):
            state = context.get_state(forces=True, groups=32)
            for got, expected in zip(flatten(state.forces[:7]), flatten(row['forces'][:7]), strict=True):
                fd_error = max(fd_error, abs(got-expected)); fd_checks += 1
                assert math.isclose(got, expected, abs_tol=5e-7, rel_tol=2e-7), (row, got, expected)
            continue
        phase = row['phase']
        if row['groups'] == 32:
            if phase == 0:
                context, force, it = create(om, row)
            elif phase == 1:
                for i in range(force.getNumParticles()):
                    p = parameters(row['setting'], i)
                    p[0] = 0.21 if p[0] == 0 else p[0]+0.013
                    for j, scale in enumerate([0.9, 1.1, 0.8, 1.2], 1): p[j] *= scale
                    force.setParticleParameters(i, *p)
                force.setSurfaceTension(2.3); force.setUsesPeriodicBoundaryConditions(not row['periodic'])
                if row['cv']: context.update_force_parameters_at_path([0, 0], force)
                else: force.updateParametersInContext(context)
            elif phase == 2:
                context.set_periodic_box_vectors(*map(tuple, row['box']))
                context.set_positions(list(map(tuple, row['positions'])))
                context.compute_virtual_sites()
            else:
                it.step(5)
        state = context.get_state(positions=True, velocities=True, forces=True, energy=True, groups=row['groups'])
        actual = {'energy': state.potential_energy, 'kinetic': state.kinetic_energy, 'time': context.time,
                  'positions': state.positions, 'velocities': state.velocities, 'forces': state.forces}
        for key, value in actual.items():
            for got, expected in zip(flatten(value), flatten(row[key]), strict=True):
                maximum[key] = max(maximum.get(key, 0), abs(got-expected)); checks += 1
                assert math.isclose(got, expected, abs_tol=2e-9, rel_tol=2e-10), (row, key, got, expected)
        states += 1
    assert states == 768 and fd_checks == 252, (states, fd_checks)
    print(json.dumps({'states': states, 'comparisons': checks, 'max_absolute_error': maximum,
                      'finite_difference': {'components': fd_checks, 'max_absolute_error': fd_error},
                      'api': api_checks(om), 'stability': stability_checks(om, stability)}, indent=2))


if __name__ == '__main__':
    main()
