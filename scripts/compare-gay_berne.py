"""Validate conservative Gay-Berne forces against finite differences of source energies."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess


def flatten(values):
    if isinstance(values, (tuple, list)):
        for value in values:
            yield from flatten(value)
    else:
        yield values


def converted(parameters):
    p = list(parameters)
    p[2], p[3] = int(p[2]), int(p[3])
    return p


def configure(f, row):
    for i, p in enumerate(row['parameters']):
        if i == f.getNumParticles():
            f.addParticle(*converted(p))
        else:
            f.setParticleParameters(i, *converted(p))
    for i, e in enumerate(row['exceptions']):
        e = [int(e[0]), int(e[1]), *e[2:]]
        if i == f.getNumExceptions():
            f.addException(*e)
        else:
            f.setExceptionParameters(i, *e)
    f.setNonbondedMethod(row['method'])
    f.setCutoffDistance(row['cutoff'])
    f.setSwitchingDistance(row['switch_distance'])
    f.setUseSwitchingFunction(bool(row['switching']))


def create(om, row):
    system = om.System()
    for i in range(8):
        system.add_particle(0 if i >= 6 else 12 if i % 3 == 0 else 1)
    system.set_virtual_site(7, om.TwoParticleAverageSite(0, 3, 0.4, 0.6))
    system.set_default_periodic_box_vectors(*map(tuple, row['box']))
    f = om.GayBerneForce()
    configure(f, row)
    if row['cv']:
        cv = om.CustomCVForce('0.7*g+0.001*g*g')
        cv.add_collective_variable('g', f)
        cv.set_force_group(5)
        system.add_force(cv)
    else:
        f.setForceGroup(5)
        system.add_force(f)
    external = om.CustomExternalForce('0.2*x*x+0.1*z')
    external.set_force_group(2)
    for i in range(8):
        external.add_particle(i, [])
    system.add_force(external)
    context = om.Context(system, om.VerletIntegrator(0.001))
    context.set_positions(list(map(tuple, row['positions'])))
    context.compute_virtual_sites()
    return context, f


def api_checks(om, probes, sphere_probe):
    f = om.GayBerneForce()
    assert f.getNumParticles() == f.getNumExceptions() == 0
    assert f.getNonbondedMethod() == f.NoCutoff == 0
    assert (f.CutoffNonPeriodic, f.CutoffPeriodic) == (1, 2)
    assert f.getCutoffDistance() == 1 and f.getSwitchingDistance() == -1
    assert not f.getUseSwitchingFunction() and not f.usesPeriodicBoundaryConditions()
    assert f.getName() == 'GayBerneForce' and f.getForceGroup() == 0
    f.setName('ellipsoids'); f.setForceGroup(31)
    assert f.get_name() == 'ellipsoids' and f.get_force_group() == 31
    for call in [lambda: f.addParticle(0.3, 1, -1, -1, 0.3, 0.2, 0.3, 1, 1, 1),
                 lambda: f.addParticle(0.3, 1, -2, -1, 0.3, 0.3, 0.3, 1, 1, 1),
                 lambda: f.getParticleParameters(0), lambda: f.getExceptionParameters(0),
                 lambda: f.setNonbondedMethod(3), lambda: f.setForceGroup(32)]:
        try:
            call()
        except (RuntimeError, IndexError):
            pass
        else:
            raise AssertionError('invalid Gay-Berne input was accepted')
    p = (0.3, 1, -1, -1, 0.3, 0.3, 0.3, 1, 1, 1)
    for _ in range(2): f.addParticle(*p)
    assert f.getParticleParameters(0) == p
    system = om.System(); system.add_particle(1); system.add_particle(1); system.add_force(f)
    c = om.Context(system, om.VerletIntegrator(0.001)); c.set_positions([(0, 0, 0), (1.5, 0, 0)])
    plain = c.get_state(energy=True).potential_energy
    f.setUseSwitchingFunction(True); f.setSwitchingDistance(0.8); f.updateParametersInContext(c)
    assert c.get_state(energy=True).potential_energy == plain
    values = {row['switching']: row['energy'] for row in probes}
    assert math.isclose(plain, values[0], abs_tol=1e-16) and abs(values[1]-plain) > 0.3
    f.setParticleParameters(0, 0.3, 0.5, -1, -1, 0.3, 0.3, 0.3, 1, 1, 1)
    assert c.get_state(energy=True).potential_energy == plain
    second = om.Context(system, om.VerletIntegrator(0.001)); second.set_positions([(0, 0, 0), (1.5, 0, 0)])
    updated = second.get_state(energy=True).potential_energy
    f.updateParametersInContext(c)
    assert c.get_state(energy=True).potential_energy == updated != plain
    assert f.addException(0, 1, 0.2, 0.7) == 0
    assert f.addException(1, 0, 0.25, 0.8, True) == 0
    assert f.getExceptionParameters(0) == (1, 0, 0.25, 0.8)
    try:
        f.addException(0, 1, 0.2, 0.1)
    except RuntimeError:
        pass
    else:
        raise AssertionError('duplicate exception was accepted')
    sphere_system = om.System(); sphere_force = om.GayBerneForce()
    for sigma in [0.2, 0.4]:
        sphere_system.add_particle(1)
        sphere_force.addParticle(sigma, 1, -1, -1, sigma, sigma, sigma, 1, 1, 1)
    sphere_system.add_force(sphere_force)
    sphere_context = om.Context(sphere_system, om.VerletIntegrator(0.001))
    sphere_context.set_positions([(0, 0, 0), (0.65, 0, 0)])
    sphere_energy = sphere_context.get_state(energy=True).potential_energy
    assert math.isclose(sphere_energy, sphere_probe['energy'], rel_tol=1e-13, abs_tol=1e-14)
    assert abs(sphere_energy-sphere_probe['lennard_jones_energy']) > 1e-3
    return {'snapshots': True, 'updates': True, 'symmetry_validation': True,
            'exception_replacement': True, 'no_cutoff_switch_ignored': True,
            'original_no_cutoff_energies': values, 'unequal_sphere_probe': sphere_probe}


def stability_checks(om, oracle):
    results = []
    for periodic in [False, True]:
        system = om.System(); f = om.GayBerneForce(); bonds = om.HarmonicBondForce()
        masses = [12, 1, 1, 12, 1, 1]
        parameters = [(0.3, 0.15, 1, 2, 0.3, 0.24, 0.2, 1, 0.8, 1.2),
                      (0.1, 0, -1, -1, 0.1, 0.1, 0.1, 1, 1, 1),
                      (0.1, 0, -1, -1, 0.1, 0.1, 0.1, 1, 1, 1),
                      (0.28, 0.8*0.15, 4, 5, 0.25, 0.3, 0.2, 1.3, 0.6, 0.9),
                      (0.1, 0, -1, -1, 0.1, 0.1, 0.1, 1, 1, 1),
                      (0.1, 0, -1, -1, 0.1, 0.1, 0.1, 1, 1, 1)]
        positions = [(0, 0, 0), (0.1, 0, 0), (0.03, 0.08, 0),
                     (0.7, 0.2, 0.1), (0.7, 0.29, 0.1), (0.75, 0.23, 0.17)]
        velocities = [(0.02*math.sin(1.7*i), 0.02*math.cos(0.9*i), 0.01*math.sin(0.3*i)) for i in range(6)]
        for mass, p in zip(masses, parameters, strict=True):
            system.add_particle(mass); f.addParticle(*p)
        f.setNonbondedMethod(f.CutoffPeriodic if periodic else f.NoCutoff)
        f.setCutoffDistance(1.2); f.setSwitchingDistance(0.9); f.setUseSwitchingFunction(periodic)
        system.set_default_periodic_box_vectors((3, 0, 0), (0.5, 3.2, 0), (-0.3, 0.4, 3.4))
        for first in [0, 3]:
            for i in range(first, first+3):
                for j in range(i+1, first+3):
                    bonds.add_bond(i, j, math.dist(positions[i], positions[j]), 5000)
        bonds.add_bond(0, 3, math.dist(positions[0], positions[3]), 50)
        system.add_force(f); system.add_force(bonds)
        it = om.VerletIntegrator(0.0001); context = om.Context(system, it)
        context.set_positions(positions); context.set_velocities(velocities)
        initial = context.get_state(energy=True); e0 = initial.potential_energy+initial.kinetic_energy
        initial_momentum = [sum(m*v[a] for m, v in zip(masses, velocities, strict=True)) for a in range(3)]
        drift, momentum = 0, 0
        samples = []
        for sample in range(200):
            it.step(100)
            state = context.get_state(positions=True, velocities=True, energy=True, forces=sample in (0, 199))
            drift = max(drift, abs(state.potential_energy+state.kinetic_energy-e0))
            momentum = max(momentum, max(abs(sum(m*v[a] for m, v in zip(masses, state.velocities, strict=True))-initial_momentum[a]) for a in range(3)))
            samples.append({'positions': state.positions, 'energy': state.potential_energy,
                            'forces': state.forces if sample in (0, 199) else None})
        assert drift < 2e-6, (periodic, drift)
        assert momentum < 1e-9, (periodic, momentum)
        assert math.isclose(context.time, 2, abs_tol=1e-9)
        payload = '\n'.join(' '.join(format(v, '.17g') for v in flatten(s['positions'])) for s in samples)+'\n'
        output = subprocess.run([str(oracle.resolve()), '--oracle', str(int(periodic))], input=payload,
                                text=True, capture_output=True, check=True)
        reference = [json.loads(line) for line in output.stdout.splitlines()]
        assert len(reference) == len(samples) == 200
        energy_error, force_error, force_checks = 0, 0, 0
        for sample, ref in zip(samples, reference, strict=True):
            energy_error = max(energy_error, abs(sample['energy']-ref['energy']))
            assert math.isclose(sample['energy'], ref['energy'], abs_tol=1e-10, rel_tol=1e-10)
            if 'forces_fd' in ref:
                for got, expected in zip(flatten(sample['forces']), flatten(ref['forces_fd']), strict=True):
                    force_error = max(force_error, abs(got-expected)); force_checks += 1
                    assert math.isclose(got, expected, abs_tol=2e-6, rel_tol=2e-7), (periodic, got, expected)
        results.append({'periodic': periodic, 'steps': 20000, 'time_ps': context.time,
                        'max_absolute_energy_drift': drift, 'max_momentum_error': momentum,
                        'source_energy_frames': len(reference), 'max_source_energy_error': energy_error,
                        'source_force_fd_components': force_checks, 'max_source_force_fd_error': force_error})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, default=Path(__file__).resolve().parents[1]/'target/reference-gay_berne')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('openmm', args.extension)
    om = importlib.util.module_from_spec(spec); spec.loader.exec_module(om)
    rows = [json.loads(line) for line in args.reference.read_text().splitlines()]
    probes = [r for r in rows if r.get('original_no_cutoff_probe')]
    sphere_probe = next(r for r in rows if r.get('original_sphere_probe'))
    rows = [r for r in rows if not r.get('original_no_cutoff_probe') and not r.get('original_sphere_probe')]
    energy_error, force_error, source_force_error, checks = 0, 0, 0, 0
    for row in rows:
        if row['groups'] == 32:
            if row['phase'] == 0:
                context, f = create(om, row)
            else:
                if row['phase'] in (1, 2):
                    configure(f, row)
                    if row['cv']: context.update_force_parameters_at_path([0, 0], f)
                    else: f.updateParametersInContext(context)
                context.set_periodic_box_vectors(*map(tuple, row['box']))
                context.set_positions(list(map(tuple, row['positions'])))
                context.compute_virtual_sites()
        state = context.get_state(energy=True, forces=True, groups=row['groups'])
        energy_error = max(energy_error, abs(state.potential_energy-row['energy']))
        assert math.isclose(state.potential_energy, row['energy'], abs_tol=1e-10, rel_tol=2e-12), (row, state.potential_energy)
        for got, expected, original in zip(flatten(state.forces[:7]), flatten(row['forces_fd']), flatten(row['forces_original'][:7]), strict=True):
            force_error = max(force_error, abs(got-expected))
            source_force_error = max(source_force_error, abs(original-expected))
            assert math.isclose(got, expected, abs_tol=2e-6, rel_tol=2e-7), (row, got, expected)
            if row['kind'] == 0:
                assert math.isclose(got, original, abs_tol=2e-9, rel_tol=2e-11), (row, got, original)
            checks += 1
    assert len(rows) == 576 and checks == 12096, (len(rows), checks)
    assert source_force_error > 0.001
    print(json.dumps({'states': len(rows), 'comparisons': checks+len(rows),
                      'max_energy_error': energy_error, 'finite_difference_components': checks,
                      'max_force_fd_error': force_error, 'original_reference_force_fd_error': source_force_error,
                      'api': api_checks(om, probes, sphere_probe), 'stability': stability_checks(om, args.oracle)}, indent=2))


if __name__ == '__main__':
    main()
