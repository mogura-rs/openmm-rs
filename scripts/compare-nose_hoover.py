"""Compare Nose-Hoover chains, corrected pair COM, full KE and live bindings."""
import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path


def create(om, row):
    system = om.System()
    for mass in [1, 2, 3, 4, 0, 5]:
        system.add_particle(mass)
    if row['constrained']:
        system.add_constraint(2, 3, 1)
    if row['virtual_site']:
        system.set_virtual_site(4, om.TwoParticleAverageSite(0, 2, 0.3, 0.7))
    bond = om.HarmonicBondForce()
    bond.add_bond(0, 1, 0.8, 2)
    bond.set_force_group(1)
    system.add_force(bond)
    force = om.CustomExternalForce('0.4*x^2+0.3*y^2+0.2*z^2')
    for a in range(5):
        force.add_particle(a, [])
    force.set_force_group(2)
    system.add_force(force)
    integrator = om.NoseHooverIntegrator(0.001)
    args = [row['length'], row['mts'], row['ys']]
    mode = row['mode']
    if mode == 0:
        integrator.addThermostat(300, 1.7, *args)
    if mode == 1:
        integrator.addSubsystemThermostat([0, 1], [], 300, 1.7, 20, 2.3, *args)
        integrator.addSubsystemThermostat([2, 3], [], 140, 0.9, 20, 2.3, *args)
    if mode in [2, 3]:
        integrator.addSubsystemThermostat([2, 3] if mode == 2 else [], [(0, 1)], 300, 1.7, 20, 2.3, *args)
    integrator.setConstraintTolerance(1e-10)
    assert integrator.computeHeatBathEnergy() == 0
    context = om.Context(system, integrator)
    assert integrator.computeHeatBathEnergy() == 0
    context.set_positions([(0, 0, 0), (0.9, 0, 0), (0, 0.7, 1.1), (0, 1.7, 1.1), (0.2, 0.4, 0.5), (0, 0, 0)])
    context.compute_virtual_sites()
    context.set_velocities([(0.2, 0.1, -0.1), (-0.3, 0.4, 0.2), (0.1, -0.2, 0.3), (0.2, 0.3, -0.1), (0.7, 0.8, 0.9), (0, 0, 0) if mode == 3 else (0.6, 0.7, 0.8)])
    context.apply_velocity_constraints(1e-10)
    context.set_time(2.7)
    context.set_step_count(17)
    integrator.step(1)
    return context, integrator


def reject(callback):
    try:
        callback()
    except (RuntimeError, OverflowError):
        return
    raise AssertionError('invalid operation was accepted')


def probes(om):
    for value in [0, -1, math.nan, math.inf]:
        reject(lambda: om.NoseHooverIntegrator(300, value, 0.001))
        reject(lambda: om.NoseHooverIntegrator(value, 1, 0.001))
        reject(lambda: om.NoseHooverIntegrator(value))
    for length, mts, ys in [(0, 3, 7), (3, 0, 7), (3, 3, 2)]:
        reject(lambda: om.NoseHooverIntegrator(300, 1, 0.001, length, mts, ys))
    s = om.System()
    for mass in [1, 2, 0]:
        s.add_particle(mass)
    s.add_constraint(0, 1, 1)
    for atoms, pairs in [([0], []), ([2], []), ([9], []), ([0, 0], []), ([], [(0, 0)]), ([], [(2, 1)])]:
        i = om.NoseHooverIntegrator(step_size=0.001)
        i.addSubsystemThermostat(atoms, pairs, 300, 1, 20, 2)
        reject(lambda: om.Context(s, i))
    i = om.NoseHooverIntegrator(300, 1, 0.001)
    reject(lambda: i.addThermostat(300, 1))
    reject(lambda: i.step(1))
    chain = i.getThermostat(0)
    assert chain.getNumDegreesOfFreedom() == 0 and chain.getChainID() == 0
    c = om.Context(s, i)
    assert chain.getThermostatedAtoms() == [0, 1] and chain.getNumDegreesOfFreedom() == 5
    assert not chain.usesPeriodicBoundaryConditions()
    assert len(chain.getYoshidaSuzukiWeights()) == 7
    assert i.getAllThermostatedIndividualParticles() == [0, 1, 2]
    assert i.getAllThermostatedPairs() == []
    assert i.get_num_thermostats() == i.getNumThermostats() == 1
    assert not i.hasSubsystemThermostats()
    reject(lambda: chain.setThermostatedAtoms([0]))
    reject(lambda: i.addThermostat(300, 1))
    reject(lambda: om.Context(s, i))
    for value in [0, -1, math.nan, math.inf]:
        reject(lambda: i.setTemperature(value))
        reject(lambda: chain.setCollisionFrequency(value))
    assert i.getTemperature() == 300 and chain.getCollisionFrequency() == 1
    del c
    gc.collect()
    root = om.CompoundIntegrator()
    root.addIntegrator(om.VerletIntegrator(0.001))
    root.addIntegrator(i)
    c = om.Context(s, root)
    c.set_positions([(0, 0, 0), (1, 0, 0), (0, 0, 0)])
    c.set_velocities([(0, 1, 0)] * 3)
    i.step(1)
    root.setCurrentIntegrator(1)
    root.step(1)
    assert c.step_count == 2 and i.computeHeatBathEnergy() != 0
    independent = om.NoseHooverChain(300, 20, 1, 2, 6, 3, 3, 7, 4, [0, 1], [])
    independent.setTemperature(200)
    assert independent.getTemperature() == 200 and independent.getChainID() == 4
    fixed = om.System()
    fixed.add_particle(0)
    inactive = om.NoseHooverIntegrator(300, 1, 0.001)
    fixed_context = om.Context(fixed, inactive)
    inactive.setTemperature(200)
    inactive.step(1)
    assert inactive.computeHeatBathEnergy() == 0
    assert inactive.getThermostat(0).getNumDegreesOfFreedom() == 0
    assert fixed_context.get_state(energy=True).kinetic_energy == 0
    # One-bead free-pair propagation has a direct scalar recurrence.
    s = om.System()
    s.add_particle(1)
    s.add_particle(1)
    i = om.NoseHooverIntegrator(0.001)
    i.addSubsystemThermostat([], [(0, 1)], 300, 1, 300, 1, 1, 1, 1)
    c = om.Context(s, i)
    c.set_positions([(0, 0, 0), (1, 0, 0)])
    c.set_velocities([(2, 0, 0), (0, 0, 0)])
    assert c.get_state(energy=True).kinetic_energy == 2
    i.step(1)
    kt = 1.380649e-23 * 6.02214076e23 / 1000 * 300
    scale = math.exp(-0.5e-6 * (2 - 3 * kt) / (3 * kt))
    v = c.get_state(velocities=True, energy=True)
    assert math.isclose(v.velocities[0][0], 2 * scale, rel_tol=2e-14)
    assert abs(v.velocities[1][0]) < 1e-14
    assert math.isclose(v.kinetic_energy, 2 * scale**2, rel_tol=2e-14)


def hard_wall_cases(om, rows):
    previous = None
    errors = []
    for row in rows:
        mass = row['parent_mass']
        if mass != previous:
            s = om.System()
            for m in [1, mass, 3]:
                s.add_particle(m)
            force = om.CustomExternalForce('-0.1*x')
            for a in range(3):
                force.add_particle(a, [])
            s.add_force(force)
            i = om.NoseHooverIntegrator(0.001)
            i.addSubsystemThermostat([2], [(0, 1)], 300, 1, 30, 2, 3, 3, 7)
            i.setMaximumPairDistance(0.8)
            c = om.Context(s, i)
            c.set_positions([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
            c.set_velocities([(-2, 0.2, 0.1), (0.3, 0.1, 0.4), (0.4, 0.5, 0.3)])
            i.setRelativeTemperature(200)
            assert i.getAllThermostatedPairs() == [(0, 1, 30)]
            assert i.getRelativeTemperature() == 200
            previous = mass
        if row['step']:
            i.setMaximumPairDistance(0.75)
        i.step(1)
        state = c.get_state(positions=True, velocities=True, energy=True)
        values = [(state.kinetic_energy, row['kinetic']), (i.computeHeatBathEnergy(), row['bath'])]
        for actual, expected in [(state.positions, row['positions']), (state.velocities, row['velocities'])]:
            values.extend((a, b) for x, y in zip(actual, expected) for a, b in zip(x, y))
        for actual, expected in values:
            assert math.isclose(actual, expected, rel_tol=2e-13, abs_tol=2e-13), (row, actual, expected)
            errors.append(abs(actual-expected))
    return {'states': len(rows), 'comparisons': len(errors), 'max_absolute_error': max(errors)}


def stability(om, row):
    system = om.System()
    force = om.CustomExternalForce('5*x^2+7.5*y^2+10*z^2')
    positions, velocities = [], []
    for a in range(24):
        system.add_particle(1 + 0.1*a)
        force.add_particle(a, [])
        positions.append((0.1*math.sin(a+0.3), 0.2*math.cos(a+0.1), 0.15*math.sin(0.7*a)))
        velocities.append((math.cos(a+0.3), math.sin(a+0.1), math.cos(0.7*a)))
    system.add_force(force)
    integrator = om.NoseHooverIntegrator(row['temperature'], row['frequency'], 0.001, 3, 3, 7)
    context = om.Context(system, integrator)
    context.set_positions(positions)
    context.set_velocities(velocities)
    integrator.step(20000)
    state = context.get_state(energy=True)
    initial = state.kinetic_energy + state.potential_energy + integrator.computeHeatBathEnergy()
    mean, m2, mean_potential, max_drift, final_energy = 0, 0, 0, 0, 0
    for j in range(1, 10001):
        integrator.step(10)
        state = context.get_state(energy=True)
        kinetic = state.kinetic_energy
        difference = kinetic - mean
        mean += difference/j
        m2 += difference*(kinetic-mean)
        mean_potential += (state.potential_energy-mean_potential)/j
        final_energy = kinetic + state.potential_energy + integrator.computeHeatBathEnergy()
        max_drift = max(max_drift, abs(final_energy-initial))
    result = {'temperature': row['temperature'], 'frequency': row['frequency'], 'mean_kinetic': mean,
              'variance_kinetic': m2/10000, 'mean_potential': mean_potential, 'max_extended_energy_drift': max_drift,
              'initial_extended_energy': initial, 'final_extended_energy': final_energy}
    kt = (1.380649e-23*6.02214076e23/1000)*row['temperature']
    for sample in [row, result]:
        assert abs(sample['mean_kinetic']/(36*kt)-1) < 0.1, sample
        assert abs(sample['variance_kinetic']/(36*kt*kt)-1) < 0.3, sample
        assert sample['max_extended_energy_drift']/(36*kt) < 0.01, sample
    # Chaotic chains are compared by block moments and bounded extended-energy drift.
    assert math.isclose(result['mean_kinetic'], row['mean_kinetic'], rel_tol=0.05), (result, row)
    result['source_mean_kinetic'] = row['mean_kinetic']
    result['source_variance_kinetic'] = row['variance_kinetic']
    result['source_max_extended_energy_drift'] = row['max_extended_energy_drift']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('openmm', args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    rows = [json.loads(line) for line in args.reference.read_text().splitlines()]
    stability_rows = [row for row in rows if row.get("stability")]
    rows = [row for row in rows if not row.get("stability")]
    assert len(stability_rows) == 6
    wall_rows = [row for row in rows if row.get("hard_wall")]
    rows = [row for row in rows if not row.get("hard_wall")]
    assert len(wall_rows) == 4
    raw_probes = {row["probe"]: row for row in rows if "probe" in row}
    rows = [row for row in rows if "probe" not in row]
    probe = raw_probes["raw_pair_only"]
    assert raw_probes["relative_only"]["velocities"] == [[1, 0, 0], [-1, 0, 0]]
    corrected_system = om.System()
    for _ in range(2):
        corrected_system.add_particle(1)
    corrected_integrator = om.NoseHooverIntegrator(0.001)
    corrected_integrator.addSubsystemThermostat([], [(0, 1)], 300, 1, 300, 1, 1, 1, 1)
    corrected_context = om.Context(corrected_system, corrected_integrator)
    corrected_context.set_positions([(0, 0, 0), (1, 0, 0)])
    corrected_context.set_velocities([(1, 0, 0), (-1, 0, 0)])
    corrected_integrator.step(1)
    corrected_velocity = corrected_context.get_state(velocities=True).velocities
    kt = 1.380649e-23*6.02214076e23/1000*300
    bath_half = 0.0005*(2-3*kt)/(3*kt)
    scale = math.exp(-0.001*bath_half)
    assert scale != 1 and math.isclose(corrected_velocity[0][0], scale, abs_tol=1e-15)
    assert math.isclose(corrected_velocity[1][0], -scale, abs_tol=1e-15)
    bath_velocity = bath_half+0.0005*(2*scale*scale-3*kt)/(3*kt)
    expected_bath = 0.5*(3*kt)*bath_velocity*bath_velocity+(3*kt)*0.001*bath_half
    assert math.isclose(corrected_integrator.computeHeatBathEnergy(), expected_bath, rel_tol=1e-12, abs_tol=1e-18)
    assert not raw_probes["zero_radial_wall"]["finite"]
    assert probe['probe'] == 'raw_pair_only' and probe['initial_reported'] == 1 and probe['final_reported'] == 0
    assert abs(sum(v[0] for v in probe['velocities'])) < 1e-15
    assert len(rows) == 3584
    errors = {key: [] for key in ['energy', 'kinetic', 'bath', 'position', 'velocity', 'force', 'time']}
    previous = None
    source_ke_gap = 0
    for row in rows:
        key = tuple(row[k] for k in ['length', 'mts', 'ys', 'mode', 'constrained', 'virtual_site'])
        if key != previous:
            context, integrator = create(om, row)
            previous = key
        event = row['event']
        if row['sample'] == 0:
            if event == 1:
                integrator.setTemperature(170)
                integrator.setCollisionFrequency(2.1)
                integrator.setRelativeTemperature(30)
                integrator.setRelativeCollisionFrequency(1.9)
            if event == 2:
                integrator.setStepSize(0.0007)
            if event == 3:
                integrator.setIntegrationForceGroups(2)
            if event == 4:
                chain = integrator.getThermostat(0)
                chain.setTemperature(230)
                chain.setRelativeTemperature(15)
                chain.setCollisionFrequency(1.3)
                chain.setRelativeCollisionFrequency(1.1)
                chain.setNumDegreesOfFreedom(chain.getNumDegreesOfFreedom() + 1)
            if event == 5:
                integrator.setMaximumPairDistance(0.85)
            if event == 6:
                context.set_velocities_to_temperature(0)
                integrator.setIntegrationForceGroups(-1)
        else:
            integrator.step(3)
        state = context.get_state(positions=True, velocities=True, forces=True, energy=True)
        assert context.step_count == row['step_count']
        assert [integrator.getThermostat(i).getNumDegreesOfFreedom() for i in range(integrator.getNumThermostats())] == row['dof']
        values = [('energy', state.potential_energy, row['energy']), ('kinetic', state.kinetic_energy, row['kinetic']),
                  ('bath', integrator.computeHeatBathEnergy(), row['bath']), ('time', context.time, row['time'])]
        for name, actual, expected in [('position', state.positions, row['positions']), ('velocity', state.velocities, row['velocities']), ('force', state.forces, row['forces'])]:
            values.extend((name, a, b) for x, y in zip(actual, expected) for a, b in zip(x, y))
        for name, actual, expected in values:
            assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=2e-8, abs_tol=2e-8), (key, event, row['sample'], name, actual, expected)
            errors[name].append(abs(actual - expected))
        source_ke_gap = max(source_ke_gap, abs(row['kinetic'] - row['source_kinetic']))
    probes(om)
    wall_result = hard_wall_cases(om, wall_rows)
    stability_results = [stability(om, row) for row in stability_rows]
    print(json.dumps({'status': 'passed', 'fixtures': len(rows), 'comparisons': sum(map(len, errors.values())),
                      'max_absolute_error': {k: max(v) for k, v in errors.items()}, 'raw_source_probes': raw_probes,
                      'source_kinetic_max_omission': source_ke_gap, 'analytic_pair_probe': 'passed',
                      'binding_and_validation_probes': 'passed', 'stability': stability_results, 'hard_wall_cases': wall_result}, indent=2))


if __name__ == '__main__':
    main()
