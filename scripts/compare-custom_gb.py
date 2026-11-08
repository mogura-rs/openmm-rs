"""Compare multistage solvation expressions and chain derivatives with Reference."""
import argparse
import importlib.util
import json
import math
from pathlib import Path


def parameters(kind, i):
    if kind == 1:
        return [0.1*(i % 7-3), 0.11+0.02*(i % 5), 0.65+0.1*(i % 4)]
    return [0.2+0.1*i if kind == 0 else i % 3]


def model(om, kind, exclusion):
    f = om.CustomGBForce()
    pair = f.ParticlePairNoExclusions if exclusion else f.ParticlePair
    single = f.SingleParticle
    if kind == 1:
        for name in ['q', 'radius', 'scale']:
            f.addPerParticleParameter(name)
        f.addGlobalParameter('solvent', 78.3)
        f.addGlobalParameter('solute', 1)
        f.addEnergyParameterDerivative('solvent')
        f.addEnergyParameterDerivative('solute')
        f.addComputedValue('I', 'step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);U=r+sr2;C=2*(1/or1-1/L)*step(sr2-r-or1);L=max(or1,D);D=abs(r-sr2);sr2=scale2*or2;or1=radius1-0.009;or2=radius2-0.009', pair)
        f.addComputedValue('B', '1/(1/or-tanh(psi-0.8*psi^2+4.85*psi^3)/radius);psi=I*or;or=radius-0.009', single)
        f.addEnergyTerm('28.391955111572114*(radius+0.14)^2*(radius/B)^6-0.5*138.93545764438198*(1/solute-1/solvent)*q^2/B', single)
        f.addEnergyTerm('-138.93545764438198*(1/solute-1/solvent)*q1*q2/sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))', f.ParticlePair)
    else:
        f.addPerParticleParameter('p')
        f.addGlobalParameter('lambda', 0.7)
        f.addGlobalParameter('beta', 0.2)
        f.addEnergyParameterDerivative('lambda')
        f.addEnergyParameterDerivative('beta')
        if kind == 0:
            f.addComputedValue('a', 'lambda*(p1+2*p2)*exp(-r)+beta*r^2', pair)
            f.addComputedValue('b', 'exp(-0.1*a)+p*x*y+lambda*z', single)
            f.addComputedValue('c', 'a*b+beta*b^2+0.2*sin(x)', single)
            f.addComputedValue('d', 'c+sqrt(b*b+1)+0.3*a', single)
            f.addEnergyTerm('0.15*d^2+lambda*b+0.2*x*x', single)
            f.addEnergyTerm('0.03*(c1+c2)*r+beta*p1*p2/(1+r)+0.07*a1*a2', f.ParticlePair)
            f.addEnergyTerm('0.04*b1*b2*cos(r)', f.ParticlePairNoExclusions)
        else:
            f.addTabulatedFunction('t', om.Continuous1DFunction([0, 0.2, 0.1, 0.4, 0.3], -1, 3))
            data2 = [0.1*(i+2*j)+0.02*i*j for j in range(3) for i in range(3)]
            data3 = [0.05*(i+2*j+3*k)+0.01*i*j*k for k in range(3) for j in range(3) for i in range(3)]
            f.addTabulatedFunction('t2', om.Continuous2DFunction(3, 3, data2, 0, 3, -1, 1))
            f.addTabulatedFunction('t3', om.Continuous3DFunction(3, 3, 3, data3, 0, 3, -1, 1, -1, 1))
            f.addTabulatedFunction('d1', om.Discrete1DFunction([0.1, 0.2, -0.1]))
            f.addTabulatedFunction('d2', om.Discrete2DFunction(3, 3, data2))
            f.addTabulatedFunction('d3', om.Discrete3DFunction(3, 3, 3, data3))
            f.addComputedValue('a', 'lambda*(t(r)+t2(r,0.3*p2)+t3(r,0.2*p1,0.2*p2))+beta*(d1(p2)+d2(p1,p2)+d3(p1,p2,1))', pair)
            f.addComputedValue('b', 'sqrt(1+a*a)+lambda*t(x)+beta*y', single)
            f.addEnergyTerm('0.1*b*b+t(x)+beta*z*z', single)
            f.addEnergyTerm('0.1*(a1+a2)*exp(-r)+lambda*b1*b2', f.ParticlePairNoExclusions)
    for i in range(8):
        f.addParticle(parameters(kind, i))
    f.addExclusion(0, 1)
    f.addExclusion(3, 5)
    return f


def create(om, row):
    kind, method, exclusion, nested, sample = (row[k] for k in ['kind', 'method', 'exclusion', 'nested', 'sample'])
    system = om.System()
    for i in range(8):
        system.add_particle(0 if i in (0, 7) else 2+i)
    system.set_virtual_site(7, om.TwoParticleAverageSite(5, 6, 0.3, 0.7))
    box = [(2.6, 0, 0), (0.5 if sample % 2 else 0, 2.8, 0), (-0.3 if sample % 2 else 0, 0.4 if sample % 2 else 0, 3)]
    system.set_default_periodic_box_vectors(*box)
    f = model(om, kind, exclusion)
    f.setNonbondedMethod(method)
    f.setCutoffDistance(1.1)
    if nested:
        cv = om.CustomCVForce('1.3*g+0.0007*g*g')
        cv.add_collective_variable('g', f)
        for i in range(f.getNumGlobalParameters()):
            name = f.getGlobalParameterName(i)
            cv.add_global_parameter(name, f.getGlobalParameterDefaultValue(i))
            cv.add_energy_parameter_derivative(name)
        cv.set_force_group(3)
        system.add_force(cv)
    else:
        f.setForceGroup(3)
        system.add_force(f)
    it = om.VerletIntegrator(0.0002)
    it.set_integration_force_groups(8)
    context = om.Context(system, it)
    context.set_positions([(0.1+0.31*(i % 3)+0.013*sample, 0.15+0.37*((i//3) % 3)+0.007*sample*i, 0.2+0.09*(i % 2)+0.011*sample) for i in range(8)])
    context.set_velocities([(0.001*i, -0.002*i, 0.001*(i % 3)) for i in range(8)])
    context.compute_virtual_sites()
    return context, it, f, box


def flatten(values):
    if isinstance(values, (tuple, list)):
        for value in values:
            yield from flatten(value)
    elif isinstance(values, dict):
        for key in sorted(values):
            yield values[key]
    else:
        yield values


def compare(om, rows):
    previous = None
    maxima = dict.fromkeys(['energy', 'kinetic', 'time', 'positions', 'velocities', 'forces', 'derivatives'], 0)
    checks = 0
    for row in rows:
        key = tuple(row[k] for k in ['kind', 'method', 'exclusion', 'nested', 'sample'])
        if key != previous:
            context, it, f, box = create(om, row)
            previous, phase = key, 0
        if row['phase'] != phase:
            phase = row['phase']
            if phase == 1:
                it.step(3)
            elif phase == 2:
                kind = row['kind']
                context.set_parameter('solvent' if kind == 1 else 'lambda', 55 if kind == 1 else 0.9)
                context.set_parameter('solute' if kind == 1 else 'beta', 1.3 if kind == 1 else 0.3)
                p = parameters(kind, 2)
                p[0] += 0.07 if kind == 1 else (1 if kind == 2 else 0.2)
                f.setParticleParameters(2, p)
                if kind == 2:
                    f.getTabulatedFunction(0).set_function_parameters([0.1, 0.3, 0.2, 0.5, 0.4], -1, 3)
                context.update_force_parameters_at_path([0, 0] if row['nested'] else [0], f)
                it.step(2)
            else:
                box[0] = tuple(v*1.03 for v in box[0])
                box[1] = tuple(v*1.02 for v in box[1])
                context.set_periodic_box_vectors(*box)
                it.step(1)
        state = context.get_state(energy=True, positions=True, velocities=True, forces=True, parameter_derivatives=True, groups=row['groups'] & 0xffffffff)
        actual = {'energy': state.potential_energy, 'kinetic': state.kinetic_energy, 'time': context.time, 'positions': state.positions, 'velocities': state.velocities, 'forces': state.forces, 'derivatives': state.energy_parameter_derivatives}
        for name, data in actual.items():
            for got, expected in zip(flatten(data), flatten(row[name]), strict=True):
                error = abs(got-expected)
                maxima[name] = max(maxima[name], error)
                checks += 1
                assert math.isclose(got, expected, abs_tol=2e-8 if name != 'time' else 1e-15, rel_tol=2e-9), (key, phase, row['groups'], name, got, expected)
    return {'states': len(rows), 'comparisons': checks, 'max_absolute_error': maxima}


def gradient_checks(om, rows):
    maxima = {'forces': 0, 'derivatives': 0}
    checks = 0
    for row in rows:
        context, _, _, _ = create(om, row)
        state = context.get_state(forces=True, parameter_derivatives=True)
        actual = {'forces': state.forces[:7], 'derivatives': state.energy_parameter_derivatives}
        for name, data in actual.items():
            for value, expected in zip(flatten(data), flatten(row[name]), strict=True):
                maxima[name] = max(maxima[name], abs(value-expected))
                assert math.isclose(value, expected, rel_tol=2e-6, abs_tol=3e-6), (row, name, value, expected)
                checks += 1
    return {'states': len(rows), 'components': checks, 'max_absolute_error': maxima}


def api_checks(om, rows):
    f = om.CustomGBForce()
    assert (f.NoCutoff, f.CutoffNonPeriodic, f.CutoffPeriodic) == (0, 1, 2)
    assert (f.SingleParticle, f.ParticlePair, f.ParticlePairNoExclusions) == (0, 1, 2)
    assert f.getName() == 'CustomGBForce' and f.getForceGroup() == 0
    assert f.getNumComputedValues() == f.getNumEnergyTerms() == 0
    for call in [lambda: f.getComputedValueParameters(0), lambda: f.getEnergyTermParameters(0), lambda: f.getParticleParameters(0), lambda: f.getExclusionParticles(0), lambda: f.getTabulatedFunctionName(0), lambda: f.getGlobalParameterName(0), lambda: f.getPerParticleParameterName(0), lambda: f.setNonbondedMethod(3), lambda: f.setForceGroup(32), lambda: f.addComputedValue('x', '1', 3), lambda: f.addEnergyParameterDerivative('missing')]:
        try:
            call()
        except (RuntimeError, IndexError):
            pass
        else:
            raise AssertionError('invalid input was accepted')
    f.addComputedValue('a', 'r', f.ParticlePair)
    f.addEnergyTerm('a+t(x)', f.SingleParticle)
    table = om.Continuous1DFunction([0, 1, 2, 3], 0, 3)
    f.addTabulatedFunction('t', table)
    s = om.System()
    for _ in range(2):
        s.add_particle(1)
        f.addParticle([])
    s.add_force(f)
    c = om.Context(s, om.VerletIntegrator(0.001))
    c.set_positions([(0.5, 0, 0), (1.5, 0, 0)])
    initial = c.get_state(energy=True).potential_energy
    f.setComputedValueParameters(0, 'a', '2*r', f.ParticlePair)
    f.setEnergyTermParameters(0, '3*a+t(x)', f.SingleParticle)
    f.updateParametersInContext(c)
    unchanged = c.get_state(energy=True).potential_energy
    table.set_function_parameters([0, 2, 4, 6], 0, 3)
    f.updateParametersInContext(c)
    updated = c.get_state(energy=True).potential_energy
    assert (initial, unchanged, updated) == (4, 4, 16)
    f.setFunctionParameters(0, 'u', [0, 3, 6, 9, 12], 0, 4)
    f.setEnergyTermParameters(0, '3*a+u(x)', f.SingleParticle)
    assert f.getFunctionParameters(0) == ('u', [0, 3, 6, 9, 12], 0, 4)
    assert f.getNumFunctions() == f.getNumTabulatedFunctions() == 1
    f.updateParametersInContext(c)
    resized = c.get_state(energy=True).potential_energy
    assert len(rows) == 1 and rows[0]['energies'] == [initial, unchanged, updated, resized] == [4, 4, 16, 18]
    legacy = om.CustomGBForce()
    assert legacy.addFunction('linear', [0, 1], 0, 1) == 0
    assert legacy.getFunctionParameters(0) == ('linear', [0, 1], 0, 1)
    legacy.addTabulatedFunction('discrete', om.Discrete1DFunction([0, 1]))
    try:
        legacy.getFunctionParameters(1)
    except RuntimeError:
        pass
    else:
        raise AssertionError('legacy function getter accepted a discrete table')
    return {'table_triggered_expression_reload': [initial, unchanged, updated, resized], 'legacy_functions': True, 'invalid_indices': True}


def stability_checks(om, rows):
    result = []
    for row in rows:
        system = om.System()
        for _ in range(8):
            system.add_particle(12)
        system.set_default_periodic_box_vectors((3, 0, 0), (0, 3, 0), (0, 0, 3))
        f = model(om, 1, 1)
        f.setNonbondedMethod(row['method'])
        f.setCutoffDistance(1.1)
        system.add_force(f)
        nb = om.NonbondedForce()
        nb.set_nonbonded_method('CutoffPeriodic' if row['method'] == 2 else 'NoCutoff')
        nb.set_cutoff_distance(1.1)
        nb.set_reaction_field_dielectric(1)
        nb.set_use_dispersion_correction(False)
        positions = [(0.2+0.33*(i % 3), 0.2+0.35*(i//3), 0.2+0.08*(i % 2)) for i in range(8)]
        velocities = [(0.01*math.sin(i), 0.01*math.cos(i), 0.005*math.sin(2*i)) for i in range(8)]
        for i in range(8):
            nb.add_particle(parameters(1, i)[0], 0.23, 0.05)
        system.add_force(nb)
        bonds = om.HarmonicBondForce()
        pairs = [(i, j, 500) for i in range(8) for j in range(i+1, 8)] if row['method'] == 2 and not row['boundary_crossing'] else [(i, i+1, 150) for i in range(7)]
        for i, j, stiffness in pairs:
            bonds.add_bond(i, j, math.dist(positions[i], positions[j]), stiffness)
        system.add_force(bonds)
        it = om.VerletIntegrator(0.0001)
        context = om.Context(system, it)
        context.set_positions(positions)
        context.set_velocities(velocities)
        initial = context.get_state(energy=True)
        e0 = initial.potential_energy+initial.kinetic_energy
        assert math.isclose(e0, row['initial_energy'], rel_tol=1e-12, abs_tol=1e-10)
        drift = 0
        max_pair = 0
        short_errors = {}
        for sample in range(500):
            it.step(100)
            state = context.get_state(energy=True, positions=True)
            max_pair = max(max_pair, max(math.dist(state.positions[i], state.positions[j]) for i in range(8) for j in range(i+1, 8)))
            drift = max(drift, abs(state.potential_energy+state.kinetic_energy-e0))
            if sample == 9:
                short = context.get_state(positions=True, velocities=True)
                for key in ['positions', 'velocities']:
                    error = max(abs(a-b) for a, b in zip(flatten(getattr(short, key)), flatten(row['short_'+key]), strict=True))
                    assert error < 5e-9, (key, error)
                    short_errors[key] = error
        state = context.get_state(positions=True, velocities=True)
        momentum0 = [12*sum(v[k] for v in velocities) for k in range(3)]
        momentum_errors = {}
        for name, v in [('rust', state.velocities), ('reference', row['velocities'])]:
            momentum_errors[name] = max(abs(12*sum(p[k] for p in v)-momentum0[k]) for k in range(3))
            assert momentum_errors[name] < 1e-8, momentum_errors
        if row['boundary_crossing']:
            assert min(max_pair, row['maximum_pair_distance']) > 1.1
            assert math.isclose(drift, row['max_absolute_drift'], abs_tol=1e-4, rel_tol=1e-5)
        else:
            assert max(drift, row['max_absolute_drift']) < 2e-3, (drift, row['max_absolute_drift'])
            if row['method'] == 2:
                assert max(max_pair, row['maximum_pair_distance']) < 1.07
        errors = {key: max(abs(a-b) for a, b in zip(flatten(getattr(state, key)), flatten(row[key]), strict=True)) for key in ['positions', 'velocities']}
        result.append({'method': row['method'], 'boundary_crossing': row['boundary_crossing'], 'maximum_pair_distance': max_pair, 'steps': 50000, 'time_ps': context.time, 'max_absolute_energy_drift': drift,
                       'reference_max_absolute_energy_drift': row['max_absolute_drift'], 'point_one_ps_errors': short_errors,
                       'five_ps_errors': errors, 'momentum_errors': momentum_errors})
    assert len(result) == 3
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
    result = compare(om, [row for row in rows if not row.get("gradient") and not row.get("stability") and not row.get("api_probe")])
    result["finite_difference"] = gradient_checks(om, [row for row in rows if row.get("gradient")])
    result['api'] = api_checks(om, [row for row in rows if row.get('api_probe')])
    result['stability'] = stability_checks(om, [row for row in rows if row.get("stability")])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
