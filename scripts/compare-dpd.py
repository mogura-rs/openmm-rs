"""Compare sequential DPD updates against OpenMM Reference, including live box changes."""
import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path


def create(om, row):
    s = om.System()
    n = 64 if row["types"] else 27
    for i in range(n):
        s.add_particle(1+i % 4)
    if row["site"]:
        s.add_particle(0)
        s.set_virtual_site(n, om.TwoParticleAverageSite(0, 26, 0.3, 0.7))
    if row["constrained"]:
        s.add_constraint(0, 1, 0.37)
    box = row["box"]
    vectors = [(2.8, 0, 0), (0.7 if box == 2 else 0, 2.6, 0),
               (-0.6 if box == 2 else 0, 0.55 if box == 2 else 0, 2.4)]
    s.set_default_periodic_box_vectors(*vectors)
    bond = om.HarmonicBondForce()
    bond.add_bond(2, 26, 0.9, 1.2)
    bond.set_uses_periodic_boundary_conditions(bool(box))
    bond.set_force_group(1)
    s.add_force(bond)
    ext = om.CustomExternalForce("0.1*x^2+0.03*y-0.05*z")
    for i in range(n+row["site"]):
        ext.add_particle(i, [])
    ext.set_force_group(3)
    s.add_force(ext)
    dpd = om.DPDIntegrator(0, 1.3, 0.85, 0.003)
    dpd.setRandomNumberSeed(191)
    dpd.setConstraintTolerance(1e-11)
    labels = [-7, 0, 12345]
    if row["types"]:
        for i in range(n):
            dpd.setParticleType(i, labels[i % 3])
        for args in [(-7, 0, 2.3, 1.05), (12345, 12345, 0.4, 1.12), (0, -7, 0.6, 0.82), (88, 89, 7, 50)]:
            dpd.addTypePair(*args)
    if row["site"]:
        dpd.setParticleType(n, 99)
        for label in labels:
            dpd.addTypePair(99, label, 0, 1e-8)
    top = dpd
    if row["routing"]:
        top = om.CompoundIntegrator()
        branch = om.CompoundIntegrator()
        branch.addIntegrator(dpd)
        top.addIntegrator(branch)
        top.addIntegrator(om.VerletIntegrator(0.001))
        top.setCurrentIntegrator(1 if row["routing"] == 2 else 0)
    context = om.Context(s, top)
    p, v = [], []
    for i in range(n):
        value = [0.37*(i % 3)-0.3, 0.39*((i//3) % 3)-0.6, 0.41*(i//9)-0.5]
        if box and i >= 2:
            for divisor, vector, sign in [(5, vectors[2], 1), (7, vectors[1], -1), (11, vectors[0], 1)]:
                if i % divisor == 0:
                    value = [x+sign*y for x, y in zip(value, vector)]
        p.append(tuple(value))
        v.append((0.12*(i % 4)-0.15, 0.09*(i % 5)-0.11, 0.07*(i % 3)-0.08))
    if row["site"]:
        p.append((0, 0, 0))
        v.append((0.2, 0.3, 0.4))
    context.set_positions(p)
    context.compute_virtual_sites()
    context.set_velocities(v)
    context.apply_velocity_constraints(1e-11)
    context.set_time(1.7)
    context.set_step_count(11)
    return context, top, dpd, vectors


def stochastic_probes(om, rows):
    kb = 1.380649e-23*6.02214076e23/1000
    results = []
    for row in rows:
        s = om.System()
        t, friction = row["temperature"], row["friction"]
        pair = row["kind"] == "pair_moments"
        for mass in ([2, 5] if pair else [1]*64):
            s.add_particle(mass)
        if not pair:
            s.set_default_periodic_box_vectors((3, 0, 0), (0, 3, 0), (0, 0, 3))
            marker = om.HarmonicBondForce()
            marker.set_uses_periodic_boundary_conditions(True)
            s.add_force(marker)
        i = om.DPDIntegrator(0 if pair else t, friction, 1 if pair else 1.2, 0.01 if pair else 0.002)
        i.setRandomNumberSeed(713 if pair else 719)
        if not pair:
            i.setIntegrationForceGroups(0)
        c = om.Context(s, i)
        i.setTemperature(t)
        values, max_momentum = [], 0
        if not pair:
            c.set_positions([(0.75*(i % 4), 0.75*((i//4) % 4), 0.75*(i//16)) for i in range(64)])
            c.set_velocities([(0, 0, 0)]*64)
            i.step(10000)
        for _ in range(row["count"]):
            if pair:
                c.set_positions([(0, 0, 0), (0.6, 0, 0)])
                c.set_velocities([(0.3, 0.2, -0.1), (-0.4, 0.2, -0.1)])
            i.step(1 if pair else 10)
            state = c.get_state(velocities=True, energy=True)
            v = state.velocities
            if pair:
                values.append(v[1][0]-v[0][0])
                max_momentum = max(max_momentum, abs(2*v[0][0]+5*v[1][0]+1.4))
                assert v[0][1:] == v[1][1:] == (0.2, -0.1)
            else:
                values.append(state.kinetic_energy)
                max_momentum = max(max_momentum, *(abs(sum(a[k] for a in v)) for k in range(3)))
        mean = sum(values)/len(values)
        variance = sum((v-mean)**2 for v in values)/len(values)
        if pair:
            radius = 0.6+0.005*(-0.4-0.3)
            scale = math.exp(-0.01*2*friction*(1-radius)**2)
            target_mean = -0.7*scale
            target_var = (1-scale**2)*kb*t/(10/7)
            for label, m, var in [("rust", mean, variance), ("reference", row["mean"], row["variance"])]:
                assert abs(m-target_mean) < 6*math.sqrt(target_var/len(values)), (row, label, m, target_mean)
                assert abs(var/target_var-1) < 0.07, (row, label, var, target_var)
        else:
            target_mean = 0.5*3*63*kb*t
            target_var = 0.5*3*63*(kb*t)**2
            for label, m, var in [("rust", mean, variance), ("reference", row["mean"], row["variance"])]:
                assert abs(m/target_mean-1) < 0.06, (row, label, m, target_mean)
                assert abs(var/target_var-1) < 0.20, (row, label, var, target_var)
        assert max_momentum < 3e-10 and row["max_momentum_error"] < 3e-10, (row, max_momentum)
        results.append({"kind": row["kind"], "temperature": t, "friction": friction,
                        "target_mean": target_mean, "target_variance": target_var,
                        "rust_mean": mean, "rust_variance": variance, "reference_mean": row["mean"],
                        "reference_variance": row["variance"], "rust_max_momentum_error": max_momentum,
                        "reference_max_momentum_error": row["max_momentum_error"]})
    return results


def api_probes(om):
    def reject(action):
        try:
            action()
        except (RuntimeError, ValueError, OverflowError):
            return
        raise AssertionError("invalid DPD operation was accepted")
    for values in [(-1, 1, 1, 0.01), (0, -1, 1, 0.01), (0, 1, 0, 0.01), (0, 1, 1, 0), (math.nan, 1, 1, 0.01)]:
        reject(lambda: om.DPDIntegrator(*values))
    s = om.System()
    s.add_particle(2)
    s.add_particle(5)
    i = om.DPDIntegrator(300, 2, 1, 0.01)
    i.setRandomNumberSeed(873)
    assert i.getRandomNumberSeed() == 873
    assert i.getTemperature() == i.get_temperature() == 300
    assert i.getDefaultFriction() == 2 and i.getDefaultCutoff() == 1
    assert i.getParticleTypes() == {} and i.getParticleType(100) == 0
    i.setParticleType(-1, -77)
    i.set_particle_type(100, 91)
    assert i.get_particle_types() == {-1: -77, 100: 91}
    pair = i.addTypePair(92, 93, -1, 1)
    assert pair == 0 and i.getNumTypePairs() == 1
    assert i.getTypePairParameters(0) == (92, 93, -1, 1)
    reject(lambda: om.Context(s, i))
    i.set_type_pair_parameters(0, 92, 93, 0, 1)
    reject(lambda: i.getTypePairParameters(1))
    reject(lambda: i.step(1))
    c = om.Context(s, i)
    reject(lambda: om.Context(s, i))
    for setter in [i.setTemperature, i.setDefaultFriction, i.setDefaultCutoff, i.setStepSize, i.setConstraintTolerance]:
        reject(lambda: setter(-1))
        reject(lambda: setter(math.inf))
    def draw():
        c.set_positions([(0, 0, 0), (0.6, 0, 0)])
        c.set_velocities([(0, 0, 0)]*2)
        i.step(1)
        return c.get_state(velocities=True).velocities
    first = draw()
    i.setRandomNumberSeed(123)
    second = draw()
    j = om.DPDIntegrator(300, 2, 1, 0.01)
    j.setRandomNumberSeed(873)
    other = om.Context(s, j)
    samples = []
    for _ in range(2):
        other.set_positions([(0, 0, 0), (0.6, 0, 0)])
        other.set_velocities([(0, 0, 0)]*2)
        j.step(1)
        samples.append(other.get_state(velocities=True).velocities)
    assert samples == [first, second]
    i.setTemperature(0)
    assert draw() == [(0, 0, 0)]*2
    i.setDefaultFriction(0)
    del c
    gc.collect()
    c = om.Context(s, i)
    i.setTemperature(300)
    assert draw() == [(0, 0, 0)]*2
    fixed = om.System()
    fixed.add_particle(1)
    fixed.add_particle(0)
    k = om.DPDIntegrator(0, 1, 1, 0.01)
    ctx = om.Context(fixed, k)
    ctx.set_positions([(0, 0, 0), (0.5, 0, 0)])
    k.step(1)
    state = ctx.get_state(positions=True, velocities=True)
    assert state.positions == [(0, 0, 0), (0.5, 0, 0)]
    assert state.velocities == [(0, 0, 0)]*2
    return "passed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("openmm", args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    rows = [json.loads(line) for line in args.reference.read_text().splitlines()]
    invalid_pairs = [row for row in rows if row.get("kind") == "invalid_pair"]
    assert len(invalid_pairs) == 4 and all(not row["finite"] for row in invalid_pairs)
    stochastic_rows = [row for row in rows if row.get("kind") in ["pair_moments", "equilibrium"]]
    rows = [row for row in rows if "kind" not in row]
    assert len(rows) == 1728 and len(stochastic_rows) == 9
    previous = None
    errors = {name: [] for name in ["position", "velocity", "force", "energy", "kinetic", "time", "dt"]}
    for row in rows:
        key = tuple(row[name] for name in ["box", "types", "constrained", "site", "routing"])
        if key != previous:
            context, top, dpd, vectors = create(om, row)
            previous = key
        event = row["event"]
        if row["sample"] == 0:
            dpd.setStepSize(0.001+0.0003*(event % 4))
            dpd.setIntegrationForceGroups([-1, 2, 8, 10][event % 4])
            dpd.setConstraintTolerance(1e-10/(1+event % 3))
            if event == 3:
                dpd.setDefaultFriction(8)
                dpd.setDefaultCutoff(0.01)
                dpd.setParticleType(3, 999)
                if row["types"]:
                    dpd.setTypePairParameters(0, 0, 0, 7, 0.01)
                dpd.addTypePair(0, 0, 50, 0.001)
            if event == 5 and row["box"]:
                vectors = [tuple(x*scale for x in v) for v, scale in zip(vectors, [1.04, 0.97, 1.02])]
                context.set_periodic_box_vectors(*vectors)
            if event == 7:
                context.set_velocities_to_temperature(0)
        else:
            (top if row["routing"] == 1 else dpd).step(2)
        state = context.get_state(positions=True, velocities=True, forces=True, energy=True)
        assert state.step_count == row["step_count"]
        pairs = [("energy", state.potential_energy, row["energy"]), ("kinetic", state.kinetic_energy, row["kinetic"]),
                 ("time", state.time, row["time"]), ("dt", dpd.getStepSize(), row["dt"])]
        for metric, field in [("position", "positions"), ("velocity", "velocities"), ("force", "forces")]:
            pairs += [(metric, got, want) for a, b in zip(getattr(state, field), row[field]) for got, want in zip(a, b)]
        for metric, got, want in pairs:
            assert math.isfinite(got) and math.isclose(got, want, rel_tol=3e-8, abs_tol=3e-9), (key, event, row["sample"], metric, got, want)
            errors[metric].append(abs(got-want))
    stochastic = stochastic_probes(om, stochastic_rows)
    print(json.dumps({"status": "passed", "stochastic_cases": stochastic, "api_probes": api_probes(om), "fixtures": len(rows), "comparisons": sum(map(len, errors.values())),
                      "max_absolute_error": {name: max(values) for name, values in errors.items()}}, indent=2))


if __name__ == "__main__":
    main()
