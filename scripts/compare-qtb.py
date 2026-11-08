"""Validate QTB splitting, corrected FFT filtering, adaptation and captured settings."""
import argparse
import cmath
import gc
import importlib.util
import json
import math
from pathlib import Path


def create(om, row):
    system = om.System()
    for mass in [1, 1, 2, 2, 0, 3]:
        system.add_particle(mass)
    if row["site"]:
        system.add_particle(0)
        system.set_virtual_site(6, om.TwoParticleAverageSite(0, 3, 0.3, 0.7))
    if row["constrained"]:
        system.add_constraint(0, 1, 0.37)
    bond = om.HarmonicBondForce()
    bond.add_bond(0, 1, 0.37, 1.2)
    bond.set_force_group(1)
    system.add_force(bond)
    external = om.CustomExternalForce("lambda*(0.1*x^2+0.03*y-0.05*z)")
    external.add_global_parameter("lambda", 0.7)
    for i in range(6+row["site"]):
        external.add_particle(i, [])
    external.set_force_group(3)
    system.add_force(external)
    integrator = om.QTBIntegrator(0, 1.3, 0.003)
    integrator.setSegmentLength((row["length"]-0.25)*0.003)
    integrator.setCutoffFrequency(1e-5)
    integrator.setDefaultAdaptationRate(0)
    integrator.setConstraintTolerance(1e-11)
    integrator.setRandomNumberSeed(191)
    if row["types"]:
        for particle, kind in [(0, -7), (1, -7), (2, 12345), (3, 12345)]:
            integrator.setParticleType(particle, kind)
    top = integrator
    if row["routing"]:
        top = om.CompoundIntegrator()
        branch = om.CompoundIntegrator()
        branch.addIntegrator(integrator)
        top.addIntegrator(branch)
        top.addIntegrator(om.VerletIntegrator(0.001))
        top.setCurrentIntegrator(1 if row["routing"] == 2 else 0)
    context = om.Context(system, top)
    context.set_positions([(0.37*i, 0.2*(i//2), -0.1*(i//2)) for i in range(6+row["site"])])
    context.compute_virtual_sites()
    context.set_velocities([(0.1*i-0.2, 0.06*i-0.1, 0.03*i-0.04) for i in range(6+row["site"])])
    context.apply_velocity_constraints(1e-11)
    context.set_time(1.7)
    context.set_step_count(11)
    return context, top, integrator


def noise_moments(om, rows):
    summaries = []
    particles = 3072
    mass = 2
    for row in rows:
        length, dt, friction = row["length"], row["dt"], row["friction"]
        system = om.System()
        for _ in range(particles):
            system.add_particle(mass)
        integrator = om.QTBIntegrator(row["temperature"], friction, dt)
        integrator.setSegmentLength((length-0.25)*dt)
        integrator.setCutoffFrequency(row["cutoff"])
        integrator.setDefaultAdaptationRate(0)
        integrator.setRandomNumberSeed(761)
        context = om.Context(system, integrator)
        context.set_positions([(0, 0, 0)]*particles)
        context.set_velocities([(0, 0, 0)]*particles)
        previous = [0.0]*(3*particles)
        forces = []
        damping = math.exp(-dt*friction)
        for _ in range(length):
            integrator.step(1)
            current = [v for xyz in context.get_state(velocities=True).velocities for v in xyz]
            forces.append([(new-damping*old)*mass/dt for old, new in zip(previous, current)])
            previous = current
        nfft = 3*length
        power = []
        for k, target in enumerate(row["deconvolved"]):
            omega = 2*math.pi*k/(nfft*dt)
            exponent = (omega-row["cutoff"])/(row["cutoff"]/100)
            cutoff = 0 if exponent > 709 else 1/(1+math.exp(exponent))
            cw = (math.expm1(-dt*friction)**2+4*damping*math.sin(0.5*omega*dt)**2)/((friction**2+omega**2)*dt**2)
            power.append(cutoff*target*cw)
        covariance = []
        for lag in range(length):
            value = sum((1 if k == 0 or 2*k == nfft else 2)*value*math.cos(2*math.pi*k*lag/nfft)
                        for k, value in enumerate(power))
            covariance.append(2*mass*friction/dt/nfft*value)
        variance = covariance[0]
        count = 3*particles
        max_mean_z, max_covariance_z = 0.0, 0.0
        for i, values in enumerate(forces):
            mean_z = abs(sum(values)/count)/math.sqrt(variance/count)
            assert mean_z < 6, (row, i, "mean", mean_z)
            max_mean_z = max(max_mean_z, mean_z)
            for j in range(i+1):
                moment = sum(x*y for x, y in zip(values, forces[j]))/count
                expected = covariance[i-j]
                error_z = abs(moment-expected)/math.sqrt((variance**2+expected**2)/count)
                assert error_z < 6, (row, i, j, "covariance", error_z, moment, expected)
                max_covariance_z = max(max_covariance_z, error_z)
        summaries.append({"length": length, "temperature": row["temperature"], "friction": friction,
                          "independent_components": count, "target_force_variance": variance,
                          "maximum_mean_z": max_mean_z, "maximum_covariance_z": max_covariance_z})
    return summaries


def adaptation(om):
    masses = [2, 2, 3, 0]
    system = om.System()
    for mass in masses:
        system.add_particle(mass)
    length, dt, friction = 5, 0.001, 10
    integrator = om.QTBIntegrator(300, friction, dt)
    integrator.setSegmentLength((length-0.25)*dt)
    integrator.setRandomNumberSeed(193)
    kinds = [-7, -7, 900, 99]
    rates = {-7: 0.3, 900: 10000, 99: 0.2}
    for particle, kind in enumerate(kinds):
        integrator.setParticleType(particle, kind)
    for kind, rate in rates.items():
        integrator.setTypeAdaptationRate(kind, rate)
    context = om.Context(system, integrator)
    context.set_positions([(0, 0, 0)]*4)
    previous = [(0.2, -0.1, 0.3)]*4
    context.set_velocities(previous)
    nfft = 3*length
    expected = {kind: [friction]*(nfft//2+1) for kind in rates}
    histories = None
    max_error, checks, clamped = 0.0, 0, 0
    for segment in range(4):
        if histories is not None:
            force_history, velocity_history = histories
            for kind, rate in rates.items():
                members = [i for i, value in enumerate(kinds) if value == kind]
                derivative = [0.0]*(nfft//2+1)
                for particle in members:
                    for axis in range(3):
                        for k in range(len(derivative)):
                            phase = [cmath.exp(-2j*math.pi*k*t/nfft) for t in range(length)]
                            v = sum(velocity_history[t][particle][axis]*phase[t] for t in range(length))
                            f = sum(force_history[t][particle][axis]*phase[t] for t in range(length))
                            derivative[k] += masses[particle]*expected[kind][k]*abs(v)**2-(v*f.conjugate()).real
                expected[kind] = [max(0, value-dt*rate/(3*len(members)*length)*change)
                                  for value, change in zip(expected[kind], derivative)]
                clamped += sum(value == 0 for value in expected[kind])
        force_history, velocity_history = [], []
        for time in range(length):
            integrator.step(1)
            current = context.get_state(velocities=True).velocities
            force_row, velocity_row = [], []
            for mass, old, new in zip(masses, previous, current):
                dv = [b-math.exp(-dt*friction)*a for a, b in zip(old, new)] if mass else [0]*3
                force_row.append([value*mass/dt for value in dv])
                velocity_row.append([math.exp(-0.5*dt*friction)*value+0.5*noise for value, noise in zip(old, dv)] if mass else [0]*3)
            force_history.append(force_row)
            velocity_history.append(velocity_row)
            previous = current
            if time == 0:
                for particle, kind in enumerate(kinds):
                    actual = integrator.getAdaptedFriction(particle)
                    for a, b in zip(actual, expected[kind], strict=True):
                        assert abs(a-b) <= 2e-8+2e-10*abs(b), (segment, particle, a, b)
                        max_error = max(max_error, abs(a-b))
                        checks += 1
                assert integrator.getAdaptedFriction(0) == integrator.getAdaptedFriction(1)
        histories = force_history, velocity_history
    assert clamped > 0
    return {"checks": checks, "maximum_absolute_error": max_error, "clamped_coefficients": clamped}


def oscillators(om, rows):
    summaries = []
    kb = 1.380649e-23*6.02214076e23/1000
    hbar = 1.054571628e-34*6.02214076e23/(1000*1e-12)
    for row in rows:
        system = om.System()
        force = om.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
        force.add_global_parameter("k", row["omega"]**2)
        particles = 32
        for i in range(particles):
            system.add_particle(1)
            force.add_particle(i, [])
        system.add_force(force)
        integrator = om.QTBIntegrator(row["temperature"], 10, 0.001)
        integrator.setSegmentLength(0.50075)
        integrator.setRandomNumberSeed(769)
        for i in range(particles):
            integrator.setParticleType(i, 0)
        context = om.Context(system, integrator)
        context.set_positions([(0, 0, 0)]*particles)
        context.set_velocities([(0, 0, 0)]*particles)
        integrator.step(10000)
        mean, m2, kinetic = 0.0, 0.0, 0.0
        for i in range(row["count"]):
            integrator.step(10)
            state = context.get_state(energy=True)
            value = state.potential_energy
            delta = value-mean
            mean += delta/(i+1)
            m2 += delta*(value-mean)
            kinetic += state.kinetic_energy/row["count"]
        energy = hbar*row["omega"]
        theta = energy*(0.5+1/math.expm1(energy/(kb*row["temperature"])))
        target_mean = 1.5*particles*theta
        target_variance = 1.5*particles*theta**2
        variance = m2/row["count"]
        for implementation, m, v, k in [("rust", mean, variance, kinetic),
                ("reference", row["mean_potential"], row["variance_potential"], row["mean_raw_kinetic"])]:
            assert all(math.isfinite(x) and x > 0 for x in [m, v, k])
            assert abs(m/target_mean-1) < 0.10, (row, implementation, m, target_mean)
            assert abs(v/target_variance-1) < 0.25, (row, implementation, v, target_variance)
        assert abs(context.get_state().time-110) < 1e-8
        summaries.append({"temperature": row["temperature"], "angular_frequency": row["omega"],
                          "target_mean_potential": target_mean, "target_variance_potential": target_variance,
                          "rust_mean_potential": mean, "reference_mean_potential": row["mean_potential"],
                          "rust_variance_potential": variance, "reference_variance_potential": row["variance_potential"],
                          "rust_mean_raw_kinetic": kinetic, "reference_mean_raw_kinetic": row["mean_raw_kinetic"]})
    return summaries


def api_probes(om):
    def reject(action):
        try:
            action()
        except (RuntimeError, ValueError, OverflowError):
            return
        raise AssertionError("invalid QTB operation was accepted")
    for values in [(-1, 1, 0.001), (300, -1, 0.001), (300, 1, 0), (math.nan, 1, 0.001)]:
        reject(lambda: om.QTBIntegrator(*values))
    def make(seed=77, temperature=300):
        system = om.System()
        system.add_particle(2)
        integrator = om.QTBIntegrator(temperature, 1, 0.001)
        integrator.setSegmentLength(0.004)
        integrator.setRandomNumberSeed(seed)
        integrator.setDefaultAdaptationRate(0)
        context = om.Context(system, integrator)
        context.set_positions([(0, 0, 0)])
        context.set_velocities([(0, 0, 0)])
        return system, integrator, context
    s, a, ca = make()
    _, b, cb = make()
    a.step(1)
    b.step(1)
    a.setTemperature(600)
    a.setFriction(9)
    a.setStepSize(0.01)
    a.setSegmentLength(0.1)
    a.setCutoffFrequency(900)
    a.setDefaultAdaptationRate(1)
    a.setRandomNumberSeed(999)
    a.step(3)
    b.step(3)
    assert ca.get_state(velocities=True).velocities == cb.get_state(velocities=True).velocities
    assert ca.get_state().time == cb.get_state().time == 0.004
    a.step(1)
    b.step(1)
    assert ca.get_state(velocities=True).velocities != cb.get_state(velocities=True).velocities
    reject(lambda: om.Context(s, a))
    reject(lambda: a.getAdaptedFriction(1))
    unbound = om.QTBIntegrator(300, 1, 0.001)
    reject(lambda: unbound.step(1))
    reject(lambda: unbound.getAdaptedFriction(0))
    reject(lambda: om.Context(s, om.QTBIntegrator(300, 0, 0.001)))
    for method in [unbound.setSegmentLength, unbound.setCutoffFrequency]:
        reject(lambda: method(0))
    for method in [unbound.setDefaultAdaptationRate, lambda v: unbound.setTypeAdaptationRate(9, v)]:
        reject(lambda: method(-1))
    unbound.setParticleType(-1, 9)
    reject(lambda: om.Context(s, unbound))
    for length in [1, 2, 3, 4, 5]:
        system, integrator, context = make(71, 0)
        del context
        gc.collect()
        integrator.setSegmentLength(length*0.001)
        integrator.setCutoffFrequency(1e-5)
        context = om.Context(system, integrator)
        context.set_positions([(0, 0, 0)])
        context.set_velocities([(0, 0, 0)])
        integrator.step(12)
        assert context.get_state(positions=True, velocities=True).positions == [(0.0, 0.0, 0.0)]
        assert context.get_state(velocities=True).velocities == [(0.0, 0.0, 0.0)]
        assert len(integrator.getAdaptedFriction(0)) == 3*length//2+1
    _, a, ca = make()
    _, b, cb = make()
    a.setRandomNumberSeed(999)
    a.step(20)
    b.step(20)
    assert ca.get_state(velocities=True).velocities == cb.get_state(velocities=True).velocities
    s = om.System()
    s.add_particle(2)
    force = om.CustomExternalForce("-3*x")
    force.add_particle(0, [])
    s.add_force(force)
    integrator = om.QTBIntegrator(0, 1, 0.001)
    integrator.setSegmentLength(0.002)
    integrator.setCutoffFrequency(1e-5)
    c = om.Context(s, integrator)
    c.set_positions([(0, 0, 0)])
    integrator.setStepSize(0.002)
    c.set_velocities_to_temperature(0)
    assert abs(c.get_state(velocities=True).velocities[0][0]+0.00075) < 1e-15
    integrator.step(1)
    assert c.get_state().time == 0.001
    return {"captured_settings": True, "temperature_changes_at_segment_boundary": True,
            "all_fft_bins_filtered": True, "actual_time_and_thermal_offset": True,
            "seed_changes_require_reinitialization": True, "ownership_and_invalid_input": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("openmm", args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    rows = [json.loads(line) for line in args.reference.read_text().splitlines()]
    unfiltered = [r for r in rows if r["kind"] == "source_unfiltered_nyquist"]
    assert len(unfiltered) == 4
    for row in unfiltered:
        if row["length"] % 2:
            assert row["maximum_speed"] == 0
        else:
            assert row["maximum_speed"] > 1e-6
    thermal = [r for r in rows if r["kind"] == "source_thermal_offset"]
    assert len(thermal) == 1 and abs(thermal[0]["velocity_x"]+0.0015) < 1e-15
    previous_key = None
    checks, states = 0, 0
    maximum = {name: 0.0 for name in ["positions", "velocities", "forces", "energy", "kinetic", "time"]}
    clock_difference = 0.0
    for row in rows:
        if row["kind"] != "deterministic":
            continue
        key = tuple(row[k] for k in ["length", "types", "constrained", "site", "routing"])
        if key != previous_key:
            context, top, integrator = create(om, row)
            previous_key = key
        if row["sample"] == 0:
            if row["phase"] == 1:
                integrator.setStepSize(0.006)
                integrator.setFriction(7)
                integrator.setSegmentLength(0.03)
                integrator.setCutoffFrequency(500)
                integrator.setDefaultAdaptationRate(999)
                integrator.setTypeAdaptationRate(-7, 999)
                integrator.setParticleType(5, 99)
            if row["phase"] == 2:
                integrator.setIntegrationForceGroups(1<<1)
                context.set_parameter("lambda", 1.4)
            if row["phase"] == 3:
                integrator.setStepSize(0.003)
                context.set_velocities_to_temperature(0)
            if row["phase"] == 4:
                integrator.setIntegrationForceGroups((1<<1)|(1<<3))
        (top if row["routing"] == 1 else integrator).step(1)
        state = context.get_state(positions=True, velocities=True, forces=True, energy=True)
        for name in ["positions", "velocities", "forces"]:
            for actual, expected in zip(getattr(state, name), row[name], strict=True):
                for a, b in zip(actual, expected, strict=True):
                    error = abs(a-b)
                    assert error < 2e-9, (key, row["phase"], name, a, b)
                    maximum[name] = max(maximum[name], error)
                    checks += 1
        for name, actual in [("energy", state.potential_energy), ("kinetic", state.kinetic_energy), ("time", state.time)]:
            error = abs(actual-row[name])
            assert error < 2e-9, (key, name, actual, row[name])
            maximum[name] = max(maximum[name], error)
            checks += 1
        assert state.step_count == row["step_count"]
        assert integrator.getAdaptedFriction(0) == row["gamma"]
        clock_difference = max(clock_difference, abs(row["time"]-row["source_time"]))
        states += 1
    assert states == 864
    assert abs(clock_difference-0.003) < 1e-12
    print(json.dumps({"states": states, "checks": checks, "maximum_error": maximum,
                      "source_clock_discrepancy": clock_difference, "source_nyquist_probes": unfiltered, "source_thermal_offset": thermal[0],
                      "noise_moments": noise_moments(om, [r for r in rows if r["kind"] == "noise_spectrum"]),
                      "adaptation": adaptation(om), "oscillators": oscillators(om, [r for r in rows if r["kind"] == "oscillator"]), "api": api_probes(om)}, indent=2))


if __name__ == "__main__":
    main()
