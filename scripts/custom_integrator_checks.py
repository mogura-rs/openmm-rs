"""Independent API and probability checks for programmable integration."""
import gc
import math
import weakref


def reject(callback):
    try:
        callback()
    except (RuntimeError, ValueError, OverflowError):
        return
    raise AssertionError("invalid CustomIntegrator operation was accepted")


def api_checks(om):
    it = om.CustomIntegrator(0)
    assert it.getStepSize() == 0 and it.getNumComputations() == 0
    kinds = ["ComputeGlobal", "ComputePerDof", "ComputeSum", "ConstrainPositions", "ConstrainVelocities", "UpdateContextState", "IfBlockStart", "WhileBlockStart", "BlockEnd"]
    assert [getattr(it, name) for name in kinds] == list(range(9))
    assert it.getKineticEnergyExpression() == "m*v*v/2"
    assert it.addGlobalVariable("g", 2) == 0
    assert it.addPerDofVariable("q", 4) == 0
    assert it.getGlobalVariableName(0) == "g" and it.getPerDofVariableName(0) == "q"
    assert it.getGlobalVariableByName("g") == 2 and it.getPerDofVariableByName("q") == [[4, 4, 4]]
    it.setPerDofVariable(0, [[5, 6, 7]])
    table = om.Continuous1DFunction([0, 0, 0], -1, 1)
    assert it.addTabulatedFunction("tab", table) == 0
    assert it.getNumTabulatedFunctions() == 1 and it.getTabulatedFunctionName(0) == "tab"
    it.addComputePerDof("q", "tab(x)")
    it.addComputeSum("g", "q")
    assert it.getComputationStep(0) == (1, "q", "tab(x)")
    system = om.System()
    for mass in [1, 1, 0]:
        system.add_particle(mass)
    context = om.Context(system, it)
    context.set_positions([(0, 0, 0)]*3)
    assert it.getPerDofVariable(0) == [[5, 6, 7]]*3
    table.set_function_parameters([3, 3, 3], -1, 1)
    # An energy request can be the first operation that compiles the program.
    assert context.get_state(energy=True).kinetic_energy == 0
    it.getTabulatedFunction(0).set_function_parameters([9, 9, 9], -1, 1)
    context.step(1)
    assert it.getPerDofVariable(0) == [[3, 3, 3], [3, 3, 3], [5, 6, 7]]
    assert it.getGlobalVariable(0) == 18 and context.time == 0 and context.step_count == 1
    it.setGlobalVariableByName("g", 13)
    assert it.getGlobalVariable(0) == 13
    for action in [lambda: om.Context(system, it), lambda: it.addGlobalVariable("late", 0),
                   lambda: it.addPerDofVariable("late", 0), lambda: it.addComputeGlobal("g", "1"),
                   lambda: it.endBlock(), lambda: it.getGlobalVariable(99),
                   lambda: it.getPerDofVariableByName("missing"), lambda: it.setPerDofVariable(0, [[1, 2, 3]]),
                   lambda: it.setStepSize(-1), lambda: it.setStepSize(float("nan"))]:
        reject(action)
    weak = weakref.ref(context)
    del context
    gc.collect()
    assert weak() is None
    reject(lambda: it.step(1))
    context = om.Context(system, it)
    context.set_positions([(0, 0, 0)]*3)
    it.step(1)
    assert it.getPerDofVariable(0)[0] == [9, 9, 9]
    for expression in ["f0+f1", "f0+energy0", "cross(x)", "deriv(x,g)", "unknown(x)"]:
        invalid = om.CustomIntegrator(0.001)
        invalid.addComputePerDof("v", expression)
        test = om.Context(system, invalid)
        reject(lambda: invalid.step(1))
        reject(lambda: test.get_state(energy=True))
        assert test.step_count == 0
    for kind in [0, 1, 2]:
        invalid = om.CustomIntegrator(0.001)
        if kind == 0:
            invalid.addGlobalVariable("dup", 0)
            invalid.addPerDofVariable("dup", 0)
        elif kind == 1:
            invalid.addComputeGlobal("not_declared", "1")
        else:
            invalid.addPerDofVariable("q", 0)
            invalid.setPerDofVariable(0, [[1, 2, 3]]*2)
        reject(lambda: om.Context(system, invalid))
    return {"table_capture": True, "ownership_rebind": True, "invalid_programs": True}


def random_checks(om):
    count = 8192
    def run(seed, change_seed=False):
        system = om.System()
        for _ in range(count):
            system.add_particle(1)
        system.add_particle(0)
        it = om.CustomIntegrator(0.001)
        for name in ["q", "r", "z", "zero"]:
            it.addPerDofVariable(name, 4)
        for name in ["hits", "sum"]:
            it.addGlobalVariable(name, 0)
        it.setRandomNumberSeed(seed)
        it.addComputePerDof("q", "gaussian")
        it.addComputePerDof("r", "uniform")
        it.addComputePerDof("z", "cross(gaussian,vector(1,0,0))")
        it.addComputePerDof("zero", "gaussian-gaussian+uniform-uniform")
        it.beginIfBlock("uniform = uniform")
        it.addComputeGlobal("hits", "hits+1")
        it.endBlock()
        it.addComputeSum("sum", "q")
        context = om.Context(system, it)
        context.set_positions([(0, 0, 0)]*(count+1))
        it.step(1)
        if change_seed:
            it.setRandomNumberSeed(seed+99)
        it.step(1)
        assert it.getGlobalVariableByName("hits") == 2
        values = {name: it.getPerDofVariableByName(name) for name in ["q", "r", "z", "zero"]}
        assert math.isclose(it.getGlobalVariableByName("sum"), sum(sum(v) for v in values["q"][:-1]), abs_tol=1e-10)
        assert all(v == [0, 0, 0] for v in values["zero"][:-1])
        assert all(values[name][-1] == [4, 4, 4] for name in values)
        return values
    values = run(7231)
    assert values == run(7231, True)
    assert values != run(7232)
    assert run(0) != run(0)
    metrics = {}
    for name, mean, variance in [("q", 0, 1), ("r", 0.5, 1/12)]:
        flat = [x for row in values[name][:-1] for x in row]
        empirical_mean = sum(flat)/len(flat)
        empirical_variance = sum((x-mean)**2 for x in flat)/len(flat)
        mean_z = abs(empirical_mean-mean)/math.sqrt(variance/len(flat))
        variance_z = abs(empirical_variance-variance)/math.sqrt(2*variance**2/len(flat))
        assert mean_z < 6 and variance_z < 6, (name, mean_z, variance_z)
        metrics[name] = {"mean_z": mean_z, "variance_z": variance_z}
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            covariance = sum((row[a]-mean)*(row[b]-mean) for row in values[name][:-1])/count
            assert abs(covariance) < 6*variance/math.sqrt(count), (name, a, b, covariance)
    z = values["z"][:-1]
    assert all(row[0] == 0 for row in z)
    for axis in [1, 2]:
        mean = sum(row[axis] for row in z)/count
        second = sum(row[axis]**2 for row in z)/count
        assert abs(mean) < 6/math.sqrt(count) and abs(second-1) < 6*math.sqrt(2/count)
    return metrics


def exception_checks(om, probes):
    assert len(probes) == 5
    for row in probes:
        if row["mode"] < 2:
            dt = 0.002 if row["mode"] == 0 else 0.003
            assert row["time"] == 0.001 and row["dt"] == dt
            assert row["x"] == 1+dt and row["ke_before"] == row["ke_after_setter"] == 7
        elif row["mode"] < 4:
            assert "No value specified for variable" in row["error"]
        else:
            assert row["d1"] == row["d2"] == 10 and row["expected1"] == 2
            assert math.isclose(row["finite_difference1"], 2, abs_tol=1e-9)
    for mode in range(4):
        system = om.System()
        system.add_particle(1)
        force = om.CustomExternalForce("x+2*y+3*z")
        force.add_particle(0, [])
        system.add_force(force)
        it = om.CustomIntegrator(0.001)
        if mode == 1:
            it.addComputeGlobal("dt", "0.003")
        if mode == 2:
            it.addComputePerDof("v", "cross(x,f0)")
        elif mode == 3:
            it.addComputePerDof("v", "vector(energy,energy,energy)")
        else:
            it.addComputePerDof("x", "x+dt")
        context = om.Context(system, it)
        context.set_positions([(1, 3, 2)])
        context.set_velocities([(1, 2, 3)])
        if mode == 0:
            it.setStepSize(0.002)
        it.step(1)
        state = context.get_state(positions=True, velocities=True, energy=True)
        assert context.time == it.getStepSize()
        if mode < 2:
            assert state.positions[0] == (1+it.getStepSize(), 3+it.getStepSize(), 2+it.getStepSize())
            it.setKineticEnergyExpression("m*v*v")
            assert context.get_state(energy=True).kinetic_energy == state.kinetic_energy == 7
        elif mode == 2:
            assert state.velocities[0] == (-5, 1, 1)
        else:
            assert state.velocities[0] == (13, 13, 13)
    system = om.System()
    system.add_particle(1)
    system.add_particle(1)
    for group, expression in [(1, "lambda*r"), (2, "5*lambda*r")]:
        force = om.CustomBondForce(expression)
        force.add_global_parameter("lambda", 1)
        force.add_energy_parameter_derivative("lambda")
        force.add_bond(0, 1, [])
        force.set_force_group(group)
        system.add_force(force)
    it = om.CustomIntegrator(0.001)
    for name in ["a", "b", "d1", "d2"]:
        it.addGlobalVariable(name, 0)
    for name, expression in [("a", "energy1"), ("b", "energy2"), ("d1", "deriv(energy1,lambda)"), ("d2", "deriv(energy2,lambda)")]:
        it.addComputeGlobal(name, expression)
    it.addComputePerDof("v", "vector(deriv(energy1,lambda),1,2)")
    context = om.Context(system, it)
    context.set_positions([(0, 0, 0), (2, 0, 0)])
    it.step(1)
    assert it.getGlobalVariableByName("d1") == 2 and it.getGlobalVariableByName("d2") == 10
    assert context.get_state(velocities=True).velocities == [(2, 1, 2)]*2
    return {"original_reference_probes": len(probes), "updated_clock": True, "vector_force_energy": True, "derivative_cache": True}


def stability_checks(om, rows):
    import statistics
    assert len(rows) == 2
    metrics = {}
    for row in rows:
        kind = row["stability"]
        n, dt = (256, 0.002) if kind else (8, 0.001)
        system = om.System()
        force = om.CustomExternalForce("12.5*(x*x+y*y+z*z)")
        for i in range(n):
            system.add_particle(1)
            force.add_particle(i, [])
        system.add_force(force)
        it = om.CustomIntegrator(dt)
        it.setRandomNumberSeed(7721)
        kt = 1.380649e-23*6.02214076e23/1000*300
        it.addGlobalVariable("kT", kt)
        it.addComputePerDof("v", "v+0.5*dt*f/m")
        if kind:
            it.addComputePerDof("x", "x+0.5*dt*v")
            it.addComputePerDof("v", "exp(-1.3*dt)*v+sqrt(kT/m*(1-exp(-2.6*dt)))*gaussian")
            it.addComputePerDof("x", "x+0.5*dt*v")
        else:
            it.addComputePerDof("x", "x+dt*v")
        it.addComputePerDof("v", "v+0.5*dt*f/m")
        context = om.Context(system, it)
        context.set_positions([(0, 0, 0)]*n if kind else [tuple(0.02*(1+(3*i+a) % 5) for a in range(3)) for i in range(n)])
        context.set_velocities([(0, 0, 0)]*n if kind else [tuple(0.03*((3*i+a) % 7-3) for a in range(3)) for i in range(n)])
        if not kind:
            initial = context.get_state(energy=True)
            e0 = initial.potential_energy+initial.kinetic_energy
            drift = 0
            for _ in range(100):
                it.step(1000)
                state = context.get_state(energy=True)
                drift = max(drift, abs((state.potential_energy+state.kinetic_energy-e0)/e0))
            state = context.get_state(positions=True, velocities=True, energy=True)
            for actual, expected in zip([x for v in state.positions+state.velocities for x in v], [x for v in row["positions"]+row["velocities"] for x in v], strict=True):
                assert math.isclose(actual, expected, abs_tol=3e-10, rel_tol=3e-10), (actual, expected)
            assert max(drift, row["max_relative_drift"]) < 7e-6
            assert math.isclose(state.potential_energy+state.kinetic_energy, row["energy"], abs_tol=1e-10)
            metrics["conservative"] = {"steps": 100000, "max_relative_drift": drift, "reference_drift": row["max_relative_drift"]}
        else:
            it.step(10000)
            blocks = []
            for _ in range(6):
                values = [0.0]*4
                for _ in range(250):
                    it.step(20)
                    state = context.get_state(positions=True, velocities=True)
                    p = [x for v in state.positions for x in v]
                    v = [x for v in state.velocities for x in v]
                    values = [a+b for a, b in zip(values, [sum(p), sum(v), sum(x*x for x in p), sum(x*x for x in v)])]
                blocks.append([value/(250*3*n) for value in values])
            expected = [0, 0, kt/25, kt*(1-25*dt*dt/4)]
            scores = {}
            for name, data in [("rust", blocks), ("reference", row["blocks"])]:
                scores[name] = []
                for column, target in zip(zip(*data), expected):
                    mean = statistics.mean(column)
                    stderr = statistics.stdev(column)/math.sqrt(len(column))
                    score = abs(mean-target)/max(stderr, 1e-12)
                    assert score < 6, (name, target, mean, stderr, score)
                    scores[name].append(score)
            metrics["baoab"] = {"particles": n, "warmup_ps": 20, "production_ps": 60, "block_mean_z": scores}
        assert math.isclose(context.time, row["time"], abs_tol=1e-10)
    return metrics
