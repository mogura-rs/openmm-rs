"""Compare programmable integrator execution against the supplied Reference backend."""
import argparse
import gc
import importlib.util
import json
import math
import weakref
from pathlib import Path
from custom_integrator_checks import api_checks, random_checks, exception_checks, stability_checks


def make(om, kind):
    it = om.CustomIntegrator(0.0005)
    for name in ["count", "before", "after", "derivative", "total", "tabvalue"]:
        it.addGlobalVariable(name, 0)
    it.addPerDofVariable("q", 0.2)
    it.addPerDofVariable("xold", 0)
    it.addTabulatedFunction("tab", om.Continuous1DFunction([25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25], -5, 5))
    it.setKineticEnergyExpression("m*(v+0.5*dt*f/m)^2/2+0.001*q*q")
    it.setConstraintTolerance(1e-10)
    it.setRandomNumberSeed(77)
    if kind != 4:
        it.addUpdateContextState()
    it.addComputeGlobal("before", "energy1")
    it.addComputeGlobal("derivative", "deriv(energy1,lambda)")
    it.addComputeGlobal("lambda", "lambda+0.01*dt")
    it.addComputeGlobal("after", "energy1")
    it.addComputePerDof("q", "f1")
    it.addComputeGlobal("tabvalue", "tab(sin(count))")
    if kind == 1:
        it.addComputePerDof("v", "v+0.5*dt*f4/m")
        it.addComputeGlobal("count", "0")
        it.beginWhileBlock("count < 2")
        it.addComputePerDof("v", "v+0.25*dt*f1/m")
        it.addComputePerDof("xold", "x")
        it.addComputePerDof("x", "x+0.5*dt*v")
        it.addConstrainPositions()
        it.addComputePerDof("v", "2*(x-xold)/dt")
        it.addComputePerDof("v", "v+0.25*dt*f1/m")
        it.addComputeGlobal("count", "count+1")
        it.endBlock()
        it.addComputePerDof("v", "v+0.5*dt*f4/m")
    else:
        it.addComputePerDof("v", "v+0.5*dt*f/m")
        it.addComputePerDof("xold", "x")
        it.addComputePerDof("x", "x+dt*v")
        it.addConstrainPositions()
        it.addComputePerDof("v", "(x-xold)/dt")
        it.addComputePerDof("v", "v+0.5*dt*f/m")
    it.addConstrainVelocities()
    if kind == 2:
        it.addComputePerDof("q", "cross(q,x)+vector(_z(v),_x(x),_y(q))")
        it.addComputeSum("total", "dot(q,q)")
    elif kind == 3:
        it.addComputeGlobal("count", "0")
        it.beginWhileBlock("count <= 2")
        it.beginIfBlock("count != 1")
        it.addComputePerDof("q", "q+0.01*tab(0.1*x)")
        it.endBlock()
        it.addComputeGlobal("count", "count+1")
        it.endBlock()
        it.addUpdateContextState()
        it.addComputeSum("total", "q*q")
    else:
        it.addComputeSum("total", "q*q")
    it.addComputeGlobal("count", "count+1")
    return it


def create(om, row):
    system = om.System()
    for mass in [1, 2, 3, 0]:
        system.add_particle(mass)
    if row["constrained"]:
        system.add_constraint(0, 1, 1)
    if row["virtual_site"]:
        system.set_virtual_site(3, om.TwoParticleAverageSite(0, 2, 0.3, 0.7))
    bond = om.CustomBondForce("lambda*0.8*(r-0.9)^2")
    bond.add_global_parameter("lambda", 1.2)
    bond.add_energy_parameter_derivative("lambda")
    bond.add_bond(0, 2, [])
    bond.set_force_group(1)
    system.add_force(bond)
    force = om.CustomExternalForce("0.4*x+0.2*y^2")
    for i in range(4):
        force.add_particle(i, [])
    force.set_force_group(4)
    system.add_force(force)
    system.add_force(om.CMMotionRemover(2))
    it = make(om, row["kind"])
    it.setIntegrationForceGroups(18 if row["kind"] % 2 == 0 else 2)
    root = om.CompoundIntegrator()
    if row["routing"] == 2:
        root.addIntegrator(om.VerletIntegrator(0.0007))
        inner = om.CompoundIntegrator()
        inner.addIntegrator(it)
        root.addIntegrator(inner)
    elif row["routing"] == 1:
        root.addIntegrator(it)
    context = om.Context(system, it if row["routing"] == 0 else root)
    context.set_positions([(0, 0, 0), (1, 0, 0), (0, 0.7, 1.1), (0.2, 0.4, 0.5)])
    context.compute_virtual_sites()
    context.set_velocities([(0.2, 0.1, -0.1), (-0.3, 0.4, 0.2), (0.1, -0.2, 0.3), (0.7, 0.8, 0.9)])
    context.apply_velocity_constraints(1e-10)
    context.set_time(2.7)
    context.set_step_count(17)
    it.setPerDofVariableByName("q", [(0.2, 0.3, 0.4), (0.4, 0.5, 0.6), (0.6, 0.7, 0.8), (0.8, 0.9, 1)])
    return context, it, root


def flatten(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("openmm", args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    maximum = {}
    count = 0
    rows = [json.loads(line) for line in args.reference.read_text().splitlines()]
    probes = [row for row in rows if row.get("source_probe")]
    stability = [row for row in rows if "stability" in row]
    rows = [row for row in rows if not row.get("source_probe") and "stability" not in row]
    for row in rows:
        event = row["event"]
        if event == 0:
            context, it, root = create(om, row)
        if event == 3:
            context.set_parameter("lambda", 0.7)
            it.setGlobalVariableByName("count", 4)
        if event == 5:
            it.setIntegrationForceGroups(16)
        if event == 7:
            it.setKineticEnergyExpression("m*v*v")
        if event:
            if row["routing"] == 1 and event % 2 == 0:
                root.step(1)
            else:
                it.step(1)
        state = context.get_state(positions=True, velocities=True, forces=True, energy=True, parameters=True, groups=row["groups"])
        actual = {"time": state.time, "step": context.step_count, "energy": state.potential_energy,
                  "kinetic": state.kinetic_energy, "lambda": context.get_parameter("lambda"), "positions": state.positions,
                  "velocities": state.velocities, "forces": state.forces,
                  "globals": [it.getGlobalVariable(i) for i in range(it.getNumGlobalVariables())], "q": it.getPerDofVariableByName("q")}
        for key, value in actual.items():
            for found, expected in zip(flatten(value), flatten(row[key]), strict=True):
                error = abs(found - expected)
                maximum[key] = max(maximum.get(key, 0), error)
                assert math.isclose(found, expected, abs_tol=2e-8, rel_tol=2e-8), (row["kind"], row["routing"], row["constrained"], row["virtual_site"], event, key, found, expected)
                count += 1
    print(json.dumps({"states": len(rows), "comparisons": count, "max_absolute_error": maximum,
                      "api": api_checks(om), "random": random_checks(om), "exceptions": exception_checks(om, probes), "stability": stability_checks(om, stability)}, indent=2))


if __name__ == "__main__":
    main()
