"""Measure CPU integration through the Python interface."""
import argparse
import importlib.util
import json
import math
import resource
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--kind", choices=["verlet", "langevin", "brownian"], required=True)
    parser.add_argument("--particles", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    if args.particles < 1 or args.steps < 1:
        parser.error("particles and steps must be positive")
    spec = importlib.util.spec_from_file_location("openmm", args.extension)
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    system = om.System()
    for _ in range(args.particles):
        system.add_particle(1)
    integrator = {"verlet": lambda: om.VerletIntegrator(0.001),
                  "langevin": lambda: om.LangevinMiddleIntegrator(0, 0.7, 0.001),
                  "brownian": lambda: om.BrownianIntegrator(0, 0.7, 0.001)}[args.kind]()
    context = system.create_context(integrator)
    context.set_positions([(0, 0, 0)]*args.particles)
    context.set_velocities([(0.1, 0.2, 0.3)]*args.particles)
    context.step(10)
    started = time.perf_counter()
    context.step(args.steps)
    elapsed = time.perf_counter()-started
    state = context.get_state(energy=True)
    expected = 0.07*args.particles
    if args.kind == "langevin":
        expected *= math.exp(-2*0.7*0.001*(args.steps+10))
    elif args.kind == "brownian":
        expected = 0
    assert math.isclose(state.kinetic_energy, expected, abs_tol=1e-9, rel_tol=1e-10)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        peak_rss /= 1024
    print(json.dumps({"kind": args.kind, "particles": args.particles, "steps": args.steps,
                      "seconds": elapsed, "steps_per_second": args.steps/elapsed,
                      "peak_rss_kib": peak_rss, "kinetic_energy": state.kinetic_energy}))


if __name__ == "__main__":
    main()
