"""Compare bonded and custom force signs, conventions and periodic geometry."""

import argparse
import importlib.util
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--extension",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location("openmm",args.extension)
    om=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)
    fixtures=[json.loads(line) for line in args.reference.read_text().splitlines()]
    assert len(fixtures)==15, "reference fixture set is incomplete"
    errors=[]
    for fixture in fixtures:
        name=fixture["case"]
        periodic=fixture["periodic"]
        if name=="bond":
            force=om.HarmonicBondForce()
            force.add_bond(0,1,0.6,10)
        elif name=="angle":
            force=om.HarmonicAngleForce()
            force.add_angle(0,1,2,1.1,2)
        elif name=="torsion":
            force=om.PeriodicTorsionForce()
            force.add_torsion((0,1,2,3),3,0.2,0.7)
        elif name=="rb":
            force=om.RBTorsionForce()
            force.add_torsion((0,1,2,3),(1,0.2,0.3,-0.4,0.1,0.2))
        elif name=="custom_bond":
            force=om.CustomBondForce("0.7*(r-0.6)^2")
            force.add_bond(0,1,[])
        elif name=="custom_angle":
            force=om.CustomAngleForce("0.7*(theta-1.1)^2")
            force.add_angle(0,1,2,[])
        elif name=="custom_torsion":
            force=om.CustomTorsionForce("0.7*(1+cos(3*theta-0.2))")
            force.add_torsion(0,1,2,3,[])
        else:
            force=om.CustomExternalForce("x*x+2*y*y+3*z*z")
            force.add_particle(0,[])
        if periodic:
            force.set_uses_periodic_boundary_conditions(True)
        system=om.System()
        for _ in range(4):
            system.add_particle(1)
        system.set_default_periodic_box_vectors((3,0,0),(0.4,3,0),(-0.2,0.3,3))
        system.add_force(force)
        context=system.create_context(om.VerletIntegrator(0.001))
        positions=[(0.1,0.2,0.3),(0.8,-0.1,0.4),(0.9,0.7,0.5),(1.2,0.8,1.4)]
        if periodic:
            positions[2]=(1.3,3.7,0.5)
            positions[3]=(1.0,1.1,4.4)
        context.set_positions(positions)
        state=context.get_state(forces=True,energy=True)
        actual=[state.potential_energy]+[v for row in state.forces for v in row]
        expected=[fixture["energy"]]+[v for row in fixture["forces"] for v in row]
        for got,want in zip(actual,expected):
            assert math.isfinite(got) and math.isclose(got,want,rel_tol=1e-8,abs_tol=1e-8),(name,periodic,got,want)
            errors.append(abs(got-want))
    print(json.dumps({"status":"passed","comparisons":len(errors),"max_absolute_error":max(errors)},indent=2))


if __name__=="__main__":
    main()
