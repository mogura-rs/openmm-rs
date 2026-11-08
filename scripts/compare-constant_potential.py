import argparse
import importlib.util
import json
import math
from pathlib import Path


def points():
    return [(0.1,0.2,0.3),(0.7,0.4,0.2),(0.3,0.8,0.6),(0.9,0.6,0.8),(1.2,0.3,0.1),(0.2,1.1,0.4),(0.6,1.3,0.9),(0,0,0)]


def create(om,row):
    solver, constrained, triclinic, periodic, nested, model = (row[k] for k in ['solver','constraint','triclinic','periodic_exception','nested','model'])
    system = om.System(); electrode = {0,1,2,4} if model else {0}
    masses = [0 if i == 7 or (solver == 2 and i in electrode) else 12 for i in range(8)]
    for mass in masses: system.add_particle(mass)
    system.set_virtual_site(7,om.OutOfPlaneSite(2,3,6,0.2,0.3,0.4))
    p = points(); system.add_constraint(5,6,math.dist(p[5],p[6]))
    system.set_default_periodic_box_vectors((3.2,0,0),(0.3 if triclinic else 0,3.4,0),(-0.2 if triclinic else 0,0.1 if triclinic else 0,3.6))
    force = om.ConstantPotentialForce()
    for i in range(8): force.addParticle(0.05*(i%3-1))
    force.addElectrode({0,2} if model else {0},2,0.12,0.01)
    if model: force.addElectrode({1,4},-1,0.15,0.015)
    force.addException(3,5,-0.01); force.addException(5,6,0)
    force.setCutoffDistance(1.2); force.setPMEParameters(2.4,12,14,16)
    force.setConstantPotentialMethod(om.ConstantPotentialForce.Matrix if solver == 2 else om.ConstantPotentialForce.CG)
    force.setUsePreconditioner(solver == 1); force.setCGErrorTolerance(1e-8)
    force.setUseChargeConstraint(bool(constrained)); force.setChargeConstraintTarget(0.25)
    force.setExternalField((0.2,-0.1,0.3)); force.setExceptionsUsePeriodicBoundaryConditions(bool(periodic)); force.setForceGroup(5)
    if nested:
        cv = om.CustomCVForce('0.8*u+0.001*u*u'); cv.add_collective_variable('u',force); cv.set_force_group(5); system.add_force(cv)
    else: system.add_force(force)
    external = om.CustomExternalForce('0.01*x*x')
    for i in range(8): external.add_particle(i,[])
    external.set_force_group(2); system.add_force(external)
    integrator = om.VerletIntegrator(0.00001); context = om.Context(system,integrator)
    context.set_positions(p); context.compute_virtual_sites()
    context.set_velocities([(0.01*math.sin(i),0.02*math.cos(i),0.01*i) if mass else (0,0,0) for i,mass in enumerate(masses)])
    context.apply_velocity_constraints(1e-10)
    return context,integrator,force


def update(context,force,row):
    solver, constrained, periodic, model = (row[k] for k in ['solver','constraint','periodic_exception','model'])
    force.setParticleParameters(3,0.12); force.setParticleParameters(5,-0.07); force.setParticleParameters(0,100)
    force.setExceptionParameters(0,3,5,-0.02)
    force.setElectrodeParameters(0,{0,2} if model else {0},3,0.14,0.02)
    if model: force.setElectrodeParameters(1,{1,4},-2,0.13,0.01)
    force.setExternalField((-0.1,0.2,-0.3)); force.setChargeConstraintTarget(-0.1)
    force.setCutoffDistance(0.7); force.setPMEParameters(3.1,8,10,12); force.setEwaldErrorTolerance(0.01)
    force.setCGErrorTolerance(0.1); force.setUsePreconditioner(solver != 1)
    force.setUseChargeConstraint(not constrained); force.setExceptionsUsePeriodicBoundaryConditions(not periodic)
    force.setConstantPotentialMethod(0 if solver == 2 else 1)
    context.update_force_parameters_at_path([0,0] if row['nested'] else [0],force)


def flatten(values):
    if isinstance(values,(list,tuple)):
        for value in values: yield from flatten(value)
    else: yield values


def api_checks(om):
    force = om.ConstantPotentialForce()
    assert force.getConstantPotentialMethod() == force.CG
    assert force.getCutoffDistance() == 1 and force.getEwaldErrorTolerance() == 5e-4
    assert force.getCGErrorTolerance() == 1e-4 and force.getUsePreconditioner()
    assert not force.getUseChargeConstraint() and force.getChargeConstraintTarget() == 0
    assert force.getExternalField() == (0,0,0) and force.getPMEParameters() == (0,0,0,0)
    for q in [0,0.2,-0.1]: force.addParticle(q)
    force.addElectrode({0},1,0.1,0.01)
    assert force.getElectrodeParameters(0) == ({0},1,0.1,0.01)
    force.addException(1,2,-0.01)
    force.addException(2,1,-0.02,True)
    assert force.getNumExceptions() == 1 and force.getExceptionParameters(0) == (2,1,-0.02)
    force.setName('electrode'); force.setForceGroup(5)
    system = om.System()
    for mass in [0,12,12]: system.add_particle(mass)
    system.add_force(force)
    context = om.Context(system,om.VerletIntegrator(0.001)); context.set_positions([(0,0,0),(0.5,0,0),(0,0.7,0)])
    initial = force.getCharges(context)
    assert len(initial) == 3 and initial[1:] == [0.2,-0.1]
    pme = force.getPMEParametersInContext(context); assert pme[0] > 0 and all(n >= 6 for n in pme[1:])
    force.setParticleParameters(1,0.4); force.setExternalField((0.1,0.2,0.3)); force.updateParametersInContext(context)
    current = force.getCharges(context); assert current[1] == 0.4 and current[0] != initial[0]
    force.setForceGroup(6)
    assert context.get_state(energy=True,groups=32).potential_energy != 0
    assert context.get_state(energy=True,groups=64).potential_energy == 0
    return {'defaults':True,'electrode_sets':True,'exception_replacement':True,'charges':True,'automatic_pme':True,'captured_group':True,'updates':True}


def update_probes(om, rows):
    maxima = {'energy': 0, 'forces': 0, 'charges': 0}
    for row in rows:
        electrode = row['probe'] == 'electrode_regrouping'; solver = row['solver']; phase = row['phase']
        if phase == 0:
            system = om.System(); force = om.ConstantPotentialForce()
            for i in range(6):
                system.add_particle(0 if electrode and i < 3 else 12); force.addParticle(0.1*(i%3-1))
            force.setCutoffDistance(1.2); force.setPMEParameters(2.4,12,14,16); force.setCGErrorTolerance(1e-8)
            force.setConstantPotentialMethod(1 if solver == 2 else 0); force.setUsePreconditioner(solver == 1)
            force.setUseChargeConstraint(electrode); force.setChargeConstraintTarget(0.1)
            if electrode:
                force.addElectrode({0,2},2,0.12,0.01); force.addElectrode({1},-1,0.15,0.015)
            else: force.addException(0,1,0.02)
            system.add_force(force); system.set_default_periodic_box_vectors((3,0,0),(0.3,3.2,0),(-0.2,0.1,3.4))
            context = om.Context(system,om.VerletIntegrator(0.001)); p=points()[:6]; context.set_positions(p)
        else:
            if electrode:
                if phase == 1:
                    force.setElectrodeParameters(0,{0},3,0.13,0.02); force.setElectrodeParameters(1,{1,2},-2,0.14,0.01)
                elif phase == 2:
                    force.setElectrodeParameters(1,set(),-2,0.14,0.01); force.addElectrode({1,2},0.5,0.16,0.015)
                elif phase == 3:
                    force.setExternalField((0.2,0.3,-0.1)); force.setChargeConstraintTarget(-0.05)
                else:
                    context.set_periodic_box_vectors((3.3,0,0),(-0.2,3.5,0),(0.1,-0.1,3.7))
                    p[0]=tuple(a+b for a,b in zip(p[0],(0.02,0.01,-0.03),strict=True)); context.set_positions(p)
            else:
                if phase == 1: force.setExceptionParameters(0,0,2,0.03)
                elif phase == 2: force.addException(2,3,0)
                elif phase == 3: force.setExceptionParameters(0,0,2,0)
                else: force.setExceptionParameters(0,0,2,0.01)
            force.updateParametersInContext(context)
        state=context.get_state(energy=True,forces=True)
        values={'energy':state.potential_energy,'forces':state.forces,'charges':force.getCharges(context)}
        for name,value in values.items():
            for got,expected in zip(flatten(value),flatten(row[name]),strict=True):
                maxima[name]=max(maxima[name],abs(got-expected))
                assert math.isclose(got,expected,abs_tol=2e-8,rel_tol=2e-9),(row['probe'],solver,phase,name,got,expected)
    assert len(rows)==30
    return {'states':len(rows),'max_absolute_error':maxima}


def stability_checks(om,rows):
    results=[]
    for row in rows:
        solver=row['solver']; constrained=row['constraint']; system=om.System()
        force=om.ConstantPotentialForce(); bonds=om.HarmonicBondForce(); p=points()[:6]
        velocities=[]
        for i,q in enumerate([0,0,0,0.2,-0.3,0.15]):
            mass=0 if solver==2 and i<3 else 12; system.add_particle(mass); force.addParticle(q)
            velocities.append((0.02*math.sin(1.7*i),0.02*math.cos(0.9*i),0.01*math.sin(0.3*i)) if mass else (0,0,0))
            for j in range(i): bonds.add_bond(i,j,math.dist(p[i],p[j]),500)
        force.addElectrode({0,2},2,0.12,0.01); force.addElectrode({1},-1,0.15,0.015)
        force.addException(3,4,-0.01); force.setCutoffDistance(1.4); force.setPMEParameters(2.4,12,14,16)
        force.setConstantPotentialMethod(1 if solver==2 else 0); force.setUsePreconditioner(solver==1)
        force.setCGErrorTolerance(1e-8); force.setUseChargeConstraint(bool(constrained)); force.setChargeConstraintTarget(0.25); force.setExternalField((0.2,-0.1,0.3))
        system.add_force(force); system.add_force(bonds); system.set_default_periodic_box_vectors((3.4,0,0),(0.3,3.6,0),(-0.2,0.1,3.8))
        integrator=om.VerletIntegrator(0.0004); context=om.Context(system,integrator); context.set_positions(p); context.set_velocities(velocities)
        initial=context.get_state(energy=True); e0=initial.potential_energy+initial.kinetic_energy; drift=max_distance=0
        for sample in range(100):
            integrator.step(50); state=context.get_state(energy=True,positions=True)
            drift=max(drift,abs(state.potential_energy+state.kinetic_energy-e0))
            max_distance=max(max_distance,max(math.dist(state.positions[i],state.positions[j]) for i in range(6) for j in range(i)))
            if sample==4:
                short=context.get_state(positions=True,velocities=True)
                short_positions=short.positions; short_velocities=short.velocities
        final=context.get_state(positions=True,velocities=True); solved=force.getCharges(context)
        values={'initial_energy':e0,'short_positions':short_positions,'short_velocities':short_velocities,'positions':final.positions,'velocities':final.velocities,'charges':solved}
        errors={}
        for name,value in values.items():
            errors[name]=max(abs(got-expected) for got,expected in zip(flatten(value),flatten(row[name]),strict=True))
            assert all(math.isclose(got,expected,abs_tol=2e-6,rel_tol=2e-8) for got,expected in zip(flatten(value),flatten(row[name]),strict=True)),(solver,constrained,name,errors[name])
        assert drift<2e-5 and row['max_absolute_drift']<2e-5,(solver,constrained,drift,row['max_absolute_drift'])
        assert max_distance<1.4 and row['max_distance']<1.4
        charge_error=abs(sum(solved)-0.25) if constrained else None
        if constrained: assert charge_error<1e-12
        results.append({'solver':solver,'constraint':constrained,'steps':5000,'duration_ps':2,'max_absolute_drift':drift,'reference_max_absolute_drift':row['max_absolute_drift'],'max_distance':max_distance,'total_charge_error':charge_error,'max_absolute_error':errors})
    assert len(results)==6
    return results


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--extension',type=Path,required=True); parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args(); spec=importlib.util.spec_from_file_location('openmm',args.extension)
    om=importlib.util.module_from_spec(spec); spec.loader.exec_module(om)
    maxima={}; states=checks=fd_checks=0; fd_max=0
    rows=list(map(json.loads,args.reference.read_text().splitlines()))
    probes=[r for r in rows if r.get("probe")]
    stability=[r for r in rows if r.get("stability")]
    for row in (r for r in rows if not r.get("probe") and not r.get("stability")):
        if row.get('fd'):
            actual=context.get_state(forces=True,groups=32).forces
            for got,expected in zip(flatten(actual[:7]),flatten(row['forces']),strict=True):
                fd_max=max(fd_max,abs(got-expected)); fd_checks+=1
                assert math.isclose(got,expected,abs_tol=3e-6,rel_tol=2e-7),(states,got,expected)
            continue
        phase=row['phase']; path=[0,0] if row['nested'] else [0]
        if row['groups']==32:
            if phase==0: context,integrator,force=create(om,row)
            elif phase==1: update(context,force,row)
            elif phase==2:
                triclinic=row['triclinic']; b=(-0.3 if triclinic else 0,3.7,0)
                context.set_periodic_box_vectors((3.5,0,0),b,(0.1 if triclinic else 0,-0.2 if triclinic else 0,3.9))
                p=points(); p[0]=tuple(a+b for a,b in zip(p[0],(0.04,-0.03,0.02),strict=True)); p[2]=tuple(a+b for a,b in zip(p[2],(-0.02,0.04,0.01),strict=True))
                p[3]=tuple(a+v for a,v in zip(p[3],b,strict=True)); context.set_positions(p); context.compute_virtual_sites()
            else: integrator.step(2)
        state=context.get_state(positions=True,velocities=True,forces=True,energy=True,groups=row['groups'])
        values={'energy':state.potential_energy,'kinetic':state.kinetic_energy,'time':context.time,'positions':state.positions,'velocities':state.velocities,'forces':state.forces,
                'charges':context.get_constant_potential_charges(path),'pme':context.get_constant_potential_pme_parameters(path)}
        key=tuple(row[k] for k in ['solver','constraint','triclinic','periodic_exception','nested','model','phase','groups'])
        for name,value in values.items():
            for actual,expected in zip(flatten(value),flatten(row[name]),strict=True):
                maxima[name]=max(maxima.get(name,0),abs(actual-expected)); checks+=1
                assert math.isclose(actual,expected,abs_tol=2e-7,rel_tol=2e-9),(key,name,actual,expected)
        states+=1
    assert states==1152 and fd_checks==2016,(states,fd_checks)
    print(json.dumps({'states':states,'comparisons':checks,'max_absolute_error':maxima,'finite_difference':{'components':fd_checks,'max_absolute_error':fd_max},'api':api_checks(om),'update_probes':update_probes(om,probes),'stability':stability_checks(om,stability)},indent=2))


if __name__=='__main__': main()
