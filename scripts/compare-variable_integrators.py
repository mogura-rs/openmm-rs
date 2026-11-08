"""Compare adaptive trajectories, corrected clocks and independent thermal moments."""
import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path

KB=1.380649e-23*6.02214076e23/1000


def make_integrator(om, kind):
    return om.VariableVerletIntegrator(1e-7) if kind==0 else om.VariableLangevinIntegrator(0,0.7,1e-7)


def make_system(om, constrained, site):
    system=om.System()
    for mass in [1,2,3,0]:system.add_particle(mass)
    if constrained:system.add_constraint(0,1,1)
    if site:system.set_virtual_site(3,om.TwoParticleAverageSite(0,2,0.3,0.7))
    bond=om.HarmonicBondForce();bond.add_bond(0,2,0.8,2);bond.set_force_group(1);system.add_force(bond)
    external=om.CustomExternalForce("scale*(0.4*x+0.2*y^2)");external.add_global_parameter("scale",1)
    for i in range(4):external.add_particle(i,[])
    external.set_force_group(4);system.add_force(external)
    return system


def rejects(callback):
    try:callback()
    except RuntimeError:return
    raise AssertionError("invalid adaptive operation was accepted")


def probes(om):
    for kind in range(2):
        system=om.System();system.add_particle(1)
        integrator=make_integrator(om,kind)
        assert integrator.getStepSize()==0 and integrator.getMaximumStepSize()==0
        rejects(lambda:integrator.stepTo(1))
        context=om.Context(system,integrator)
        rejects(lambda:om.Context(system,integrator))
        context.set_positions([(0,0,0)]);context.set_velocities([(1,0,0)])
        rejects(lambda:integrator.step(1))
        assert context.time==0 and context.step_count==0
        assert context.get_state(positions=True).positions==[(0,0,0)]
        integrator.setMaximumStepSize(0.001)
        integrator.step(1)
        assert context.time==0.001 and context.step_count==1
        integrator.stepTo(0.0037)
        assert context.time==0.0037 and context.step_count==4
        integrator.step_to(0.0037);integrator.stepTo(0)
        assert context.step_count==4
        for value in [math.nan,math.inf,-math.inf]:rejects(lambda:integrator.stepTo(value))
        for value in [0,-1,math.nan,math.inf]:rejects(lambda:integrator.setErrorTolerance(value))
        for value in [-1,math.nan,math.inf]:rejects(lambda:integrator.setMaximumStepSize(value))
        if kind:
            for value in [-1,math.nan,math.inf]:
                rejects(lambda:integrator.setTemperature(value));rejects(lambda:integrator.setFriction(value))
        integrator.setMaximumStepSize(0)
        assert integrator.getMaximumStepSize()==0
        del context;gc.collect()
        rejects(lambda:integrator.step(1))
        context=om.Context(system,integrator);context.set_positions([(0,0,0)])
        integrator.stepTo(0.00073)
        assert context.time==0.00073 and context.step_count==1
    def seeded(change):
        system=om.System()
        for mass in [1,2,3,0]:system.add_particle(mass)
        integrator=om.VariableLangevinIntegrator(300,0.7,1e-5)
        integrator.setMaximumStepSize(0.001);integrator.setRandomNumberSeed(-17)
        context=om.Context(system,integrator);context.set_positions([(0,0,0)]*4)
        if change==1:integrator.setRandomNumberSeed(31)
        integrator.step(3)
        integrator.setTemperature(250);integrator.setFriction(1.2)
        if change==2:integrator.setRandomNumberSeed(31)
        integrator.stepTo(0.00723)
        state=context.get_state(positions=True,velocities=True)
        return state.positions,state.velocities
    assert seeded(0)==seeded(1)==seeded(2)==seeded(0)


def statistics(om,rows):
    reports=[]
    for friction in [0.7,1000000.]:
        n=12000;system=om.System()
        for i in range(n):system.add_particle(1+i%5)
        integrator=om.VariableLangevinIntegrator(300,friction,1e-5)
        integrator.setMaximumStepSize(0.001);integrator.setRandomNumberSeed(17)
        context=om.Context(system,integrator)
        for phase,temperature in enumerate([100,500,200]):
            integrator.setTemperature(temperature)
            context.set_positions([(0,0,0)]*n);context.set_velocities([(0,0,0)]*n);integrator.step(5)
            state=context.get_state(positions=True,velocities=True)
            decay=math.exp(-friction*0.001);noise2=-math.expm1(-2*friction*0.001)
            pvar=vvar=covar=0
            for step in range(5):
                a=0.0005*(1+decay);b=0.0005
                pvar+=a*a*vvar+2*a*covar+b*b*noise2
                covar=decay*covar+a*decay*vvar+b*noise2
                vvar=decay*decay*vvar+noise2
            moments=[]
            for field,variance in [(state.positions,pvar),(state.velocities,vvar)]:
                z=[value/math.sqrt(KB*temperature/(1+i%5)*variance) for i,vector in enumerate(field) for value in vector]
                moments.append([sum(value**power for value in z)/(3*n) for power in [1,2,4]])
            reference=next(row for row in rows if row['friction']==friction and row['phase']==phase)
            for label,values in [('rust',moments),('reference',reference['moments'])]:
                for group in values:
                    for value,mean,var in zip(group,[0,1,3],[1,2,96]):
                        assert abs(value-mean)<6*math.sqrt(var/(3*n)),(label,friction,phase,value,mean)
            reports.append({'friction':friction,'temperature':temperature,'samples':3*n,'rust_moments':moments,'reference_moments':reference['moments']})
    return reports


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extension',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True);args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('openmm',args.extension);om=importlib.util.module_from_spec(spec);spec.loader.exec_module(om)
    rows=[json.loads(line) for line in args.reference.read_text().splitlines()]
    fixtures=[r for r in rows if 'event' in r]
    assert len(fixtures)==336 and len(rows)==344
    errors={key:[] for key in ['dt','time','energy','kinetic','positions','velocities','forces']}
    previous=None
    for row in fixtures:
        key=row['kind'],row['constrained'],row['site']
        if key!=previous:
            kind,constrained,site=key;integrator=make_integrator(om,kind);system=make_system(om,constrained,site);context=om.Context(system,integrator)
            context.set_positions([(0,0,0),(1,0,0),(0,0.7,1.1),(0.2,0.4,0.5)])
            context.set_velocities([(0.2,0.1,-0.1),(-0.3,0.4,0.2),(0.1,-0.2,0.3),(0.7,0.8,0.9)])
            context.compute_virtual_sites();context.apply_velocity_constraints(1e-10);context.set_time(2.7);context.set_step_count(17);previous=key
        phase=row['phase'];event=row['event']
        if event==0:
            integrator.setMaximumStepSize([0,0.0002,0.0005,0.0003,0.0001,0,0][phase])
            integrator.setErrorTolerance([1e-7,1e-7,1e-7,1e-5,1e-5,1e-6,1e-8][phase])
            integrator.setIntegrationForceGroups([-1,18,0,2,18,-1,16][phase]);integrator.setConstraintTolerance(1e-10)
            integrator.setStepSize(0.001+phase*0.0001);context.set_parameter('scale',0.3 if phase%2 else 1.0)
            if kind:integrator.setFriction([0.7,0.7,0.7,0.7,1.3,1.3,0][phase])
            if phase==6:context.set_velocities_to_temperature(0)
        elif event<5:
            if event%2:integrator.step(1)
            else:context.step(1)
        else:integrator.stepTo(context.time+0.000731)
        state=context.get_state(positions=True,velocities=True,forces=True,energy=True)
        assert state.step_count==row['step_count'],(key,phase,event,state.step_count,row['step_count'])
        actual={'dt':integrator.getStepSize(),'time':state.time,'energy':state.potential_energy,'kinetic':state.kinetic_energy}
        actual.update({field:[v for vector in getattr(state,field) for v in vector] for field in ['positions','velocities','forces']})
        for field,got in actual.items():
            if field in ['positions','velocities','forces']:want=[v for vector in row[field] for v in vector]
            else:got=[got];want=[row[field]]
            for a,b in zip(got,want):
                assert math.isfinite(a) and math.isclose(a,b,rel_tol=3e-8,abs_tol=3e-8),(key,phase,event,field,a,b)
                errors[field].append(abs(a-b))
    probes(om)
    clock=[r for r in rows if r.get('clock_exception')]
    assert len(clock)==2 and all(r['source_time_infinite'] and r['dt']==0.001 and r['source_step_to_count']==1 and r['source_step_to_time']==0.003 for r in clock)
    reports=statistics(om,[r for r in rows if r.get('statistic')])
    print(json.dumps({'status':'passed','fixtures':len(fixtures),'comparisons':sum(map(len,errors.values())),'clock_exception':clock,'statistics':reports,'max_absolute_error':{k:max(v) for k,v in errors.items()}},indent=2))


if __name__=='__main__':main()
