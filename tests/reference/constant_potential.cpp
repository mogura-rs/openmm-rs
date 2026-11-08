#include <openmm/ConstantPotentialForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/CustomCVForce.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/Context.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/VirtualSite.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <vector>
using namespace OpenMM;
void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (size_t i=0; i<values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']';
    }
    std::cout << ']';
}
std::vector<Vec3> points() {
    return {Vec3(.1,.2,.3),Vec3(.7,.4,.2),Vec3(.3,.8,.6),Vec3(.9,.6,.8),Vec3(1.2,.3,.1),Vec3(.2,1.1,.4),Vec3(.6,1.3,.9),Vec3()};
}
void run(int solver, bool constrained, bool triclinic, bool periodicExceptions, bool nested, int model) {
    System system;
    const std::set<int> electrodes = model ? std::set<int>{0,1,2,4} : std::set<int>{0};
    for (int i=0; i<8; ++i) system.addParticle(i==7 || (solver==2 && electrodes.count(i)) ? 0. : 12.);
    system.setVirtualSite(7,new OutOfPlaneSite(2,3,6,.2,.3,.4));
    auto p=points(); const Vec3 constraintVector=p[5]-p[6];
    system.addConstraint(5,6,std::sqrt(constraintVector.dot(constraintVector)));
    Vec3 a(3.2,0,0),b(triclinic?.3:0,3.4,0),c(triclinic?-.2:0,triclinic?.1:0,3.6);
    system.setDefaultPeriodicBoxVectors(a,b,c);
    auto* force=new ConstantPotentialForce();
    for(int i=0;i<8;++i) force->addParticle(.05*(i%3-1));
    force->addElectrode(model?std::set<int>{0,2}:std::set<int>{0},2.,.12,.01);
    if(model) force->addElectrode({1,4},-1.,.15,.015);
    force->addException(3,5,-.01); force->addException(5,6,0.);
    force->setCutoffDistance(1.2); force->setPMEParameters(2.4,12,14,16);
    force->setConstantPotentialMethod(solver==2?ConstantPotentialForce::Matrix:ConstantPotentialForce::CG);
    force->setUsePreconditioner(solver==1); force->setCGErrorTolerance(1e-8);
    force->setUseChargeConstraint(constrained); force->setChargeConstraintTarget(.25);
    force->setExternalField(Vec3(.2,-.1,.3)); force->setExceptionsUsePeriodicBoundaryConditions(periodicExceptions);
    force->setForceGroup(5);
    CustomCVForce* cv=nullptr;
    if(nested) { cv=new CustomCVForce("0.8*u+0.001*u*u"); cv->addCollectiveVariable("u",force); cv->setForceGroup(5); system.addForce(cv); }
    else system.addForce(force);
    auto* external=new CustomExternalForce("0.01*x*x");
    for(int i=0;i<8;++i) external->addParticle(i); external->setForceGroup(2); system.addForce(external);
    VerletIntegrator integrator(.00001); Context context(system,integrator,Platform::getPlatformByName("Reference"));
    std::vector<Vec3> velocities(8);
    for(int i=0;i<8;++i) if(system.getParticleMass(i)>0) velocities[i]=Vec3(.01*std::sin(i),.02*std::cos(i),.01*i);
    context.setPositions(p); context.computeVirtualSites(); context.setVelocities(velocities); context.applyVelocityConstraints(1e-10);
    Context* inner=nested?&cv->getInnerContext(context):&context;
    if(nested) force=&dynamic_cast<ConstantPotentialForce&>(const_cast<Force&>(inner->getSystem().getForce(0)));
    for(int phase=0;phase<4;++phase) {
        if(phase==1) {
            force->setParticleParameters(3,.12); force->setParticleParameters(5,-.07);
            force->setParticleParameters(0,100.);
            force->setExceptionParameters(0,3,5,-.02);
            force->setElectrodeParameters(0,model?std::set<int>{0,2}:std::set<int>{0},3.,.14,.02);
            if(model) force->setElectrodeParameters(1,{1,4},-2.,.13,.01);
            force->setExternalField(Vec3(-.1,.2,-.3)); force->setChargeConstraintTarget(-.1);
            force->setCutoffDistance(.7); force->setPMEParameters(3.1,8,10,12); force->setEwaldErrorTolerance(.01);
            force->setCGErrorTolerance(.1); force->setUsePreconditioner(solver!=1);
            force->setUseChargeConstraint(!constrained); force->setExceptionsUsePeriodicBoundaryConditions(!periodicExceptions);
            force->setConstantPotentialMethod(solver==2?ConstantPotentialForce::CG:ConstantPotentialForce::Matrix);
            force->updateParametersInContext(*inner);
        }
        if(phase==2) {
            a=Vec3(3.5,0,0); b=Vec3(triclinic?-.3:0,3.7,0); c=Vec3(triclinic?.1:0,triclinic?-.2:0,3.9);
            context.setPeriodicBoxVectors(a,b,c); p[0]+=Vec3(.04,-.03,.02); p[2]+=Vec3(-.02,.04,.01); p[3]+=b;
            context.setPositions(p); context.computeVirtualSites();
        }
        if(phase==3) integrator.step(2);
        for(int groups:{32,4,36}) {
            auto state=context.getState(State::Positions|State::Velocities|State::Forces|State::Energy,false,groups);
            std::vector<double> charges; force->getCharges(*inner,charges);
            double alpha;int nx,ny,nz;force->getPMEParametersInContext(*inner,alpha,nx,ny,nz);
            std::cout << "{\"solver\":" << solver << ",\"constraint\":" << constrained << ",\"triclinic\":" << triclinic << ",\"periodic_exception\":" << periodicExceptions << ",\"nested\":" << nested << ",\"model\":" << model << ",\"phase\":" << phase << ",\"groups\":" << groups << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy() << ",\"time\":" << state.getTime() << ",\"positions\":";
            vectors(state.getPositions());std::cout << ",\"velocities\":";vectors(state.getVelocities());std::cout << ",\"forces\":";vectors(state.getForces());
            std::cout << ",\"charges\":[";for(size_t i=0;i<charges.size();++i){if(i)std::cout<<',';std::cout<<charges[i];}
            std::cout << "],\"pme\":[" << alpha << ',' << nx << ',' << ny << ',' << nz << "]}\n";
        }
        if(phase==0) {
            auto original=context.getState(State::Positions).getPositions(); std::vector<Vec3> fd(7);
            for(int i=0;i<7;++i) for(int axis=0;axis<3;++axis) {
                auto q=original;q[i][axis]+=1e-5;context.setPositions(q);context.computeVirtualSites();double ep=context.getState(State::Energy,false,32).getPotentialEnergy();
                q=original;q[i][axis]-=1e-5;context.setPositions(q);context.computeVirtualSites();double em=context.getState(State::Energy,false,32).getPotentialEnergy();
                fd[i][axis]=-(ep-em)/2e-5;
            }
            context.setPositions(original);context.computeVirtualSites();std::cout << "{\"fd\":true,\"forces\":";vectors(fd);std::cout << "}\n";
        }
    }
}
void probeSnapshot(const std::string& kind,int solver,int phase,ConstantPotentialForce& force,Context& context) {
    auto state=context.getState(State::Energy|State::Forces);std::vector<double> charges;force.getCharges(context,charges);
    std::cout << "{\"probe\":" << std::quoted(kind) << ",\"solver\":" << solver << ",\"phase\":" << phase << ",\"energy\":" << state.getPotentialEnergy() << ",\"forces\":";vectors(state.getForces());
    std::cout << ",\"charges\":[";for(size_t i=0;i<charges.size();++i){if(i)std::cout<<',';std::cout<<charges[i];}std::cout << "]}\n";
}
void updateProbes(int solver,bool electrodes) {
    System system;auto* force=new ConstantPotentialForce();
    for(int i=0;i<6;++i){system.addParticle(electrodes && i<3 ? 0. : 12.);force->addParticle(.1*(i%3-1));}
    force->setCutoffDistance(1.2);force->setPMEParameters(2.4,12,14,16);force->setCGErrorTolerance(1e-8);
    force->setConstantPotentialMethod(solver==2?ConstantPotentialForce::Matrix:ConstantPotentialForce::CG);force->setUsePreconditioner(solver==1);
    force->setUseChargeConstraint(electrodes);force->setChargeConstraintTarget(.1);
    if(electrodes){force->addElectrode({0,2},2.,.12,.01);force->addElectrode({1},-1.,.15,.015);}
    else force->addException(0,1,.02);
    system.addForce(force);system.setDefaultPeriodicBoxVectors(Vec3(3,0,0),Vec3(.3,3.2,0),Vec3(-.2,.1,3.4));
    VerletIntegrator integrator(.001);Context context(system,integrator,Platform::getPlatformByName("Reference"));auto p=points();p.resize(6);context.setPositions(p);
    std::string kind=electrodes?"electrode_regrouping":"exception_topology";
    for(int phase=0;phase<5;++phase) {
        if(phase>0) {
            if(electrodes) {
                if(phase==1){force->setElectrodeParameters(0,{0},3.,.13,.02);force->setElectrodeParameters(1,{1,2},-2.,.14,.01);}
                if(phase==2){force->setElectrodeParameters(1,{},-2.,.14,.01);force->addElectrode({1,2},.5,.16,.015);}
                if(phase==3){force->setExternalField(Vec3(.2,.3,-.1));force->setChargeConstraintTarget(-.05);}
                if(phase==4){context.setPeriodicBoxVectors(Vec3(3.3,0,0),Vec3(-.2,3.5,0),Vec3(.1,-.1,3.7));p[0]+=Vec3(.02,.01,-.03);context.setPositions(p);}
            } else {
                if(phase==1)force->setExceptionParameters(0,0,2,.03);
                if(phase==2)force->addException(2,3,0.);
                if(phase==3)force->setExceptionParameters(0,0,2,0.);
                if(phase==4)force->setExceptionParameters(0,0,2,.01);
            }
            force->updateParametersInContext(context);
        }
        probeSnapshot(kind,solver,phase,*force,context);
    }
}

void stability(int solver,bool constrained) {
    System system;auto* force=new ConstantPotentialForce();auto* bonds=new HarmonicBondForce();
    auto p=points();p.resize(6);std::vector<Vec3> velocity(6);
    const double charges[6]={0,0,0,.2,-.3,.15};
    for(int i=0;i<6;++i) {
        double mass=solver==2 && i<3?0.:12.;system.addParticle(mass);force->addParticle(charges[i]);
        if(mass)velocity[i]=Vec3(.02*std::sin(1.7*i),.02*std::cos(.9*i),.01*std::sin(.3*i));
        for(int j=0;j<i;++j){auto d=p[i]-p[j];bonds->addBond(i,j,std::sqrt(d.dot(d)),500.);}
    }
    force->addElectrode({0,2},2.,.12,.01);force->addElectrode({1},-1.,.15,.015);
    force->addException(3,4,-.01);force->setCutoffDistance(1.4);force->setPMEParameters(2.4,12,14,16);
    force->setConstantPotentialMethod(solver==2?ConstantPotentialForce::Matrix:ConstantPotentialForce::CG);
    force->setUsePreconditioner(solver==1);force->setCGErrorTolerance(1e-8);force->setUseChargeConstraint(constrained);force->setChargeConstraintTarget(.25);force->setExternalField(Vec3(.2,-.1,.3));
    system.addForce(force);system.addForce(bonds);system.setDefaultPeriodicBoxVectors(Vec3(3.4,0,0),Vec3(.3,3.6,0),Vec3(-.2,.1,3.8));
    VerletIntegrator integrator(.0004);Context context(system,integrator,Platform::getPlatformByName("Reference"));context.setPositions(p);context.setVelocities(velocity);
    auto initial=context.getState(State::Energy);double e0=initial.getPotentialEnergy()+initial.getKineticEnergy(),drift=0,maxDistance=0;
    std::vector<Vec3> shortPositions,shortVelocities;
    for(int sample=0;sample<100;++sample){
        integrator.step(50);auto state=context.getState(State::Energy|State::Positions);drift=std::max(drift,std::abs(state.getPotentialEnergy()+state.getKineticEnergy()-e0));
        for(int i=0;i<6;++i)for(int j=0;j<i;++j){auto d=state.getPositions()[i]-state.getPositions()[j];maxDistance=std::max(maxDistance,std::sqrt(d.dot(d)));}
        if(sample==4){auto shortState=context.getState(State::Positions|State::Velocities);shortPositions=shortState.getPositions();shortVelocities=shortState.getVelocities();}
    }
    auto state=context.getState(State::Positions|State::Velocities);std::vector<double> solved;force->getCharges(context,solved);
    std::cout << "{\"stability\":true,\"solver\":" << solver << ",\"constraint\":" << constrained << ",\"initial_energy\":" << e0 << ",\"max_absolute_drift\":" << drift << ",\"max_distance\":" << maxDistance << ",\"short_positions\":";vectors(shortPositions);std::cout << ",\"short_velocities\":";vectors(shortVelocities);std::cout << ",\"positions\":";vectors(state.getPositions());std::cout << ",\"velocities\":";vectors(state.getVelocities());
    std::cout << ",\"charges\":[";for(size_t i=0;i<solved.size();++i){if(i)std::cout<<',';std::cout<<solved[i];}std::cout << "]}\n";
}

int main() {
    std::cout << std::setprecision(17);
    for(int solver=0;solver<3;++solver)for(bool electrodes:{false,true})updateProbes(solver,electrodes);
    for(int solver=0;solver<3;++solver) for(bool constrained:{false,true}) for(bool triclinic:{false,true})
    for(bool periodic:{false,true}) for(bool nested:{false,true}) for(int model:{0,1}) {
        try { run(solver,constrained,triclinic,periodic,nested,model); }
        catch(const std::exception& error) {
            std::cerr << solver << ' ' << constrained << ' ' << triclinic << ' ' << periodic << ' ' << nested << ' ' << model << ": " << error.what() << '\n';
            return 1;
        }
    }
    for(int solver=0;solver<3;++solver)for(bool constrained:{false,true})stability(solver,constrained);
}
