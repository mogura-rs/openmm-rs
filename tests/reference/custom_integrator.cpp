#include <openmm/CMMotionRemover.h>
#include <openmm/CompoundIntegrator.h>
#include <openmm/Context.h>
#include <openmm/CustomBondForce.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/CustomIntegrator.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/TabulatedFunction.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/VirtualSite.h>
#include <iomanip>
#include <cmath>
#include <iostream>
using namespace OpenMM;

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for(size_t i=0;i<values.size();++i) {
        if(i)std::cout << ',';
        std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']';
    }
    std::cout << ']';
}

CustomIntegrator* make(int kind) {
    auto* it=new CustomIntegrator(0.0005);
    for(const auto& name : {"count","before","after","derivative","total","tabvalue"})it->addGlobalVariable(name,0);
    it->addPerDofVariable("q",0.2);it->addPerDofVariable("xold",0);
    it->addTabulatedFunction("tab",new Continuous1DFunction({25,16,9,4,1,0,1,4,9,16,25},-5,5));
    it->setKineticEnergyExpression("m*(v+0.5*dt*f/m)^2/2+0.001*q*q");
    it->setConstraintTolerance(1e-10);it->setRandomNumberSeed(77);
    if(kind!=4)it->addUpdateContextState();
    it->addComputeGlobal("before","energy1");
    it->addComputeGlobal("derivative","deriv(energy1,lambda)");
    it->addComputeGlobal("lambda","lambda+0.01*dt");
    it->addComputeGlobal("after","energy1");
    it->addComputePerDof("q","f1");
    it->addComputeGlobal("tabvalue","tab(sin(count))");
    if(kind==1) {
        it->addComputePerDof("v","v+0.5*dt*f4/m");
        it->addComputeGlobal("count","0");
        it->beginWhileBlock("count < 2");
        it->addComputePerDof("v","v+0.25*dt*f1/m");
        it->addComputePerDof("xold","x");
        it->addComputePerDof("x","x+0.5*dt*v");
        it->addConstrainPositions();
        it->addComputePerDof("v","2*(x-xold)/dt");
        it->addComputePerDof("v","v+0.25*dt*f1/m");
        it->addComputeGlobal("count","count+1");
        it->endBlock();
        it->addComputePerDof("v","v+0.5*dt*f4/m");
    } else {
        it->addComputePerDof("v","v+0.5*dt*f/m");
        it->addComputePerDof("xold","x");
        it->addComputePerDof("x","x+dt*v");
        it->addConstrainPositions();
        it->addComputePerDof("v","(x-xold)/dt");
        it->addComputePerDof("v","v+0.5*dt*f/m");
    }
    it->addConstrainVelocities();
    if(kind==2) {
        it->addComputePerDof("q","cross(q,x)+vector(_z(v),_x(x),_y(q))");
        it->addComputeSum("total","dot(q,q)");
    } else if(kind==3) {
        it->addComputeGlobal("count","0");
        it->beginWhileBlock("count <= 2");
        it->beginIfBlock("count != 1");
        it->addComputePerDof("q","q+0.01*tab(0.1*x)");
        it->endBlock();it->addComputeGlobal("count","count+1");it->endBlock();
        it->addUpdateContextState();it->addComputeSum("total","q*q");
    } else it->addComputeSum("total","q*q");
    it->addComputeGlobal("count","count+1");
    return it;
}

void run(int kind,int routing,bool constrained,bool virtualSite) {
    System system; for(double mass:{1.,2.,3.,0.})system.addParticle(mass);
    if(constrained)system.addConstraint(0,1,1);
    if(virtualSite)system.setVirtualSite(3,new TwoParticleAverageSite(0,2,0.3,0.7));
    auto* bond=new CustomBondForce("lambda*0.8*(r-0.9)^2");
    bond->addGlobalParameter("lambda",1.2);bond->addEnergyParameterDerivative("lambda");
    bond->addBond(0,2,{});bond->setForceGroup(1);system.addForce(bond);
    auto* force=new CustomExternalForce("0.4*x+0.2*y^2");
    for(int i=0;i<4;++i)force->addParticle(i,{});
    force->setForceGroup(4);system.addForce(force);system.addForce(new CMMotionRemover(2));
    CustomIntegrator* it=make(kind);
    it->setIntegrationForceGroups(kind%2==0?18:2);
    CompoundIntegrator root;
    if(routing==2) {
        root.addIntegrator(new VerletIntegrator(0.0007));
        auto* inner=new CompoundIntegrator();inner->addIntegrator(it);root.addIntegrator(inner);
    } else root.addIntegrator(it);
    Integrator& active=(routing==0?static_cast<Integrator&>(*it):static_cast<Integrator&>(root));
    Context context(system,active,Platform::getPlatformByName("Reference"));
    context.setPositions({Vec3(0,0,0),Vec3(1,0,0),Vec3(0,0.7,1.1),Vec3(0.2,0.4,0.5)});
    context.computeVirtualSites();
    context.setVelocities({Vec3(0.2,0.1,-0.1),Vec3(-0.3,0.4,0.2),Vec3(0.1,-0.2,0.3),Vec3(0.7,0.8,0.9)});
    context.applyVelocityConstraints(1e-10);context.setTime(2.7);context.setStepCount(17);
    it->setPerDofVariableByName("q",{Vec3(0.2,0.3,0.4),Vec3(0.4,0.5,0.6),Vec3(0.6,0.7,0.8),Vec3(0.8,0.9,1.0)});
    for(int event=0;event<10;++event) {
        if(event==3) {context.setParameter("lambda",0.7);it->setGlobalVariableByName("count",4);}
        if(event==5)it->setIntegrationForceGroups(16);
        if(event==7)it->setKineticEnergyExpression("m*v*v");
        if(event) {if(routing==1 && event%2==0)root.step(1);else it->step(1);}
        int groups=event%3==0?-1:(event%3==1?2:16);
        auto state=context.getState(State::Positions|State::Velocities|State::Forces|State::Energy|State::Parameters, false,groups);
        std::cout << "{\"kind\":" << kind << ",\"routing\":" << routing << ",\"constrained\":" << constrained << ",\"virtual_site\":" << virtualSite
            << ",\"event\":" << event << ",\"groups\":" << groups << ",\"time\":" << state.getTime() << ",\"step\":" << context.getStepCount()
            << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy() << ",\"lambda\":" << context.getParameter("lambda") << ",\"positions\":";
        vectors(state.getPositions());std::cout << ",\"velocities\":";vectors(state.getVelocities());std::cout << ",\"forces\":";vectors(state.getForces());
        std::cout << ",\"globals\":[";for(int i=0;i<it->getNumGlobalVariables();++i) {if(i)std::cout << ',';std::cout << it->getGlobalVariable(i);}std::cout << ']';
        std::vector<Vec3> q;it->getPerDofVariableByName("q",q);std::cout << ",\"q\":";vectors(q);std::cout << "}\n";
    }
}

void original_probe(int mode) {
 System system;system.addParticle(1);
 auto* force=new CustomExternalForce("x+2*y+3*z");force->addParticle(0,{});system.addForce(force);
 CustomIntegrator integrator(.001);
 if(mode==1)integrator.addComputeGlobal("dt","0.003");
 if(mode==2)integrator.addComputePerDof("v","cross(x,f0)");
 else if(mode==3)integrator.addComputePerDof("v","vector(energy,energy,energy)");
 else integrator.addComputePerDof("x","x+dt");
 Context context(system,integrator,Platform::getPlatformByName("Reference"));
 context.setPositions({Vec3(1,2,3)});context.setVelocities({Vec3(1,2,3)});
 if(mode==0)integrator.setStepSize(.002);
 try {
  integrator.step(1);
  auto first=context.getState(State::Positions|State::Velocities|State::Energy);
  integrator.setKineticEnergyExpression("m*v*v");
  double second=context.getState(State::Energy).getKineticEnergy();
  std::cout<<"{\"source_probe\":true,\"mode\":"<<mode<<",\"time\":"<<first.getTime()<<",\"dt\":"<<integrator.getStepSize()<<",\"x\":"<<first.getPositions()[0][0]<<",\"ke_before\":"<<first.getKineticEnergy()<<",\"ke_after_setter\":"<<second<<"}\n";
 } catch(const std::exception& e) { std::cout<<"{\"source_probe\":true,\"mode\":"<<mode<<",\"error\":"<<std::quoted(e.what())<<"}\n"; }
}

void derivative_cache_probe() {
    System system;system.addParticle(1);system.addParticle(1);
    for(int group=1;group<=2;++group) {
        auto* force=new CustomBondForce(group==1 ? "lambda*r" : "5*lambda*r");
        force->addGlobalParameter("lambda",1);force->addEnergyParameterDerivative("lambda");
        force->addBond(0,1,{});force->setForceGroup(group);system.addForce(force);
    }
    CustomIntegrator it(0.001);for(const auto& name:{"a","b","d1","d2"})it.addGlobalVariable(name,0);
    it.addComputeGlobal("a","energy1");it.addComputeGlobal("b","energy2");
    it.addComputeGlobal("d1","deriv(energy1,lambda)");it.addComputeGlobal("d2","deriv(energy2,lambda)");
    Context context(system,it,Platform::getPlatformByName("Reference"));context.setPositions({Vec3(),Vec3(2,0,0)});it.step(1);
    double d1=it.getGlobalVariableByName("d1"),d2=it.getGlobalVariableByName("d2");
    double expected=context.getState(State::ParameterDerivatives,false,2).getEnergyParameterDerivatives().at("lambda");
    context.setParameter("lambda",1+1e-5);double plus=context.getState(State::Energy,false,2).getPotentialEnergy();
    context.setParameter("lambda",1-1e-5);double minus=context.getState(State::Energy,false,2).getPotentialEnergy();
    std::cout<<"{\"source_probe\":true,\"mode\":4,\"d1\":"<<d1<<",\"d2\":"<<d2<<",\"expected1\":"<<expected<<",\"finite_difference1\":"<<(plus-minus)/2e-5<<"}\n";
}

void stability(int kind) {
    int n=kind?256:8;double dt=kind?0.002:0.001;
    System system;auto* force=new CustomExternalForce("12.5*(x*x+y*y+z*z)");
    for(int i=0;i<n;++i){system.addParticle(1);force->addParticle(i,{});}system.addForce(force);
    CustomIntegrator it(dt);it.setRandomNumberSeed(7721);
    it.addGlobalVariable("kT",1.380649e-23*6.02214076e23/1000*300);
    it.addComputePerDof("v","v+0.5*dt*f/m");
    if(kind) {
        it.addComputePerDof("x","x+0.5*dt*v");
        it.addComputePerDof("v","exp(-1.3*dt)*v+sqrt(kT/m*(1-exp(-2.6*dt)))*gaussian");
        it.addComputePerDof("x","x+0.5*dt*v");
    } else it.addComputePerDof("x","x+dt*v");
    it.addComputePerDof("v","v+0.5*dt*f/m");
    Context context(system,it,Platform::getPlatformByName("Reference"));
    std::vector<Vec3> p(n),v(n);
    if(!kind)for(int i=0;i<n;++i)for(int axis=0;axis<3;++axis) {p[i][axis]=0.02*(1+(3*i+axis)%5);v[i][axis]=0.03*((3*i+axis)%7-3);}
    context.setPositions(p);context.setVelocities(v);
    if(!kind) {
        auto initial=context.getState(State::Energy);double e0=initial.getPotentialEnergy()+initial.getKineticEnergy(),maxDrift=0;
        for(int sample=0;sample<100;++sample) {it.step(1000);auto s=context.getState(State::Energy);maxDrift=std::max(maxDrift,std::abs((s.getPotentialEnergy()+s.getKineticEnergy()-e0)/e0));}
        auto s=context.getState(State::Positions|State::Velocities|State::Energy);
        std::cout<<"{\"stability\":0,\"time\":"<<s.getTime()<<",\"energy\":"<<s.getPotentialEnergy()+s.getKineticEnergy()<<",\"initial_energy\":"<<e0<<",\"max_relative_drift\":"<<maxDrift<<",\"positions\":";
        vectors(s.getPositions());std::cout<<",\"velocities\":";vectors(s.getVelocities());std::cout<<"}\n";
    } else {
        it.step(10000);std::cout<<"{\"stability\":1,\"blocks\":[";
        for(int block=0;block<6;++block) {
            double x=0,v=0,x2=0,v2=0;
            for(int sample=0;sample<250;++sample) {
                it.step(20);auto s=context.getState(State::Positions|State::Velocities);
                for(auto p:s.getPositions())for(int axis=0;axis<3;++axis){x+=p[axis];x2+=p[axis]*p[axis];}
                for(auto p:s.getVelocities())for(int axis=0;axis<3;++axis){v+=p[axis];v2+=p[axis]*p[axis];}
            }
            double scale=1.0/(250*3*n);if(block)std::cout<<',';
            std::cout<<'['<<x*scale<<','<<v*scale<<','<<x2*scale<<','<<v2*scale<<']';
        }
        std::cout<<"],\"time\":"<<context.getTime()<<"}\n";
    }
}

int main() {
    std::cout << std::setprecision(17);
    for(int kind=0;kind<5;++kind)for(int routing=0;routing<3;++routing)
        for(bool constrained:{false,true})for(bool virtualSite:{false,true})run(kind,routing,constrained,virtualSite);
    for(int mode=0;mode<4;++mode)original_probe(mode);
    derivative_cache_probe();stability(0);stability(1);
}
