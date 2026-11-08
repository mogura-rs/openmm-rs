#include <openmm/AndersenThermostat.h>
#include <openmm/BrownianIntegrator.h>
#include <openmm/CompoundIntegrator.h>
#include <openmm/Context.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/LangevinIntegrator.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VariableLangevinIntegrator.h>
#include <openmm/VariableVerletIntegrator.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/VirtualSite.h>
#include <iomanip>
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

Integrator* make(int kind) {
    if(kind==0)return new VerletIntegrator(0.0002);
    if(kind==1)return new LangevinMiddleIntegrator(0,0.7,0.0003);
    if(kind==2)return new BrownianIntegrator(0,0.9,0.0002);
    if(kind==3) {
        auto* value=new VariableVerletIntegrator(1e-5);
        value->setMaximumStepSize(0.0003);return value;
    }
    if(kind==4) {
        auto* value=new VariableLangevinIntegrator(0,0.8,1e-5);
        value->setMaximumStepSize(0.0003);return value;
    }
    return new LangevinIntegrator(0,0.6,0.0004);
}

void run(int rotation, bool nested, bool constrained, bool virtualSite) {
    System system;
    for(double mass : {1.,2.,3.,0.})system.addParticle(mass);
    if(constrained)system.addConstraint(0,1,1);
    if(virtualSite)system.setVirtualSite(3,new TwoParticleAverageSite(0,2,0.3,0.7));
    auto* bond=new HarmonicBondForce();bond->addBond(0,2,0.8,2);bond->setForceGroup(1);system.addForce(bond);
    auto* force=new CustomExternalForce("0.4*x+0.2*y^2");
    for(int i=0;i<4;++i)force->addParticle(i,{});
    force->setForceGroup(4);system.addForce(force);
    CompoundIntegrator root;
    CompoundIntegrator* branches[2]={nullptr,nullptr};
    if(nested)for(int i=0;i<2;++i) {branches[i]=new CompoundIntegrator();root.addIntegrator(branches[i]);}
    Integrator* leaves[6];
    for(int i=0;i<6;++i) {
        leaves[i]=make((i+rotation)%6);leaves[i]->setConstraintTolerance(1e-10);
        if(nested)branches[i/3]->addIntegrator(leaves[i]);else root.addIntegrator(leaves[i]);
    }
    Context context(system,root,Platform::getPlatformByName("Reference"));
    context.setPositions({Vec3(0,0,0),Vec3(1,0,0),Vec3(0,0.7,1.1),Vec3(0.2,0.4,0.5)});
    context.computeVirtualSites();
    context.setVelocities({Vec3(0.2,0.1,-0.1),Vec3(-0.3,0.4,0.2),Vec3(0.1,-0.2,0.3),Vec3(0.7,0.8,0.9)});
    context.applyVelocityConstraints(1e-10);context.setTime(2.7);context.setStepCount(17);
    int masks[]={-1,2,16,18};
    for(int event=0;event<24;++event) {
        int active=event%6;
        if(nested) {root.setCurrentIntegrator(active/3);branches[active/3]->setCurrentIntegrator(active%3);}
        else root.setCurrentIntegrator(active);
        root.setStepSize(0.0001+0.00003*(event%5));
        root.setConstraintTolerance(1e-9/(1+event%3));
        if(event%4==0)root.setIntegrationForceGroups(masks[(event/4)%4]);
        int called=(event%2==0?active:(active+2)%6);
        leaves[called]->setIntegrationForceGroups(masks[(event+1)%4]);
        if(event==10)context.setVelocitiesToTemperature(0,71);
        for(int sample=0;sample<2;++sample) {
            if(sample) {
                double time=context.getTime();
                if(event%2==0)root.step(1);else leaves[called]->step(1);
                // Repair only the known Reference clock defect after an adaptive step.
                int kind=(called+rotation)%6;
                if(kind==3 || kind==4)context.setTime(time+leaves[called]->getStepSize());
            }
            auto state=context.getState(State::Positions|State::Velocities|State::Forces|State::Energy);
            std::cout << "{\"rotation\":" << rotation << ",\"nested\":" << nested << ",\"constrained\":" << constrained
                << ",\"virtual_site\":" << virtualSite << ",\"event\":" << event << ",\"sample\":" << sample
                << ",\"current\":" << root.getCurrentIntegrator() << ",\"time\":" << state.getTime()
                << ",\"step_count\":" << state.getStepCount() << ",\"energy\":" << state.getPotentialEnergy()
                << ",\"kinetic\":" << state.getKineticEnergy() << ",\"dt\":[";
            for(int i=0;i<6;++i) {if(i)std::cout << ',';std::cout << leaves[i]->getStepSize();}
            std::cout << "],\"groups\":[";
            for(int i=0;i<6;++i) {if(i)std::cout << ',';std::cout << leaves[i]->getIntegrationForceGroups();}
            std::cout << "],\"tolerance\":[";
            for(int i=0;i<6;++i) {if(i)std::cout << ',';std::cout << leaves[i]->getConstraintTolerance();}
            std::cout << "],\"positions\":";vectors(state.getPositions());
            std::cout << ",\"velocities\":";vectors(state.getVelocities());
            std::cout << ",\"forces\":";vectors(state.getForces());std::cout << "}\n";
        }
    }
}

void updateProbe() {
    System system;system.addParticle(1.);system.addForce(new AndersenThermostat(0,1e9));
    CompoundIntegrator root;
    auto* variable=new VariableVerletIntegrator(1e-5);variable->setMaximumStepSize(0.001);
    auto* fixed=new VerletIntegrator(0.001);
    root.addIntegrator(variable);root.addIntegrator(fixed);
    Context context(system,root,Platform::getPlatformByName("Reference"));
    context.setPositions({Vec3()});context.setVelocities({Vec3(1,2,3)});
    fixed->step(1);
    auto v=context.getState(State::Velocities).getVelocities()[0];
    if(v!=Vec3(1,2,3))throw std::runtime_error("state update did not use active zero step size");
    root.setCurrentIntegrator(1);variable->step(1);
    v=context.getState(State::Velocities).getVelocities()[0];
    if(v!=Vec3())throw std::runtime_error("state update did not use active fixed step size");
}

int main() {
    updateProbe();std::cout << std::setprecision(17);
    for(int rotation : {0,1,3})for(bool nested : {false,true})
        for(bool constrained : {false,true})for(bool virtualSite : {false,true})run(rotation,nested,constrained,virtualSite);
}
