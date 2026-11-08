#include <openmm/AndersenThermostat.h>
#include <openmm/Context.h>
#include <openmm/CustomBondForce.h>
#include <openmm/CustomCVForce.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/CustomNonbondedForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/TabulatedFunction.h>
#include <openmm/VerletIntegrator.h>
#include <iomanip>
#include <iostream>
#include <memory>
using namespace OpenMM;

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (size_t i=0;i<values.size();++i) {
        if(i) std::cout << ',';
        std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']';
    }
    std::cout << ']';
}

void run(int kind) {
    System system;
    for (int i=0;i<4;++i) system.addParticle(i+1);
    auto* bond = kind==0 ? new HarmonicBondForce() : nullptr;
    auto* nonbonded = kind==1 ? new CustomNonbondedForce("a*s1*s2*f(r)") : nullptr;
    auto* cv = kind==2 ? new CustomCVForce("scale*q^2") : nullptr;
    auto* nested = kind==2 ? new CustomCVForce("c*v") : nullptr;
    auto* child = kind==2 ? new CustomBondForce("k*(r-b)^2") : nullptr;
    auto* thermostat = kind==3 ? new AndersenThermostat(100,2) : nullptr;
    Continuous1DFunction* table=nullptr;
    Force* root=nullptr;
    if(bond) {
        bond->addBond(0,1,0.3,1);
        root=bond;
    } else if(nonbonded) {
        nonbonded->addPerParticleParameter("s");
        nonbonded->addGlobalParameter("a",0.4);
        for(int i=0;i<4;++i) nonbonded->addParticle({0.2*(i+1)});
        table=new Continuous1DFunction({0,0.2,0.1,0.3},0,3);
        nonbonded->addTabulatedFunction("f",table);
        root=nonbonded;
    } else if(cv) {
        child->addGlobalParameter("b",0.1);
        child->addPerBondParameter("k");
        child->addBond(0,1,{0.2});
        nested->addCollectiveVariable("v",child);
        nested->addGlobalParameter("c",0.3);
        cv->addCollectiveVariable("q",nested);
        cv->addGlobalParameter("scale",0.4);
        root=cv;
    } else {
        root=thermostat;
        auto* external=new CustomExternalForce("0.2*x^2");
        for(int i=0;i<4;++i) external->addParticle(i,{});
        system.addForce(external);
    }
    system.addForce(root);
    std::vector<std::unique_ptr<VerletIntegrator>> integrators;
    std::vector<std::unique_ptr<Context>> contexts;
    for(int phase=0;phase<3;++phase) {
        root->setForceGroup(phase+1);
        root->setName("edited force");
        system.setParticleMass(0,2+phase);
        double side=4+0.2*phase;
        system.setDefaultPeriodicBoxVectors(Vec3(side,0,0),Vec3(0,side,0),Vec3(0,0,side));
        if(bond) {
            bond->setBondParameters(0,phase,3,0.4+0.1*phase,3+phase);
            bond->setUsesPeriodicBoundaryConditions(phase!=1);
        } else if(nonbonded) {
            nonbonded->setGlobalParameterDefaultValue(0,0.7+phase);
            nonbonded->setParticleParameters(1,{1.5+phase});
            nonbonded->setNonbondedMethod(phase==1 ? CustomNonbondedForce::CutoffPeriodic : CustomNonbondedForce::NoCutoff);
            nonbonded->setCutoffDistance(1.6);
            table->setFunctionParameters({0.1+phase,0.4,0.8,0.3+phase},0,3);
        } else if(cv) {
            cv->setGlobalParameterDefaultValue(0,0.7+phase);
            nested->setGlobalParameterDefaultValue(0,1.1+0.2*phase);
            child->setGlobalParameterDefaultValue(0,0.4+0.1*phase);
            child->setBondParameters(0,phase,3,{0.5+phase});
            child->setUsesPeriodicBoundaryConditions(phase!=1);
        } else {
            thermostat->setDefaultTemperature(200+50*phase);
            thermostat->setDefaultCollisionFrequency(7+phase);
        }
        integrators.push_back(std::make_unique<VerletIntegrator>(0.0002));
        contexts.push_back(std::make_unique<Context>(system,*integrators.back(),Platform::getPlatformByName("Reference")));
        for(int c=0;c<=phase;++c) for(int point=0;point<3;++point) {
            auto& context=*contexts[c];
            context.setPositions({Vec3(0.1,0.2,0.3),Vec3(0.7+point*0.03,0.4,0.5),Vec3(1.4,0.9+point*0.02,0.2),Vec3(3.5,0.3,0.7+point*0.01)});
            context.setVelocities({Vec3(0.2,-0.1,0.3),Vec3(-0.3,0.2,0.1),Vec3(0.1,0.3,-0.2),Vec3(0.4,-0.2,0.1)});
            for(int groups : {-1,2,4,8}) {
                auto state=context.getState(State::Energy|State::Forces|State::Parameters,false,groups);
                std::cout << "{\"kind\":" << kind << ",\"phase\":" << phase << ",\"context\":" << c
                          << ",\"point\":" << point << ",\"groups\":" << groups
                          << ",\"system_pbc\":" << (system.usesPeriodicBoundaryConditions()?"true":"false")
                          << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy()
                          << ",\"forces\":";
                vectors(state.getForces());
                std::cout << ",\"parameters\":{";
                bool first=true;
                for(auto pair:state.getParameters()) {
                    if(!first)std::cout << ',';
                    first=false;
                    std::cout << '"' << pair.first << "\":" << pair.second;
                }
                std::cout << "}}\n";
            }
        }
    }
}

int main() {
    std::cout << std::setprecision(17);
    for(int kind=0;kind<4;++kind)run(kind);
}
