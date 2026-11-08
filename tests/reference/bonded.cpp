#include <openmm/Context.h>
#include <openmm/CustomAngleForce.h>
#include <openmm/CustomBondForce.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/CustomTorsionForce.h>
#include <openmm/HarmonicAngleForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/PeriodicTorsionForce.h>
#include <openmm/Platform.h>
#include <openmm/RBTorsionForce.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <iomanip>
#include <iostream>
#include <string>

using namespace OpenMM;

void run(const std::string& name, bool periodic) {
    System system;
    for (int i=0;i<4;++i) system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(3,0,0),Vec3(0.4,3,0),Vec3(-0.2,0.3,3));
    if (name == "bond") {
        auto* f=new HarmonicBondForce(); f->addBond(0,1,0.6,10); f->setUsesPeriodicBoundaryConditions(periodic); system.addForce(f);
    } else if (name == "angle") {
        auto* f=new HarmonicAngleForce(); f->addAngle(0,1,2,1.1,2); f->setUsesPeriodicBoundaryConditions(periodic); system.addForce(f);
    } else if (name == "torsion") {
        auto* f=new PeriodicTorsionForce(); f->addTorsion(0,1,2,3,3,0.2,0.7); f->setUsesPeriodicBoundaryConditions(periodic); system.addForce(f);
    } else if (name == "rb") {
        auto* f=new RBTorsionForce(); f->addTorsion(0,1,2,3,1,0.2,0.3,-0.4,0.1,0.2); f->setUsesPeriodicBoundaryConditions(periodic); system.addForce(f);
    } else if (name == "custom_bond") {
        auto* f=new CustomBondForce("0.7*(r-0.6)^2"); f->addBond(0,1); f->setUsesPeriodicBoundaryConditions(periodic); system.addForce(f);
    } else if (name == "custom_angle") {
        auto* f=new CustomAngleForce("0.7*(theta-1.1)^2"); f->addAngle(0,1,2); f->setUsesPeriodicBoundaryConditions(periodic); system.addForce(f);
    } else if (name == "custom_torsion") {
        auto* f=new CustomTorsionForce("0.7*(1+cos(3*theta-0.2))"); f->addTorsion(0,1,2,3); f->setUsesPeriodicBoundaryConditions(periodic); system.addForce(f);
    } else {
        auto* f=new CustomExternalForce("x*x+2*y*y+3*z*z"); f->addParticle(0); system.addForce(f);
    }
    VerletIntegrator integrator(0.001);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    std::vector<Vec3> positions={Vec3(0.1,0.2,0.3),Vec3(0.8,-0.1,0.4),Vec3(0.9,0.7,0.5),Vec3(1.2,0.8,1.4)};
    if (periodic) { positions[2]+=Vec3(0.4,3,0); positions[3]+=Vec3(-0.2,0.3,3); }
    context.setPositions(positions);
    State state=context.getState(State::Forces|State::Energy);
    std::cout << "{\"case\":\"" << name << "\",\"periodic\":" << (periodic ? "true" : "false") << ",\"energy\":" << state.getPotentialEnergy() << ",\"forces\":[";
    for (int i=0;i<4;++i) {
        if (i) std::cout << ',';
        const auto& f=state.getForces()[i];
        std::cout << '[' << f[0] << ',' << f[1] << ',' << f[2] << ']';
    }
    std::cout << "]}\n";
}

int main() {
    std::cout << std::setprecision(17);
    for (const auto& name : {"bond","angle","torsion","rb","custom_bond","custom_angle","custom_torsion"}) { run(name,false); run(name,true); }
    run("external",false);
}
