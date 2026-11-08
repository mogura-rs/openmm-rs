#include <openmm/Context.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/NonbondedForce.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <array>
#include <iomanip>
#include <iostream>
#include <string>

using namespace OpenMM;

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (int i=0; i<values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']';
    }
    std::cout << ']';
}

void run(NonbondedForce::NonbondedMethod method, const std::string& name, bool direct) {
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(3,0,0), Vec3(0,3,0), Vec3(0,0,3));
    auto* force = new NonbondedForce();
    force->setNonbondedMethod(method);
    force->setCutoffDistance(1.2);
    force->setEwaldErrorTolerance(1e-5);
    force->setForceGroup(1);
    force->setReciprocalSpaceForceGroup(5);
    force->setIncludeDirectSpace(direct);
    const double charges[] = {0.4,-0.3,0.1,-0.2};
    auto* external = new CustomExternalForce("lambda*(x*x+2*y*y+3*z*z)");
    external->addGlobalParameter("lambda", 0.2);
    external->setForceGroup(31);
    for (int i=0; i<4; ++i) {
        system.addParticle(2.0);
        force->addParticle(charges[i], 0.3+0.02*i, 0.5+0.1*i);
        external->addParticle(i, {});
    }
    force->addException(0,2,-0.02,0.32,0.4);
    force->addException(0,1,0,1,0);
    system.addForce(force);
    system.addForce(external);
    auto* bond = new HarmonicBondForce();
    bond->addBond(1,3,0.7,10);
    bond->setForceGroup(3);
    system.addForce(bond);
    VerletIntegrator integrator(0.001);
    integrator.setIntegrationForceGroups((1<<1)|(1<<3));
    Context context(system, integrator, Platform::getPlatformByName("Reference"));
    context.setPositions({Vec3(0.1,0.2,0.3),Vec3(0.6,0.4,0.2),Vec3(2.8,0.7,0.4),Vec3(1.3,1.1,0.9)});
    context.setVelocities({Vec3(0.01,-0.02,0.03),Vec3(0.02,0.03,-0.01),Vec3(-0.03,0.01,0.02),Vec3(0.01,0.01,-0.02)});
    for (int step : {0,3}) {
        if (step) integrator.step(step);
        for (unsigned mask : std::array<unsigned,6>{0,2,32,8,0x80000000u,0xffffffffu}) {
            State state = context.getState(State::Forces|State::Energy|State::Positions|State::Velocities, false, static_cast<int>(mask));
            std::cout << "{\"method\":\"" << name << "\",\"direct\":" << (direct ? "true" : "false") << ",\"step\":" << step << ",\"groups\":" << mask << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy() << ",\"forces\":";
            vectors(state.getForces());
            std::cout << ",\"positions\":";
            vectors(state.getPositions());
            std::cout << ",\"velocities\":";
            vectors(state.getVelocities());
            std::cout << "}\n";
        }
    }
}

int main() {
    std::cout << std::setprecision(17);
    for (bool direct : {true,false}) {
        run(NonbondedForce::NoCutoff, "NoCutoff", direct);
        run(NonbondedForce::CutoffPeriodic, "CutoffPeriodic", direct);
        run(NonbondedForce::Ewald, "Ewald", direct);
        run(NonbondedForce::PME, "PME", direct);
        run(NonbondedForce::LJPME, "LJPME", direct);
    }
}
