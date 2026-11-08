#include <openmm/Context.h>
#include <openmm/NonbondedForce.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <iomanip>
#include <iostream>
#include <string>

using namespace OpenMM;

void run(const std::string& name) {
    const bool pme = name.find("pme") == 0;
    const bool periodic = pme || name.find("periodic") == 0;
    const bool triclinic = name.find("triclinic") != std::string::npos;
    const bool exceptions = name.find("exception") != std::string::npos;
    const bool switching = name.find("switch") != std::string::npos;
    const bool charged = name.find("charged") != std::string::npos;
    const bool dispersion = name.find("dispersion") != std::string::npos;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(3,0,0), Vec3(triclinic ? 0.4 : 0,3,0), Vec3(triclinic ? -0.2 : 0,triclinic ? 0.3 : 0,3));
    auto* force = new NonbondedForce();
    force->setNonbondedMethod(pme ? NonbondedForce::PME : periodic ? NonbondedForce::CutoffPeriodic : name == "cutoff" ? NonbondedForce::CutoffNonPeriodic : NonbondedForce::NoCutoff);
    force->setCutoffDistance(1.2);
    force->setEwaldErrorTolerance(1e-5);
    force->setUseDispersionCorrection(dispersion);
    force->setUseSwitchingFunction(switching);
    force->setSwitchingDistance(0.8);
    force->setExceptionsUsePeriodicBoundaryConditions(name.find("wrapped") != std::string::npos);
    const double charges[] = {0.4,-0.3,0.1,charged ? 0.2 : -0.2};
    for (int i=0; i<4; ++i) {
        system.addParticle(2.0);
        force->addParticle(charges[i], 0.3+0.02*i, 0.5+0.1*i);
    }
    if (exceptions) {
        force->addException(0,2,-0.02,0.32,0.4);
        force->addException(0,1,0,1,0);
    }
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, Platform::getPlatformByName("Reference"));
    context.setPositions({Vec3(0.1,0.2,0.3),Vec3(0.6,0.4,0.2),Vec3(2.8,0.7,0.4),Vec3(1.3,1.1,0.9)});
    State state = context.getState(State::Forces | State::Energy);
    std::cout << "{\"case\":\"" << name << "\",\"energy\":" << state.getPotentialEnergy() << ",\"forces\":[";
    for (int i=0; i<4; ++i) {
        if (i) std::cout << ',';
        const auto& f = state.getForces()[i];
        std::cout << '[' << f[0] << ',' << f[1] << ',' << f[2] << ']';
    }
    std::cout << "]}\n";
}

int main() {
    std::cout << std::setprecision(17);
    for (const auto& name : {"nocutoff", "cutoff", "periodic", "periodic_triclinic", "periodic_exception", "periodic_exception_wrapped", "periodic_dispersion", "periodic_switch_dispersion", "pme", "pme_triclinic", "pme_exception", "pme_exception_wrapped", "pme_charged", "pme_switch_dispersion"}) run(name);
}
