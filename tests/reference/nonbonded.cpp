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
    const bool ljpme = name.find("ljpme") == 0;
    const bool ewald = name.find("ewald") == 0;
    const bool periodic = pme || ljpme || ewald || name.find("periodic") == 0;
    const bool triclinic = name.find("triclinic") != std::string::npos;
    const bool exceptions = name.find("exception") != std::string::npos;
    const bool switching = name.find("switch") != std::string::npos;
    const bool charged = name.find("charged") != std::string::npos;
    const bool dispersion = name.find("dispersion") != std::string::npos;
    const bool offsets = name.find("offset") != std::string::npos;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(3,0,0), Vec3(triclinic ? 0.4 : 0,3,0), Vec3(triclinic ? -0.2 : 0,triclinic ? 0.3 : 0,3));
    auto* force = new NonbondedForce();
    force->setNonbondedMethod(ljpme ? NonbondedForce::LJPME : ewald ? NonbondedForce::Ewald : pme ? NonbondedForce::PME : periodic ? NonbondedForce::CutoffPeriodic : name == "cutoff" ? NonbondedForce::CutoffNonPeriodic : NonbondedForce::NoCutoff);
    force->setCutoffDistance(1.2);
    force->setEwaldErrorTolerance(1e-5);
    force->setUseDispersionCorrection(dispersion);
    force->setUseSwitchingFunction(switching);
    force->setSwitchingDistance(0.8);
    force->setExceptionsUsePeriodicBoundaryConditions(name.find("wrapped") != std::string::npos);
    if (name.find("explicit") != std::string::npos) {
        force->setPMEParameters(2.7, 24, 25, 28);
        force->setLJPMEParameters(2.5, 18, 20, 21);
    }
    const double charges[] = {0.4,-0.3,0.1,charged ? 0.2 : -0.2};
    for (int i=0; i<4; ++i) {
        system.addParticle(2.0);
        force->addParticle(name.find("neutral") != std::string::npos ? 0 : charges[i], 0.3+0.02*i, 0.5+0.1*i);
    }
    if (exceptions) {
        force->addException(0,2,-0.02,0.32,0.4);
        force->addException(0,1,0,1,0);
    }
    if (offsets) {
        force->addGlobalParameter("lambda", 0.25);
        force->addGlobalParameter("mu", 0.1);
        force->addParticleParameterOffset("lambda", 0, 0.2, 0.04, 0.3);
        force->addParticleParameterOffset("mu", 0, -0.1, 0.01, -0.2);
        force->addParticleParameterOffset("lambda", 2, -0.15, -0.02, 0.2);
        force->addExceptionParameterOffset("lambda", 0, 0.04, 0.05, 0.1);
        force->addExceptionParameterOffset("mu", 1, 0.03, -0.2, 0.15);
    }
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, Platform::getPlatformByName("Reference"));
    context.setPositions({Vec3(0.1,0.2,0.3),Vec3(0.6,0.4,0.2),Vec3(2.8,0.7,0.4),Vec3(1.3,1.1,0.9)});
    if (name.find("runtime") != std::string::npos) {
        context.setParameter("lambda", 0.7);
        context.setParameter("mu", 0.2);
    }
    if (name.find("update") != std::string::npos) {
        context.getState(State::Energy);
        force->setParticleParameterOffset(0, "lambda", 0, 0.3, 0.06, -0.1);
        force->setExceptionParameterOffset(0, "lambda", 0, -0.02, 0.01, 0.2);
        force->setParticleParameters(3, 0.1, 0.38, 0.7);
        force->setExceptionParameters(0, 0, 2, 0.03, 0.36, 0.2);
        force->updateParametersInContext(context);
    }
    if (name.find("resize") != std::string::npos)
        context.setPeriodicBoxVectors(Vec3(3.4,0,0), Vec3(triclinic ? 0.4 : 0,3.2,0), Vec3(triclinic ? -0.2 : 0,triclinic ? 0.3 : 0,3.3));
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
    for (const auto& name : {"nocutoff", "cutoff", "periodic", "periodic_triclinic", "periodic_exception", "periodic_exception_wrapped", "periodic_dispersion", "periodic_switch_dispersion", "pme", "pme_triclinic", "pme_exception", "pme_exception_wrapped", "pme_charged", "pme_switch_dispersion",
        "ewald", "ewald_exception", "ewald_exception_wrapped", "ewald_charged_switch_dispersion",
        "ljpme", "ljpme_triclinic", "ljpme_exception", "ljpme_exception_wrapped", "ljpme_charged", "ljpme_explicit_triclinic", "ljpme_neutral",
        "nocutoff_exception_offset", "nocutoff_exception_offset_runtime", "nocutoff_exception_offset_runtime_update",
        "periodic_exception_offset_dispersion", "periodic_exception_offset_runtime_dispersion", "periodic_exception_offset_runtime_update_dispersion",
        "pme_exception_offset", "pme_exception_offset_runtime", "pme_exception_offset_runtime_update_dispersion",
        "ljpme_exception_offset", "ljpme_exception_offset_runtime", "ljpme_exception_offset_runtime_update",
        "ewald_resize", "pme_triclinic_resize", "ljpme_triclinic_resize", "ljpme_switch"}) run(name);
}
