#include <openmm/BrownianIntegrator.h>
#include <openmm/Context.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/LangevinMiddleIntegrator.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

using namespace OpenMM;

void printVectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']';
    }
    std::cout << ']';
}

void run(const std::string& name, std::unique_ptr<Integrator> integrator, bool constrained = false) {
    System system;
    system.addParticle(2.0);
    system.addParticle(constrained ? 3.0 : 0.0);
    if (constrained) system.addConstraint(0, 1, 0.1);
    auto* force = new CustomExternalForce("2*x*x+3*y*y+4*z*z");
    force->addParticle(0);
    force->addParticle(1);
    system.addForce(force);
    Context context(system, *integrator, Platform::getPlatformByName("Reference"));
    if (constrained) {
        context.setPositions({Vec3(0, 0, 0), Vec3(0.1, 0, 0)});
        context.setVelocities({Vec3(0.3, 0.2, -0.1), Vec3(0.3, -0.1, 0.2)});
    } else {
        context.setPositions({Vec3(0.2, -0.3, 0.4), Vec3(1, 2, 3)});
        context.setVelocities({Vec3(0.3, 0.2, -0.1), Vec3(4, 5, 6)});
    }
    int previous = 0;
    for (int step : {0, 1, 10, 100}) {
        if (step > previous) integrator->step(step-previous);
        previous = step;
        State state = context.getState(State::Positions | State::Velocities | State::Forces | State::Energy);
        std::cout << "{\"case\":\"" << name << "\",\"step\":" << step << ",\"positions\":";
        printVectors(state.getPositions());
        std::cout << ",\"velocities\":";
        printVectors(state.getVelocities());
        std::cout << ",\"forces\":";
        printVectors(state.getForces());
        std::cout << ",\"potential_energy\":" << state.getPotentialEnergy()
                  << ",\"kinetic_energy\":" << state.getKineticEnergy() << "}\n";
    }
}

int main() {
    std::cout << std::setprecision(17);
    run("verlet", std::make_unique<VerletIntegrator>(0.001));
    run("langevin", std::make_unique<LangevinMiddleIntegrator>(0.0, 2.0, 0.001));
    run("langevin_zero_friction", std::make_unique<LangevinMiddleIntegrator>(0.0, 0.0, 0.001));
    run("brownian", std::make_unique<BrownianIntegrator>(0.0, 2.0, 0.001));
    run("verlet_constrained", std::make_unique<VerletIntegrator>(0.001), true);
    run("langevin_constrained", std::make_unique<LangevinMiddleIntegrator>(0.0, 2.0, 0.001), true);
    run("brownian_constrained", std::make_unique<BrownianIntegrator>(0.0, 2.0, 0.001), true);
}
