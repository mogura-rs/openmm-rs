#include <openmm/Context.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/VirtualSite.h>
#include <iomanip>
#include <iostream>
#include <string>

using namespace OpenMM;

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (size_t i=0;i<values.size();++i) {
        if (i) std::cout << ',';
        std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']';
    }
    std::cout << ']';
}

void run(const std::string& name) {
    System system;
    for (double mass : {2.0,1.0,3.0,0.0}) system.addParticle(mass);
    system.setDefaultPeriodicBoxVectors(Vec3(2,0,0),Vec3(0.3,2.4,0),Vec3(-0.2,0.4,2.8));
    if (name == "two" || name == "nested") system.setVirtualSite(3,new TwoParticleAverageSite(0,1,0.3,0.7));
    if (name == "three") system.setVirtualSite(3,new ThreeParticleAverageSite(0,1,2,0.2,0.3,0.5));
    if (name == "out") system.setVirtualSite(3,new OutOfPlaneSite(0,1,2,0.2,0.3,0.4));
    if (name == "local") system.setVirtualSite(3,new LocalCoordinatesSite(0,1,2,Vec3(0.2,0.3,0.5),Vec3(-1,1,0),Vec3(-1,0,1),Vec3(0.1,0.2,0.3)));
    if (name == "symmetry" || name == "fractional") system.setVirtualSite(3,new SymmetrySite(0,Vec3(0,-1,0),Vec3(1,0,0),Vec3(0,0,1),Vec3(0.1,0.2,0.3),name == "fractional"));
    auto* force = new CustomExternalForce("x*x+2*y*y+3*z*z");
    force->addParticle(3);
    std::vector<Vec3> positions = {Vec3(0.1,0.2,0.3),Vec3(0.5,0.1,0.4),Vec3(0.2,0.7,0.8),Vec3()};
    if (name == "nested") {
        system.addParticle(0);
        system.setVirtualSite(4,new TwoParticleAverageSite(3,1,0.6,0.4));
        force->addParticle(4);
        positions.emplace_back();
    }
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    context.setPositions(positions);
    context.computeVirtualSites();
    Vec3 initialGradientForce;
    if (name == "fractional") {
        for (int axis=0;axis<3;++axis) {
            auto plus = positions;
            auto minus = positions;
            plus[0][axis] += 1e-6;
            minus[0][axis] -= 1e-6;
            context.setPositions(plus);
            context.computeVirtualSites();
            double ep = context.getState(State::Energy).getPotentialEnergy();
            context.setPositions(minus);
            context.computeVirtualSites();
            double em = context.getState(State::Energy).getPotentialEnergy();
            initialGradientForce[axis] = -(ep-em)/2e-6;
        }
        context.setPositions(positions);
        context.computeVirtualSites();
    }
    int previous=0;
    for (int step : {0,1,10}) {
        if (step > previous) integrator.step(step-previous);
        previous=step;
        State state=context.getState(State::Positions|State::Forces|State::Energy);
        std::cout << "{\"case\":\"" << name << "\",\"step\":" << step << ",\"energy\":" << state.getPotentialEnergy() << ",\"positions\":";
        vectors(state.getPositions());
        std::cout << ",\"forces\":";
        vectors(state.getForces());
        if (name == "fractional" && step == 0) {
            std::cout << ",\"initial_energy_gradient_force\":[" << initialGradientForce[0] << ',' << initialGradientForce[1] << ',' << initialGradientForce[2] << ']';
        }
        std::cout << "}\n";
    }
}

int main() {
    std::cout << std::setprecision(17);
    for (const auto& name : {"two","three","out","local","symmetry","fractional","nested"}) run(name);
}
