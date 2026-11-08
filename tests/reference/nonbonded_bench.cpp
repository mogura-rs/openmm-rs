#include <openmm/Context.h>
#include <openmm/NonbondedForce.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

using namespace OpenMM;

int main(int argc, char** argv) {
    const int side = argc > 1 ? std::stoi(argv[1]) : 8;
    const int iterations = argc > 2 ? std::stoi(argv[2]) : 10;
    const int n = side*side*side;
    const double length = side*0.35;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(length,0,0),Vec3(0,length,0),Vec3(0,0,length));
    auto* force = new NonbondedForce();
    force->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    force->setCutoffDistance(0.9);
    std::vector<Vec3> positions;
    for (int x=0;x<side;++x) for (int y=0;y<side;++y) for (int z=0;z<side;++z) {
        system.addParticle(1);
        force->addParticle((x+y+z)%2 == 0 ? 0.1 : -0.1,0.3,0.1);
        positions.emplace_back(float(x)*0.35f,float(y)*0.35f,float(z)*0.35f);
    }
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    context.setPositions(positions);
    context.getState(State::Forces|State::Energy);
    auto start = std::chrono::steady_clock::now();
    double energy = 0;
    for (int i=0;i<iterations;++i) energy += context.getState(State::Forces|State::Energy).getPotentialEnergy();
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << std::setprecision(17) << "{\"implementation\":\"OpenMM\",\"platform\":\"Reference\",\"precision\":\"double_with_f32_input\",\"particles\":" << n << ",\"iterations\":" << iterations << ",\"seconds\":" << seconds << ",\"evaluations_per_second\":" << iterations/seconds << ",\"energy\":" << energy/iterations << "}\n";
}
