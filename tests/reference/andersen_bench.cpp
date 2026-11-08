#include <openmm/AndersenThermostat.h>
#include <openmm/Context.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <chrono>
#include <iomanip>
#include <iostream>
using namespace OpenMM;
int main(int argc, char** argv) {
    const int n=argc>1 ? std::stoi(argv[1]) : 4096;
    const int iterations=argc>2 ? std::stoi(argv[2]) : 1000;
    System system;
    for (int i=0;i<n;++i) system.addParticle((i%3+1)*(i%3+1));
    auto* thermostat=new AndersenThermostat(300.0,37.0);
    thermostat->setRandomNumberSeed(3721);
    system.addForce(thermostat);
    VerletIntegrator integrator(0.001);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    context.setPositions(std::vector<Vec3>(n));
    integrator.step(10);
    const auto start=std::chrono::steady_clock::now();
    integrator.step(iterations);
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const double energy=context.getState(State::Energy).getKineticEnergy();
    std::cout << std::setprecision(17) << "{\"implementation\":\"openmm-reference\",\"precision\":\"f64\",\"api\":\"Context.step\",\"particles\":" << n << ",\"iterations\":" << iterations << ",\"seconds\":" << seconds << ",\"evaluations_per_second\":" << iterations/seconds << ",\"energy\":0,\"kinetic_energy\":" << energy << "}\n";
}
