#include <openmm/Context.h>
#include <openmm/CustomCVForce.h>
#include <openmm/Platform.h>
#include <openmm/RGForce.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace OpenMM;
int main(int argc,char** argv) {
    const int n=argc>1?std::stoi(argv[1]):4096;
    const int variables=argc>2?std::stoi(argv[2]):8;
    const int iterations=argc>3?std::stoi(argv[3]):100;
    std::string terms;
    for (int i=0;i<variables;++i) { if (i) terms+="+"; terms+=std::to_string(i+1)+"*v"+std::to_string(i); }
    auto* cv=new CustomCVForce("s^2;s=("+terms+")/"+std::to_string(variables*(variables+1)/2));
    for (int i=0;i<variables;++i) {
        std::vector<int> particles; for (int p=i;p<n;p+=variables) particles.push_back(p);
        cv->addCollectiveVariable("v"+std::to_string(i),new RGForce(particles));
    }
    System system; std::vector<Vec3> positions;
    for (int i=0;i<n;++i) { system.addParticle(1); positions.emplace_back(2*std::sin(0.17*i),std::cos(0.31*i),0.7*std::sin(0.73*i)); }
    system.addForce(cv);
    VerletIntegrator integrator(0.001); Context context(system,integrator,Platform::getPlatformByName("Reference"));
    context.setPositions(positions); context.getState(State::Forces|State::Energy);
    const auto start=std::chrono::steady_clock::now();
    double energy=0;
    for (int i=0;i<iterations;++i) energy+=context.getState(State::Forces|State::Energy).getPotentialEnergy();
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << std::setprecision(17) << "{\"implementation\":\"openmm-reference\",\"precision\":\"f64\",\"api\":\"Context.getState\",\"variables\":" << variables << ",\"particles\":" << n << ",\"iterations\":" << iterations << ",\"seconds\":" << seconds << ",\"evaluations_per_second\":" << iterations/seconds << ",\"energy\":" << energy/iterations << "}\n";
}
