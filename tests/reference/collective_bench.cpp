#include <openmm/Context.h>
#include <openmm/OrientationRestraintForce.h>
#include <openmm/Platform.h>
#include <openmm/RGForce.h>
#include <openmm/RMSDForce.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace OpenMM;
int main(int argc, char** argv) {
    const int n=argc>1?std::stoi(argv[1]):512;
    const int iterations=argc>2?std::stoi(argv[2]):1000;
    const std::string kind=argc>3?argv[3]:"rmsd";
    System system;
    std::vector<Vec3> reference, positions;
    for (int i=0;i<n;++i) {
        Vec3 p(2*std::sin(0.17*i),std::cos(0.31*i),0.7*std::sin(0.73*i));
        reference.push_back(p);
        positions.emplace_back(std::cos(0.7)*p[0]-std::sin(0.7)*p[1]+0.3,std::sin(0.7)*p[0]+std::cos(0.7)*p[1]-0.4,p[2]+0.03*std::cos(0.11*i));
        system.addParticle(1);
    }
    if (kind=="rmsd") system.addForce(new RMSDForce(reference));
    else if (kind=="orientation") system.addForce(new OrientationRestraintForce(1.7,reference));
    else if (kind=="rg") system.addForce(new RGForce());
    else return 2;
    VerletIntegrator integrator(0.001); Context context(system,integrator,Platform::getPlatformByName("Reference"));
    context.setPositions(positions); context.getState(State::Forces|State::Energy);
    const auto start=std::chrono::steady_clock::now();
    double energy=0;
    for (int i=0;i<iterations;++i) energy+=context.getState(State::Forces|State::Energy).getPotentialEnergy();
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << std::setprecision(17) << "{\"implementation\":\"openmm-reference\",\"precision\":\"f64\",\"api\":\"Context.getState\",\"force\":\"" << kind << "\",\"particles\":" << n << ",\"iterations\":" << iterations << ",\"seconds\":" << seconds << ",\"evaluations_per_second\":" << iterations/seconds << ",\"energy\":" << energy/iterations << "}\n";
}
