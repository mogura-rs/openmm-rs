#include <openmm/Context.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/MonteCarloBarostat.h>
#include <openmm/MonteCarloMembraneBarostat.h>
#include <openmm/MonteCarloAnisotropicBarostat.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace OpenMM;
int main(int argc, char** argv) {
    const int n=argc>1?std::stoi(argv[1]):4096;
    const int iterations=argc>2?std::stoi(argv[2]):1000;
    const bool rigid=argc<4 || std::string(argv[3])=="rigid";
    const std::string kind=argc>4?argv[4]:"isotropic";
    if (kind!="isotropic" && kind!="anisotropic" && kind!="membrane") throw std::runtime_error("unknown barostat kind");
    const int scaled=rigid?n/2:n;
    const double kt=1.380649e-23*6.02214076e23/1000*300;
    const double pressure=kind=="membrane"?20-7.0/3.0:20;
    const double thermalVolume=kt/(pressure*0.0602214076);
    const double expectedVolume=(scaled+1)*thermalVolume;
    const double volumeStddev=std::sqrt(scaled+1)*thermalVolume;
    const double height=kind=="membrane"?3:std::cbrt(expectedVolume);
    const double length=kind=="membrane"?std::sqrt(expectedVolume/height):height;
    System system;
    for (int i=0;i<n;++i) system.addParticle(1);
    system.setDefaultPeriodicBoxVectors(Vec3(length,0,0),Vec3(0,length,0),Vec3(0,0,height));
    auto* bond=new HarmonicBondForce();
    for (int i=0;i<n;i+=2) bond->addBond(i,i+1,0.02,0);
    bond->setUsesPeriodicBoundaryConditions(true);system.addForce(bond);
    if (kind=="membrane") {
        auto* barostat=new MonteCarloMembraneBarostat(20,7,300,MonteCarloMembraneBarostat::XYIsotropic,MonteCarloMembraneBarostat::ZFixed,1,rigid);
        barostat->setRandomNumberSeed(3721);system.addForce(barostat);
    } else if (kind=="anisotropic") {
        auto* barostat=new MonteCarloAnisotropicBarostat(Vec3(20,20,20),300,true,false,false,1,rigid);
        barostat->setRandomNumberSeed(3721);system.addForce(barostat);
    } else {
        auto* barostat=new MonteCarloBarostat(20,300,1,rigid);barostat->setRandomNumberSeed(3721);system.addForce(barostat);
    }
    VerletIntegrator integrator(0.001);integrator.setIntegrationForceGroups(0);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    const int side=std::ceil(std::cbrt(n/2));
    std::vector<Vec3> positions;
    for (int i=0;i<n;++i) {
        int cell=i/2;
        positions.push_back(Vec3((cell%side)*0.1+(i%2)*0.02,((cell/side)%side)*0.1,(cell/(side*side))*0.1));
    }
    context.setPositions(positions);integrator.step(100);
    const auto start=std::chrono::steady_clock::now();integrator.step(iterations);
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    Vec3 a,b,c;context.getState(0).getPeriodicBoxVectors(a,b,c);
    const double volume=a[0]*b[1]*c[2];
    const double kinetic=context.getState(State::Energy).getKineticEnergy();
    std::cout << std::setprecision(17) << "{\"implementation\":\"openmm-reference\",\"precision\":\"f64\",\"api\":\"Context.step\",\"barostat\":true,\"barostat_kind\":\"" << kind << "\",\"rigid\":" << (rigid?"true":"false") << ",\"particles\":" << n << ",\"iterations\":" << iterations << ",\"seconds\":" << seconds << ",\"evaluations_per_second\":" << iterations/seconds << ",\"energy\":0,\"kinetic_energy\":" << kinetic << ",\"expected_kinetic_energy\":0,\"kinetic_energy_stddev\":0,\"volume\":" << volume << ",\"expected_volume\":" << expectedVolume << ",\"volume_stddev\":" << volumeStddev << "}\n";
}
