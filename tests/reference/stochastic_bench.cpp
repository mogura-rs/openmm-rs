#include <openmm/BrownianIntegrator.h>
#include <openmm/Context.h>
#include <openmm/LangevinMiddleIntegrator.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
using namespace OpenMM;
int main(int argc,char**argv) {
    std::string kind=argc>1?argv[1]:"langevin";
    int n=argc>2?std::stoi(argv[2]):4096;
    int iterations=argc>3?std::stoi(argv[3]):1500;
    double temperature=argc>4?std::stod(argv[4]):300;
    System system;
    for(int i=0;i<n;++i)system.addParticle(1);
    std::unique_ptr<Integrator> integrator;
    if(kind=="verlet")integrator=std::make_unique<VerletIntegrator>(0.001);
    else if(kind=="langevin") {
        auto* value=new LangevinMiddleIntegrator(temperature,0.7,0.001);
        value->setRandomNumberSeed(3721);integrator.reset(value);
    } else if(kind=="brownian") {
        auto* value=new BrownianIntegrator(temperature,0.7,0.001);
        value->setRandomNumberSeed(3721);integrator.reset(value);
    } else throw std::runtime_error("unknown integrator");
    Context context(system,*integrator,Platform::getPlatformByName("Reference"));
    context.setPositions(std::vector<Vec3>(n));
    context.setVelocities(std::vector<Vec3>(n,Vec3(0.1,0.2,0.3)));
    integrator->step(10);
    auto start=std::chrono::steady_clock::now();integrator->step(iterations);
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    double energy=context.getState(State::Energy).getKineticEnergy();
    double kb=1.380649e-23*6.02214076e23/1000,variance=0,meanSquared=0.14*n;
    if(kind=="langevin") {
        double decay=std::exp(-2*0.7*0.001*(iterations+10));
        variance=kb*temperature*(1-decay);meanSquared*=decay;
    } else if(kind=="brownian") {
        variance=2*kb*temperature/(0.7*0.001);meanSquared=0;
    }
    double expected=0.5*(3*n*variance+meanSquared);
    double stddev=std::sqrt(1.5*n*variance*variance+variance*meanSquared);
    std::cout<<std::setprecision(17)<<"{\"implementation\":\"openmm-reference\",\"precision\":\"f64\",\"api\":\"Context.step\",\"integrator\":\""<<kind<<"\",\"temperature\":"<<temperature<<",\"particles\":"<<n<<",\"iterations\":"<<iterations<<",\"seconds\":"<<seconds<<",\"evaluations_per_second\":"<<iterations/seconds<<",\"energy\":0,\"kinetic_energy\":"<<energy<<",\"expected_kinetic_energy\":"<<expected<<",\"kinetic_energy_stddev\":"<<stddev<<"}\n";
}
