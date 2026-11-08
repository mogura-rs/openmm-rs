#include <openmm/Context.h>
#include <openmm/GBSAOBCForce.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace OpenMM;
int main(int argc,char** argv) {
    int n=std::stoi(argv[1]),iterations=std::stoi(argv[2]),method=std::stoi(argv[3]);
    int grid=static_cast<int>(std::ceil(std::cbrt(n)));double side=0.3*grid;
    System system;system.setDefaultPeriodicBoxVectors(Vec3(side,0,0),Vec3(0,side,0),Vec3(0,0,side));
    auto* gb=new GBSAOBCForce();gb->setNonbondedMethod(static_cast<GBSAOBCForce::NonbondedMethod>(method));gb->setCutoffDistance(0.8);
    std::vector<Vec3> p(n);
    for(int i=0;i<n;++i){system.addParticle(1);gb->addParticle(0.1*(i%7-3),0.11+0.02*(i%5),0.65+0.1*(i%4));p[i]=Vec3(0.3*(i%grid),0.3*((i/grid)%grid),0.3*(i/(grid*grid)));}
    system.addForce(gb);VerletIntegrator it(0.001);Context context(system,it,Platform::getPlatformByName("Reference"));context.setPositions(p);
    int flags=State::Forces|State::Energy;context.getState(flags);
    auto start=std::chrono::steady_clock::now();for(int i=0;i<iterations;++i)context.getState(flags);
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();auto state=context.getState(flags);
    double norm=0;for(auto f:state.getForces())norm+=f.dot(f);
    std::cout<<std::setprecision(17)<<"{\"implementation\":\"openmm-reference\",\"precision\":\"f64\",\"api\":\"Context.getState\",\"force\":\"gbsa_obc\",\"method\":"<<method<<",\"particles\":"<<n<<",\"iterations\":"<<iterations<<",\"seconds\":"<<seconds<<",\"evaluations_per_second\":"<<iterations/seconds<<",\"energy\":"<<state.getPotentialEnergy()<<",\"force_squared_norm\":"<<norm<<"}\n";
}
