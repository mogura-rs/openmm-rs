#include <openmm/ATMForce.h>
#include <openmm/CustomManyParticleForce.h>
#include <openmm/NonbondedForce.h>
#include <openmm/Context.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace OpenMM;
int main(int argc,char** argv){
    int n=std::stoi(argv[1]),iterations=std::stoi(argv[2]),model=std::stoi(argv[3]);int grid=std::max(5,static_cast<int>(std::ceil(std::cbrt(n))));double side=0.32*grid;
    System system;system.setDefaultPeriodicBoxVectors(Vec3(side,0,0),Vec3(0,side,0),Vec3(0,0,side));auto* atm=new ATMForce(0.2,0.6,0.1,0,0,50,1,0.5,1);
    auto* local=new CustomManyParticleForce(1,"q1*(x1*x1+y1*y1+z1*z1)");local->addPerParticleParameter("q");auto* nb=new NonbondedForce();nb->setNonbondedMethod(NonbondedForce::CutoffPeriodic);nb->setCutoffDistance(0.8);nb->setUseDispersionCorrection(false);nb->setUseSwitchingFunction(true);nb->setSwitchingDistance(0.6);
    std::vector<Vec3> p(n);for(int i=0;i<n;++i){system.addParticle(1);local->addParticle({0.1+0.01*(i%7)});nb->addParticle(i%2==0?0.02:-0.02,0.2+0.002*(i%5),0.05);
        if(model==0)atm->addParticle(Vec3(0.03*std::sin(i),0.02*std::cos(i),0.01));else atm->addParticle(new ATMForce::ParticleOffsetDisplacement(i%2==0?i+1:i-1,i));
        p[i]=Vec3(0.05+0.32*(i%grid)+0.015*std::sin(1.7*i),0.05+0.32*((i/grid)%grid)+0.017*std::cos(0.9*i),0.05+0.32*(i/(grid*grid))+0.013*std::sin(1.3*i));}
    if(model==0){atm->addForce(local);delete nb;}else{atm->addForce(nb);delete local;}system.addForce(atm);VerletIntegrator it(0.001);Context context(system,it,Platform::getPlatformByName("Reference"));context.setPositions(p);int flags=State::Forces|State::Energy;context.getState(flags);auto start=std::chrono::steady_clock::now();for(int i=0;i<iterations;++i)context.getState(flags);double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();auto state=context.getState(flags);double norm=0;for(auto f:state.getForces())norm+=f.dot(f);
    std::cout<<std::setprecision(17)<<"{\"implementation\":\"openmm-reference\",\"precision\":\"f64\",\"api\":\"Context.getState\",\"force\":\"atm\",\"model\":"<<model<<",\"particles\":"<<n<<",\"iterations\":"<<iterations<<",\"seconds\":"<<seconds<<",\"evaluations_per_second\":"<<iterations/seconds<<",\"energy\":"<<state.getPotentialEnergy()<<",\"force_squared_norm\":"<<norm<<"}\n";
}
