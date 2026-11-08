#include <openmm/Context.h>
#include <openmm/DrudeForce.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace OpenMM;
extern "C" void registerDrudeReferenceKernelFactories();
int main(int argc,char** argv) {
    registerDrudeReferenceKernelFactories();
    int n=std::stoi(argv[1]),iterations=std::stoi(argv[2]),model=std::stoi(argv[3]),periodic=std::stoi(argv[4]),count=n/2;
    int grid=static_cast<int>(std::ceil(std::cbrt(count)));double side=.32*grid;
    System system;system.setDefaultPeriodicBoxVectors(Vec3(side,0,0),Vec3(.1*side,side,0),Vec3(-.08*side,.05*side,side));
    auto* force=new DrudeForce();force->setUsesPeriodicBoundaryConditions(periodic);std::vector<Vec3> points(n);
    for(int i=0;i<count;++i){system.addParticle(12.);system.addParticle(.4);
        force->addParticle(2*i+1,2*i,model?2*((i+1)%count):-1,model?2*((i+2)%count):-1,model?2*((i+3)%count):-1,-.1-.01*(i%4),.001+.0001*(i%5),.7,1.2);
        if(i>0)force->addScreenedPair(i-1,i,1.7+.1*(i%3));
        points[2*i]=Vec3(.05+.32*(i%grid)+.005*std::sin(1.7*i),.05+.32*((i/grid)%grid)+.005*std::cos(.9*i),.05+.32*(i/(grid*grid)));
        points[2*i+1]=points[2*i]+Vec3(.01,-.015,.012);
    }
    system.addForce(force);VerletIntegrator it(.001);Context context(system,it,Platform::getPlatformByName("Reference"));context.setPositions(points);int flags=State::Energy|State::Forces;context.getState(flags);
    auto start=std::chrono::steady_clock::now();for(int i=0;i<iterations;++i)context.getState(flags);double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();auto state=context.getState(flags);double norm=0;for(auto f:state.getForces())norm+=f.dot(f);
    std::cout<<std::setprecision(17)<<"{\"implementation\":\"openmm-reference\",\"api\":\"Context.getState\",\"precision\":\"f64\",\"force\":\"drude\",\"model\":"<<model<<",\"periodic\":"<<(periodic?"true":"false")<<",\"particles\":"<<n<<",\"drude_particles\":"<<count<<",\"iterations\":"<<iterations<<",\"seconds\":"<<seconds<<",\"evaluations_per_second\":"<<iterations/seconds<<",\"energy\":"<<state.getPotentialEnergy()<<",\"force_squared_norm\":"<<norm<<"}\n";
}
