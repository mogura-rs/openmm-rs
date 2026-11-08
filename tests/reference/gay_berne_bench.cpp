#include <openmm/Context.h>
#include <openmm/GayBerneForce.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace OpenMM;
int main(int argc,char** argv) {
    int n=std::stoi(argv[1]),iterations=std::stoi(argv[2]),method=std::stoi(argv[3]),model=std::stoi(argv[4]);
    int stride=model+1,grid=static_cast<int>(std::ceil(std::cbrt(n/stride)));double side=0.45*grid;
    System system;system.setDefaultPeriodicBoxVectors(Vec3(side,0,0),Vec3(0,side,0),Vec3(0,0,side));
    auto* gb=new GayBerneForce();gb->setNonbondedMethod(static_cast<GayBerneForce::NonbondedMethod>(method));
    gb->setCutoffDistance(0.9);gb->setSwitchingDistance(0.7);gb->setUseSwitchingFunction(method!=0);
    std::vector<Vec3> p(n);
    for(int i=0;i<n/stride;++i){
        Vec3 center(0.45*(i%grid),0.45*((i/grid)%grid),0.45*(i/(grid*grid)));
        system.addParticle(1);p[stride*i]=center;
        double width=0.25+0.01*(i%4);
        gb->addParticle(0.26+0.005*(i%3),0.7+0.1*(i%5),model==1?stride*i+1:-1,-1,model==0?width:0.32,model==0?width:0.24,model==0?width:0.24,1,1,1);
        if(model==1){system.addParticle(1);p[stride*i+1]=center+Vec3(0.08,0.02,0.03);gb->addParticle(0.1,0,-1,-1,0.1,0.1,0.1,1,1,1);}
    }
    system.addForce(gb);VerletIntegrator it(0.001);Context context(system,it,Platform::getPlatformByName("Reference"));context.setPositions(p);
    int flags=State::Forces|State::Energy;context.getState(flags);
    auto start=std::chrono::steady_clock::now();for(int i=0;i<iterations;++i)context.getState(flags);
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();auto state=context.getState(flags);
    double norm=0;for(auto f:state.getForces())norm+=f.dot(f);
    std::cout<<std::setprecision(17)<<"{\"implementation\":\"openmm-reference\",\"precision\":\"f64\",\"api\":\"Context.getState\",\"force\":\"gay_berne\",\"model\":"<<model<<",\"method\":"<<method<<",\"particles\":"<<n<<",\"iterations\":"<<iterations<<",\"seconds\":"<<seconds<<",\"evaluations_per_second\":"<<iterations/seconds<<",\"energy\":"<<state.getPotentialEnergy()<<",\"force_squared_norm\":"<<norm<<"}\n";
}
