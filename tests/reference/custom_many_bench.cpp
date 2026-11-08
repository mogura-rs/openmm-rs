#include <openmm/Context.h>
#include <openmm/CustomManyParticleForce.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace OpenMM;
int main(int argc,char** argv){
 int n=std::stoi(argv[1]),iterations=std::stoi(argv[2]),method=std::stoi(argv[3]),model=std::stoi(argv[4]);int grid=std::max(5,static_cast<int>(std::ceil(std::cbrt(n))));double side=0.32*grid;
 System system;system.setDefaultPeriodicBoxVectors(Vec3(side,0,0),Vec3(0,side,0),Vec3(0,0,side));
 auto* f=new CustomManyParticleForce(3,model==0?"lambda*(q1+q2+q3)*exp(-distance(p1,p2)-distance(p2,p3)-distance(p3,p1))":"lambda*q1*(q2+q3)*(cos(angle(p2,p1,p3))+1/3)^2*exp(-distance(p1,p2)-distance(p1,p3))");
 f->addPerParticleParameter("q");f->addGlobalParameter("lambda",0.7);f->setPermutationMode(model==0?CustomManyParticleForce::SinglePermutation:CustomManyParticleForce::UniqueCentralParticle);f->setNonbondedMethod(static_cast<CustomManyParticleForce::NonbondedMethod>(method));f->setCutoffDistance(0.65);
 std::vector<Vec3> p(n);for(int i=0;i<n;++i){system.addParticle(1);f->addParticle({0.2+0.03*(i%7)});p[i]=Vec3(0.05+0.32*(i%grid)+0.015*std::sin(1.7*i),0.05+0.32*((i/grid)%grid)+0.017*std::cos(0.9*i),0.05+0.32*(i/(grid*grid))+0.013*std::sin(1.3*i));}
 system.addForce(f);VerletIntegrator it(0.001);Context context(system,it,Platform::getPlatformByName("Reference"));context.setPositions(p);int flags=State::Forces|State::Energy;context.getState(flags);auto start=std::chrono::steady_clock::now();for(int i=0;i<iterations;++i)context.getState(flags);double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();auto state=context.getState(flags);double norm=0;for(auto f:state.getForces())norm+=f.dot(f);
 std::cout<<std::setprecision(17)<<"{\"implementation\":\"openmm-reference\",\"precision\":\"f64\",\"api\":\"Context.getState\",\"force\":\"custom_many\",\"model\":"<<model<<",\"method\":"<<method<<",\"particles\":"<<n<<",\"iterations\":"<<iterations<<",\"seconds\":"<<seconds<<",\"evaluations_per_second\":"<<iterations/seconds<<",\"energy\":"<<state.getPotentialEnergy()<<",\"force_squared_norm\":"<<norm<<"}\n";
}
