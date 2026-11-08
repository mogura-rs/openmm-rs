#include <openmm/ConstantPotentialForce.h>
#include <openmm/Context.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
using namespace OpenMM;
int main(int argc,char** argv) {
    int n=std::stoi(argv[1]),iterations=std::stoi(argv[2]),solver=std::stoi(argv[3]);
    int grid=std::max(5,static_cast<int>(std::ceil(std::cbrt(n)))),k=n/4;double side=.32*grid;
    int mesh=16;while(mesh<static_cast<int>(std::ceil(6.*side)))mesh*=2;
    auto setupStart=std::chrono::steady_clock::now();System system;system.setDefaultPeriodicBoxVectors(Vec3(side,0,0),Vec3(0,side,0),Vec3(0,0,side));
    auto* force=new ConstantPotentialForce();force->setConstantPotentialMethod(solver==2?ConstantPotentialForce::Matrix:ConstantPotentialForce::CG);force->setUsePreconditioner(solver==1);force->setCGErrorTolerance(1e-8);
    force->setCutoffDistance(.8);force->setPMEParameters(3.,mesh,mesh,mesh);force->setUseChargeConstraint(true);force->setChargeConstraintTarget(.25);force->setExternalField(Vec3(.1,-.05,.08));
    std::set<int> first,second;for(int i=0;i<k;++i)(i%2==0?first:second).insert(i);force->addElectrode(first,2.,.08,.005);force->addElectrode(second,-2.,.10,.005);
    std::vector<Vec3> base(n);for(int i=0;i<n;++i){system.addParticle(i<k?0.:12.);force->addParticle(i<k?0.:(i%2==0?.02:-.02));
        base[i]=Vec3(.05+.32*(i%grid)+.015*std::sin(1.7*i),.05+.32*((i/grid)%grid)+.017*std::cos(.9*i),.05+.32*(i/(grid*grid))+.013*std::sin(1.3*i));}
    system.addForce(force);VerletIntegrator integrator(.001);Context context(system,integrator,Platform::getPlatformByName("Reference"));context.setPositions(base);int flags=State::Energy|State::Forces;
    auto firstStart=std::chrono::steady_clock::now();context.getState(flags);double firstSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-firstStart).count();double setupSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setupStart).count();
    auto p=base;auto start=std::chrono::steady_clock::now();for(int iteration=0;iteration<iterations;++iteration){
        for(int i=k;i<n;++i){p[i][0]=base[i][0]+.002*std::sin(.37*(iteration+1)+.11*i);p[i][1]=base[i][1]+.001*std::cos(.23*(iteration+1)+.07*i);}
        context.setPositions(p);context.getState(flags);
    }
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();auto state=context.getState(flags);double norm=0;for(auto f:state.getForces())norm+=f.dot(f);
    std::vector<double> charges;force->getCharges(context,charges);double chargeNorm=0,totalCharge=0;for(double q:charges){chargeNorm+=q*q;totalCharge+=q;}
    std::cout<<std::setprecision(17)<<"{\"implementation\":\"openmm-reference\",\"precision\":\"f64\",\"api\":\"Context.setPositions/getState\",\"force\":\"constant_potential\",\"solver\":"<<solver<<",\"particles\":"<<n<<",\"electrode_particles\":"<<k<<",\"mesh\":"<<mesh<<",\"iterations\":"<<iterations<<",\"setup_seconds\":"<<setupSeconds<<",\"first_evaluation_seconds\":"<<firstSeconds<<",\"seconds\":"<<seconds<<",\"evaluations_per_second\":"<<iterations/seconds<<",\"energy\":"<<state.getPotentialEnergy()<<",\"force_squared_norm\":"<<norm<<",\"charge_squared_norm\":"<<chargeNorm<<",\"total_charge\":"<<totalCharge<<"}\n";
}
