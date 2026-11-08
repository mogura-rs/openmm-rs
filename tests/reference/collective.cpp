#include <openmm/Context.h>
#include <openmm/CustomVolumeForce.h>
#include <openmm/OrientationRestraintForce.h>
#include <openmm/Platform.h>
#include <openmm/RGForce.h>
#include <openmm/RMSDForce.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace OpenMM;
void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (size_t i=0; i<values.size(); ++i) { if (i) std::cout << ','; std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']'; }
    std::cout << ']';
}
std::vector<Vec3> reference(int phase) {
    std::vector<Vec3> result;
    for (int i=0;i<9;++i) result.push_back(Vec3(0.2*i+0.15*std::sin(i+phase),std::cos(i*1.1)+0.02*phase*i,std::sin(i*0.7)-0.03*phase*i));
    return result;
}
std::vector<int> selection(int mode, int phase) {
    return mode==0 ? std::vector<int>{} : (phase==0 ? std::vector<int>{0,2,4,6,8} : std::vector<int>{1,2,3,5,8});
}
void run(int type, int subset) {
    System system;
    for (int i=0;i<9;++i) system.addParticle(i==8?0:1+0.2*i);
    Force* force;
    if (type==0) force=new RMSDForce(reference(0),selection(subset,0));
    else if (type==1) force=new OrientationRestraintForce(1.7,reference(0),selection(subset,0));
    else if (type==2) force=new RGForce(selection(subset,0));
    else {
        auto* volume=new CustomVolumeForce("g*(v-7)^2+0.2*ax*bx+0.3*by*cx+0.4*cy*cz");
        volume->addGlobalParameter("g",0.7); volume->addGlobalParameter("v",999.0); force=volume;
    }
    force->setForceGroup(12); system.addForce(force);
    VerletIntegrator integrator(0.0002);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    for (int phase=0;phase<3;++phase) {
        if (phase>0) {
            if (type==0) {
                auto* f=dynamic_cast<RMSDForce*>(force); f->setReferencePositions(reference(phase)); f->setParticles(selection(subset,phase)); f->updateParametersInContext(context);
            }
            if (type==1) {
                auto* f=dynamic_cast<OrientationRestraintForce*>(force); f->setReferencePositions(reference(phase)); f->setParticles(selection(subset,phase)); f->setK(1.7+0.4*phase); f->updateParametersInContext(context);
            }
            if (type==3) context.setParameter("g",0.7+0.2*phase);
        }
        context.setPeriodicBoxVectors(Vec3(2+0.1*phase,0,0),Vec3(0.3,2.2+0.1*phase,0),Vec3(-0.2,0.4,2.4+0.1*phase));
        for (int point=0;point<7;++point) {
            auto positions=reference(0); std::vector<Vec3> velocities;
            double theta=0.31+0.47*point;
            for (int i=0;i<9;++i) {
                auto p=positions[i];
                positions[i]=Vec3(std::cos(theta)*p[0]-std::sin(theta)*p[1]+0.5*point,std::sin(theta)*p[0]+std::cos(theta)*p[1]-0.3*point,(point==6?-1:1)*p[2]+0.04*std::sin(i+point));
                velocities.push_back(Vec3(0.01*i,-0.02,0.03-0.004*i));
            }
            context.setPositions(positions); context.setVelocities(velocities); context.setTime(0);
            for (int steps:{0,4}) {
                if (steps) integrator.step(steps);
                auto state=context.getState(State::Energy|State::Forces|State::Positions|State::Velocities,false,1<<12);
                std::cout << "{\"type\":" << type << ",\"subset\":" << subset << ",\"phase\":" << phase << ",\"point\":" << point << ",\"steps\":" << steps << ",\"initial_positions\":";
                vectors(positions); std::cout << ",\"initial_velocities\":"; vectors(velocities);
                std::cout << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy() << ",\"forces\":"; vectors(state.getForces());
                std::cout << ",\"positions\":"; vectors(state.getPositions()); std::cout << ",\"velocities\":"; vectors(state.getVelocities()); std::cout << "}\n";
            }
        }
    }
}
void collapsed() {
    System system; system.addParticle(1); system.addParticle(1); system.addForce(new RGForce());
    VerletIntegrator integrator(0.001); Context context(system,integrator,Platform::getPlatformByName("Reference"));
    context.setPositions({Vec3(1,1,1),Vec3(1,1,1)});
    auto state=context.getState(State::Forces|State::Energy);
    bool finite=true; for (const auto& f:state.getForces()) for (int axis=0;axis<3;++axis) finite &= std::isfinite(f[axis]);
    std::cout << "{\"record\":\"collapsed_rg\",\"energy\":" << state.getPotentialEnergy() << ",\"forces_finite\":" << (finite?"true":"false") << "}\n";
}
void rigid() {
    System system;
    for (int i=0;i<9;++i) system.addParticle(1);
    system.addForce(new RMSDForce(reference(0)));
    VerletIntegrator integrator(0.001); Context context(system,integrator,Platform::getPlatformByName("Reference"));
    for (int point=0;point<12;++point) {
        auto positions=reference(0);
        double angle=0.2*point;
        for (auto& p:positions) p=Vec3(std::cos(angle)*p[0]-std::sin(angle)*p[1]+point,std::sin(angle)*p[0]+std::cos(angle)*p[1]-0.3*point,p[2]+0.4*point);
        context.setPositions(positions);
        auto state=context.getState(State::Forces|State::Energy);
        double maximum=0;
        for (const auto& f:state.getForces()) for (int axis=0;axis<3;++axis) maximum=std::max(maximum,std::abs(f[axis]));
        std::cout << "{\"record\":\"rigid_rmsd\",\"energy\":" << state.getPotentialEnergy() << ",\"force_max\":" << maximum << ",\"positions\":";
        vectors(positions); std::cout << "}\n";
    }
}
int main() {
    std::cout << std::setprecision(17); collapsed(); rigid();
    for (int type=0;type<4;++type) for (int subset=0;subset<2;++subset) run(type,subset);
}
