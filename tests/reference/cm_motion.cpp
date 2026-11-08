#include <openmm/BrownianIntegrator.h>
#include <openmm/CMMotionRemover.h>
#include <openmm/Context.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/LangevinMiddleIntegrator.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
using namespace OpenMM;

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (size_t i=0; i<values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']';
    }
    std::cout << ']';
}

void run(int kind, int frequency, int start, bool constrained) {
    System system;
    for (double mass : {1.0,2.0,3.0,4.0,0.0}) system.addParticle(mass);
    if (constrained) {
        system.addConstraint(0,1,1.0);
        system.addConstraint(2,3,1.0);
    }
    auto* external=new CustomExternalForce("-k*x+0.1*y^2");
    external->addPerParticleParameter("k");
    for (int i=0;i<5;++i) external->addParticle(i,{0.2*(i+1)});
    external->setForceGroup(2);
    system.addForce(external);
    auto* remover=new CMMotionRemover(frequency);
    remover->setForceGroup(31);
    system.addForce(remover);
    std::unique_ptr<Integrator> integrator;
    if (kind==0) integrator=std::make_unique<VerletIntegrator>(0.0003);
    else if (kind==1) integrator=std::make_unique<LangevinMiddleIntegrator>(0,0.7,0.0003);
    else integrator=std::make_unique<BrownianIntegrator>(0,1.3,0.0003);
    integrator->setConstraintTolerance(1e-9);
    integrator->setIntegrationForceGroups(1<<2);
    Context context(system,*integrator,Platform::getPlatformByName("Reference"));
    std::vector<Vec3> initial={{0,0,0},{1,0,0},{0,1,0},{0,2,0},{0.5,0.3,0.9}};
    std::vector<Vec3> velocity={{0.3,-0.2,0.1},{-0.1,0.4,0.2},{0.2,0.1,-0.3},{0.1,-0.3,0.4},{0.7,0.8,0.9}};
    for (int phase=0;phase<2;++phase) {
        if (phase) remover->setFrequency(frequency+11);
        context.setPositions(initial);
        context.setVelocities(velocity);
        context.applyVelocityConstraints(1e-9);
        context.setStepCount(start);
        context.setTime(9.7);
        int completed=0;
        for (int steps : {0,1,4,10}) {
            integrator->step(steps-completed);
            completed=steps;
            auto state=context.getState(State::Energy|State::Forces|State::Positions|State::Velocities);
            auto excluded=context.getState(State::Energy|State::Forces,false,1U<<31);
            if (excluded.getPotentialEnergy()!=0) throw std::runtime_error("state updater has nonzero energy");
            for (auto f:excluded.getForces()) if (f.dot(f)!=0) throw std::runtime_error("state updater has nonzero forces");
            std::cout << "{\"kind\":" << kind << ",\"frequency\":" << frequency
                      << ",\"start\":" << start << ",\"constrained\":" << (constrained?"true":"false")
                      << ",\"phase\":" << phase << ",\"steps\":" << steps
                      << ",\"time\":" << state.getTime() << ",\"step_count\":" << state.getStepCount()
                      << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy()
                      << ",\"forces\":";
            vectors(state.getForces()); std::cout << ",\"positions\":"; vectors(state.getPositions());
            std::cout << ",\"velocities\":"; vectors(state.getVelocities()); std::cout << "}\n";
        }
    }
}

int main() {
    std::cout << std::setprecision(17);
    for (int kind=0;kind<3;++kind) for (int frequency : {1,3,7})
        for (int start : {0,1,2,17}) for (bool constrained : {false,true})
            run(kind,frequency,start,constrained);
}
