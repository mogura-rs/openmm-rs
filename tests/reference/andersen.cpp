#include <openmm/BrownianIntegrator.h>
#include <openmm/AndersenThermostat.h>
#include <openmm/Context.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/LangevinMiddleIntegrator.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <cmath>
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

void run(int kind, int mode, bool constrained) {
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
    auto* remover=new AndersenThermostat(mode==0 ? 300.0 : 0.0, mode==1 ? 1e20 : 0.0);
    remover->setRandomNumberSeed(123);
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
    for (int phase=0;phase<3;++phase) {
        if (phase==1) {
            remover->setDefaultTemperature(777);
            remover->setDefaultCollisionFrequency(88);
        }
        if (phase==2) {
            context.setParameter(AndersenThermostat::Temperature(),0.0);
            context.setParameter(AndersenThermostat::CollisionFrequency(),mode==1 ? 0.0 : 1e20);
        }
        context.setPositions(initial);
        context.setVelocities(velocity);
        context.applyVelocityConstraints(1e-9);
        context.setStepCount(17);
        context.setTime(9.7);
        int completed=0;
        for (int steps : {0,1,4,10}) {
            integrator->step(steps-completed);
            completed=steps;
            auto state=context.getState(State::Energy|State::Forces|State::Positions|State::Velocities);
            auto excluded=context.getState(State::Energy|State::Forces,false,1U<<31);
            if (excluded.getPotentialEnergy()!=0) throw std::runtime_error("state updater has nonzero energy");
            for (auto f:excluded.getForces()) if (f.dot(f)!=0) throw std::runtime_error("state updater has nonzero forces");
            std::cout << "{\"kind\":" << kind << ",\"mode\":" << mode
                      << ",\"temperature\":" << context.getParameter(AndersenThermostat::Temperature())
                      << ",\"frequency\":" << context.getParameter(AndersenThermostat::CollisionFrequency()) << ",\"constrained\":" << (constrained?"true":"false")
                      << ",\"phase\":" << phase << ",\"steps\":" << steps
                      << ",\"time\":" << state.getTime() << ",\"step_count\":" << state.getStepCount()
                      << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy()
                      << ",\"forces\":";
            vectors(state.getForces()); std::cout << ",\"positions\":"; vectors(state.getPositions());
            std::cout << ",\"velocities\":"; vectors(state.getVelocities()); std::cout << "}\n";
        }
    }
}

void statistics(bool constrained) {
    const int groups=12000, count=3*groups;
    const double temperature=constrained ? 0.0 : 310.0;
    System system;
    for (int i=0;i<count;++i) system.addParticle(constrained ? 1.0 : double((i%3+1)*(i%3+1)));
    if (constrained) for (int i=0;i<groups;++i) {
        system.addConstraint(i,i+groups,1.0);
        system.addConstraint(i+groups,i+2*groups,1.0);
    }
    auto* thermostat=new AndersenThermostat(temperature,constrained ? 37.0 : 1e20);
    thermostat->setRandomNumberSeed(3721);
    system.addForce(thermostat);
    VerletIntegrator integrator(0.01);
    integrator.setConstraintTolerance(1e-9);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    std::vector<Vec3> positions(count), velocities(count,Vec3(1,1,1));
    if (constrained) for (int i=0;i<count;++i) positions[i]=Vec3(i/groups,0,0);
    context.setPositions(positions);
    context.setVelocities(velocities);
    integrator.step(1);
    velocities=context.getState(State::Velocities).getVelocities();
    if (constrained) {
        int collisions=0;
        for (int i=0;i<groups;++i) {
            const bool collided=std::abs(velocities[i][0])<1e-9;
            collisions+=collided;
            for (int a : {i,i+groups,i+2*groups}) for (int axis=0;axis<3;++axis)
                if (std::abs(velocities[a][axis]-(collided ? 0.0 : 1.0))>1e-8)
                    throw std::runtime_error("inconsistent constraint-group collisions");
        }
        std::cout << "{\"statistics\":\"collision\",\"groups\":" << groups
                  << ",\"rate\":" << double(collisions)/groups << "}\n";
    } else {
        std::cout << "{\"statistics\":\"maxwell\",\"moments\":[";
        for (int mass=0;mass<3;++mass) {
            if (mass) std::cout << ',';
            double sums[3]={0,0,0};
            const double scale=std::sqrt((1.380649e-23*6.02214076e23/1000.0)*temperature/((mass+1)*(mass+1)));
            for (int i=mass;i<count;i+=3) for (int axis=0;axis<3;++axis) {
                const double value=velocities[i][axis]/scale;
                sums[0]+=value; sums[1]+=value*value; sums[2]+=value*value*value*value;
            }
            std::cout << '[' << sums[0]/count << ',' << sums[1]/count << ',' << sums[2]/count << ']';
        }
        std::cout << "]}\n";
    }
}

int main() {
    std::cout << std::setprecision(17);
    statistics(true);
    statistics(false);
    for (int kind=0;kind<3;++kind) for (int mode=0;mode<3;++mode)
        for (bool constrained : {false,true}) run(kind,mode,constrained);
}
