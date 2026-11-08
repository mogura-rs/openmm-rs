#include <openmm/BrownianIntegrator.h>
#include <openmm/Context.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/LangevinMiddleIntegrator.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
using namespace OpenMM;

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for(size_t i=0;i<values.size();++i) {
        if(i)std::cout << ',';
        std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']';
    }
    std::cout << ']';
}

void run(int kind, bool constrained) {
    System system;
    for(double mass : {1.,2.,3.,0.}) system.addParticle(mass);
    if(constrained)system.addConstraint(0,1,1);
    auto* bond=new HarmonicBondForce();
    bond->addBond(0,2,0.8,2);
    bond->setForceGroup(1);
    system.addForce(bond);
    auto* external=new CustomExternalForce("0.4*x+0.2*y^2");
    for(int i=0;i<4;++i)external->addParticle(i,{});
    external->setForceGroup(4);
    system.addForce(external);
    std::unique_ptr<Integrator> integrator;
    if(kind==0)integrator=std::make_unique<VerletIntegrator>(0.0002);
    else if(kind==1)integrator=std::make_unique<LangevinMiddleIntegrator>(0,0.7,0.0002);
    else integrator=std::make_unique<BrownianIntegrator>(0,0.7,0.0002);
    Context context(system,*integrator,Platform::getPlatformByName("Reference"));
    context.setPositions({Vec3(0,0,0),Vec3(1,0,0),Vec3(0,0.7,1.1),Vec3(0.2,0.4,0.5)});
    context.setVelocities({Vec3(0.2,0.1,-0.1),Vec3(-0.3,0.4,0.2),Vec3(0.1,-0.2,0.3),Vec3(0.7,0.8,0.9)});
    context.applyVelocityConstraints(1e-10);
    context.setTime(2.7);
    context.setStepCount(17);
    double dt[]={0.0002,0.00035,0.0001,0.0005};
    int masks[]={-1,2,0,18};
    double friction[]={0.7,1.4,2.1,0.3};
    double tolerance[]={1e-7,1e-8,1e-9,1e-10};
    for(int phase=0;phase<4;++phase) {
        context.getIntegrator().setStepSize(dt[phase]);
        integrator->setIntegrationForceGroups(masks[phase]);
        integrator->setConstraintTolerance(tolerance[phase]);
        if(kind==1)dynamic_cast<LangevinMiddleIntegrator&>(*integrator).setFriction(friction[phase]);
        if(kind==2)dynamic_cast<BrownianIntegrator&>(*integrator).setFriction(friction[phase]);
        int completed=0;
        for(int steps : {0,1,4}) {
            integrator->step(steps-completed);
            completed=steps;
            auto state=context.getState(State::Energy|State::Forces|State::Positions|State::Velocities);
            std::cout << "{\"kind\":" << kind << ",\"constrained\":" << (constrained?"true":"false")
                      << ",\"phase\":" << phase << ",\"steps\":" << steps
                      << ",\"time\":" << state.getTime() << ",\"step_count\":" << state.getStepCount()
                      << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy()
                      << ",\"forces\":";
            vectors(state.getForces());std::cout << ",\"positions\":";vectors(state.getPositions());
            std::cout << ",\"velocities\":";vectors(state.getVelocities());std::cout << "}\n";
        }
    }
}

void statistics(int kind) {
    const int n=12000;
    const double kb=1.380649e-23*6.02214076e23/1000;
    System system;
    for(int i=0;i<n;++i)system.addParticle(1+i%5);
    std::unique_ptr<Integrator> integrator;
    if(kind==1)integrator=std::make_unique<LangevinMiddleIntegrator>(1,1e6,0.001);
    else integrator=std::make_unique<BrownianIntegrator>(1,0.7,0.001);
    Context context(system,*integrator,Platform::getPlatformByName("Reference"));
    int phase=0;
    for(double temperature : {100.,500.,200.}) {
        if(kind==1)dynamic_cast<LangevinMiddleIntegrator&>(*integrator).setTemperature(temperature);
        else dynamic_cast<BrownianIntegrator&>(*integrator).setTemperature(temperature);
        context.setPositions(std::vector<Vec3>(n));
        context.setVelocities(std::vector<Vec3>(n));
        integrator->step(1);
        auto velocity=context.getState(State::Velocities).getVelocities();
        double first=0,second=0,fourth=0;
        for(int i=0;i<n;++i)for(int axis=0;axis<3;++axis) {
            double variance=kb*temperature/(1+i%5);
            if(kind==2)variance*=2/(0.7*0.001);
            double z=velocity[i][axis]/std::sqrt(variance);
            first+=z;second+=z*z;fourth+=z*z*z*z;
        }
        std::cout << "{\"statistic\":true,\"kind\":" << kind << ",\"phase\":" << phase++
                  << ",\"samples\":" << 3*n << ",\"temperature\":" << temperature
                  << ",\"moments\":[" << first/(3*n) << ',' << second/(3*n) << ',' << fourth/(3*n) << "]}\n";
    }
}

int main() {
    std::cout << std::setprecision(17);
    for(int kind=0;kind<3;++kind)for(bool constrained : {false,true})run(kind,constrained);
    statistics(1);statistics(2);
}
