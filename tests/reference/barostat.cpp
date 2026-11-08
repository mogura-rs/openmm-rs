#include <openmm/Context.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/CustomVolumeForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/MonteCarloBarostat.h>
#include <openmm/NonbondedForce.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/VirtualSite.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace OpenMM;

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (size_t i=0;i<values.size();++i) {
        if (i) std::cout << ',';
        std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']';
    }
    std::cout << ']';
}

void pressure(bool rigid, int variant, int mask) {
    System system;
    for (double mass : {1.0,3.0,2.0,0.0,0.0,4.0}) system.addParticle(mass);
    system.setDefaultPeriodicBoxVectors(Vec3(3,0,0),Vec3(0.4,3.3,0),Vec3(-0.2,0.3,3.5));
    auto* bond = new HarmonicBondForce();
    bond->addBond(0,1,0.55,17);
    bond->addBond(2,5,0.7,23);
    bond->setUsesPeriodicBoundaryConditions(true);
    bond->setForceGroup(1);
    system.addForce(bond);
    auto* external = new CustomExternalForce("k*(x*x+0.7*y*y+0.3*z*z)");
    external->addGlobalParameter("k",0.4);
    for (int i=0;i<6;++i) external->addParticle(i);
    external->setForceGroup(2);
    system.addForce(external);
    auto* volume = new CustomVolumeForce("0.03*v+0.07*ax*by");
    volume->setForceGroup(5);
    system.addForce(volume);
    if (variant>=1) system.addConstraint(0,1,0.7);
    if (variant==2) {
        system.setVirtualSite(4,new TwoParticleAverageSite(0,1,0.3,0.7));
        auto* nb=new NonbondedForce();
        nb->setNonbondedMethod(NonbondedForce::PME);
        nb->setCutoffDistance(0.9);
        nb->setPMEParameters(3.2,16,18,20);
        for (int i=0;i<6;++i) nb->addParticle((i%2 ? -1 : 1)*0.08,0.2,0.01);
        nb->addException(0,1,0,0.2,0);
        nb->setForceGroup(3);
        nb->setReciprocalSpaceForceGroup(4);
        system.addForce(nb);
    }
    auto* barostat=new MonteCarloBarostat(1.3,300,0,rigid);
    barostat->setRandomNumberSeed(17);
    barostat->setForceGroup(31);
    system.addForce(barostat);
    VerletIntegrator integrator(0.00002);
    integrator.setConstraintTolerance(1e-10);
    integrator.setIntegrationForceGroups(mask);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    for (int phase=0;phase<3;++phase) {
        double s=1+phase*0.1;
        context.setPeriodicBoxVectors(Vec3(3,0,0)*s,Vec3(0.4,3.3,0)*s,Vec3(-0.2,0.3,3.5)*s);
        context.setPositions({{0.2,0.3,0.4},{0.9,0.3,0.4},{1.3,1.1,0.7},{0.5,1.7,1.3},{1.4,0.8,1.5},{1.9,1.2,0.8}});
        context.setVelocities({{0.3,-0.2,0.1},{-0.1,0.4,0.2},{0.2,0.1,-0.3},{0.7,0.8,0.9},{-0.3,0.6,0.5},{0.1,-0.3,0.4}});
        context.computeVirtualSites();
        context.applyVelocityConstraints(1e-10);
        context.setTime(4.7);
        context.setStepCount(19);
        context.setParameter("k",0.4+phase*0.1);
        context.setParameter(MonteCarloBarostat::Pressure(),2+phase);
        context.setParameter(MonteCarloBarostat::Temperature(),260+10*phase);
        barostat->setDefaultPressure(11+phase);
        barostat->setDefaultTemperature(410+phase);
        int done=0;
        for (int steps : {0,1,4}) {
            integrator.step(steps-done);done=steps;
            auto before=context.getState(State::Positions|State::Velocities|State::Forces|State::Energy);
            double value=barostat->computeCurrentPressure(context);
            auto after=context.getState(State::Positions|State::Velocities);
            if (before.getPositions()!=after.getPositions() || before.getVelocities()!=after.getVelocities() || before.getTime()!=after.getTime())
                throw std::runtime_error("pressure changed state");
            std::cout << "{\"case\":\"pressure\",\"rigid\":" << (rigid?"true":"false")
                << ",\"variant\":" << variant << ",\"mask\":" << mask << ",\"phase\":" << phase
                << ",\"steps\":" << steps << ",\"pressure\":" << value
                << ",\"energy\":" << before.getPotentialEnergy() << ",\"kinetic\":" << before.getKineticEnergy()
                << ",\"positions\":"; vectors(before.getPositions());
            std::cout << ",\"velocities\":"; vectors(before.getVelocities());
            std::cout << ",\"forces\":"; vectors(before.getForces());
            std::cout << "}\n";
        }
    }
}

void schedule(bool rigid) {
    System system;
    system.addParticle(1);system.addParticle(3);
    auto* bond=new HarmonicBondForce();bond->addBond(0,1,0.7,0);
    bond->setUsesPeriodicBoundaryConditions(true);system.addForce(bond);
    auto* barostat=new MonteCarloBarostat(0,0,0,rigid);
    barostat->setRandomNumberSeed(73);system.addForce(barostat);
    VerletIntegrator integrator(0.001);integrator.setIntegrationForceGroups(0);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    context.setPositions({{0.2,0.3,0.4},{0.9,0.3,0.4}});
    for (int step=0;step<30;++step) {
        if (step==5) barostat->setFrequency(4);
        if (step==13) barostat->setFrequency(0);
        if (step==19) barostat->setFrequency(3);
        context.setStepCount(step%3);
        auto before=context.getState(State::Positions);
        Vec3 a,b,c;context.getState(0).getPeriodicBoxVectors(a,b,c);double old=a[0];
        integrator.step(1);
        context.getState(0).getPeriodicBoxVectors(a,b,c);double scale=a[0]/old;
        auto after=context.getState(State::Positions);
        Vec3 displacement=after.getPositions()[1]-after.getPositions()[0];
        double distance=std::sqrt(displacement.dot(displacement));
        displacement=before.getPositions()[1]-before.getPositions()[0];
        double previous=std::sqrt(displacement.dot(displacement));
        if (std::abs(distance/previous-(rigid?1.0:scale))>1e-13) throw std::runtime_error("incorrect coordinate scaling");
        std::cout << "{\"case\":\"schedule\",\"rigid\":" << (rigid?"true":"false")
                  << ",\"step\":" << step << ",\"changed\":" << (scale!=1?"true":"false") << "}\n";
    }
}

void idealGas(bool rigid) {
    const int count=128, blocks=40, samples=250, stride=20;
    const double kt=1.380649e-23*6.02214076e23/1000*300;
    const double pressure=20*0.0602214076+0.2;
    const int scaled=rigid?count/2:count;
    double length=std::cbrt((scaled+1)*kt/pressure);
    System system;
    std::vector<Vec3> positions;
    for (int i=0;i<count;++i) {system.addParticle(1);positions.push_back(Vec3(0.1+0.02*i,0.2,0.3));}
    system.setDefaultPeriodicBoxVectors(Vec3(length,0,0),Vec3(0,length,0),Vec3(0,0,length));
    auto* bond=new HarmonicBondForce();
    for (int i=0;i<count;i+=2) bond->addBond(i,i+1,0.02,0);
    bond->setUsesPeriodicBoundaryConditions(true);system.addForce(bond);
    auto* bias=new CustomVolumeForce("q*v");bias->addGlobalParameter("q",0.2);system.addForce(bias);
    auto* barostat=new MonteCarloBarostat(20,300,1,rigid);barostat->setRandomNumberSeed(91);system.addForce(barostat);
    VerletIntegrator integrator(0.001);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));context.setPositions(positions);
    integrator.step(10000);
    std::cout << "{\"case\":\"ideal\",\"rigid\":" << (rigid?"true":"false") << ",\"blocks\":[";
    for (int block=0;block<blocks;++block) {
        double sum=0,sum2=0;
        for (int j=0;j<samples;++j) {
            integrator.step(stride);
            Vec3 a,b,c;context.getState(0).getPeriodicBoxVectors(a,b,c);double v=a[0]*b[1]*c[2];sum+=v;sum2+=v*v;
        }
        if (block) std::cout << ',';
        std::cout << '[' << sum/samples << ',' << sum2/samples << ']';
    }
    std::cout << "]}\n";
}

int main() {
    std::cout << std::setprecision(17);
    for (bool rigid:{false,true}) {
        for (int variant=0;variant<3;++variant) for (int mask:{-1,2,8,16,32,0}) pressure(rigid,variant,mask);
        schedule(rigid);idealGas(rigid);
    }
}
