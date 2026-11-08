#include <openmm/Context.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/CustomVolumeForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/MonteCarloFlexibleBarostat.h>
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

void scalars(const std::vector<double>& values) {
    std::cout << '[';
    for (int i=0;i<values.size();++i) {if(i) std::cout << ',';std::cout << values[i];}
    std::cout << ']';
}
Vec3 basisB(int cell) {return Vec3(cell==0?0:cell==1?0.4:1.4999,3.3,0);}
Vec3 basisC(int cell) {return Vec3(cell==0?0:cell==1?-0.2:-1.4998,cell==0?0:cell==1?0.3:1.6499,3.5);}

void pressure(bool rigid, int variant, int mask, int cell) {
    System system;
    for (double mass : {1.0,3.0,2.0,0.0,0.0,4.0}) system.addParticle(mass);
    system.setDefaultPeriodicBoxVectors(Vec3(3,0,0),basisB(cell),basisC(cell));
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
    auto* volume = new CustomVolumeForce("0.03*v+0.07*ax*by+0.02*bx-0.03*cx+0.04*cy");
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
    auto* barostat=new MonteCarloFlexibleBarostat(1.3,300,0,rigid);
    barostat->setRandomNumberSeed(17);
    barostat->setForceGroup(31);
    system.addForce(barostat);
    VerletIntegrator integrator(0.00002);
    integrator.setConstraintTolerance(1e-10);
    integrator.setIntegrationForceGroups(mask);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    for (int phase=0;phase<3;++phase) {
        double s=1+phase*0.1;
        context.setPeriodicBoxVectors(Vec3(3,0,0)*s,basisB(cell)*s,basisC(cell)*s);
        context.setPositions({{0.2,0.3,0.4},{0.9,0.3,0.4},{1.3,1.1,0.7},{0.5,1.7,1.3},{1.4,0.8,1.5},{1.9,1.2,0.8}});
        context.setVelocities({{0.3,-0.2,0.1},{-0.1,0.4,0.2},{0.2,0.1,-0.3},{0.7,0.8,0.9},{-0.3,0.6,0.5},{0.1,-0.3,0.4}});
        context.computeVirtualSites();
        context.applyVelocityConstraints(1e-10);
        context.setTime(4.7);
        context.setStepCount(19);
        context.setParameter("k",0.4+phase*0.1);
        context.setParameter(MonteCarloFlexibleBarostat::Pressure(),2+phase);
        context.setParameter(MonteCarloFlexibleBarostat::Temperature(),260+10*phase);
        barostat->setDefaultPressure(11+phase);
        barostat->setDefaultTemperature(410+phase);
        int done=0;
        for (int steps : {0,1,4}) {
            integrator.step(steps-done);done=steps;
            auto before=context.getState(State::Positions|State::Velocities|State::Forces|State::Energy);
            std::vector<double> pressure,stress,total;
            barostat->computeCurrentPressure(context,pressure);
            barostat->computeStressTensor(context,stress,false);
            barostat->computeStressTensor(context,total,true);
            auto after=context.getState(State::Positions|State::Velocities);
            if (before.getPositions()!=after.getPositions() || before.getVelocities()!=after.getVelocities() || before.getTime()!=after.getTime())
                throw std::runtime_error("pressure changed state");
            std::cout << "{\"case\":\"pressure\",\"rigid\":" << (rigid?"true":"false")
                << ",\"cell\":" << cell << ",\"variant\":" << variant << ",\"mask\":" << mask << ",\"phase\":" << phase
                << ",\"steps\":" << steps << ",\"pressure\":";scalars(pressure);
            std::cout << ",\"stress\":";scalars(stress);
            std::cout << ",\"total_stress\":";scalars(total);
            std::cout
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
    auto* barostat=new MonteCarloFlexibleBarostat(0,0,0,rigid);
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
        Vec3 old[3],next[3];context.getState(0).getPeriodicBoxVectors(old[0],old[1],old[2]);
        integrator.step(1);
        context.getState(0).getPeriodicBoxVectors(next[0],next[1],next[2]);
        bool changed=false;
        for (int axis=0;axis<3;++axis) changed|=old[axis][axis]!=next[axis][axis];
        std::cout << "{\"case\":\"schedule\",\"rigid\":" << (rigid?"true":"false")
                  << ",\"step\":" << step << ",\"changed\":" << (changed?"true":"false") << "}\n";
    }
}

void moments(bool rigid) {
    const int count=128,blocks=40,samples=250,stride=20;
    const double kt=1.380649e-23*6.02214076e23/1000*300;
    const int scaled=rigid?count/2:count;
    System system;
    std::vector<Vec3> positions;
    for(int i=0;i<count;++i) {system.addParticle(1);positions.push_back(Vec3(0.1+0.02*i,0.2,0.3));}
    system.setDefaultPeriodicBoxVectors(Vec3((scaled+1)*kt/50,0,0),Vec3(0,scaled*kt/60,0),Vec3(0,0,(scaled-1)*kt/70));
    auto* bond=new HarmonicBondForce();for(int i=0;i<count;i+=2) bond->addBond(i,i+1,0.02,0);
    bond->setUsesPeriodicBoundaryConditions(true);system.addForce(bond);
    auto* potential=new CustomVolumeForce("-pc*v+50*ax+60*by+70*cz+125*(bx^2+cx^2+cy^2)");
    potential->addGlobalParameter("pc",20*0.0602214076);system.addForce(potential);
    auto* barostat=new MonteCarloFlexibleBarostat(5,150,1,rigid);barostat->setRandomNumberSeed(91);system.addForce(barostat);
    VerletIntegrator integrator(0.001);Context context(system,integrator,Platform::getPlatformByName("Reference"));context.setPositions(positions);
    context.setParameter(MonteCarloFlexibleBarostat::Pressure(),20);context.setParameter(MonteCarloFlexibleBarostat::Temperature(),300);
    barostat->setDefaultPressure(99);barostat->setDefaultTemperature(50);
    integrator.step(10000);
    std::cout << "{\"case\":\"moments\",\"rigid\":" << (rigid?"true":"false") << ",\"blocks\":[";
    for(int block=0;block<blocks;++block) {
        std::vector<double> sums(12,0.0);
        for(int sample=0;sample<samples;++sample) {
            integrator.step(stride);Vec3 a,b,c;context.getState(0).getPeriodicBoxVectors(a,b,c);
            const double values[6]={a[0],b[1],c[2],b[0],c[0],c[1]};
            for(int k=0;k<6;++k) {sums[2*k]+=values[k];sums[2*k+1]+=values[k]*values[k];}
        }
        for(auto& value:sums) value/=samples;
        if(block) std::cout << ',';scalars(sums);
    }
    std::cout << "]}\n";
}

int main() {
    std::cout << std::setprecision(17);
    for (bool rigid:{false,true}) {
        for (int cell=0;cell<3;++cell) for (int variant=0;variant<3;++variant) for (int mask:{-1,2,8,16,32,0}) pressure(rigid,variant,mask,cell);
        schedule(rigid);moments(rigid);
    }
}
