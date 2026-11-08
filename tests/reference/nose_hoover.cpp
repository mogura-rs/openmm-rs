#include <openmm/Context.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/NoseHooverIntegrator.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VirtualSite.h>
#include <iomanip>
#include <cmath>
#include <iostream>
using namespace OpenMM;

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for(size_t i=0;i<values.size();++i) {
        if(i)std::cout << ',';
        std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']';
    }
    std::cout << ']';
}
void run(int length,int mts,int ys,int mode,bool constrained,bool virtualSite) {
    System system;
    for(double mass : {1.,2.,3.,4.,0.,5.})system.addParticle(mass);
    if(constrained)system.addConstraint(2,3,1);
    if(virtualSite)system.setVirtualSite(4,new TwoParticleAverageSite(0,2,0.3,0.7));
    auto* bond=new HarmonicBondForce();bond->addBond(0,1,0.8,2);bond->setForceGroup(1);system.addForce(bond);
    auto* force=new CustomExternalForce("0.4*x^2+0.3*y^2+0.2*z^2");
    for(int i=0;i<5;++i)force->addParticle(i,{});
    force->setForceGroup(2);system.addForce(force);
    NoseHooverIntegrator integrator(0.001);
    if(mode==0)integrator.addThermostat(300,1.7,length,mts,ys);
    if(mode==1) {
        integrator.addSubsystemThermostat({0,1},{},300,1.7,20,2.3,length,mts,ys);
        integrator.addSubsystemThermostat({2,3},{},140,0.9,20,2.3,length,mts,ys);
    }
    if(mode==2)integrator.addSubsystemThermostat({2,3},{{0,1}},300,1.7,20,2.3,length,mts,ys);
    // A stationary, force-free atom enables the missing absolute propagation branch.
    // Removing its three DOFs makes this an oracle for the approved pair-only fix.
    if(mode==3)integrator.addSubsystemThermostat({5},{{0,1}},300,1.7,20,2.3,length,mts,ys);
    integrator.setConstraintTolerance(1e-10);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    if(mode==3)const_cast<NoseHooverChain&>(integrator.getThermostat(0)).setNumDegreesOfFreedom(3);
    context.setPositions({Vec3(0,0,0),Vec3(0.9,0,0),Vec3(0,0.7,1.1),Vec3(0,1.7,1.1),Vec3(0.2,0.4,0.5),Vec3(0,0,0)});
    context.computeVirtualSites();
    context.setVelocities({Vec3(0.2,0.1,-0.1),Vec3(-0.3,0.4,0.2),Vec3(0.1,-0.2,0.3),Vec3(0.2,0.3,-0.1),Vec3(0.7,0.8,0.9),mode==3?Vec3():Vec3(0.6,0.7,0.8)});
    context.applyVelocityConstraints(1e-10);context.setTime(2.7);context.setStepCount(17);
    integrator.step(1);
    for(int event=0;event<7;++event) {
        if(event==1) {integrator.setTemperature(170);integrator.setCollisionFrequency(2.1);integrator.setRelativeTemperature(30);integrator.setRelativeCollisionFrequency(1.9);}
        if(event==2)integrator.setStepSize(0.0007);
        if(event==3)integrator.setIntegrationForceGroups(2);
        if(event==4) {
            auto& chain=const_cast<NoseHooverChain&>(integrator.getThermostat(0));
            chain.setTemperature(230);chain.setRelativeTemperature(15);chain.setCollisionFrequency(1.3);chain.setRelativeCollisionFrequency(1.1);
            chain.setNumDegreesOfFreedom(chain.getNumDegreesOfFreedom()+1);
        }
        if(event==5)integrator.setMaximumPairDistance(0.85);
        if(event==6) {context.setVelocitiesToTemperature(0,71);integrator.setIntegrationForceGroups(-1);}
        for(int sample=0;sample<2;++sample) {
            if(sample)integrator.step(3);
            auto state=context.getState(State::Positions|State::Velocities|State::Forces|State::Energy);
            double kinetic=0;
            for(int a=0;a<system.getNumParticles();++a)kinetic+=0.5*system.getParticleMass(a)*state.getVelocities()[a].dot(state.getVelocities()[a]);
            std::cout << "{\"length\":" << length << ",\"mts\":" << mts << ",\"ys\":" << ys << ",\"mode\":" << mode
                << ",\"constrained\":" << constrained << ",\"virtual_site\":" << virtualSite << ",\"event\":" << event << ",\"sample\":" << sample
                << ",\"time\":" << state.getTime() << ",\"step_count\":" << state.getStepCount()
                << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << kinetic << ",\"source_kinetic\":" << state.getKineticEnergy()
                << ",\"bath\":" << integrator.computeHeatBathEnergy() << ",\"dof\":[";
            for(int i=0;i<integrator.getNumThermostats();++i) {if(i)std::cout << ',';std::cout << integrator.getThermostat(i).getNumDegreesOfFreedom();}
            std::cout << "],\"positions\":";vectors(state.getPositions());std::cout << ",\"velocities\":";vectors(state.getVelocities());
            std::cout << ",\"forces\":";vectors(state.getForces());std::cout << "}\n";
        }
    }
}
void rawPairProbe() {
    System s;s.addParticle(1);s.addParticle(1);NoseHooverIntegrator i(0.001);
    i.addSubsystemThermostat({},{{0,1}},300,1,300,1,1,1,1);
    Context c(s,i,Platform::getPlatformByName("Reference"));
    c.setPositions({Vec3(),Vec3(1,0,0)});c.setVelocities({Vec3(2,0,0),Vec3()});
    double initial=c.getState(State::Energy).getKineticEnergy();i.step(1);
    auto state=c.getState(State::Energy|State::Velocities);
    std::cout << "{\"probe\":\"raw_pair_only\",\"initial_reported\":" << initial << ",\"final_reported\":" << state.getKineticEnergy()
        << ",\"velocities\":";vectors(state.getVelocities());std::cout << "}\n";
}

void zeroMotionProbes() {
    for(bool wall : {false,true}) {
        System s;s.addParticle(1);s.addParticle(1);NoseHooverIntegrator i(0.001);
        i.addSubsystemThermostat({},{{0,1}},300,1,300,1,1,1,1);
        if(wall)i.setMaximumPairDistance(0.8);
        Context c(s,i,Platform::getPlatformByName("Reference"));
        c.setPositions({Vec3(),Vec3(1,0,0)});
        c.setVelocities(wall ? std::vector<Vec3>{Vec3(),Vec3()} : std::vector<Vec3>{Vec3(1,0,0),Vec3(-1,0,0)});
        i.step(1);auto state=c.getState(State::Positions|State::Velocities);
        bool finite=true;
        for(const auto& v : state.getVelocities())for(int k=0;k<3;++k)finite=finite&&std::isfinite(v[k]);
        std::cout<<"{\"probe\":\""<<(wall?"zero_radial_wall":"relative_only")<<"\",\"finite\":"<<(finite?"true":"false");
        if(!wall){std::cout<<",\"velocities\":";vectors(state.getVelocities());}
        std::cout<<"}\n";
    }
}

void hardWallCases(double parentMass) {
    System s;s.addParticle(1);s.addParticle(parentMass);s.addParticle(3);
    auto* force=new CustomExternalForce("-0.1*x");
    for(int a=0;a<3;++a)force->addParticle(a,{});s.addForce(force);
    NoseHooverIntegrator i(0.001);
    i.addSubsystemThermostat({2},{{0,1}},300,1,30,2,3,3,7);i.setMaximumPairDistance(0.8);
    Context c(s,i,Platform::getPlatformByName("Reference"));
    c.setPositions({Vec3(),Vec3(1,0,0),Vec3(0,1,0)});
    c.setVelocities({Vec3(-2,0.2,0.1),Vec3(0.3,0.1,0.4),Vec3(0.4,0.5,0.3)});
    i.setRelativeTemperature(200);
    for(int step=0;step<2;++step) {
        if(step)i.setMaximumPairDistance(0.75);
        i.step(1);auto state=c.getState(State::Positions|State::Velocities|State::Energy);
        double kinetic=0;
        for(int a=0;a<3;++a)kinetic+=0.5*s.getParticleMass(a)*state.getVelocities()[a].dot(state.getVelocities()[a]);
        std::cout<<"{\"hard_wall\":true,\"parent_mass\":"<<parentMass<<",\"step\":"<<step
                 <<",\"kinetic\":"<<kinetic<<",\"bath\":"<<i.computeHeatBathEnergy()<<",\"positions\":";
        vectors(state.getPositions());std::cout<<",\"velocities\":";vectors(state.getVelocities());std::cout<<"}\n";
    }
}
void stability(double temperature,double frequency) {
    System system;
    const int n=24, sampleCount=10000, stride=10;
    std::vector<Vec3> positions,velocities;
    auto* force=new CustomExternalForce("5*x^2+7.5*y^2+10*z^2");
    for(int a=0;a<n;++a) {
        system.addParticle(1+0.1*a);force->addParticle(a,{});
        positions.emplace_back(0.1*std::sin(a+0.3),0.2*std::cos(a+0.1),0.15*std::sin(0.7*a));
        velocities.emplace_back(std::cos(a+0.3),std::sin(a+0.1),std::cos(0.7*a));
    }
    system.addForce(force);
    NoseHooverIntegrator integrator(temperature,frequency,0.001,3,3,7);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    context.setPositions(positions);context.setVelocities(velocities);integrator.step(20000);
    auto state=context.getState(State::Energy);
    double initial=state.getKineticEnergy()+state.getPotentialEnergy()+integrator.computeHeatBathEnergy();
    double mean=0,m2=0, maxDrift=0, finalEnergy=0, meanPotential=0;
    for(int j=1;j<=sampleCount;++j) {
        integrator.step(stride);state=context.getState(State::Energy);
        double kinetic=state.getKineticEnergy(),difference=kinetic-mean;
        mean+=difference/j;m2+=difference*(kinetic-mean);
        meanPotential+=(state.getPotentialEnergy()-meanPotential)/j;
        finalEnergy=kinetic+state.getPotentialEnergy()+integrator.computeHeatBathEnergy();
        maxDrift=std::max(maxDrift,std::abs(finalEnergy-initial));
    }
    std::cout<<"{\"stability\":true,\"temperature\":"<<temperature<<",\"frequency\":"<<frequency
       <<",\"samples\":"<<sampleCount<<",\"mean_kinetic\":"<<mean<<",\"variance_kinetic\":"<<m2/sampleCount
       <<",\"mean_potential\":"<<meanPotential<<",\"max_extended_energy_drift\":"<<maxDrift
       <<",\"final_extended_energy\":"<<finalEnergy<<",\"initial_extended_energy\":"<<initial<<"}\n";
}

int main() {
    std::cout << std::setprecision(17);
    for(int length : {1,3})for(int mts : {1,3})for(int ys : {1,3,5,7})for(int mode=0;mode<4;++mode)
        for(bool constrained : {false,true})for(bool virtualSite : {false,true})run(length,mts,ys,mode,constrained,virtualSite);
    rawPairProbe();zeroMotionProbes();
    hardWallCases(0);hardWallCases(2);
    for(double temperature : {70.,300.,600.})for(double frequency : {1.,10.})stability(temperature,frequency);
}
