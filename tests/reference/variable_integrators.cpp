#include <openmm/VariableVerletIntegrator.h>
#include <openmm/VariableLangevinIntegrator.h>
#include <openmm/Context.h>
#include <openmm/System.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/VirtualSite.h>
#include <openmm/Platform.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
using namespace OpenMM;

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (size_t i=0; i<values.size(); ++i) {
        if(i) std::cout << ',';
        std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']';
    }
    std::cout << ']';
}

template<class I> void correctedStep(I& integrator, Context& context, double limit=0) {
    double time=context.getTime();
    double maximum=integrator.getMaximumStepSize();
    if (limit>0) integrator.setMaximumStepSize(maximum>0?std::min(maximum,limit):limit);
    integrator.step(1);
    integrator.setMaximumStepSize(maximum);
    context.setTime(time+integrator.getStepSize());
}

template<class I> void correctedStepTo(I& integrator, Context& context, double target) {
    while(target>context.getTime()) {
        double remaining=target-context.getTime();
        correctedStep(integrator,context,remaining);
        if(integrator.getStepSize()==remaining)context.setTime(target);
    }
}

template<class I> void run(I& integrator, int kind, bool constrained, bool site) {
    System system;
    for(double mass:{1.,2.,3.,0.})system.addParticle(mass);
    if(constrained)system.addConstraint(0,1,1);
    if(site)system.setVirtualSite(3,new TwoParticleAverageSite(0,2,0.3,0.7));
    auto* bond=new HarmonicBondForce();bond->addBond(0,2,0.8,2);bond->setForceGroup(1);system.addForce(bond);
    auto* external=new CustomExternalForce("scale*(0.4*x+0.2*y^2)");external->addGlobalParameter("scale",1);
    for(int i=0;i<4;++i)external->addParticle(i,{});
    external->setForceGroup(4);system.addForce(external);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    context.setPositions({Vec3(0,0,0),Vec3(1,0,0),Vec3(0,0.7,1.1),Vec3(0.2,0.4,0.5)});
    context.setVelocities({Vec3(0.2,0.1,-0.1),Vec3(-0.3,0.4,0.2),Vec3(0.1,-0.2,0.3),Vec3(0.7,0.8,0.9)});
    context.computeVirtualSites();context.applyVelocityConstraints(1e-10);
    context.setTime(2.7);context.setStepCount(17);
    double caps[]={0,0.0002,0.0005,0.0003,0.0001,0,0};
    double errors[]={1e-7,1e-7,1e-7,1e-5,1e-5,1e-6,1e-8};
    int masks[]={-1,18,0,2,18,-1,16};
    double frictions[]={0.7,0.7,0.7,0.7,1.3,1.3,0};
    for(int phase=0;phase<7;++phase) {
        integrator.setMaximumStepSize(caps[phase]);integrator.setErrorTolerance(errors[phase]);
        integrator.setIntegrationForceGroups(masks[phase]);integrator.setConstraintTolerance(1e-10);
        context.setParameter("scale",phase%2?0.3:1.0);
        integrator.setStepSize(0.001+phase*0.0001);
        if(auto* stochastic=dynamic_cast<VariableLangevinIntegrator*>(&integrator))stochastic->setFriction(frictions[phase]);
        if(phase==6)context.setVelocitiesToTemperature(0,17);
        for(int event=0;event<6;++event) {
            if(event>0 && event<5)correctedStep(integrator,context);
            if(event==5)correctedStepTo(integrator,context,context.getTime()+0.000731);
            auto state=context.getState(State::Positions|State::Velocities|State::Forces|State::Energy);
            std::cout << "{\"kind\":" << kind << ",\"constrained\":" << constrained << ",\"site\":" << site
                << ",\"phase\":" << phase << ",\"event\":" << event << ",\"dt\":" << integrator.getStepSize()
                << ",\"time\":" << state.getTime() << ",\"step_count\":" << state.getStepCount()
                << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy()
                << ",\"positions\":";vectors(state.getPositions());std::cout << ",\"velocities\":";vectors(state.getVelocities());
            std::cout << ",\"forces\":";vectors(state.getForces());std::cout << "}\n";
        }
    }
}

void statistics() {
    const int n=12000;const double kb=1.380649e-23*6.02214076e23/1000;
    for(double friction:{0.7,1000000.}) {
        System system;for(int i=0;i<n;++i)system.addParticle(1+i%5);
        VariableLangevinIntegrator integrator(300,friction,1e-5);integrator.setMaximumStepSize(0.001);integrator.setRandomNumberSeed(17);
        Context context(system,integrator,Platform::getPlatformByName("Reference"));
        int phase=0;
        for(double temperature:{100.,500.,200.}) {
            integrator.setTemperature(temperature);
            context.setPositions(std::vector<Vec3>(n));context.setVelocities(std::vector<Vec3>(n));
            for(int step=0;step<5;++step)correctedStep(integrator,context);
            auto state=context.getState(State::Velocities|State::Positions);
            double mean[2]={0,0},second[2]={0,0},fourth[2]={0,0};
            double decay=exp(-friction*0.001),noise2=-expm1(-2*friction*0.001);
            double pvar=0,vvar=0,covar=0;
            for(int step=0;step<5;++step) {
                double a=0.0005*(1+decay),b=0.0005;
                pvar+=a*a*vvar+2*a*covar+b*b*noise2;
                covar=decay*covar+a*decay*vvar+b*noise2;
                vvar=decay*decay*vvar+noise2;
            }
            for(int i=0;i<n;++i)for(int axis=0;axis<3;++axis) {
                double scale=kb*temperature/(1+i%5);
                double z[2]={state.getPositions()[i][axis]/sqrt(scale*pvar),state.getVelocities()[i][axis]/sqrt(scale*vvar)};
                for(int j=0;j<2;++j){mean[j]+=z[j];second[j]+=z[j]*z[j];fourth[j]+=pow(z[j],4);}
            }
            std::cout << "{\"statistic\":true,\"friction\":" << friction << ",\"temperature\":" << temperature << ",\"phase\":" << phase++ << ",\"samples\":" << 3*n << ",\"moments\":[";
            for(int j=0;j<2;++j){if(j)std::cout<<',';std::cout<<'['<<mean[j]/(3*n)<<','<<second[j]/(3*n)<<','<<fourth[j]/(3*n)<<']';}
            std::cout << "]}\n";
        }
    }
}

int main() {
    std::cout << std::setprecision(17);
    for(bool constrained:{false,true})for(bool site:{false,true}) {
        VariableVerletIntegrator verlet(1e-7);run(verlet,0,constrained,site);
        VariableLangevinIntegrator langevin(0,0.7,1e-7);run(langevin,1,constrained,site);
    }
    statistics();
    for(int kind=0;kind<2;++kind) {
        System system;system.addParticle(1);
        std::unique_ptr<Integrator> integrator;
        if(kind==0){auto* i=new VariableVerletIntegrator(1e-5);i->setMaximumStepSize(0.001);integrator.reset(i);}
        else{auto* i=new VariableLangevinIntegrator(0,1,1e-5);i->setMaximumStepSize(0.001);integrator.reset(i);}
        Context context(system,*integrator,Platform::getPlatformByName("Reference"));context.setPositions({Vec3()});integrator->step(1);
        bool infinite=std::isinf(context.getTime());
        context.setTime(0);context.setStepCount(0);
        if(kind==0)dynamic_cast<VariableVerletIntegrator&>(*integrator).stepTo(0.003);
        else dynamic_cast<VariableLangevinIntegrator&>(*integrator).stepTo(0.003);
        std::cout << "{\"clock_exception\":true,\"kind\":"<<kind<<",\"source_time_infinite\":"<<(infinite?"true":"false")<<",\"dt\":"<<integrator->getStepSize()<<",\"source_step_to_count\":"<<context.getStepCount()<<",\"source_step_to_time\":"<<context.getTime()<<"}\n";
    }
}
