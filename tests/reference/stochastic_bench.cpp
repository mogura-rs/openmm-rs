#include <openmm/BrownianIntegrator.h>
#include <openmm/CustomIntegrator.h>
#include <openmm/QTBIntegrator.h>
#include <ReferencePlatform.h>
#include <openmm/DPDIntegrator.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/Context.h>
#include <openmm/CompoundIntegrator.h>
#include <openmm/LangevinMiddleIntegrator.h>
#include <openmm/NoseHooverIntegrator.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/VariableVerletIntegrator.h>
#include <openmm/VariableLangevinIntegrator.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
using namespace OpenMM;
class SerialReference : public ReferencePlatform {
public:
    void contextCreated(ContextImpl& context, const std::map<std::string, std::string>&) const override {
        ReferencePlatform::contextCreated(context, {{"Threads", "1"}});
    }
};
int main(int argc,char**argv) {
    std::string kind=argc>1?argv[1]:"langevin";
    int n=argc>2?std::stoi(argv[2]):4096;
    int iterations=argc>3?std::stoi(argv[3]):1500;
    double temperature=argc>4?std::stod(argv[4]):300;
    System system;
    for(int i=0;i<n;++i)system.addParticle(1);
    bool dpd=kind=="dpd";
    bool qtb=kind.substr(0,4)=="qtb_";
    int side=std::max(5,(int)std::ceil(std::cbrt(n)));
    if(dpd) {
        double length=0.3*side;
        system.setDefaultPeriodicBoxVectors(Vec3(length,0,0),Vec3(0,length,0),Vec3(0,0,length));
        auto* marker=new HarmonicBondForce();marker->setUsesPeriodicBoundaryConditions(true);system.addForce(marker);
    }
    std::unique_ptr<Integrator> integrator;
    bool compound=kind=="compound_verlet" || kind=="compound_langevin";
    if(compound) {
        auto* root=new CompoundIntegrator();
        for(int seed : {3721,3722}) {
            if(kind=="compound_verlet")root->addIntegrator(new VerletIntegrator(0.001));
            else {
                auto* value=new LangevinMiddleIntegrator(temperature,0.7,0.001);
                value->setRandomNumberSeed(seed);root->addIntegrator(value);
            }
        }
        integrator.reset(root);
    }
    else if(kind=="nose_hoover" || kind=="nose_pairs") {
        auto* value=new NoseHooverIntegrator(0.001);
        if(kind=="nose_hoover")value->addThermostat(temperature,0.7,3,3,7);
        else {
            std::vector<std::pair<int,int>> pairs;
            for(int a=0;a<n-2;a+=2)pairs.emplace_back(a,a+1);
            value->addSubsystemThermostat({n-2,n-1},pairs,temperature,0.7,30,2,3,3,7);
        }
        integrator.reset(value);
    }
    else if(kind=="custom_scalar" || kind=="custom_vector" || kind=="custom_langevin") {
        auto* value=new CustomIntegrator(0.001);
        value->addPerDofVariable("q",0);value->addGlobalVariable("total",0);
        value->addGlobalVariable("kT",1.380649e-23*6.02214076e23/1000*temperature);
        value->setRandomNumberSeed(3721);
        value->addComputePerDof("v",kind=="custom_langevin" ? "exp(-0.7*dt)*v+sqrt((1-exp(-1.4*dt))*kT/m)*gaussian" : "v+dt*f/m");
        value->addComputePerDof("x","x+dt*v");
        value->addComputePerDof("q",kind=="custom_vector" ? "cross(x,v)+vector(_x(x),_y(v),_z(x+v))" : "x+v+dt");
        value->addComputeSum("total",kind=="custom_vector" ? "dot(q,q)" : "q*q");
        integrator.reset(value);
    }
    else if(qtb) {
        auto* value=new QTBIntegrator(temperature,0.7,0.001);
        value->setSegmentLength(0.03275);value->setDefaultAdaptationRate(kind=="qtb_fixed" ? 0 : 0.01);
        value->setRandomNumberSeed(3721);
        for(int particle=0;particle<n;++particle)value->setParticleType(particle,0);
        integrator.reset(value);
    }
    else if(dpd) {
        auto* value=new DPDIntegrator(temperature,0.7,0.7,0.001);value->setRandomNumberSeed(3721);value->setIntegrationForceGroups(0);integrator.reset(value);
    }
    else if(kind=="verlet")integrator=std::make_unique<VerletIntegrator>(0.001);
    else if(kind=="variable_verlet") {
        auto* value=new VariableVerletIntegrator(1e-5);value->setMaximumStepSize(0.001);integrator.reset(value);
    } else if(kind=="variable_langevin") {
        auto* value=new VariableLangevinIntegrator(temperature,0.7,1e-5);value->setMaximumStepSize(0.001);value->setRandomNumberSeed(3721);integrator.reset(value);
    } else if(kind=="langevin") {
        auto* value=new LangevinMiddleIntegrator(temperature,0.7,0.001);
        value->setRandomNumberSeed(3721);integrator.reset(value);
    } else if(kind=="brownian") {
        auto* value=new BrownianIntegrator(temperature,0.7,0.001);
        value->setRandomNumberSeed(3721);integrator.reset(value);
    } else throw std::runtime_error("unknown integrator");
    SerialReference serial;
    Context context(system,*integrator,qtb ? serial : Platform::getPlatformByName("Reference"));
    std::vector<Vec3> positions(n);
    if(dpd)for(int i=0;i<n;++i)positions[i]=Vec3(0.3*(i%side),0.3*((i/side)%side),0.3*(i/(side*side)));
    context.setPositions(positions);
    std::vector<Vec3> initialVelocities(n,Vec3(0.1,0.2,0.3));
    if(kind.substr(0,5)=="nose_" || dpd)for(int a=0;a<n;++a)initialVelocities[a]*=(1+a%2);
    context.setVelocities(initialVelocities);
    auto advance=[&](int steps) {
        if(compound) {
            auto& root=dynamic_cast<CompoundIntegrator&>(*integrator);
            for(int i=0;i<steps;++i) {root.setCurrentIntegrator(i%2);root.step(1);}
        } else if(kind=="variable_verlet" || kind=="variable_langevin") {
            for(int i=0;i<steps;++i) {
                double time=context.getTime();integrator->step(1);
                context.setTime(time+integrator->getStepSize());
            }
        } else integrator->step(steps);
    };
    advance(10);
    auto start=std::chrono::steady_clock::now();advance(iterations);
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    double energy=context.getState(State::Energy).getKineticEnergy();
    if(kind=="nose_pairs") {
        energy=0;
        auto state=context.getState(State::Velocities);
        for(const auto& v : state.getVelocities())energy+=0.5*v.dot(v);
    }
    double kb=1.380649e-23*6.02214076e23/1000,variance=0,meanSquared=0.14*n;
    if(kind=="langevin" || kind=="variable_langevin" || kind=="compound_langevin" || kind=="custom_langevin") {
        double decay=std::exp(-2*0.7*0.001*(iterations+10));
        variance=kb*temperature*(1-decay);meanSquared*=decay;
    } else if(kind=="brownian") {
        variance=2*kb*temperature/(0.7*0.001);meanSquared=0;
    }
    double expected=0.5*(3*n*variance+meanSquared);
    double stddev=std::sqrt(1.5*n*variance*variance+variance*meanSquared);
    Vec3 momentum;
    if(dpd) {auto state=context.getState(State::Velocities);for(auto v: state.getVelocities())momentum+=v;}
    std::ostringstream expectedStream;expectedStream<<std::setprecision(17)<<expected;
    std::cout<<std::setprecision(17)<<"{\"implementation\":\"openmm-reference\",\"precision\":\"f64\",\"api\":\"Context.step\",\"integrator\":\""<<kind<<"\",\"temperature\":"<<temperature<<",\"particles\":"<<n<<",\"iterations\":"<<iterations<<",\"seconds\":"<<seconds<<",\"evaluations_per_second\":"<<iterations/seconds<<",\"energy\":0,\"kinetic_energy\":"<<energy<<",\"expected_kinetic_energy\":"<<((kind.substr(0,5)=="nose_" || dpd || qtb) ? "null" : expectedStream.str())<<",\"kinetic_energy_stddev\":"<<stddev<<",\"simulation_time\":"<<context.getTime()<<",\"step_count\":"<<context.getStepCount()<<",\"adaptive\":"<<((kind=="variable_verlet" || kind=="variable_langevin")?"true":"false")<<",\"maximum_step_size\":0.001,\"error_tolerance\":"<<((kind=="variable_verlet" || kind=="variable_langevin")?"0.00001":"null")<<",\"compound_switch_each_step\":"<<(compound?"true":"false")<<",\"clock_repair\":"<<((kind=="variable_verlet" || kind=="variable_langevin")?"true":"false")<<",\"momentum\":["<<momentum[0]<<","<<momentum[1]<<","<<momentum[2]<<"]}\n";
}
