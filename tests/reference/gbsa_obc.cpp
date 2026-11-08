#include <openmm/Context.h>
#include <openmm/GBSAOBCForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/CustomCVForce.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/NonbondedForce.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace OpenMM;

void vectors(const std::vector<Vec3>& points) {
    std::cout << '[';
    for (size_t i=0; i<points.size(); ++i) {
        if(i) std::cout << ',';
        std::cout << '[' << points[i][0] << ',' << points[i][1] << ',' << points[i][2] << ']';
    }
    std::cout << ']';
}

void run(int method, int geometry, int settings, bool cv) {
    int n=geometry==0?1:geometry==1?2:geometry==2?8:125;
    System system;
    std::vector<Vec3> box={Vec3(3,0,0),Vec3(0.7,3.2,0),Vec3(-0.6,0.4,3.4)};
    system.setDefaultPeriodicBoxVectors(box[0],box[1],box[2]);
    auto* gb=new GBSAOBCForce();
    gb->setNonbondedMethod(static_cast<GBSAOBCForce::NonbondedMethod>(method));
    gb->setCutoffDistance(0.8);
    if(settings==1) {gb->setSoluteDielectric(2.1);gb->setSolventDielectric(35);gb->setSurfaceAreaEnergy(0.75);}
    if(settings==2) gb->setSurfaceAreaEnergy(0);
    if(settings==3) gb->setSolventDielectric(0);
    std::vector<Vec3> positions(n),parameters(n);
    for(int i=0;i<n;++i) {
        system.addParticle(i==n-1 && n>2?0:2+i%4);
        double charge=0.1*(i%7-3),radius=0.11+0.02*(i%5),scale=0.65+0.1*(i%4);
        if(geometry==1) {radius=i==0?0.3:0.1;scale=1.2;}
        parameters[i]=Vec3(charge,radius,scale);gb->addParticle(charge,radius,scale);
        positions[i]=Vec3(0.37*(i%5)+0.01*std::sin(i),0.41*((i/5)%5)+0.02*std::cos(i),0.43*(i/25)+0.015*std::sin(2*i));
    }
    if(geometry==1) positions[1]=positions[0]+Vec3(0.035,0.025,0.02);
    CustomCVForce* outer=nullptr;
    if(cv) {outer=new CustomCVForce("0.7*g+0.001*g*g");outer->addCollectiveVariable("g",gb);outer->setForceGroup(5);system.addForce(outer);}
    else {gb->setForceGroup(5);system.addForce(gb);}
    auto* external=new CustomExternalForce("0.2*x*x+0.1*z");external->setForceGroup(2);
    for(int i=0;i<n;++i)external->addParticle(i,{});system.addForce(external);
    VerletIntegrator it(0.0001);
    Context context(system,it,Platform::getPlatformByName("Reference"));context.setPositions(positions);
    if(cv)gb=&dynamic_cast<GBSAOBCForce&>(const_cast<Force&>(outer->getInnerContext(context).getSystem().getForce(0)));
    context.setVelocities(std::vector<Vec3>(n,Vec3(0.01,-0.02,0.03)));
    for(int phase=0;phase<4;++phase) {
        if(phase==1) {
            for(int i=0;i<n;++i) {parameters[i][0]*=0.8;parameters[i][1]+=0.013;parameters[i][2]*=0.97;gb->setParticleParameters(i,parameters[i][0],parameters[i][1],parameters[i][2]);}
            gb->setSoluteDielectric(7);gb->setSolventDielectric(11);gb->setSurfaceAreaEnergy(9);gb->setCutoffDistance(0.4);gb->setNonbondedMethod(GBSAOBCForce::NoCutoff);
            gb->updateParametersInContext(cv?outer->getInnerContext(context):context);
        }
        if(phase==2) {
            box={Vec3(3.4,0,0),Vec3(-0.4,3.1,0),Vec3(0.8,-0.6,3.8)};
            context.setPeriodicBoxVectors(box[0],box[1],box[2]);
            for(int i=0;i<n;++i) {positions[i]+=Vec3(0.02*std::sin(i+1),-0.01*std::cos(i),0.015*std::sin(i+2));if(i%3==1)positions[i]+=box[1];}
            context.setPositions(positions);
        }
        if(phase==3)it.step(5);
        for(int groups : {32,4,36,0}) {
            auto state=context.getState(State::Energy|State::Forces|State::Positions|State::Velocities,false,groups);
            std::cout<<"{\"method\":"<<method<<",\"geometry\":"<<geometry<<",\"settings\":"<<settings<<",\"cv\":"<<cv<<",\"phase\":"<<phase<<",\"groups\":"<<groups<<",\"energy\":"<<state.getPotentialEnergy()<<",\"kinetic\":"<<state.getKineticEnergy()<<",\"time\":"<<state.getTime()<<",\"parameters\":";
            vectors(parameters);std::cout<<",\"box\":";vectors(box);std::cout<<",\"positions\":";vectors(state.getPositions());
            std::cout<<",\"velocities\":";vectors(state.getVelocities());std::cout<<",\"forces\":";vectors(state.getForces());std::cout<<"}\n";
        }
    }
}

void cutoff_boundary(int method,double displacement) {
    System system;auto* gb=new GBSAOBCForce();gb->setNonbondedMethod(static_cast<GBSAOBCForce::NonbondedMethod>(method));gb->setCutoffDistance(0.8);
    for(int i=0;i<2;++i){system.addParticle(1);gb->addParticle(i?0.4:-0.3,0.15,0.8);}system.addForce(gb);
    system.setDefaultPeriodicBoxVectors(Vec3(2,0,0),Vec3(0,2,0),Vec3(0,0,2));VerletIntegrator it(0.001);
    Context context(system,it,Platform::getPlatformByName("Reference"));context.setPositions({Vec3(),Vec3(0.8+displacement,0,0)});
    auto state=context.getState(State::Energy|State::Forces);
    std::cout<<"{\"boundary\":true,\"method\":"<<method<<",\"distance\":"<<0.8+displacement<<",\"energy\":"<<state.getPotentialEnergy()<<",\"forces\":";vectors(state.getForces());std::cout<<"}\n";
}

void small_radius(double radius) {
    System system;system.addParticle(1);auto* gb=new GBSAOBCForce();gb->addParticle(0.5,radius,1);system.addForce(gb);
    VerletIntegrator it(0.001);Context context(system,it,Platform::getPlatformByName("Reference"));context.setPositions({Vec3()});
    auto state=context.getState(State::Energy|State::Forces);
    std::cout<<"{\"single_radius\":"<<radius<<",\"energy\":"<<state.getPotentialEnergy()<<",\"forces\":";vectors(state.getForces());std::cout<<"}\n";
}

void stability(int settings, double perturbation=0) {
    int n=16;System system;auto* gb=new GBSAOBCForce();auto* nb=new NonbondedForce();auto* bonds=new HarmonicBondForce();
    if(settings) {gb->setSoluteDielectric(2.1);gb->setSolventDielectric(35);gb->setSurfaceAreaEnergy(0);}
    std::vector<Vec3> p(n),v(n);
    for(int i=0;i<n;++i) {
        system.addParticle(12);double q=i%2?0.1:-0.1;gb->addParticle(q,0.15,0.8);nb->addParticle(q,0.25,0.2);
        p[i]=Vec3(0.32*(i%4),0.32*(i/4),0.03*std::sin(i));v[i]=Vec3(0.05*std::sin(1.7*i),0.04*std::cos(0.9*i),0.03*std::sin(0.3*i));
    }
    for(int i=0;i<n;++i)for(int j : {i+1,i+4})if(j<n && (j==i+4 || i%4!=3)) {Vec3 d=p[j]-p[i];bonds->addBond(i,j,std::sqrt(d.dot(d)),1000);}
    system.addForce(gb);system.addForce(nb);system.addForce(bonds);
    p[0][0]+=perturbation;
    VerletIntegrator it(0.0005);Context context(system,it,Platform::getPlatformByName("Reference"));context.setPositions(p);context.setVelocities(v);
    auto initial=context.getState(State::Energy);double e0=initial.getPotentialEnergy()+initial.getKineticEnergy(),maxDrift=0;
    State shortState;
    for(int sample=0;sample<200;++sample) {it.step(100);auto s=context.getState(State::Energy);maxDrift=std::max(maxDrift,std::abs((s.getPotentialEnergy()+s.getKineticEnergy()-e0)/e0));if(sample==19)shortState=context.getState(State::Positions|State::Velocities);}
    auto s=context.getState(State::Energy|State::Positions|State::Velocities);
    std::cout<<"{\"stability\":true,\"perturbation\":"<<perturbation<<",\"settings\":"<<settings<<",\"initial_energy\":"<<e0<<",\"final_energy\":"<<s.getPotentialEnergy()+s.getKineticEnergy()<<",\"max_relative_drift\":"<<maxDrift<<",\"time\":"<<s.getTime()<<",\"positions\":";
    vectors(s.getPositions());std::cout<<",\"velocities\":";vectors(s.getVelocities());
    std::cout<<",\"short_positions\":";vectors(shortState.getPositions());std::cout<<",\"short_velocities\":";vectors(shortState.getVelocities());std::cout<<"}\n";
}

int main() {
    std::cout<<std::setprecision(17);
    for(int method=0;method<3;++method)for(int geometry=0;geometry<4;++geometry)for(int settings=0;settings<4;++settings)for(bool cv : {false,true})run(method,geometry,settings,cv);
    for(int method=1;method<=2;++method)for(double displacement : {-1e-8,0.,1e-8})cutoff_boundary(method,displacement);
    stability(0);stability(1);stability(0,1e-12);stability(1,1e-12);
    small_radius(0.005);small_radius(0.008);
}
