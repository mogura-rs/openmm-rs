#include <openmm/Context.h>
#include <openmm/LCPOForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/CustomCVForce.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/VirtualSite.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace OpenMM;
void vectors(const std::vector<Vec3>& points) {
    std::cout<<'['; for(size_t i=0;i<points.size();++i){if(i)std::cout<<',';std::cout<<'['<<points[i][0]<<','<<points[i][1]<<','<<points[i][2]<<']';}std::cout<<']';
}
void run(bool periodic,int geometry,int setting,bool cv) {
    int n=geometry==0?1:geometry==1?2:geometry==2?8:125;
    System system;std::vector<Vec3> box={Vec3(3,0,0),Vec3(0.7,3.2,0),Vec3(-0.6,0.4,3.4)};
    system.setDefaultPeriodicBoxVectors(box[0],box[1],box[2]);
    auto* lcpo=new LCPOForce();lcpo->setSurfaceTension(setting==0?1:setting==1?5:-2);lcpo->setUsesPeriodicBoundaryConditions(periodic);
    std::vector<Vec3> positions(n),velocities(n,Vec3(0.01,-0.02,0.03));
    std::vector<std::array<double,5>> parameters;
    for(int i=0;i<n;++i) {
        system.addParticle(n==8 && (i==0 || i==7)?0:2+i%4);
        double radius=0.18+0.02*(i%5);if(setting==1 && i%3==1)radius=0;
        std::array<double,5> p={radius,0.7+0.05*(i%4),-0.2+0.03*(i%3),0.08+0.01*(i%3),0.1+0.02*(i%5)};
        if(setting==2 && i%3==1)for(int k=1;k<5;++k)p[k]=0;
        parameters.push_back(p);lcpo->addParticle(p[0],p[1],p[2],p[3],p[4]);
        positions[i]=Vec3(0.13*(i%5)+0.01*std::sin(i),0.14*((i/5)%5)+0.02*std::cos(i),0.15*(i/25)+0.015*std::sin(2*i));
    }
    if(n==2)positions[1]=positions[0]+Vec3(0.035,0.025,0.02);
    if(n==8)system.setVirtualSite(7,new TwoParticleAverageSite(2,6,0.4,0.6));
    CustomCVForce* outer=nullptr;
    if(cv){outer=new CustomCVForce("0.7*a+0.001*a*a");outer->addCollectiveVariable("a",lcpo);outer->setForceGroup(5);system.addForce(outer);}
    else {lcpo->setForceGroup(5);system.addForce(lcpo);}
    auto* external=new CustomExternalForce("0.2*x*x+0.1*z");external->setForceGroup(2);for(int i=0;i<n;++i)external->addParticle(i,{});system.addForce(external);
    VerletIntegrator it(0.00001);Context context(system,it,Platform::getPlatformByName("Reference"));
    context.setPositions(positions);context.computeVirtualSites();context.setVelocities(velocities);
    if(cv)lcpo=&dynamic_cast<LCPOForce&>(const_cast<Force&>(outer->getInnerContext(context).getSystem().getForce(0)));
    for(int phase=0;phase<4;++phase) {
        if(phase==1){for(int i=0;i<n;++i){auto& p=parameters[i];p[0]=p[0]==0?0.21:p[0]+0.013;p[1]*=0.9;p[2]*=1.1;p[3]*=0.8;p[4]*=1.2;lcpo->setParticleParameters(i,p[0],p[1],p[2],p[3],p[4]);}lcpo->setSurfaceTension(2.3);lcpo->setUsesPeriodicBoundaryConditions(!periodic);lcpo->updateParametersInContext(cv?outer->getInnerContext(context):context);}
        if(phase==2){box={Vec3(3.4,0,0),Vec3(-0.4,3.1,0),Vec3(0.8,-0.6,3.8)};context.setPeriodicBoxVectors(box[0],box[1],box[2]);for(int i=0;i<n;++i){positions[i]+=Vec3(0.01*std::sin(i+1),-0.008*std::cos(i),0.006*std::sin(i+2));if(i%3==1)positions[i]+=box[1];}context.setPositions(positions);context.computeVirtualSites();}
        if(phase==3)it.step(5);
        for(int groups : {32,4,36,0}) {
            auto state=context.getState(State::Energy|State::Forces|State::Positions|State::Velocities,false,groups);
            std::cout<<"{\"periodic\":"<<periodic<<",\"geometry\":"<<geometry<<",\"setting\":"<<setting<<",\"cv\":"<<cv<<",\"phase\":"<<phase<<",\"groups\":"<<groups<<",\"energy\":"<<state.getPotentialEnergy()<<",\"kinetic\":"<<state.getKineticEnergy()<<",\"time\":"<<state.getTime()<<",\"box\":";vectors(box);
            std::cout<<",\"positions\":";vectors(state.getPositions());std::cout<<",\"velocities\":";vectors(state.getVelocities());std::cout<<",\"forces\":";vectors(state.getForces());std::cout<<"}\n";
        }
        if(phase==0 && n==8){auto base=context.getState(State::Positions|State::Forces,false,32);auto p=base.getPositions();std::vector<Vec3> fd(n);for(int i=0;i<n-1;++i)for(int axis=0;axis<3;++axis){auto plus=p;plus[i][axis]+=1e-6;context.setPositions(plus);context.computeVirtualSites();double ep=context.getState(State::Energy,false,32).getPotentialEnergy();plus=p;plus[i][axis]-=1e-6;context.setPositions(plus);context.computeVirtualSites();double em=context.getState(State::Energy,false,32).getPotentialEnergy();fd[i][axis]=-(ep-em)/2e-6;}context.setPositions(p);std::cout<<"{\"fd\":true,\"periodic\":"<<periodic<<",\"setting\":"<<setting<<",\"cv\":"<<cv<<",\"forces\":";vectors(fd);std::cout<<"}\n";}
    }
}
void stability(bool periodic) {
    int n=8;System system;auto* force=new LCPOForce();force->setSurfaceTension(5.);force->setUsesPeriodicBoundaryConditions(periodic);auto* bonds=new HarmonicBondForce();std::vector<Vec3> p(n),v(n);
    system.setDefaultPeriodicBoxVectors(Vec3(3,0,0),Vec3(0.7,3.2,0),Vec3(-0.6,0.4,3.4));
    for(int i=0;i<n;++i){system.addParticle(12);force->addParticle(0.3,0.7,-0.2,0.08,0.1);p[i]=Vec3(0.12*(i%2),0.12*((i/2)%2),0.12*(i/4));v[i]=Vec3(0.02*std::sin(1.7*i),0.02*std::cos(0.9*i),0.01*std::sin(0.3*i));}
    for(int i=0;i<n;++i)for(int j=i+1;j<n;++j){auto d=p[j]-p[i];bonds->addBond(i,j,std::sqrt(d.dot(d)),1000);}
    system.addForce(force);system.addForce(bonds);VerletIntegrator it(0.0002);Context context(system,it,Platform::getPlatformByName("Reference"));context.setPositions(p);context.setVelocities(v);
    auto initial=context.getState(State::Energy);double e0=initial.getPotentialEnergy()+initial.getKineticEnergy(),maxDrift=0,maxDistance=0;State shortState;
    for(int sample=0;sample<250;++sample){it.step(100);auto s=context.getState(State::Energy|State::Positions);maxDrift=std::max(maxDrift,std::abs(s.getPotentialEnergy()+s.getKineticEnergy()-e0));for(int i=0;i<n;++i)for(int j=i+1;j<n;++j){auto d=s.getPositions()[j]-s.getPositions()[i];maxDistance=std::max(maxDistance,std::sqrt(d.dot(d)));}if(sample==4)shortState=context.getState(State::Positions|State::Velocities);}
    auto s=context.getState(State::Energy|State::Positions|State::Velocities);std::cout<<"{\"stability\":true,\"periodic\":"<<periodic<<",\"max_absolute_drift\":"<<maxDrift<<",\"max_distance\":"<<maxDistance<<",\"time\":"<<s.getTime()<<",\"positions\":";vectors(s.getPositions());std::cout<<",\"velocities\":";vectors(s.getVelocities());std::cout<<",\"short_positions\":";vectors(shortState.getPositions());std::cout<<",\"short_velocities\":";vectors(shortState.getVelocities());std::cout<<"}\n";
}
int main(){std::cout<<std::setprecision(17);for(bool periodic : {false,true})for(int geometry=0;geometry<4;++geometry)for(int setting=0;setting<3;++setting)for(bool cv : {false,true})run(periodic,geometry,setting,cv);stability(false);stability(true);}
