#include <openmm/Context.h>
#include <openmm/CustomCVForce.h>
#include <openmm/CustomIntegrator.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/DrudeForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/VirtualSite.h>
#include <openmm/serialization/XmlSerializer.h>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace OpenMM;
extern "C" void registerDrudeReferenceKernelFactories();
void vectors(const std::vector<Vec3>& points) {std::cout<<'[';for(size_t i=0;i<points.size();++i){if(i)std::cout<<',';std::cout<<'['<<points[i][0]<<','<<points[i][1]<<','<<points[i][2]<<']';}std::cout<<']';}
void configure(DrudeForce& force,int mode,bool update) {
    for(int i=0;i<2;++i) {
        int p=2*i+1,parent=2*i;
        int p2=(mode==1 || mode==3 || mode>=6)?(i==0?4:6):-1;
        int p3=(mode==2 || mode==3 || mode==5 || mode>=6)?(i==0?5:4):-1;
        int p4=(mode==2 || mode==3 || mode==4 || mode>=6)?7:-1;
        double charge=(mode==6 && i==0?0.:(i==0?-.12:.08))*(update?1.2:1.);
        double polar=(.001+.0004*i)*(update?1.1:1.);
        double a1=(mode==7?1.8:.7)+.03*i+(update?.04:0.);
        double a2=(mode==7?1.5:1.2)-.02*i-(update?.05:0.);
        if(update)force.setParticleParameters(i,p,parent,p2,p3,p4,charge,polar,a1,a2);
        else force.addParticle(p,parent,p2,p3,p4,charge,polar,a1,a2);
    }
    if(update)force.setScreenedPairParameters(0,0,1,2.1);else force.addScreenedPair(0,1,1.7);
}
void run(bool periodic,bool triclinic,int mode,bool cv) {
    System system;std::vector<Vec3> box={Vec3(3,0,0),Vec3(triclinic?.4:0,3.2,0),Vec3(triclinic?-.3:0,triclinic?.2:0,3.4)};
    system.setDefaultPeriodicBoxVectors(box[0],box[1],box[2]);
    std::vector<Vec3> p={Vec3(.1,.2,.3),Vec3(.115,.185,.31),Vec3(2.85,.45,.55),Vec3(2.84,.47,.53),Vec3(.45,.3,.7),Vec3(.6,.85,.4),Vec3(.2,.7,.9),Vec3()};
    std::vector<Vec3> v(8,Vec3(.01,-.02,.03));
    for(int i=0;i<8;++i)system.addParticle(i>=6?0.:i%2==0?12.:.4);
    system.setVirtualSite(7,new TwoParticleAverageSite(4,6,.4,.6));
    auto* force=new DrudeForce();configure(*force,mode,false);force->setUsesPeriodicBoundaryConditions(periodic);force->setForceGroup(5);
    CustomCVForce* outer=nullptr;
    if(cv){outer=new CustomCVForce("0.7*a+0.001*a*a");outer->addCollectiveVariable("a",force);outer->setForceGroup(5);system.addForce(outer);}else system.addForce(force);
    auto* external=new CustomExternalForce(".2*x*x+.1*z");external->setForceGroup(2);for(int i=0;i<8;++i)external->addParticle(i,{});system.addForce(external);
    VerletIntegrator it(.000001);Context context(system,it,Platform::getPlatformByName("Reference"));context.setPositions(p);context.computeVirtualSites();context.setVelocities(v);
    if(cv)force=&dynamic_cast<DrudeForce&>(const_cast<Force&>(outer->getInnerContext(context).getSystem().getForce(0)));
    for(int phase=0;phase<4;++phase) {
        if(phase==1){configure(*force,mode,true);force->setUsesPeriodicBoundaryConditions(!periodic);if(!cv)force->setForceGroup(6);force->updateParametersInContext(cv?outer->getInnerContext(context):context);}
        if(phase==2){box={Vec3(3.4,0,0),Vec3(-.4,3.1,0),Vec3(.6,-.3,3.8)};context.setPeriodicBoxVectors(box[0],box[1],box[2]);for(int i=0;i<8;++i){p[i]+=Vec3(.01*std::sin(i+1),-.008*std::cos(i),.006*std::sin(i+2));if(i==2 || i==3)p[i]+=box[1];}context.setPositions(p);context.computeVirtualSites();}
        if(phase==3)it.step(5);
        for(int groups:{32,64,4,-1}) {
            auto state=context.getState(State::Energy|State::Forces|State::Positions|State::Velocities,false,groups);
            std::cout<<"{\"periodic\":"<<periodic<<",\"triclinic\":"<<triclinic<<",\"mode\":"<<mode<<",\"cv\":"<<cv<<",\"phase\":"<<phase<<",\"groups\":"<<groups<<",\"energy\":"<<state.getPotentialEnergy()<<",\"kinetic\":"<<state.getKineticEnergy()<<",\"time\":"<<state.getTime()<<",\"box\":";vectors(box);
            std::cout<<",\"positions\":";vectors(state.getPositions());std::cout<<",\"velocities\":";vectors(state.getVelocities());std::cout<<",\"forces\":";vectors(state.getForces());std::cout<<"}\n";
        }
        if(phase==0){auto base=context.getState(State::Positions,false,32).getPositions();std::vector<Vec3> fd(8);for(int i=0;i<7;++i)for(int axis=0;axis<3;++axis){auto q=base;q[i][axis]+=1e-6;context.setPositions(q);context.computeVirtualSites();double ep=context.getState(State::Energy,false,32).getPotentialEnergy();q=base;q[i][axis]-=1e-6;context.setPositions(q);context.computeVirtualSites();double em=context.getState(State::Energy,false,32).getPotentialEnergy();fd[i][axis]=(em-ep)/2e-6;}context.setPositions(base);std::cout<<"{\"fd\":true,\"forces\":";vectors(fd);std::cout<<"}\n";}
    }
}
void serializationProbe() {
    DrudeForce force;force.setUsesPeriodicBoundaryConditions(true);force.addParticle(1,0,-1,-1,-1,-.1,.001,1,1);
    auto* copy=XmlSerializer::clone<DrudeForce>(force);
    System system;system.addParticle(12);system.addParticle(.4);
    auto* cv=new CustomCVForce("a");cv->addCollectiveVariable("a",copy);system.addForce(cv);
    VerletIntegrator it(.001);Context context(system,it,Platform::getPlatformByName("Reference"));
    const Force& child=cv->getInnerContext(context).getSystem().getForce(0);
    std::cout<<"{\"serialization_probe\":true,\"original_periodic\":"<<force.usesPeriodicBoundaryConditions()<<",\"copy_periodic\":"<<copy->usesPeriodicBoundaryConditions()<<",\"inner_periodic\":"<<child.usesPeriodicBoundaryConditions()<<"}\n";
}
void stability(bool periodic,bool anisotropic) {
    System system;system.setDefaultPeriodicBoxVectors(Vec3(3,0,0),Vec3(.4,3.2,0),Vec3(-.3,.2,3.4));
    std::vector<Vec3> p={Vec3(.1,.2,.3),Vec3(.11,.185,.31),Vec3(2.2,.45,.55),Vec3(2.19,.46,.53),Vec3(.45,.3,.7),Vec3(.6,.85,.4)},v;
    for(int i=0;i<6;++i){system.addParticle(i==1 || i==3?.4:12.);v.push_back(Vec3(.01*std::sin(1.7*i),.012*std::cos(.9*i),.008*std::sin(.3*i)));}
    auto* drude=new DrudeForce();drude->addParticle(1,0,anisotropic?4:-1,anisotropic?4:-1,anisotropic?5:-1,-.12,.001,.7,1.2);drude->addParticle(3,2,anisotropic?5:-1,anisotropic?0:-1,anisotropic?4:-1,.08,.0014,.75,1.15);drude->addScreenedPair(0,1,1.7);drude->setUsesPeriodicBoundaryConditions(periodic);system.addForce(drude);
    auto* bonds=new HarmonicBondForce();for(int i:{0,2,4,5})for(int j:{0,2,4,5})if(j>i){auto d=p[j]-p[i];bonds->addBond(i,j,std::sqrt(d.dot(d)),500.);}system.addForce(bonds);
    VerletIntegrator it(.0001);Context context(system,it,Platform::getPlatformByName("Reference"));context.setPositions(p);context.setVelocities(v);
    auto initial=context.getState(State::Energy);double e0=initial.getPotentialEnergy()+initial.getKineticEnergy(),drift=0,maxDrude=0;State shortState;
    for(int sample=0;sample<500;++sample){it.step(100);auto s=context.getState(State::Energy|State::Positions);drift=std::max(drift,std::abs(s.getPotentialEnergy()+s.getKineticEnergy()-e0));for(int i:{0,2}){auto delta=s.getPositions()[i+1]-s.getPositions()[i];maxDrude=std::max(maxDrude,std::sqrt(delta.dot(delta)));}if(sample==9)shortState=context.getState(State::Positions|State::Velocities);}
    auto state=context.getState(State::Positions|State::Velocities);
    std::cout<<"{\"stability\":true,\"periodic\":"<<periodic<<",\"anisotropic\":"<<anisotropic<<",\"initial_energy\":"<<e0<<",\"max_absolute_drift\":"<<drift<<",\"max_drude_distance\":"<<maxDrude<<",\"time\":"<<state.getTime()<<",\"positions\":";vectors(state.getPositions());std::cout<<",\"velocities\":";vectors(state.getVelocities());std::cout<<",\"short_positions\":";vectors(shortState.getPositions());std::cout<<",\"short_velocities\":";vectors(shortState.getVelocities());std::cout<<"}\n";
}

void groupCacheProbe() {
    System system;system.addParticle(12);system.addParticle(.4);auto* force=new DrudeForce();force->addParticle(1,0,-1,-1,-1,-.1,.001,1,1);system.addForce(force);
    CustomIntegrator it(.001);it.addGlobalVariable("a",0);it.addComputeGlobal("a","energy0");Context context(system,it,Platform::getPlatformByName("Reference"));std::vector<Vec3> p={Vec3(),Vec3(.01,.02,.03)};context.setPositions(p);
    it.step(1);double first=it.getGlobalVariable(0);force->setForceGroup(1);it.step(1);double changed=it.getGlobalVariable(0);context.setPositions(p);it.step(1);
    std::cout<<std::setprecision(17)<<"{\"group_cache_probe\":true,\"energies\":["<<first<<','<<changed<<','<<it.getGlobalVariable(0)<<"]}\n";
}
int main(){registerDrudeReferenceKernelFactories();serializationProbe();groupCacheProbe();std::cout<<std::setprecision(17);for(bool periodic:{false,true})for(bool triclinic:{false,true})for(int mode=0;mode<8;++mode)for(bool cv:{false,true})run(periodic,triclinic,mode,cv);for(bool periodic:{false,true})for(bool aniso:{false,true})stability(periodic,aniso);}
