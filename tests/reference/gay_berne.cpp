#include <openmm/Context.h>
#include <openmm/GayBerneForce.h>
#include <openmm/CustomCVForce.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/HarmonicBondForce.h>
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
using Parameters=std::array<double,10>;
void vectors(const std::vector<Vec3>& points){std::cout<<'[';for(size_t i=0;i<points.size();++i){if(i)std::cout<<',';std::cout<<'['<<points[i][0]<<','<<points[i][1]<<','<<points[i][2]<<']';}std::cout<<']';}
Parameters particle(const GayBerneForce& f,int i){Parameters p;int x,y;f.getParticleParameters(i,p[0],p[1],x,y,p[4],p[5],p[6],p[7],p[8],p[9]);p[2]=x;p[3]=y;return p;}
void setParticle(GayBerneForce& f,int i,const Parameters& p){f.setParticleParameters(i,p[0],p[1],int(p[2]),int(p[3]),p[4],p[5],p[6],p[7],p[8],p[9]);}
void addParticle(GayBerneForce& f,const Parameters& p){f.addParticle(p[0],p[1],int(p[2]),int(p[3]),p[4],p[5],p[6],p[7],p[8],p[9]);}
GayBerneForce* makeForce(int kind,int n=8){
    auto* f=new GayBerneForce();
    for(int i=0;i<n;++i){
        Parameters p={0.3,i==0?1.0:i==3?0.8:i==6?0.2:i==7?0.1:0.0,-1,-1,0.3,0.3,0.3,1,1,1};
        if(i==1 || i==2 || i==4 || i==5)p={0.1,0,-1,-1,0.1,0.1,0.1,1,1,1};
        if(kind==1 && i==0)p={0.3,1,1,-1,0.3,0.24,0.24,1,0.8,0.8};
        if(kind==1 && i==3)p={0.28,0.8,4,-1,0.25,0.3,0.3,1.3,0.6,0.6};
        if(kind==2 && i==0)p={0.3,1,1,2,0.3,0.24,0.2,1,0.8,1.2};
        if(kind==2 && i==3)p={0.28,0.8,4,5,0.25,0.3,0.2,1.3,0.6,0.9};
        addParticle(*f,p);
    }
    return f;
}
std::vector<Vec3> initialPositions(){return {Vec3(),Vec3(0.1,0,0),Vec3(0.03,0.08,0),Vec3(0.7,0.2,0.1),Vec3(0.7,0.29,0.1),Vec3(0.75,0.23,0.17),Vec3(1.2,0.4,0.2),Vec3()};}
std::vector<Vec3> finiteDifference(Context& context,const std::vector<Vec3>& p,int independent,int groups,bool sites){
    std::vector<Vec3> result(independent);
    for(int i=0;i<independent;++i)for(int axis=0;axis<3;++axis){auto q=p;q[i][axis]+=1e-6;context.setPositions(q);if(sites)context.computeVirtualSites();double ep=context.getState(State::Energy,false,groups).getPotentialEnergy();q=p;q[i][axis]-=1e-6;context.setPositions(q);if(sites)context.computeVirtualSites();double em=context.getState(State::Energy,false,groups).getPotentialEnergy();result[i][axis]=-(ep-em)/2e-6;}
    context.setPositions(p);if(sites)context.computeVirtualSites();return result;
}
void run(int initialMethod,int kind,bool initialSwitch,bool cv){
    System system;auto* f=makeForce(kind);int method=initialMethod;bool switching=initialSwitch;
    f->setNonbondedMethod(static_cast<GayBerneForce::NonbondedMethod>(method));f->setCutoffDistance(1.2);f->setSwitchingDistance(0.8);f->setUseSwitchingFunction(switching && method!=0);
    f->addException(0,3,0.31,0.9);f->addException(6,7,0.3,0);
    for(int i=0;i<8;++i)system.addParticle(i>=6?0:(i%3==0?12:1));
    system.setVirtualSite(7,new TwoParticleAverageSite(0,3,0.4,0.6));
    std::vector<Vec3> box={Vec3(3,0,0),Vec3(0.5,3.2,0),Vec3(-0.3,0.4,3.4)};system.setDefaultPeriodicBoxVectors(box[0],box[1],box[2]);
    CustomCVForce* outer=nullptr;
    if(cv){outer=new CustomCVForce("0.7*g+0.001*g*g");outer->addCollectiveVariable("g",f);outer->setForceGroup(5);system.addForce(outer);}else{f->setForceGroup(5);system.addForce(f);}
    auto* external=new CustomExternalForce("0.2*x*x+0.1*z");external->setForceGroup(2);for(int i=0;i<8;++i)external->addParticle(i,{});system.addForce(external);
    VerletIntegrator it(0.001);Context context(system,it,Platform::getPlatformByName("Reference"));auto positions=initialPositions();context.setPositions(positions);context.computeVirtualSites();
    if(cv)f=&dynamic_cast<GayBerneForce&>(const_cast<Force&>(outer->getInnerContext(context).getSystem().getForce(0)));
    for(int phase=0;phase<4;++phase){
        if(phase==1){
            for(int i=0;i<8;++i){auto p=particle(*f,i);p[0]*=1.03;p[1]*=0.8;for(int a=4;a<7;++a)p[a]*=1.02;for(int a=7;a<10;++a)p[a]*=0.9;if(i==0 || i==3){if(kind==1)p[2]+=1;if(kind==2)std::swap(p[2],p[3]);}setParticle(*f,i,p);}
            f->setExceptionParameters(0,0,6,0.29,0.4);f->addException(1,6,0.2,0.15);f->updateParametersInContext(cv?outer->getInnerContext(context):context);
        }
        if(phase==2){
            box={Vec3(3.5,0,0),Vec3(-0.6,3.3,0),Vec3(0.7,-0.4,3.7)};context.setPeriodicBoxVectors(box[0],box[1],box[2]);
            method=(initialMethod+1)%3;switching=!initialSwitch;f->setNonbondedMethod(static_cast<GayBerneForce::NonbondedMethod>(method));f->setCutoffDistance(1.1);f->setSwitchingDistance(0.65);f->setUseSwitchingFunction(switching && method!=0);f->updateParametersInContext(cv?outer->getInnerContext(context):context);
            for(int i=3;i<6;++i)positions[i]+=box[1];context.setPositions(positions);context.computeVirtualSites();
        }
        if(phase==3){for(int i=0;i<7;++i)positions[i]+=Vec3(0.005*std::sin(i+1),0.007*std::cos(i+2),-0.004*std::sin(i));context.setPositions(positions);context.computeVirtualSites();}
        for(int groups:{32,4,36,0}){
            auto state=context.getState(State::Positions|State::Energy|State::Forces,false,groups);auto p=state.getPositions();auto fd=finiteDifference(context,p,7,groups,true);
            std::cout<<"{\"initial_method\":"<<initialMethod<<",\"kind\":"<<kind<<",\"initial_switching\":"<<initialSwitch<<",\"cv\":"<<cv<<",\"phase\":"<<phase<<",\"groups\":"<<groups<<",\"method\":"<<method<<",\"switching\":"<<switching<<",\"cutoff\":"<<f->getCutoffDistance()<<",\"switch_distance\":"<<f->getSwitchingDistance()<<",\"energy\":"<<state.getPotentialEnergy()<<",\"positions\":";vectors(p);std::cout<<",\"forces_original\":";vectors(state.getForces());std::cout<<",\"forces_fd\":";vectors(fd);std::cout<<",\"box\":";vectors(box);
            std::cout<<",\"parameters\":[";for(int i=0;i<8;++i){if(i)std::cout<<',';auto params=particle(*f,i);std::cout<<'[';for(int j=0;j<10;++j){if(j)std::cout<<',';std::cout<<params[j];}std::cout<<']';}
            std::cout<<"],\"exceptions\":[";for(int i=0;i<f->getNumExceptions();++i){if(i)std::cout<<',';int a,b;double sigma,epsilon;f->getExceptionParameters(i,a,b,sigma,epsilon);std::cout<<'['<<a<<','<<b<<','<<sigma<<','<<epsilon<<']';}std::cout<<"]}\n";
        }
    }
}
void oracle(bool periodic){
    System system;auto* f=makeForce(2,6);for(int i=0;i<6;++i){system.addParticle(i%3==0?12:1);auto p=particle(*f,i);p[1]*=0.15;setParticle(*f,i,p);}
    f->setNonbondedMethod(periodic?GayBerneForce::CutoffPeriodic:GayBerneForce::NoCutoff);f->setCutoffDistance(1.2);f->setSwitchingDistance(0.9);f->setUseSwitchingFunction(periodic);
    system.setDefaultPeriodicBoxVectors(Vec3(3,0,0),Vec3(0.5,3.2,0),Vec3(-0.3,0.4,3.4));auto initial=initialPositions();initial.resize(6);auto* bonds=new HarmonicBondForce();
    for(int first:{0,3})for(int i=first;i<first+3;++i)for(int j=i+1;j<first+3;++j){auto d=initial[j]-initial[i];bonds->addBond(i,j,std::sqrt(d.dot(d)),5000);}
    auto d=initial[3]-initial[0];bonds->addBond(0,3,std::sqrt(d.dot(d)),50);system.addForce(f);system.addForce(bonds);
    VerletIntegrator it(0.0001);Context context(system,it,Platform::getPlatformByName("Reference"));std::vector<Vec3> p(6);int sample=0;
    while(std::cin>>p[0][0]){for(int i=0;i<6;++i)for(int axis=0;axis<3;++axis)if(i || axis)std::cin>>p[i][axis];context.setPositions(p);std::cout<<"{\"energy\":"<<context.getState(State::Energy).getPotentialEnergy();if(sample==0 || sample==199){auto fd=finiteDifference(context,p,6,-1,false);std::cout<<",\"forces_fd\":";vectors(fd);}std::cout<<"}\n";++sample;}
}
void originalSwitchProbe(){
    for(bool switching:{false,true}){System s;auto* f=new GayBerneForce();for(int i=0;i<2;++i){s.addParticle(1);f->addParticle(0.3,1,-1,-1,0.3,0.3,0.3,1,1,1);}f->setUseSwitchingFunction(switching);f->setSwitchingDistance(0.8);s.addForce(f);VerletIntegrator it(0.001);Context c(s,it,Platform::getPlatformByName("Reference"));c.setPositions({Vec3(),Vec3(1.5,0,0)});std::cout<<"{\"original_no_cutoff_probe\":true,\"switching\":"<<switching<<",\"energy\":"<<c.getState(State::Energy).getPotentialEnergy()<<"}\n";}
}
void originalSphereProbe(){
    System s;auto* f=new GayBerneForce();
    for(double sigma:{0.2,0.4}){s.addParticle(1);f->addParticle(sigma,1,-1,-1,sigma,sigma,sigma,1,1,1);}
    s.addForce(f);VerletIntegrator it(0.001);Context c(s,it,Platform::getPlatformByName("Reference"));c.setPositions({Vec3(),Vec3(0.65,0,0)});
    double rho6=std::pow(0.3/0.65,6);
    std::cout<<"{\"original_sphere_probe\":true,\"energy\":"<<c.getState(State::Energy).getPotentialEnergy()<<",\"lennard_jones_energy\":"<<4*(rho6*rho6-rho6)<<"}\n";
}
int main(int argc,char** argv){std::cout<<std::setprecision(17);if(argc>1){oracle(argc>2 && std::stoi(argv[2])!=0);return 0;}for(int method=0;method<3;++method)for(int kind=0;kind<3;++kind)for(bool switching:{false,true})for(bool cv:{false,true})run(method,kind,switching,cv);originalSwitchProbe();originalSphereProbe();}
