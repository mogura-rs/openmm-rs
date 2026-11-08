#include <openmm/Context.h>
#include <openmm/CustomManyParticleForce.h>
#include <openmm/CustomCVForce.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/TabulatedFunction.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/VirtualSite.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
using namespace OpenMM;
void vectors(const std::vector<Vec3>& p) { std::cout<<'[';for(size_t i=0;i<p.size();++i){if(i)std::cout<<',';std::cout<<'['<<p[i][0]<<','<<p[i][1]<<','<<p[i][2]<<']';}std::cout<<']'; }
std::string sum(const std::string& prefix,int k) { std::string s;for(int i=1;i<=k;++i){if(i>1)s+='+';s+=prefix+std::to_string(i);}return s; }
std::string expression(int k,int mode,int filter,bool changed) {
    std::string body;
    if(k==1)body="0.1*lambda*(x1*x1+y1*y1+z1*z1)+q1";
    if(k==2)body="lambda*(q1+q2)*exp(-distance(p1,p2))+0.02*(x1+x2)^2";
    if(k==3)body=mode==0?"lambda*(1+cos(angle(p1,p2,p3))*cos(angle(p2,p3,p1))*cos(angle(p3,p1,p2)))/(1+distance(p1,p2)*distance(p2,p3)*distance(p3,p1))^3+0.01*qs":"lambda*q1*(q2+q3)*(cos(angle(p2,p1,p3))+1/3)^2*exp(-distance(p1,p2)-distance(p1,p3))+0.01*qs";
    if(k==4){body="0.01*qs";for(int i=1;i<=4;++i)for(int j=i+1;j<=4;++j)body+="+lambda*exp(-distance(p"+std::to_string(i)+",p"+std::to_string(j)+"))";if(filter==1)body+="+0.2*cos(dihedral(p1,p2,p3,p4))";}
    std::string count=std::to_string(k);
    return std::string(changed?"1.2*(":"(")+body+")+0.01*(c1(sx/"+count+")+c2(lambda,qs/"+count+")+c3(sx/"+count+",sy/"+count+",lambda)+d1(1)+d2(1,0)+d3(1,0,1));sx="+sum("x",k)+";sy="+sum("y",k)+";qs="+sum("q",k);
}
void tables(CustomManyParticleForce& f) {
    f.addTabulatedFunction("c1",new Continuous1DFunction({0.3,-0.1,0.6,0.2,0.8},-2,5));
    std::vector<double> v2(9),v3(8);for(int i=0;i<9;++i)v2[i]=0.03*i+0.01*std::sin(i);for(int i=0;i<8;++i)v3[i]=0.05*i-0.01*std::cos(i);
    f.addTabulatedFunction("c2",new Continuous2DFunction(3,3,v2,-1,2,-1,2));
    f.addTabulatedFunction("c3",new Continuous3DFunction(2,2,2,v3,-1,5,-1,5,-1,2));
    f.addTabulatedFunction("d1",new Discrete1DFunction({0.2,-0.1,0.4}));
    f.addTabulatedFunction("d2",new Discrete2DFunction(2,2,{0.1,0.3,-0.2,0.5}));
    f.addTabulatedFunction("d3",new Discrete3DFunction(2,2,2,{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8}));
}
void filters(CustomManyParticleForce& f,int setting,bool changed) {
    const int types[4]={-7,2,5,9};
    for(int i=0;i<f.getNumParticlesPerSet();++i){
        std::set<int> allowed;
        if(setting==1)allowed.insert(types[(i+(changed?1:0))%4]);
        if(setting==2 && i%3!=2){allowed.insert(types[(i+(changed?1:0))%4]);allowed.insert(types[(i+2+(changed?1:0))%4]);}
        f.setTypeFilter(i,allowed);
    }
}
std::vector<Vec3> points() { return {Vec3(),Vec3(0.26,0.03,0.01),Vec3(0.1,0.28,-0.04),Vec3(0.23,0.31,0.21),Vec3(0.6,0.08,0.1),Vec3(0.52,0.4,0.23),Vec3(1.03,0.16,0.27),Vec3()}; }
void definition(const CustomManyParticleForce& f) {
    std::cout<<"{\"energy\":"<<std::quoted(f.getEnergyFunction())<<",\"method\":"<<f.getNonbondedMethod()<<",\"mode\":"<<f.getPermutationMode()<<",\"cutoff\":"<<f.getCutoffDistance()<<",\"particles\":[";
    for(int i=0;i<f.getNumParticles();++i){if(i)std::cout<<',';std::vector<double> p;int type;f.getParticleParameters(i,p,type);std::cout<<'['<<p[0]<<','<<type<<']';}
    std::cout<<"],\"filters\":[";for(int i=0;i<f.getNumParticlesPerSet();++i){if(i)std::cout<<',';std::set<int> types;f.getTypeFilter(i,types);std::cout<<'[';bool first=true;for(int t:types){if(!first)std::cout<<',';first=false;std::cout<<t;}std::cout<<']';}
    std::cout<<"],\"exclusions\":[";for(int i=0;i<f.getNumExclusions();++i){if(i)std::cout<<',';int a,b;f.getExclusionParticles(i,a,b);std::cout<<'['<<a<<','<<b<<']';}
    std::vector<double> values;double min,max;dynamic_cast<const Continuous1DFunction&>(f.getTabulatedFunction(0)).getFunctionParameters(values,min,max);
    std::cout<<"],\"table_values\":[";for(size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}std::cout<<"],\"table_bounds\":["<<min<<','<<max<<"]}";
}
void finiteDifference(Context& context) {
    auto p=context.getState(State::Positions).getPositions();std::vector<Vec3> fd(7);
    for(int i=0;i<7;++i)for(int axis=0;axis<3;++axis){auto q=p;q[i][axis]+=1e-6;context.setPositions(q);context.computeVirtualSites();double ep=context.getState(State::Energy,false,32).getPotentialEnergy();q=p;q[i][axis]-=1e-6;context.setPositions(q);context.computeVirtualSites();double em=context.getState(State::Energy,false,32).getPotentialEnergy();fd[i][axis]=-(ep-em)/2e-6;}
    context.setPositions(p);context.computeVirtualSites();std::cout<<"{\"fd\":true,\"forces\":";vectors(fd);std::cout<<"}\n";
}
void run(int k,int mode,int method,int filter,bool cv) {
    System system;auto* force=new CustomManyParticleForce(k,expression(k,mode,filter,false));force->addPerParticleParameter("q");force->addGlobalParameter("lambda",0.7);tables(*force);
    force->setNonbondedMethod(static_cast<CustomManyParticleForce::NonbondedMethod>(method));force->setPermutationMode(static_cast<CustomManyParticleForce::PermutationMode>(mode));force->setCutoffDistance(1.05);
    const int types[4]={-7,2,5,9};for(int i=0;i<8;++i){system.addParticle(i>=6?0:12);force->addParticle({0.1+0.03*i},types[i%4]);}
    filters(*force,filter,false);force->addExclusion(0,1);system.setVirtualSite(7,new OutOfPlaneSite(0,2,4,0.2,0.3,0.4));
    std::vector<Vec3> box={Vec3(3,0,0),Vec3(0.5,3.2,0),Vec3(-0.3,0.4,3.4)};system.setDefaultPeriodicBoxVectors(box[0],box[1],box[2]);
    CustomCVForce* outer=nullptr;
    if(cv){outer=new CustomCVForce("0.8*m+0.002*m*m");outer->addGlobalParameter("lambda",0.7);outer->addEnergyParameterDerivative("lambda");outer->addCollectiveVariable("m",force);outer->setForceGroup(5);system.addForce(outer);}else{force->setForceGroup(5);system.addForce(force);}
    auto* external=new CustomExternalForce("0.01*lambda*x*x");external->addGlobalParameter("lambda",0.7);external->setForceGroup(2);for(int i=0;i<8;++i)external->addParticle(i,{});system.addForce(external);
    VerletIntegrator it(0.00001);Context context(system,it,Platform::getPlatformByName("Reference"));auto p=points();std::vector<Vec3> v(8);for(int i=0;i<6;++i)v[i]=Vec3(0.01*std::sin(i+1),-0.02*std::cos(i),0.015*std::sin(0.7*i));context.setPositions(p);context.computeVirtualSites();context.setVelocities(v);
    if(cv)force=&dynamic_cast<CustomManyParticleForce&>(const_cast<Force&>(outer->getInnerContext(context).getSystem().getForce(0)));
    for(int phase=0;phase<4;++phase){
        if(phase==1){
            for(int i=0;i<8;++i)force->setParticleParameters(i,{0.12+0.04*i},types[(i+1)%4]);filters(*force,filter,true);
            force->setPermutationMode(static_cast<CustomManyParticleForce::PermutationMode>(1-mode));force->setEnergyFunction(expression(k,1-mode,filter,true));
            force->setNonbondedMethod(static_cast<CustomManyParticleForce::NonbondedMethod>(method==2?2:1-method));force->setCutoffDistance(0.85);
            force->setExclusionParticles(0,0,4);force->addExclusion(2,6);force->setGlobalParameterDefaultValue(0,9);
            force->updateParametersInContext(cv?outer->getInnerContext(context):context);context.setParameter("lambda",1.1);
        }
        if(phase==2){
            dynamic_cast<Continuous1DFunction&>(force->getTabulatedFunction(0)).setFunctionParameters({0.4,0.1,0.7,-0.2,0.5,0.9},-2.5,6);
            force->updateParametersInContext(cv?outer->getInnerContext(context):context);
            box={Vec3(3.4,0,0),Vec3(-0.4,3.6,0),Vec3(0.3,-0.2,3.8)};context.setPeriodicBoxVectors(box[0],box[1],box[2]);
            for(int i:{4,5})p[i]+=box[1];context.setPositions(p);context.computeVirtualSites();
        }
        if(phase==3)it.step(3);
        for(int groups:{32,4,36}){
            auto state=context.getState(State::Positions|State::Velocities|State::Forces|State::Energy|State::ParameterDerivatives,false,groups);
            std::cout<<"{\"count\":"<<k<<",\"initial_mode\":"<<mode<<",\"initial_method\":"<<method<<",\"filter\":"<<filter<<",\"cv\":"<<cv<<",\"phase\":"<<phase<<",\"groups\":"<<groups<<",\"definition\":";definition(*force);
            std::cout<<",\"energy\":"<<state.getPotentialEnergy()<<",\"kinetic\":"<<state.getKineticEnergy()<<",\"time\":"<<state.getTime()<<",\"positions\":";vectors(state.getPositions());std::cout<<",\"velocities\":";vectors(state.getVelocities());std::cout<<",\"forces\":";vectors(state.getForces());std::cout<<",\"box\":";vectors(box);
            auto derivatives=state.getEnergyParameterDerivatives();std::cout<<",\"derivative\":"<<derivatives["lambda"]<<"}\n";
        }
        if(phase==0)finiteDifference(context);
    }
}
void probes(){
    for(int count:{1,2,3})for(int mode:{0,1}){
        System s;auto* f=new CustomManyParticleForce(count,"1");f->setPermutationMode(static_cast<CustomManyParticleForce::PermutationMode>(mode));
        for(int i=0;i<3;++i){s.addParticle(1);f->addParticle({},0);}f->setTypeFilter(0,{1});s.addForce(f);VerletIntegrator it(0.001);Context c(s,it,Platform::getPlatformByName("Reference"));c.setPositions({Vec3(),Vec3(0.2,0,0),Vec3(0,0.3,0)});
        std::cout<<"{\"filter_probe\":true,\"count\":"<<count<<",\"mode\":"<<mode<<",\"energy\":"<<c.getState(State::Energy).getPotentialEnergy()<<"}\n";
    }
}
void stability(int mode,bool periodic){
    System s;auto* f=new CustomManyParticleForce(3,mode==0?"lambda*(q1+q2+q3)*exp(-distance(p1,p2)-distance(p2,p3)-distance(p3,p1))":"lambda*q1*(q2+q3)*(cos(angle(p2,p1,p3))+1/3)^2*exp(-distance(p1,p2)-distance(p1,p3))");
    f->addPerParticleParameter("q");f->addGlobalParameter("lambda",0.7);f->setPermutationMode(static_cast<CustomManyParticleForce::PermutationMode>(mode));f->setNonbondedMethod(periodic?CustomManyParticleForce::CutoffPeriodic:CustomManyParticleForce::NoCutoff);f->setCutoffDistance(0.9);
    std::vector<Vec3> p(8),v(8);for(int i=0;i<8;++i){s.addParticle(12);f->addParticle({0.2+0.03*i});p[i]=Vec3(0.14*(i%2)+0.008*std::sin(1.1*i),0.14*((i/2)%2)+0.007*std::cos(0.9*i),0.14*(i/4)+0.006*std::sin(1.3*i));v[i]=Vec3(0.02*std::sin(1.7*i),0.02*std::cos(0.9*i),0.01*std::sin(0.3*i));}
    auto* bonds=new HarmonicBondForce();for(int i=0;i<8;++i)for(int j=i+1;j<8;++j){Vec3 d=p[i]-p[j];bonds->addBond(i,j,std::sqrt(d.dot(d)),1000);}
    s.addForce(f);s.addForce(bonds);s.setDefaultPeriodicBoxVectors(Vec3(3,0,0),Vec3(0.4,3.2,0),Vec3(-0.2,0.3,3.4));VerletIntegrator it(0.0002);Context c(s,it,Platform::getPlatformByName("Reference"));c.setPositions(p);c.setVelocities(v);
    auto initial=c.getState(State::Energy);double e0=initial.getPotentialEnergy()+initial.getKineticEnergy(),drift=0,maxDistance=0;std::vector<Vec3> shortP,shortV;
    for(int sample=0;sample<250;++sample){it.step(100);auto state=c.getState(State::Energy|State::Positions);drift=std::max(drift,std::abs(state.getPotentialEnergy()+state.getKineticEnergy()-e0));auto q=state.getPositions();for(int i=0;i<8;++i)for(int j=i+1;j<8;++j){Vec3 d=q[i]-q[j];maxDistance=std::max(maxDistance,std::sqrt(d.dot(d)));}if(sample==4){auto shortState=c.getState(State::Positions|State::Velocities);shortP=shortState.getPositions();shortV=shortState.getVelocities();}}
    auto final=c.getState(State::Positions|State::Velocities);std::cout<<"{\"stability\":true,\"mode\":"<<mode<<",\"periodic\":"<<periodic<<",\"initial_energy\":"<<e0<<",\"max_absolute_drift\":"<<drift<<",\"max_distance\":"<<maxDistance<<",\"short_positions\":";vectors(shortP);std::cout<<",\"short_velocities\":";vectors(shortV);std::cout<<",\"positions\":";vectors(final.getPositions());std::cout<<",\"velocities\":";vectors(final.getVelocities());std::cout<<"}\n";
}
void boundaryProbes(){
    for(int next:{0,1}){System s;auto* f=new CustomManyParticleForce(2,"1");for(int i=0;i<2;++i){s.addParticle(1);f->addParticle();}double cutoff=std::sqrt(0.3*0.3+0.2*0.2);if(next)cutoff=std::nextafter(cutoff,1.0);f->setNonbondedMethod(CustomManyParticleForce::CutoffNonPeriodic);f->setCutoffDistance(cutoff);s.addForce(f);VerletIntegrator it(0.001);Context c(s,it,Platform::getPlatformByName("Reference"));c.setPositions({Vec3(),Vec3(0.3,0.2,0)});std::cout<<"{\"boundary_probe\":true,\"cutoff\":"<<cutoff<<",\"energy\":"<<c.getState(State::Energy).getPotentialEnergy()<<"}\n";}
}
void periodicUpdateProbe(){
    System s;auto* f=new CustomManyParticleForce(2,"distance(p1,p2)+t(0)");for(int i=0;i<2;++i){s.addParticle(1);f->addParticle();}f->addTabulatedFunction("t",new Continuous1DFunction({0,0},0,1));f->setNonbondedMethod(CustomManyParticleForce::CutoffPeriodic);f->setCutoffDistance(0.9);s.addForce(f);s.setDefaultPeriodicBoxVectors(Vec3(3,0,0),Vec3(0,3,0),Vec3(0,0,3));VerletIntegrator it(0.001);Context c(s,it,Platform::getPlatformByName("Reference"));c.setPositions({Vec3(),Vec3(2.8,0,0)});auto before=c.getState(State::Energy|State::Forces);f->setNonbondedMethod(CustomManyParticleForce::CutoffNonPeriodic);dynamic_cast<Continuous1DFunction&>(f->getTabulatedFunction(0)).setFunctionParameters({1,1},0,1);f->updateParametersInContext(c);auto after=c.getState(State::Energy|State::Forces);std::cout<<"{\"periodic_update_probe\":true,\"energies\":["<<before.getPotentialEnergy()<<','<<after.getPotentialEnergy()<<"],\"force_x\":["<<before.getForces()[0][0]<<','<<after.getForces()[0][0]<<"]}\n";
}
int main(){std::cout<<std::setprecision(17);for(int k=1;k<=4;++k)for(int mode=0;mode<2;++mode)for(int method=0;method<3;++method)for(int filter=0;filter<3;++filter)for(bool cv:{false,true})run(k,mode,method,filter,cv);probes();boundaryProbes();periodicUpdateProbe();for(int mode:{0,1})for(bool periodic:{false,true})stability(mode,periodic);}
