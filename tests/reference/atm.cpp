#include <openmm/ATMForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/CustomBondForce.h>
#include <openmm/CustomCVForce.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/NonbondedForce.h>
#include <openmm/RGForce.h>
#include <openmm/VirtualSite.h>
#include <openmm/Context.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using namespace OpenMM;
void vectors(const std::vector<Vec3>& values){std::cout<<'[';for(size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<'['<<values[i][0]<<','<<values[i][1]<<','<<values[i][2]<<']';}std::cout<<']';}
std::vector<Vec3> points(){return {Vec3(0.1,0.2,0.3),Vec3(0.55,0.13,0.24),Vec3(0.24,0.68,0.17),Vec3(0.64,0.59,0.72),Vec3(0.86,0.35,0.41),Vec3()};}
Force* child(bool periodic){
    auto* bond=new CustomBondForce("g*r*r+0.1*sin(r)");bond->addGlobalParameter("g",2.);bond->addBond(0,1);bond->addBond(2,4);bond->addBond(3,5);bond->setForceGroup(11);
    auto* nb=new NonbondedForce();nb->setNonbondedMethod(periodic?NonbondedForce::CutoffPeriodic:NonbondedForce::NoCutoff);nb->setCutoffDistance(1.2);nb->setUseDispersionCorrection(false);nb->setReactionFieldDielectric(1.);nb->setForceGroup(9);
    for(int i=0;i<6;++i)nb->addParticle(0.02*(i%3-1),0.1+0.004*i,0.015+0.001*i);nb->addException(0,2,0,0.1,0);
    auto* cv=new CustomCVForce("b+n+0.03*r");cv->addCollectiveVariable("b",bond);cv->addCollectiveVariable("n",nb);cv->addCollectiveVariable("r",new RGForce());return cv;
}
void transforms(ATMForce& force,int mode,int phase){
    for(int i=0;i<6;++i){
        if(mode==0){force.setParticleParameters(i,Vec3(0.05*std::sin(i+phase),0.02*i,-0.01*(i+phase)),Vec3(-0.01*i,0.02*phase,0.01*std::cos(i)));}
        else if(mode==1){force.setParticleTransformation(i,new ATMForce::ParticleOffsetDisplacement((i+1+phase)%5,(i+3)%5,i%2?(phase?1:0):-1,i%2?(phase?2:4):-1));}
        else if(i<2){force.setParticleTransformation(i,new ATMForce::ParticleOffsetDisplacement(1-i,i,phase?1-i:-1,phase?i:-1));}
        else force.setParticleParameters(i,Vec3(),Vec3());
    }
}
ATMForce* model(int kind,int mode,bool periodic,bool nested){
    ATMForce* force;
    if(kind<2){force=new ATMForce(kind==0?"u0+lambda*(u1-u0)":"0.7*u0+lambda*u1+0.003*u0*u1+0.1*g");force->addGlobalParameter("lambda",0.4);force->addEnergyParameterDerivative("lambda");}
    else {force=new ATMForce(0.2,0.7,0.15,-0.3,0.1,30.,kind==2?10.:-1.,0.5,kind==4?-1.:1.);force->addEnergyParameterDerivative(ATMForce::Lambda2());}
    force->addGlobalParameter("g",1.7);
    for(int i=0;i<6;++i)force->addParticle();transforms(*force,mode,0);
    if(nested){auto* inner=new ATMForce("0.6*u0+0.4*u1");for(int i=0;i<6;++i)inner->addParticle(Vec3(0.01*i,0.02*std::sin(i),-0.01));inner->addForce(child(periodic));inner->setForceGroup(12);force->addForce(inner);}
    else force->addForce(child(periodic));
    force->setForceGroup(5);return force;
}
void run(int kind,int mode,bool periodic,bool nested,bool outerCV){
    System system;for(int i=0;i<6;++i)system.addParticle(i>=4?0:12.);
    system.setVirtualSite(5,new OutOfPlaneSite(0,2,4,0.2,0.3,0.4));
    std::vector<Vec3> box={Vec3(3.4,0,0),Vec3(0.4,3.6,0),Vec3(-0.2,0.3,3.8)};system.setDefaultPeriodicBoxVectors(box[0],box[1],box[2]);
    ATMForce* atm=model(kind,mode,periodic,nested);CustomCVForce* cv=nullptr;
    if(outerCV){cv=new CustomCVForce("0.8*a+0.002*a*a");cv->addCollectiveVariable("a",atm);cv->setForceGroup(5);system.addForce(cv);}else system.addForce(atm);
    auto* external=new CustomExternalForce("0.01*g*x*x");external->addGlobalParameter("g",1.7);for(int i=0;i<6;++i)external->addParticle(i,{});external->setForceGroup(2);system.addForce(external);
    VerletIntegrator it(0.00001);Context c(system,it,Platform::getPlatformByName("Reference"));auto p=points();std::vector<Vec3> v(6);for(int i=0;i<4;++i)v[i]=Vec3(0.01*std::sin(i),0.02*std::cos(i),0.01*i);c.setPositions(p);c.computeVirtualSites();c.setVelocities(v);
    Context* inner=outerCV?&cv->getInnerContext(c):&c;
    if(outerCV)atm=&dynamic_cast<ATMForce&>(const_cast<Force&>(inner->getSystem().getForce(0)));
    for(int phase=0;phase<4;++phase){
        if(phase==1){transforms(*atm,mode,1);atm->setEnergyFunction("100*u1");atm->setGlobalParameterDefaultValue(0,9.);atm->updateParametersInContext(*inner);}
        if(phase==2){c.setParameter(kind<2?"lambda":ATMForce::Lambda2(),0.9);c.setParameter("g",1.1);box={Vec3(3.5,0,0),Vec3(-0.3,3.7,0),Vec3(0.1,-0.2,3.9)};c.setPeriodicBoxVectors(box[0],box[1],box[2]);if(periodic)p[3]+=box[1];c.setPositions(p);c.computeVirtualSites();}
        if(phase==3)it.step(3);
        for(int groups:{32,4,36}){
            auto state=c.getState(State::Energy|State::Positions|State::Velocities|State::Forces|State::ParameterDerivatives,false,groups);
            double u1,u0,energy;atm->getPerturbationEnergy(*inner,u1,u0,energy);
            std::cout<<"{\"kind\":"<<kind<<",\"mode\":"<<mode<<",\"periodic\":"<<periodic<<",\"nested\":"<<nested<<",\"cv\":"<<outerCV<<",\"phase\":"<<phase<<",\"groups\":"<<groups<<",\"energy\":"<<state.getPotentialEnergy()<<",\"kinetic\":"<<state.getKineticEnergy()<<",\"time\":"<<state.getTime()<<",\"positions\":";vectors(state.getPositions());std::cout<<",\"velocities\":";vectors(state.getVelocities());std::cout<<",\"forces\":";vectors(state.getForces());std::cout<<",\"perturbation\":["<<u1<<','<<u0<<','<<energy<<"],\"derivatives\":{";bool first=true;for(auto d:state.getEnergyParameterDerivatives()){if(!first)std::cout<<',';first=false;std::cout<<std::quoted(d.first)<<':'<<d.second;}std::cout<<"}}\n";
        }
        if(phase==0){
            auto original=c.getState(State::Positions).getPositions();std::vector<Vec3> forces(5);
            for(int i=0;i<5;++i)for(int axis=0;axis<3;++axis){auto q=original;q[i][axis]+=1e-6;c.setPositions(q);c.computeVirtualSites();double ep=c.getState(State::Energy,false,32).getPotentialEnergy();q=original;q[i][axis]-=1e-6;c.setPositions(q);c.computeVirtualSites();double em=c.getState(State::Energy,false,32).getPotentialEnergy();forces[i][axis]=-(ep-em)/2e-6;}
            c.setPositions(original);c.computeVirtualSites();std::cout<<"{\"fd\":true,\"forces\":";vectors(forces);std::cout<<"}\n";
        }
    }
}
void stability(int mode,bool periodic){
    System system;auto p=points();p[5]=Vec3(0.43,0.48,0.31);std::vector<Vec3> v(6);
    ATMForce* atm=mode==0?new ATMForce(0.2,0.7,0.15,-0.3,0.1,30,0,0.5,1):new ATMForce("0.3*u0+0.7*u1+0.003*u0*u1");
    auto* child=new CustomBondForce("g*r*r+0.1*sin(r)");child->addGlobalParameter("g",0.7);child->setUsesPeriodicBoundaryConditions(periodic);
    auto* bonds=new HarmonicBondForce();bonds->setUsesPeriodicBoundaryConditions(periodic);
    for(int i=0;i<6;++i){system.addParticle(12.);v[i]=Vec3(0.02*std::sin(1.7*i),0.02*std::cos(0.9*i),0.01*std::sin(0.3*i));atm->addParticle();
        if(mode==0)atm->setParticleParameters(i,Vec3(0.05*std::sin(i),0.02*i,-0.01*i),Vec3(-0.01*i,0,0.01*std::cos(i)));
        else if(i<2)atm->setParticleTransformation(i,new ATMForce::ParticleOffsetDisplacement(1-i,i));
        for(int j=0;j<i;++j){Vec3 d=p[i]-p[j];bonds->addBond(i,j,std::sqrt(d.dot(d)),500.);child->addBond(i,j);}}
    atm->addForce(child);system.addForce(atm);system.addForce(bonds);system.setDefaultPeriodicBoxVectors(Vec3(3,0,0),Vec3(0.3,3.2,0),Vec3(-0.2,0.1,3.4));
    VerletIntegrator it(0.0002);Context c(system,it,Platform::getPlatformByName("Reference"));c.setPositions(p);c.setVelocities(v);
    auto initial=c.getState(State::Energy);double e0=initial.getPotentialEnergy()+initial.getKineticEnergy(),drift=0;std::vector<Vec3> shortP,shortV;
    for(int sample=0;sample<250;++sample){it.step(100);auto state=c.getState(State::Energy);drift=std::max(drift,std::abs(state.getPotentialEnergy()+state.getKineticEnergy()-e0));if(sample==4){auto shortState=c.getState(State::Positions|State::Velocities);shortP=shortState.getPositions();shortV=shortState.getVelocities();}}
    auto state=c.getState(State::Positions|State::Velocities);std::cout<<"{\"stability\":true,\"mode\":"<<mode<<",\"periodic\":"<<periodic<<",\"initial_energy\":"<<e0<<",\"max_absolute_drift\":"<<drift<<",\"short_positions\":";vectors(shortP);std::cout<<",\"short_velocities\":";vectors(shortV);std::cout<<",\"positions\":";vectors(state.getPositions());std::cout<<",\"velocities\":";vectors(state.getVelocities());std::cout<<"}\n";
}

void derivativeProbe(const std::vector<std::string>& order){
    System system;for(int i=0;i<2;++i)system.addParticle(12.);
    auto* atm=new ATMForce("u0+lambda*(u1-u0)+0.5*k*k*u1");atm->addGlobalParameter("lambda",0.5);atm->addGlobalParameter("k",1.5);
    atm->addParticle(Vec3(),Vec3());atm->addParticle(Vec3(1,0,0),Vec3());for(const auto& name:order)atm->addEnergyParameterDerivative(name);
    auto* bond=new CustomBondForce("g*r*r");bond->addGlobalParameter("g",2.);bond->addBond(0,1);atm->addForce(bond);system.addForce(atm);
    VerletIntegrator it(0.001);Context c(system,it,Platform::getPlatformByName("Reference"));c.setPositions({Vec3(),Vec3(1,0,0)});
    auto state=c.getState(State::ParameterDerivatives);std::cout<<"{\"derivative_probe\":true,\"order\":[";
    for(size_t i=0;i<order.size();++i){if(i)std::cout<<',';std::cout<<std::quoted(order[i]);}
    std::cout<<"],\"derivatives\":{";bool first=true;for(auto d:state.getEnergyParameterDerivatives()){if(!first)std::cout<<',';first=false;std::cout<<std::quoted(d.first)<<':'<<d.second;}std::cout<<"}}\n";
}

int main(){std::cout<<std::setprecision(17);for(int kind=0;kind<5;++kind)for(int mode=0;mode<3;++mode)for(bool periodic:{false,true})for(bool nested:{false,true})for(bool cv:{false,true})run(kind,mode,periodic,nested,cv);for(int mode:{0,1})for(bool periodic:{false,true})stability(mode,periodic);for(const auto& order:std::vector<std::vector<std::string>>{{"lambda"},{"k"},{"lambda","k"},{"k","lambda"}})derivativeProbe(order);}
