#include <openmm/Context.h>
#include <openmm/NonbondedForce.h>
#include <openmm/HarmonicBondForce.h>
#include <openmm/System.h>
#include <openmm/CustomGBForce.h>
#include <openmm/CustomCVForce.h>
#include <openmm/TabulatedFunction.h>
#include <openmm/VirtualSite.h>
#include <openmm/VerletIntegrator.h>
#include <openmm/Platform.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>
using namespace OpenMM;
using namespace std;
void vectors(const vector<Vec3>& p) {cout<<'[';for(int i=0;i<p.size();++i){if(i)cout<<',';cout<<'['<<p[i][0]<<','<<p[i][1]<<','<<p[i][2]<<']';}cout<<']';}
vector<double> params(int model,int i) {
    if(model==1)return {0.1*(i%7-3),0.11+0.02*(i%5),0.65+0.1*(i%4)};
    return {model==0 ? 0.2+0.1*i : double(i%3)};
}
CustomGBForce* model(int kind,int exclusion) {
    auto f=new CustomGBForce();
    auto pair=exclusion?CustomGBForce::ParticlePairNoExclusions:CustomGBForce::ParticlePair;
    if(kind==1) {
        for(auto name:{"q","radius","scale"})f->addPerParticleParameter(name);
        f->addGlobalParameter("solvent",78.3);f->addGlobalParameter("solute",1.0);
        f->addEnergyParameterDerivative("solvent");f->addEnergyParameterDerivative("solute");
        f->addComputedValue("I","step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);U=r+sr2;C=2*(1/or1-1/L)*step(sr2-r-or1);L=max(or1,D);D=abs(r-sr2);sr2=scale2*or2;or1=radius1-0.009;or2=radius2-0.009",pair);
        f->addComputedValue("B","1/(1/or-tanh(psi-0.8*psi^2+4.85*psi^3)/radius);psi=I*or;or=radius-0.009",CustomGBForce::SingleParticle);
        f->addEnergyTerm("28.391955111572114*(radius+0.14)^2*(radius/B)^6-0.5*138.93545764438198*(1/solute-1/solvent)*q^2/B",CustomGBForce::SingleParticle);
        f->addEnergyTerm("-138.93545764438198*(1/solute-1/solvent)*q1*q2/sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))",CustomGBForce::ParticlePair);
    } else {
        f->addPerParticleParameter("p");f->addGlobalParameter("lambda",0.7);f->addGlobalParameter("beta",0.2);
        f->addEnergyParameterDerivative("lambda");f->addEnergyParameterDerivative("beta");
        if(kind==0) {
            f->addComputedValue("a","lambda*(p1+2*p2)*exp(-r)+beta*r^2",pair);
            f->addComputedValue("b","exp(-0.1*a)+p*x*y+lambda*z",CustomGBForce::SingleParticle);
            f->addComputedValue("c","a*b+beta*b^2+0.2*sin(x)",CustomGBForce::SingleParticle);
            f->addComputedValue("d","c+sqrt(b*b+1)+0.3*a",CustomGBForce::SingleParticle);
            f->addEnergyTerm("0.15*d^2+lambda*b+0.2*x*x",CustomGBForce::SingleParticle);
            f->addEnergyTerm("0.03*(c1+c2)*r+beta*p1*p2/(1+r)+0.07*a1*a2",CustomGBForce::ParticlePair);
            f->addEnergyTerm("0.04*b1*b2*cos(r)",CustomGBForce::ParticlePairNoExclusions);
        } else {
            f->addTabulatedFunction("t",new Continuous1DFunction({0.,0.2,0.1,0.4,0.3},-1,3));
            vector<double> data2,data3;for(int j=0;j<3;++j)for(int i=0;i<3;++i)data2.push_back(0.1*(i+2*j)+0.02*i*j);
            for(int k=0;k<3;++k)for(int j=0;j<3;++j)for(int i=0;i<3;++i)data3.push_back(0.05*(i+2*j+3*k)+0.01*i*j*k);
            f->addTabulatedFunction("t2",new Continuous2DFunction(3,3,data2,0,3,-1,1));
            f->addTabulatedFunction("t3",new Continuous3DFunction(3,3,3,data3,0,3,-1,1,-1,1));
            f->addTabulatedFunction("d1",new Discrete1DFunction({0.1,0.2,-0.1}));
            f->addTabulatedFunction("d2",new Discrete2DFunction(3,3,data2));
            f->addTabulatedFunction("d3",new Discrete3DFunction(3,3,3,data3));
            f->addComputedValue("a","lambda*(t(r)+t2(r,0.3*p2)+t3(r,0.2*p1,0.2*p2))+beta*(d1(p2)+d2(p1,p2)+d3(p1,p2,1))",pair);
            f->addComputedValue("b","sqrt(1+a*a)+lambda*t(x)+beta*y",CustomGBForce::SingleParticle);
            f->addEnergyTerm("0.1*b*b+t(x)+beta*z*z",CustomGBForce::SingleParticle);
            f->addEnergyTerm("0.1*(a1+a2)*exp(-r)+lambda*b1*b2",CustomGBForce::ParticlePairNoExclusions);
        }
    }
    for(int i=0;i<8;++i)f->addParticle(params(kind,i));
    f->addExclusion(0,1);f->addExclusion(3,5);return f;
}
void run(int kind,int method,int exclusion,int nested,int sample) {
    System system; for(int i=0;i<8;++i)system.addParticle(i==0||i==7?0:2+i);
    system.setVirtualSite(7,new TwoParticleAverageSite(5,6,0.3,0.7));
    Vec3 box[3]={Vec3(2.6,0,0),Vec3(sample%2?0.5:0,2.8,0),Vec3(sample%2?-0.3:0,sample%2?0.4:0,3.0)};
    system.setDefaultPeriodicBoxVectors(box[0],box[1],box[2]);
    auto f=model(kind,exclusion);f->setNonbondedMethod(CustomGBForce::NonbondedMethod(method));f->setCutoffDistance(1.1);
    CustomCVForce* cv=nullptr;
    if(nested){cv=new CustomCVForce("1.3*g+0.0007*g*g");cv->addCollectiveVariable("g",f);for(int i=0;i<f->getNumGlobalParameters();++i){cv->addGlobalParameter(f->getGlobalParameterName(i),f->getGlobalParameterDefaultValue(i));cv->addEnergyParameterDerivative(f->getGlobalParameterName(i));}cv->setForceGroup(3);system.addForce(cv);}
    else {f->setForceGroup(3);system.addForce(f);}
    VerletIntegrator it(0.0002);it.setIntegrationForceGroups(8);
    Context context(system,it,Platform::getPlatformByName("Reference"));
    vector<Vec3> p,v;
    for(int i=0;i<8;++i){p.push_back(Vec3(0.1+0.31*(i%3)+0.013*sample,0.15+0.37*((i/3)%3)+0.007*sample*i,0.2+0.09*(i%2)+0.011*sample));v.push_back(Vec3(0.001*i,-0.002*i,0.001*(i%3)));}
    context.setPositions(p);context.setVelocities(v);context.computeVirtualSites();
    if(sample<2) {
        auto original=context.getState(State::Positions).getPositions();vector<Vec3> fd(7);
        for(int i=0;i<7;++i)for(int axis=0;axis<3;++axis){
            auto plus=original,minus=original;plus[i][axis]+=1e-6;minus[i][axis]-=1e-6;
            context.setPositions(plus);context.computeVirtualSites();double ep=context.getState(State::Energy).getPotentialEnergy();
            context.setPositions(minus);context.computeVirtualSites();double em=context.getState(State::Energy).getPotentialEnergy();
            fd[i][axis]=-(ep-em)/2e-6;
        }
        context.setPositions(original);context.computeVirtualSites();
        cout<<"{\"gradient\":true,\"kind\":"<<kind<<",\"method\":"<<method<<",\"exclusion\":"<<exclusion<<",\"nested\":"<<nested<<",\"sample\":"<<sample<<",\"forces\":";vectors(fd);cout<<",\"derivatives\":{";
        for(int i=0;i<f->getNumGlobalParameters();++i){auto name=f->getGlobalParameterName(i);double value=context.getParameter(name),h=1e-5;
            context.setParameter(name,value+h);double ep=context.getState(State::Energy).getPotentialEnergy();
            context.setParameter(name,value-h);double em=context.getState(State::Energy).getPotentialEnergy();context.setParameter(name,value);
            if(i)cout<<',';cout<<'"'<<name<<"\":"<<(ep-em)/(2*h);
        }
        cout<<"}}\n";
    }
    for(int phase=0;phase<4;++phase){
        if(phase==1)it.step(3);
        if(phase==2){
            context.setParameter(kind==1?"solvent":"lambda",kind==1?55.:0.9);
            context.setParameter(kind==1?"solute":"beta",kind==1?1.3:0.3);
            auto& inner=nested?cv->getInnerContext(context):context;
            auto& bound=nested?dynamic_cast<CustomGBForce&>(const_cast<Force&>(inner.getSystem().getForce(0))):*f;
            auto parameters=params(kind,2);parameters[0]+=(kind==1?0.07:(kind==2?1.:0.2));bound.setParticleParameters(2,parameters);
            if(kind==2)dynamic_cast<Continuous1DFunction&>(bound.getTabulatedFunction(0)).setFunctionParameters({0.1,0.3,0.2,0.5,0.4},-1,3);
            bound.updateParametersInContext(inner);
            it.step(2);
        }
        if(phase==3){box[0]*=1.03;box[1]*=1.02;context.setPeriodicBoxVectors(box[0],box[1],box[2]);it.step(1);}
        for(int groups:{8,16,-1}){
            auto state=context.getState(State::Positions|State::Velocities|State::Forces|State::Energy|State::ParameterDerivatives,false,groups);
            cout<<"{\"kind\":"<<kind<<",\"method\":"<<method<<",\"exclusion\":"<<exclusion<<",\"nested\":"<<nested<<",\"sample\":"<<sample<<",\"phase\":"<<phase<<",\"groups\":"<<groups<<",\"energy\":"<<state.getPotentialEnergy()<<",\"kinetic\":"<<state.getKineticEnergy()<<",\"time\":"<<state.getTime()<<",\"positions\":";vectors(state.getPositions());cout<<",\"velocities\":";vectors(state.getVelocities());cout<<",\"forces\":";vectors(state.getForces());cout<<",\"derivatives\":{";bool first=true;for(auto& d:state.getEnergyParameterDerivatives()){if(!first)cout<<',';first=false;cout<<'"'<<d.first<<"\":"<<d.second;}cout<<"}}\n";
        }
    }
}
void stability(int method, bool crossing=false) {
    System system;for(int i=0;i<8;++i)system.addParticle(12.);
    system.setDefaultPeriodicBoxVectors(Vec3(3,0,0),Vec3(0,3,0),Vec3(0,0,3));
    auto f=model(1,1);f->setNonbondedMethod(CustomGBForce::NonbondedMethod(method));f->setCutoffDistance(1.1);system.addForce(f);
    auto nb=new NonbondedForce();nb->setNonbondedMethod(method==2?NonbondedForce::CutoffPeriodic:NonbondedForce::NoCutoff);nb->setCutoffDistance(1.1);nb->setReactionFieldDielectric(1);nb->setUseDispersionCorrection(false);
    vector<Vec3> p,v;for(int i=0;i<8;++i){p.push_back(Vec3(0.2+0.33*(i%3),0.2+0.35*(i/3),0.2+0.08*(i%2)));v.push_back(Vec3(0.01*sin(i),0.01*cos(i),0.005*sin(2*i)));nb->addParticle(params(1,i)[0],0.23,0.05);}
    system.addForce(nb);auto bonds=new HarmonicBondForce();
    if(method==2&&!crossing){for(int i=0;i<8;++i)for(int j=i+1;j<8;++j){auto d=p[j]-p[i];bonds->addBond(i,j,sqrt(d.dot(d)),500.);}}
    else {for(int i=0;i<7;++i){auto d=p[i+1]-p[i];bonds->addBond(i,i+1,sqrt(d.dot(d)),150.);}}
    system.addForce(bonds);
    VerletIntegrator it(0.0001);Context c(system,it,Platform::getPlatformByName("Reference"));c.setPositions(p);c.setVelocities(v);
    auto initial=c.getState(State::Energy);double e0=initial.getPotentialEnergy()+initial.getKineticEnergy(),drift=0,maxPair=0;State shortState;
    for(int sample=0;sample<500;++sample){it.step(100);auto st=c.getState(State::Energy|State::Positions);drift=max(drift,abs(st.getPotentialEnergy()+st.getKineticEnergy()-e0));for(int i=0;i<8;++i)for(int j=i+1;j<8;++j){auto d=st.getPositions()[j]-st.getPositions()[i];maxPair=max(maxPair,sqrt(d.dot(d)));}if(sample==9)shortState=c.getState(State::Positions|State::Velocities);}
    auto state=c.getState(State::Positions|State::Velocities|State::Energy);
    cout<<"{\"stability\":true,\"method\":"<<method<<",\"boundary_crossing\":"<<(crossing?"true":"false")<<",\"maximum_pair_distance\":"<<maxPair<<",\"initial_energy\":"<<e0<<",\"max_absolute_drift\":"<<drift<<",\"positions\":";vectors(state.getPositions());cout<<",\"velocities\":";vectors(state.getVelocities());cout<<",\"short_positions\":";vectors(shortState.getPositions());cout<<",\"short_velocities\":";vectors(shortState.getVelocities());cout<<"}\n";
}
void apiProbe() {
    System system;system.addParticle(1);system.addParticle(1);auto f=new CustomGBForce();f->addParticle({});f->addParticle({});
    f->addComputedValue("a","r",CustomGBForce::ParticlePair);f->addEnergyTerm("a+t(x)",CustomGBForce::SingleParticle);f->addFunction("t",{0,1,2,3},0,3);system.addForce(f);
    VerletIntegrator it(0.001);Context c(system,it,Platform::getPlatformByName("Reference"));c.setPositions({Vec3(0.5,0,0),Vec3(1.5,0,0)});
    vector<double> energies;energies.push_back(c.getState(State::Energy).getPotentialEnergy());
    f->setComputedValueParameters(0,"a","2*r",CustomGBForce::ParticlePair);f->setEnergyTermParameters(0,"3*a+t(x)",CustomGBForce::SingleParticle);f->updateParametersInContext(c);energies.push_back(c.getState(State::Energy).getPotentialEnergy());
    f->setFunctionParameters(0,"t",{0,2,4,6},0,3);f->updateParametersInContext(c);energies.push_back(c.getState(State::Energy).getPotentialEnergy());
    f->setFunctionParameters(0,"u",{0,3,6,9,12},0,4);f->setEnergyTermParameters(0,"3*a+u(x)",CustomGBForce::SingleParticle);f->updateParametersInContext(c);energies.push_back(c.getState(State::Energy).getPotentialEnergy());
    cout<<"{\"api_probe\":true,\"energies\":[";for(int i=0;i<energies.size();++i){if(i)cout<<',';cout<<energies[i];}cout<<"]}\n";
}
int main(){cout<<setprecision(17);for(int kind=0;kind<3;++kind)for(int method=0;method<3;++method)for(int exclusion=0;exclusion<2;++exclusion)for(int nested=0;nested<2;++nested)for(int sample=0;sample<6;++sample)run(kind,method,exclusion,nested,sample);stability(0);stability(2);stability(2,true);apiProbe();}
