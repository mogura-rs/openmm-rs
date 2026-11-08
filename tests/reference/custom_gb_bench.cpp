#include <openmm/Context.h>
#include <openmm/CustomGBForce.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace OpenMM;
CustomGBForce* model(int kind);
int main(int argc,char** argv) {
    int n=std::stoi(argv[1]),iterations=std::stoi(argv[2]),method=std::stoi(argv[3]),kind=std::stoi(argv[4]);
    int grid=static_cast<int>(std::ceil(std::cbrt(n)));double side=0.3*grid;
    System system;system.setDefaultPeriodicBoxVectors(Vec3(side,0,0),Vec3(0,side,0),Vec3(0,0,side));
    auto* gb=model(kind);gb->setNonbondedMethod(static_cast<CustomGBForce::NonbondedMethod>(method));gb->setCutoffDistance(0.8);
    std::vector<Vec3> p(n);
    for(int i=0;i<n;++i){system.addParticle(1);gb->addParticle(kind==0?std::vector<double>{0.1*(i%7-3),0.11+0.02*(i%5),0.65+0.1*(i%4)}:std::vector<double>{0.01*(i%5)});p[i]=Vec3(0.3*(i%grid),0.3*((i/grid)%grid),0.3*(i/(grid*grid)));}
    system.addForce(gb);VerletIntegrator it(0.001);Context context(system,it,Platform::getPlatformByName("Reference"));context.setPositions(p);
    int flags=State::Forces|State::Energy;context.getState(flags);
    auto start=std::chrono::steady_clock::now();for(int i=0;i<iterations;++i)context.getState(flags);
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();auto state=context.getState(flags);
    double norm=0;for(auto f:state.getForces())norm+=f.dot(f);
    auto derivatives=context.getState(State::ParameterDerivatives).getEnergyParameterDerivatives();double derivativeNorm=0;for(auto d:derivatives)derivativeNorm+=d.second*d.second;
    std::cout<<std::setprecision(17)<<"{\"implementation\":\"openmm-reference\",\"precision\":\"f64\",\"api\":\"Context.getState\",\"force\":\"custom_gb\",\"model\":"<<kind<<",\"derivative_squared_norm\":"<<derivativeNorm<<",\"method\":"<<method<<",\"particles\":"<<n<<",\"iterations\":"<<iterations<<",\"seconds\":"<<seconds<<",\"evaluations_per_second\":"<<iterations/seconds<<",\"energy\":"<<state.getPotentialEnergy()<<",\"force_squared_norm\":"<<norm<<"}\n";
}

CustomGBForce* model(int kind) {
    auto f=new CustomGBForce();
    if(kind==0) {
        for(auto name:{"q","radius","scale"})f->addPerParticleParameter(name);
        f->addGlobalParameter("solvent",78.3);f->addGlobalParameter("solute",1.);
        f->addEnergyParameterDerivative("solvent");f->addEnergyParameterDerivative("solute");
        f->addComputedValue("I","step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);U=r+sr2;C=2*(1/or1-1/L)*step(sr2-r-or1);L=max(or1,D);D=abs(r-sr2);sr2=scale2*or2;or1=radius1-0.009;or2=radius2-0.009",CustomGBForce::ParticlePairNoExclusions);
        f->addComputedValue("B","1/(1/or-tanh(psi-0.8*psi^2+4.85*psi^3)/radius);psi=I*or;or=radius-0.009",CustomGBForce::SingleParticle);
        f->addEnergyTerm("28.391955111572114*(radius+0.14)^2*(radius/B)^6-0.5*138.93545764438198*(1/solute-1/solvent)*q^2/B",CustomGBForce::SingleParticle);
        f->addEnergyTerm("-138.93545764438198*(1/solute-1/solvent)*q1*q2/sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))",CustomGBForce::ParticlePair);
    } else {
        f->addPerParticleParameter("p");std::string sum;
        for(int i=0;i<24;++i){auto name="g"+std::to_string(i);f->addGlobalParameter(name,0.01*(i+1));f->addEnergyParameterDerivative(name);if(i)sum+="+";sum+=name;}
        f->addComputedValue("a","("+sum+")*exp(-r)*p2",CustomGBForce::ParticlePair);
        for(auto p:std::vector<std::pair<std::string,std::string>>{{"b","1+0.1*a*a"},{"c","log(b)+g0*a"},{"d","c+0.01*c*c+g1*b"},{"e","sqrt(1+d*d)+g2*a"},{"f","0.5*e+0.03*b"}})f->addComputedValue(p.first,p.second,CustomGBForce::SingleParticle);
        f->addEnergyTerm("0.01*f*f",CustomGBForce::SingleParticle);
        f->addEnergyTerm("0.001*(c1+c2)*exp(-r)",CustomGBForce::ParticlePair);
    }
    return f;
}
