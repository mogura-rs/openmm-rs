#include <openmm/Context.h>
#include <openmm/CustomAngleForce.h>
#include <openmm/CustomBondForce.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/CustomNonbondedForce.h>
#include <openmm/CustomTorsionForce.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <iomanip>
#include <iostream>
#include <string>

using namespace OpenMM;

template<class F> void globals(F* force) {
    force->addGlobalParameter("g",1.3);
    force->addGlobalParameter("y",0.7);
    force->addGlobalParameter("r0",0.3);
    force->addEnergyParameterDerivative("g");
    force->addEnergyParameterDerivative("y");
    force->addEnergyParameterDerivative("r0");
}

void run(const std::string& kind, const std::string& expression, double radius, bool periodic=false) {
    System system;
    for (int i=0; i<4; ++i) system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(3,0,0),Vec3(0.4,3,0),Vec3(-0.2,0.3,3));
    if (kind=="bond") {
        auto* force=new CustomBondForce(expression); globals(force);
        force->addBond(0,1); force->setUsesPeriodicBoundaryConditions(periodic); system.addForce(force);
    } else if (kind=="angle") {
        auto* force=new CustomAngleForce(expression); globals(force);
        force->addPerAngleParameter("k"); force->addAngle(0,1,2,{0.8}); force->addAngle(1,2,3,{1.2});
        force->setUsesPeriodicBoundaryConditions(periodic); system.addForce(force);
    } else if (kind=="torsion") {
        auto* force=new CustomTorsionForce(expression); globals(force);
        force->addPerTorsionParameter("k"); force->addTorsion(0,1,2,3,{0.8});
        force->setUsesPeriodicBoundaryConditions(periodic); system.addForce(force);
    } else if (kind=="external") {
        auto* force=new CustomExternalForce(expression);
        force->addGlobalParameter("g",1.3); force->addGlobalParameter("y",0.7); force->addGlobalParameter("r0",0.3);
        force->addPerParticleParameter("k"); force->addParticle(0,{0.8}); force->addParticle(2,{1.2}); system.addForce(force);
    } else {
        auto* force=new CustomNonbondedForce(expression); globals(force);
        force->addPerParticleParameter("a");
        for (int i=0; i<4; ++i) force->addParticle({0.8+0.2*i});
        force->addExclusion(0,2);
        if (periodic) { force->setNonbondedMethod(CustomNonbondedForce::CutoffPeriodic); force->setCutoffDistance(1.4); }
        system.addForce(force);
    }
    VerletIntegrator integrator(0.001);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    std::vector<Vec3> positions={Vec3(0,0,0),Vec3(radius,0,0),Vec3(0.9,0.7,0.5),Vec3(1.2,0.8,1.4)};
    if (periodic) { positions[2]+=Vec3(0.4,3,0); positions[3]+=Vec3(-0.2,0.3,3); }
    context.setPositions(positions);
    for (int iteration=0; iteration<2; ++iteration) {
        if (iteration) { context.setParameter("g",0.9); context.setParameter("y",0.6); }
        State state=context.getState(State::Forces|State::Energy|State::ParameterDerivatives);
        std::cout << "{\"kind\":\"" << kind << "\",\"expression\":\"" << expression << "\",\"radius\":" << radius << ",\"periodic\":" << (periodic?"true":"false") << ",\"iteration\":" << iteration << ",\"energy\":" << state.getPotentialEnergy() << ",\"forces\":[";
        for (int i=0;i<4;++i) { if (i) std::cout << ','; const auto& f=state.getForces()[i]; std::cout << '[' << f[0] << ',' << f[1] << ',' << f[2] << ']'; }
        std::cout << "],\"derivatives\":{";
        bool first=true;
        if (kind!="external") for (const auto& d: state.getEnergyParameterDerivatives()) { if (!first) std::cout << ','; first=false; std::cout << '"' << d.first << "\":" << d.second; }
        std::cout << "}}\n";
    }
}

int main() {
    std::cout << std::setprecision(17);
    for (const std::string& expr: {"sin(x)","cos(x)","tan(x)","sec(x)","csc(x)","cot(x)","asin(x)","acos(x)","atan(x)","atan2(x,y)","sinh(x)","cosh(x)","tanh(x)","exp(x)","log(x)","sqrt(x)","erf(x)","erfc(x)","square(x)","cube(x)","recip(x)","x^y","x/y","x*y+x-y","min(x,y)","max(x,y)","abs(x)","floor(x)","ceil(x)","step(x)","delta(x)","select(x,y,r0)","x==y","x!=y","x<y","x<=y","x>y","x>=y","2+x<y","-x^2","2^3^2"}) {
        run("bond","g*("+expr+"); x=r-r0",0.7);
    }
    for (double radius: {0.1,0.3,1.0}) {
        for (const std::string& expr: {"abs(x)","min(x,y)","max(x,y)","step(x)","delta(x)","select(x,y,r0)","x^3","x^0","x==y","x!=y","x<y","x<=y","x>y","x>=y"}) run("bond","g*("+expr+"); x=r-r0",radius);
    }
    run("bond","g*(a*a+a); a=x*y; x=r-r0",0.7);
    run("bond","g*select(x,sqrt(y),r^2); x=r-r0",0.3);
    for (bool periodic: {false,true}) {
        run("angle","g*k*(sin(theta)+cos(theta*y)+tanh(theta-r0))",0.7,periodic);
        run("torsion","g*k*(1+cos(theta*y-r0))+erf(theta)",0.7,periodic);
        run("nonbonded","g*sqrt(a1*a2)*erf(r)+cos(r*y)+r0*r",0.7,periodic);
    }
    run("external","g*k*(x*x+2*y*y+3*z*z)+erfc(x-r0)",0.7);
}
