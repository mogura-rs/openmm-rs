#include <openmm/Context.h>
#include <openmm/CustomNonbondedForce.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/TabulatedFunction.h>
#include <openmm/VerletIntegrator.h>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace OpenMM;

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (size_t i=0; i<values.size(); ++i) { if (i) std::cout << ','; const auto& v=values[i]; std::cout << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']'; }
    std::cout << ']';
}

void run(int id, int method, bool switching, int groups, bool computed, bool longrange) {
    auto expression = computed ? "g*q1*q2/r^6+s1*s2*exp(-r)" : "g*a1*a2/r^6+(1+u)*b1*b2*exp(-r)";
    auto* force=new CustomNonbondedForce(expression);
    force->setNonbondedMethod(static_cast<CustomNonbondedForce::NonbondedMethod>(method));
    force->setCutoffDistance(1.25);
    force->setUseSwitchingFunction(switching);
    force->setSwitchingDistance(0.75);
    force->setUseLongRangeCorrection(longrange);
    force->addGlobalParameter("g",1.2); force->addGlobalParameter("u",0.3);
    force->addPerParticleParameter("a"); force->addPerParticleParameter("b");
    if (computed) {
        force->addComputedValue("q","a*(1+u)+lookup(b)");
        force->addComputedValue("s","sqrt(b)*(1+0.1*g)");
        force->addTabulatedFunction("lookup",new Continuous1DFunction({0,0.04,0.05,0.08},0,0.3));
    } else { force->addEnergyParameterDerivative("g"); force->addEnergyParameterDerivative("u"); }
    System system;
    for (int i=0; i<6; ++i) { system.addParticle(1); force->addParticle({0.1*(1+i%3),0.04*(2+i%2)}); }
    force->addExclusion(0,1); force->addExclusion(2,4);
    if (groups==1 || groups==2) force->addInteractionGroup({0,1,2,3},{2,3,4,5});
    if (groups==2) { force->addInteractionGroup({0,2,4},{1,2,4,5}); force->addInteractionGroup({0,1,2,3},{2,3,4,5}); }
    if (groups==3) force->addInteractionGroup({}, {0,1,2,3,4,5});
    system.addForce(force);
    system.setDefaultPeriodicBoxVectors(Vec3(4,0,0),Vec3(0.3,4,0),Vec3(-0.2,0.4,4));
    VerletIntegrator integrator(0.001);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    std::vector<Vec3> positions={Vec3(0,0,0),Vec3(0.75,0,0),Vec3(1.25,0,0),Vec3(0.4,0.9,0.2),Vec3(0.9,0.6,0.85),Vec3(1.8,1.2,0.6)};
    if (method==2) { positions[1]+=Vec3(4,0,0); positions[4]+=Vec3(0.3,4,0); }
    context.setPositions(positions);
    for (int phase=0; phase<4; ++phase) {
        if (phase==1) { context.setParameter("g",0.9); context.setParameter("u",0.2); }
        if (phase==2) {
            for (int i=0; i<6; ++i) force->setParticleParameters(i,{0.1*(1+i%3)+0.02,0.04*(2+i%2)+0.012});
            if (computed) dynamic_cast<Continuous1DFunction&>(force->getTabulatedFunction(0)).setFunctionParameters({0.005,0.049,0.060,0.093},0,0.3);
            force->updateParametersInContext(context);
        }
        double scale=phase==3?1.2:1.0;
        context.setPeriodicBoxVectors(Vec3(4*scale,0,0),Vec3(0.3*scale,4*scale,0),Vec3(-0.2*scale,0.4*scale,4*scale));
        auto state=context.getState(State::Energy|State::Forces|State::ParameterDerivatives);
        std::cout << "{\"case\":" << id << ",\"method\":" << method << ",\"switching\":" << (switching?"true":"false") << ",\"groups\":" << groups << ",\"computed\":" << (computed?"true":"false") << ",\"longrange\":" << (longrange?"true":"false") << ",\"phase\":" << phase << ",\"expression\":\"" << expression << "\",\"positions\":";
        vectors(positions); std::cout << ",\"energy\":" << state.getPotentialEnergy() << ",\"forces\":"; vectors(state.getForces());
        std::cout << ",\"derivatives\":{";
        bool first=true;
        if (!computed) for (auto& d:state.getEnergyParameterDerivatives()) { if (!first) std::cout << ','; first=false; std::cout << '"' << d.first << "\":" << d.second; }
        std::cout << "}}\n";
    }
}

int main() {
    std::cout << std::setprecision(17);
    int id=0;
    for (int method=0; method<3; ++method) for (bool switching:{false,true}) for (int groups=0; groups<4; ++groups) for (bool computed:{false,true}) for (bool longrange:{false,true}) run(id++,method,switching,groups,computed,longrange);
}
