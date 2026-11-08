#include <openmm/Context.h>
#include <openmm/CustomHbondForce.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/TabulatedFunction.h>
#include <openmm/VerletIntegrator.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace OpenMM;

void vectors(const std::vector<Vec3>& values) {
    std::cout << '[';
    for (size_t i=0; i<values.size(); ++i) { if (i) std::cout << ','; std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']'; }
    std::cout << ']';
}
std::vector<double> tableValues(int phase) {
    std::vector<double> result;
    for (int i=0;i<9;++i) result.push_back((1+0.2*phase)*std::sin(i-4.0));
    return result;
}

void run(int variant, int method, bool excluded) {
    int expression=variant<6 ? variant : (variant==6?0:(variant==7?3:5));
    bool omitted=variant>=6;
    std::vector<std::string> expressions={"distance(d1,a1)^2","angle(d2,d1,a1)","sin(dihedral(d3,d2,a2,a3))","f(distance(d1,a1))","distance(d2,a2)*angle(a1,a2,a3)+cos(dihedral(a3,a2,d1,d2))","distance(d1,a1)+x1"};
    std::string energy="g*kd*ka*("+expressions[expression]+")";
    System system;
    for (int i=0;i<16;++i) system.addParticle(1+0.2*i);
    system.setDefaultPeriodicBoxVectors(Vec3(6,0,0),Vec3(0.3,6.2,0),Vec3(-0.2,0.4,6.4));
    auto* force=new CustomHbondForce(energy);
    force->addGlobalParameter("g",1.3); force->addGlobalParameter("x1",0.2);
    force->addPerDonorParameter("kd"); force->addPerAcceptorParameter("ka");
    std::vector<std::vector<int>> donors={{0,1,2},{3,4,5},{0,4,2}},acceptors={{6,7,8},{9,10,11},{12,13,14}};
    for (int i=0;i<3;++i) {
        force->addDonor(donors[i][0],omitted?-1:donors[i][1],omitted?-1:donors[i][2],{0.7+0.1*i});
        force->addAcceptor(acceptors[i][0],omitted?-1:acceptors[i][1],omitted?-1:acceptors[i][2],{0.4+0.2*i});
    }
    if (excluded) { force->addExclusion(0,0); force->addExclusion(2,2); }
    force->setNonbondedMethod((CustomHbondForce::NonbondedMethod)method); force->setCutoffDistance(1.4);
    auto* table=new Continuous1DFunction(tableValues(0),-4,4); force->addTabulatedFunction("f",table);
    force->setForceGroup(9); system.addForce(force);
    VerletIntegrator integrator(0.0001);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    for (int phase=0;phase<3;++phase) {
        if (phase==1) { context.setParameter("g",0.8); context.setParameter("x1",0.4); }
        if (phase==2) {
            for (int i=0;i<3;++i) {
                force->setDonorParameters(i,donors[i][0],omitted?-1:donors[i][1],omitted?-1:donors[i][2],{1.2-0.1*i});
                force->setAcceptorParameters(i,acceptors[i][0],omitted?-1:acceptors[i][1],omitted?-1:acceptors[i][2],{0.8-0.1*i});
            }
            if (excluded) force->setExclusionParticles(0,1,1);
            table->setFunctionParameters(tableValues(2),-4,4); force->updateParametersInContext(context);
        }
        double scale=1+0.1*phase;
        context.setPeriodicBoxVectors(Vec3(6*scale,0,0),Vec3(0.3,6.2*scale,0),Vec3(-0.2,0.4,6.4*scale));
        for (int point=0;point<4;++point) {
            std::vector<Vec3> positions;
            for (int i=0;i<16;++i) positions.push_back(Vec3(0.15*((7*i+1)%17)+0.013*std::sin(0.7*i+0.3),0.13*((11*i+3)%19)+0.017*std::cos(0.3*i+0.7),0.11*((5*i+2)%13)+0.019*std::sin(1.3*i+0.2)));
            positions[6]+=Vec3(0.1*point,-0.07*point,0.04*point);
            if (point>=2) { for (int i:{0,1,2}) positions[i]+=Vec3(6,0,0); for (int i:{9,10,11}) positions[i]+=Vec3(0.3,6.2,0); }
            std::vector<Vec3> velocities;
            for (int i=0;i<16;++i) velocities.push_back(Vec3(0.01*i,-0.02,0.03-0.004*i));
            context.setPositions(positions); context.setVelocities(velocities); context.setTime(0);
            for (int steps:{0,3}) {
                if (steps) integrator.step(steps);
                auto state=context.getState(State::Energy|State::Forces|State::Positions|State::Velocities,false,1<<9);
                std::cout << "{\"variant\":" << variant << ",\"method\":" << method << ",\"excluded\":" << (excluded?"true":"false") << ",\"energy_function\":\"" << energy << "\",\"phase\":" << phase << ",\"point\":" << point << ",\"steps\":" << steps << ",\"initial_positions\":";
                vectors(positions); std::cout << ",\"initial_velocities\":"; vectors(velocities);
                std::cout << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy() << ",\"forces\":"; vectors(state.getForces());
                std::cout << ",\"positions\":"; vectors(state.getPositions()); std::cout << ",\"velocities\":"; vectors(state.getVelocities()); std::cout << "}\n";
            }
        }
    }
}
int main() {
    std::cout << std::setprecision(17);
    for (int variant=0;variant<9;++variant) for (int method=0;method<3;++method) for (bool excluded:{false,true}) run(variant,method,excluded);
}
