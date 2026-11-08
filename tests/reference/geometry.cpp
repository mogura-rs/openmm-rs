#include <openmm/Context.h>
#include <openmm/CustomCompoundBondForce.h>
#include <openmm/CustomExternalForce.h>
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
void run(int kind, int expression, bool periodic) {
    std::vector<std::string> compound={"distance(p1,p2)^2", "angle(p1,p2,p3)", "sin(dihedral(p1,p2,p3,p4))", "cos(dihedral(p4,p3,p2,p1))", "distance(p1,p4)*angle(p2,p3,p4)", "pointdistance(x1+0.2,y1,z1,x3,y3,z3)", "pointangle(x1,y1,z1,x2,y2+0.1,z2,x4,y4,z4)", "pointdihedral(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4)", "f(distance(p1,p3))", "distance(p1,p1)", "pointdistance(x1,y1,z1,x2+distance(p2,p3),y2,z2)", "x1*y2-z3*x4"};
    std::vector<std::string> external={"periodicdistance(x,y,z,a,b,c)^2", "sin(periodicdistance(x,y,z,a,b,c))", "periodicdistance(2*x,y+0.1,z,a,b,c)", "periodicdistance(x,y,z,x,y,z)", "periodicdistance(x,y,z,a,b,c)+0.01*x*y", "periodicdistance(x+periodicdistance(x,y,z,a,b,c),y,z,a,b,c)"};
    std::string energy=kind==0 ? "g*k*("+compound[expression]+")+0.02*x1*z4" : "g*k*("+external[expression]+")";
    System system;
    for (int i=0;i<8;++i) system.addParticle(1+0.2*i);
    system.setDefaultPeriodicBoxVectors(Vec3(4,0,0),Vec3(0.3,4.2,0),Vec3(-0.2,0.4,4.4));
    CustomCompoundBondForce* cf=nullptr;
    CustomExternalForce* ef=nullptr;
    Continuous1DFunction* table=nullptr;
    if (kind==0) {
        cf=new CustomCompoundBondForce(4,energy);
        cf->addGlobalParameter("g",1.3); cf->addEnergyParameterDerivative("g"); cf->addPerBondParameter("k");
        cf->addBond({0,1,2,3},{0.7}); cf->addBond({2,4,5,6},{0.9});
        cf->setUsesPeriodicBoundaryConditions(periodic);
        table=new Continuous1DFunction(tableValues(0),-4,4); cf->addTabulatedFunction("f",table);
        cf->setForceGroup(9); system.addForce(cf);
    } else {
        ef=new CustomExternalForce(energy); ef->addGlobalParameter("g",1.3);
        for (auto name:{"k","a","b","c"}) ef->addPerParticleParameter(name);
        ef->addParticle(0,{0.7,0.1,0.2,0.3}); ef->addParticle(3,{0.9,-0.2,0.4,0.1});
        ef->setForceGroup(9); system.addForce(ef);
    }
    VerletIntegrator integrator(0.0001);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    for (int phase=0;phase<3;++phase) {
        if (phase==1) context.setParameter("g",0.8);
        if (phase==2) {
            if (cf) { cf->setBondParameters(0,{0,1,2,3},{1.2}); cf->setBondParameters(1,{2,4,5,6},{0.4}); table->setFunctionParameters(tableValues(2),-4,4); cf->updateParametersInContext(context); }
            else { ef->setParticleParameters(0,0,{1.2,-0.4,0.1,0.3}); ef->setParticleParameters(1,3,{0.4,-0.2,0.4,-0.1}); ef->updateParametersInContext(context); }
        }
        double scale=1+0.1*phase;
        context.setPeriodicBoxVectors(Vec3(4*scale,0,0),Vec3(0.3,4.2*scale,0),Vec3(-0.2,0.4,4.4*scale));
        for (int point=0;point<4;++point) {
            double phi=-2.7+1.7*point;
            std::vector<Vec3> positions={Vec3(0,1,0),Vec3(0,0,0),Vec3(1,0,0),Vec3(1,std::cos(phi),std::sin(phi)),Vec3(0.3,1.7,0.2),Vec3(1.2,1.6,0.4),Vec3(1.7,2.3,1.1),Vec3(2,2,2)};
            if (periodic) { positions[0]+=Vec3(4,0,0); positions[3]+=Vec3(0.3,4.2,0); positions[6]+=Vec3(-0.2,0.4,4.4); }
            std::vector<Vec3> velocities;
            for (int i=0;i<8;++i) velocities.push_back(Vec3(0.01*i,-0.02,0.03-0.004*i));
            context.setPositions(positions); context.setVelocities(velocities); context.setTime(0);
            for (int steps:{0,3}) {
                if (steps) integrator.step(steps);
                auto state=context.getState(State::Energy|State::Forces|State::Positions|State::Velocities|State::ParameterDerivatives,false,1<<9);
                std::cout << "{\"kind\":" << kind << ",\"expression\":" << expression << ",\"energy_function\":\"" << energy << "\",\"periodic\":" << (periodic?"true":"false") << ",\"phase\":" << phase << ",\"point\":" << point << ",\"steps\":" << steps << ",\"initial_positions\":";
                vectors(positions); std::cout << ",\"initial_velocities\":"; vectors(velocities);
                std::cout << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy() << ",\"parameter_derivative\":" << state.getEnergyParameterDerivatives().at("g") << ",\"forces\":"; vectors(state.getForces());
                std::cout << ",\"positions\":"; vectors(state.getPositions()); std::cout << ",\"velocities\":"; vectors(state.getVelocities()); std::cout << "}\n";
            }
        }
    }
}
int main() {
    std::cout << std::setprecision(17);
    for (int kind=0;kind<2;++kind) for (int expression=0;expression<(kind==0?12:6);++expression) for (bool periodic:{false,true}) run(kind,expression,periodic);
}
