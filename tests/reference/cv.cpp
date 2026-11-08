#include <openmm/Context.h>
#include <openmm/CustomBondForce.h>
#include <openmm/CustomCVForce.h>
#include <openmm/CustomVolumeForce.h>
#include <openmm/NonbondedForce.h>
#include <openmm/OrientationRestraintForce.h>
#include <openmm/Platform.h>
#include <openmm/RGForce.h>
#include <openmm/RMSDForce.h>
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
    for (size_t i=0;i<values.size();++i) { if (i) std::cout << ','; std::cout << '[' << values[i][0] << ',' << values[i][1] << ',' << values[i][2] << ']'; }
    std::cout << ']';
}
void scalars(const std::vector<double>& values) {
    std::cout << '[';
    for (size_t i=0;i<values.size();++i) { if (i) std::cout << ','; std::cout << values[i]; }
    std::cout << ']';
}
std::vector<Vec3> reference(int phase) {
    std::vector<Vec3> result;
    for (int i=0;i<8;++i) result.emplace_back(0.23*i+0.12*std::sin(i+phase),std::cos(1.1*i)+0.02*phase*i,std::sin(0.7*i)-0.03*phase*i);
    return result;
}
std::vector<double> tableValues(int phase) {
    std::vector<double> values;
    for (int i=0;i<9;++i) values.push_back((1+0.1*phase)*std::sin(0.5*i-2));
    return values;
}
CustomBondForce* bond() {
    auto* force=new CustomBondForce("g*k*r^2"); force->addGlobalParameter("g",2); force->addEnergyParameterDerivative("g"); force->addPerBondParameter("k");
    force->addBond(0,1,{0.4}); force->addBond(2,5,{0.6}); force->setForceGroup(17); return force;
}
CustomCVForce* build(int variant, int derivatives) {
    auto* cv=new CustomCVForce(variant==4?"scale+g^2":"scale*(a-b)^2+0.2*f(c)");
    cv->addGlobalParameter("scale",1.2); cv->addGlobalParameter("g",0.7);
    if (derivatives>0) cv->addEnergyParameterDerivative("scale");
    if (derivatives==2) cv->addEnergyParameterDerivative("g");
    if (variant!=4) {
        if (variant==0) {
            cv->addCollectiveVariable("a",bond()); cv->addCollectiveVariable("b",new RGForce({0,2,4,6}));
        } else if (variant==1) {
            cv->addCollectiveVariable("a",new OrientationRestraintForce(0.9,reference(0),{0,1,2,3,4})); cv->addCollectiveVariable("b",bond());
        } else if (variant==2) {
            auto* nb=new NonbondedForce(); nb->setNonbondedMethod(NonbondedForce::PME); nb->setCutoffDistance(1.2); nb->setPMEParameters(2.7,16,16,16); nb->setUseDispersionCorrection(false);
            nb->addGlobalParameter("g",2); nb->setForceGroup(17); nb->setReciprocalSpaceForceGroup(19);
            for (int i=0;i<8;++i) { nb->addParticle(i%2?-0.1:0.1,0.3,0.05); nb->addParticleParameterOffset("g",i,i%2?-0.01:0.01,0,0); }
            cv->addCollectiveVariable("a",nb);
            auto* volume=new CustomVolumeForce("0.01*v+0.1*g"); volume->addGlobalParameter("g",2); cv->addCollectiveVariable("b",volume);
        } else {
            auto* nested=new CustomCVForce("g*bb+0.2*rr^2"); nested->addGlobalParameter("g",2); nested->addEnergyParameterDerivative("g");
            nested->addCollectiveVariable("bb",bond()); nested->addCollectiveVariable("rr",new RGForce({0,1,2,3}));
            cv->addCollectiveVariable("a",nested); cv->addCollectiveVariable("b",new RGForce({0,2,4,6}));
        }
        cv->addCollectiveVariable("c",new RMSDForce(reference(0),{0,2,4,6}));
    }
    cv->addTabulatedFunction("f",new Continuous1DFunction(tableValues(0),-2,2)); cv->setForceGroup(11); return cv;
}
template<class F> F& innerForce(Context& context,int index) {
    return dynamic_cast<F&>(const_cast<Force&>(context.getSystem().getForce(index)));
}
void updateBond(Context& context,int index) {
    auto& force=innerForce<CustomBondForce>(context,index);
    force.setBondParameters(0,0,1,{0.8}); force.setBondParameters(1,2,5,{0.3}); force.updateParametersInContext(context);
}
void update(CustomCVForce& cv, Context& context, int variant) {
    auto& inner=cv.getInnerContext(context);
    if (variant==0) updateBond(inner,0);
    else if (variant==1) {
        auto& force=innerForce<OrientationRestraintForce>(inner,0); force.setK(1.3); force.setReferencePositions(reference(2)); force.updateParametersInContext(inner); updateBond(inner,1);
    } else if (variant==2) {
        auto& force=innerForce<NonbondedForce>(inner,0);
        for (int i=0;i<8;++i) force.setParticleParameters(i,i%2?-0.11:0.11,0.3,0.05);
        force.updateParametersInContext(inner);
    } else if (variant==3) {
        auto& nested=innerForce<CustomCVForce>(inner,0); updateBond(nested.getInnerContext(inner),0);
    }
    if (variant!=4) {
        auto& force=innerForce<RMSDForce>(inner,2); force.setReferencePositions(reference(2)); force.setParticles({1,3,5,7}); force.updateParametersInContext(inner);
    }
    dynamic_cast<Continuous1DFunction&>(cv.getTabulatedFunction(0)).setFunctionParameters(tableValues(2),-2,2);
    cv.setEnergyFunction("999"); cv.updateParametersInContext(context);
}
void run(int variant,int derivatives) {
    System system;
    for (int i=0;i<8;++i) system.addParticle(i==7?0:1+0.2*i);
    system.setDefaultPeriodicBoxVectors(Vec3(4,0,0),Vec3(0.3,4.2,0),Vec3(-0.2,0.4,4.4));
    auto* cv=build(variant,derivatives); system.addForce(cv);
    VerletIntegrator integrator(0.00005); Context context(system,integrator,Platform::getPlatformByName("Reference"));
    for (int phase=0;phase<3;++phase) {
        if (phase==1) { context.setParameter("g",1.1); context.setParameter("scale",0.8); }
        if (phase==2) update(*cv,context,variant);
        context.setPeriodicBoxVectors(Vec3(4+0.2*phase,0,0),Vec3(0.3,4.2+0.1*phase,0),Vec3(-0.2,0.4,4.4+0.15*phase));
        for (int point=0;point<4;++point) {
            auto positions=reference(0); std::vector<Vec3> velocities;
            double theta=0.3+0.6*point;
            for (int i=0;i<8;++i) {
                auto p=positions[i]; positions[i]=Vec3(std::cos(theta)*p[0]-std::sin(theta)*p[1]+0.3*point,std::sin(theta)*p[0]+std::cos(theta)*p[1]-0.2*point,p[2]+0.02*std::sin(i+0.3*point)+0.003*i*point);
                velocities.emplace_back(0.01*i,-0.02,0.03-0.004*i);
            }
            context.setPositions(positions); context.setVelocities(velocities); context.setTime(0);
            for (int steps:{0,3}) {
                if (steps) integrator.step(steps);
                auto state=context.getState(State::Energy|State::Forces|State::Positions|State::Velocities|State::ParameterDerivatives,false,1<<11);
                auto deriv=state.getEnergyParameterDerivatives(); std::vector<double> values; cv->getCollectiveVariableValues(context,values);
                std::cout << "{\"variant\":" << variant << ",\"derivatives\":" << derivatives << ",\"phase\":" << phase << ",\"point\":" << point << ",\"steps\":" << steps << ",\"initial_positions\":";
                vectors(positions); std::cout << ",\"initial_velocities\":"; vectors(velocities);
                std::cout << ",\"energy\":" << state.getPotentialEnergy() << ",\"kinetic\":" << state.getKineticEnergy() << ",\"derivative_g\":" << deriv["g"] << ",\"derivative_scale\":" << deriv["scale"] << ",\"values\":"; scalars(values);
                std::cout << ",\"forces\":"; vectors(state.getForces()); std::cout << ",\"positions\":"; vectors(state.getPositions()); std::cout << ",\"velocities\":"; vectors(state.getVelocities()); std::cout << "}\n";
            }
        }
    }
}
int main() {
    std::cout << std::setprecision(17);
    for (int variant=0;variant<5;++variant) for (int derivatives=0;derivatives<3;++derivatives) run(variant,derivatives);
}
