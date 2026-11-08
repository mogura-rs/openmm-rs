#include <ReferenceTabulatedFunction.h>
#include <openmm/Context.h>
#include <openmm/CustomNonbondedForce.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/TabulatedFunction.h>
#include <openmm/VerletIntegrator.h>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace OpenMM;
using Shape = std::array<int,3>;
using Bounds = std::array<std::array<double,2>,3>;

std::vector<double> tableValues(Shape shape, bool periodic, int phase) {
    std::vector<double> values;
    for (int z=0; z<shape[2]; ++z) for (int y=0; y<shape[1]; ++y) for (int x=0; x<shape[0]; ++x) {
        int i=periodic && shape[0]>1 ? x%(shape[0]-1):x;
        int j=periodic && shape[1]>1 ? y%(shape[1]-1):y;
        int k=periodic && shape[2]>1 ? z%(shape[2]-1):z;
        double value=std::sin(0.7*i+0.3*j-0.2*k)+0.2*i*j+0.05*j*k+0.3*std::cos(k);
        values.push_back((1+0.3*phase)*value+0.1*phase);
    }
    return values;
}

TabulatedFunction* create(int dim, bool discrete, bool periodic, Shape s, const std::vector<double>& v, Bounds b) {
    if (discrete) {
        if (dim==1) return new Discrete1DFunction(v);
        if (dim==2) return new Discrete2DFunction(s[0],s[1],v);
        return new Discrete3DFunction(s[0],s[1],s[2],v);
    }
    if (dim==1) return new Continuous1DFunction(v,b[0][0],b[0][1],periodic);
    if (dim==2) return new Continuous2DFunction(s[0],s[1],v,b[0][0],b[0][1],b[1][0],b[1][1],periodic);
    return new Continuous3DFunction(s[0],s[1],s[2],v,b[0][0],b[0][1],b[1][0],b[1][1],b[2][0],b[2][1],periodic);
}

void update(TabulatedFunction& f, int dim, bool discrete, Shape s, const std::vector<double>& v, Bounds b) {
    if (discrete) {
        if (dim==1) dynamic_cast<Discrete1DFunction&>(f).setFunctionParameters(v);
        if (dim==2) dynamic_cast<Discrete2DFunction&>(f).setFunctionParameters(s[0],s[1],v);
        if (dim==3) dynamic_cast<Discrete3DFunction&>(f).setFunctionParameters(s[0],s[1],s[2],v);
    } else {
        if (dim==1) dynamic_cast<Continuous1DFunction&>(f).setFunctionParameters(v,b[0][0],b[0][1]);
        if (dim==2) dynamic_cast<Continuous2DFunction&>(f).setFunctionParameters(s[0],s[1],v,b[0][0],b[0][1],b[1][0],b[1][1]);
        if (dim==3) dynamic_cast<Continuous3DFunction&>(f).setFunctionParameters(s[0],s[1],s[2],v,b[0][0],b[0][1],b[1][0],b[1][1],b[2][0],b[2][1]);
    }
}

template<class T> void jsonArray(const T& v) {
    std::cout << '[';
    for (size_t i=0; i<v.size(); ++i) { if (i) std::cout << ','; std::cout << v[i]; }
    std::cout << ']';
}

void run(int dim, bool discrete, bool periodic, bool small, int id) {
    Shape shape={1,1,1};
    for (int axis=0; axis<dim; ++axis) shape[axis]=small ? (discrete?1:(periodic?3:2)) : std::array<int,3>{5,4,6}[axis];
    Bounds bounds={{{-1,2},{-2,1},{-1.5,2.5}}};
    std::string expression="g*f(sin(r)+u";
    if (dim>1) expression+=",v+0.2*r";
    if (dim>2) expression+=",w-0.3*r";
    expression+=")+0.1*r^2";
    auto* force=new CustomNonbondedForce(expression);
    for (auto name:{"g","u","v","w"}) { force->addGlobalParameter(name,0); force->addEnergyParameterDerivative(name); }
    auto values=tableValues(shape,periodic,0);
    force->addTabulatedFunction("f",create(dim,discrete,periodic,shape,values,bounds));
    force->addParticle(); force->addParticle();
    System system; system.addParticle(1); system.addParticle(1); system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system,integrator,Platform::getPlatformByName("Reference"));
    const double radius=0.731;
    context.setPositions({Vec3(0,0,0),Vec3(radius,0,0)});
    context.setParameter("g",1.2);
    for (int phase=0; phase<3; ++phase) {
        if (phase) {
            if (phase==2) for (int axis=0; axis<dim; ++axis) { shape[axis]++; bounds[axis][0]-=0.25; bounds[axis][1]+=0.5; }
            values=tableValues(shape,periodic,phase);
            update(force->getTabulatedFunction(0),dim,discrete,shape,values,bounds);
            force->updateParametersInContext(context);
        }
        std::vector<std::array<double,3>> points={{0.17,-0.23,0.39},{0,0,0},{0.5,0.5,0.5},{1.5,1.5,1.5},{-0.5,-0.5,-0.5}};
        for (int axis=0; axis<dim; ++axis) for (double t:{-1.1,-1.0,-0.1,0.0,0.4,1.0,1.1,2.0}) {
            std::array<double,3> point={0.17,-0.23,0.39};
            point[axis]=discrete ? t*(shape[axis]-1) : bounds[axis][0]+t*(bounds[axis][1]-bounds[axis][0]);
            points.push_back(point);
        }
        for (size_t point=0; point<points.size(); ++point) {
            auto p=points[point];
            context.setParameter("u",p[0]-std::sin(radius));
            context.setParameter("v",p[1]-0.2*radius);
            context.setParameter("w",p[2]+0.3*radius);
            auto state=context.getState(State::Energy|State::Forces|State::ParameterDerivatives);
            std::cout << "{\"case\":" << id << ",\"phase\":" << phase << ",\"point\":" << point << ",\"dimension\":" << dim << ",\"discrete\":" << (discrete?"true":"false") << ",\"periodic\":" << (periodic?"true":"false") << ",\"expression\":\"" << expression << "\",\"shape\":";
            jsonArray(shape); std::cout << ",\"bounds\":[";
            for (int axis=0; axis<3; ++axis) { if (axis) std::cout << ','; jsonArray(bounds[axis]); }
            std::cout << "],\"values\":"; jsonArray(values);
            std::cout << ",\"coordinates\":"; jsonArray(p);
            std::cout << ",\"energy\":" << state.getPotentialEnergy() << ",\"forces\":[";
            for (int i=0; i<2; ++i) { if (i) std::cout << ','; const auto& f=state.getForces()[i]; std::cout << '[' << f[0] << ',' << f[1] << ',' << f[2] << ']'; }
            std::cout << "],\"derivatives\":{";
            bool first=true;
            for (const auto& d:state.getEnergyParameterDerivatives()) { if (!first) std::cout << ','; first=false; std::cout << '"' << d.first << "\":" << d.second; }
            std::cout << "}}\n";
        }
    }
}

int main() {
    std::cout << std::setprecision(17);
    int id=0;
    for (int dim=1; dim<=3; ++dim) for (bool small:{true,false}) {
        run(dim,false,false,small,id++);
        run(dim,false,true,small,id++);
        run(dim,true,false,small,id++);
    }
    // The source can round the final knot below the declared upper bound.
    Continuous1DFunction function({1,2,3,4,5,6},-1.1,2.2);
    ReferenceContinuous1DFunction compiled(function);
    const double argument=2.2;
    try {
        double energy=compiled.evaluate(&argument);
        std::cout << "{\"endpoint_roundoff\":true,\"source_rejected\":false,\"energy\":" << energy << "}\n";
    } catch (const std::exception& e) {
        std::cout << "{\"endpoint_roundoff\":true,\"source_rejected\":true,\"error\":\"" << e.what() << "\"}\n";
    }
}
