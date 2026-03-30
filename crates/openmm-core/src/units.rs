// OpenMM internal units: length=nm, energy=kJ/mol, mass=Da, time=ps, charge=e
// uom types are used at system boundaries (file I/O, Python API) for unit safety.
// Internal computation uses bare f64 with the unit documented in comments.

pub use uom::si::f64::{
    Energy,   // kJ/mol
    Length,   // nm
    Mass,     // Da
    Time,     // ps
    Velocity, // nm/ps
};

pub use uom::si::{
    length::nanometer, mass::dalton, molar_energy::kilojoule_per_mole, time::picosecond,
    velocity::meter_per_second,
};

/// Box vectors: three lattice vectors [a, b, c], each [x, y, z] in nm.
/// Row-major: bv[0] = a-vector, bv[1] = b-vector, bv[2] = c-vector.
pub type BoxVectors = [[f64; 3]; 3];
