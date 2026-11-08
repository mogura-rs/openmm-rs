use crate::forces::harmonic_bond::apply_minimum_image_ortho;
use crate::forces::periodic_torsion::{cross3, dot3};

#[derive(Debug, Clone, Copy)]
pub(super) enum GeometryFunction {
    Distance,
    Angle,
    Dihedral,
}

impl GeometryFunction {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "pointdistance" | "periodicdistance" => Some(Self::Distance),
            "pointangle" => Some(Self::Angle),
            "pointdihedral" => Some(Self::Dihedral),
            _ => None,
        }
    }

    pub fn arity(self) -> usize {
        match self {
            Self::Distance => 6,
            Self::Angle => 9,
            Self::Dihedral => 12,
        }
    }

    pub fn evaluate(self, arguments: &[f64], vectors: Option<&[[f64; 3]; 3]>) -> (f64, [f64; 12]) {
        let delta = |a: usize, b: usize| {
            let mut d =
                std::array::from_fn(|axis| arguments[3 * a + axis] - arguments[3 * b + axis]);
            if let Some(vectors) = vectors {
                apply_minimum_image_ortho(&mut d, vectors);
            }
            d
        };
        let mut gradient = [0.0; 12];
        let value = match self {
            Self::Distance => {
                let d = delta(0, 1);
                let distance = dot3(&d, &d).sqrt();
                if distance != 0.0 {
                    for axis in 0..3 {
                        gradient[axis] = d[axis] / distance;
                        gradient[3 + axis] = -gradient[axis];
                    }
                }
                distance
            }
            Self::Angle => {
                let a = delta(1, 0);
                let b = delta(1, 2);
                let norm_a = dot3(&a, &a);
                let norm_b = dot3(&b, &b);
                let cross = cross3(&a, &b);
                let cross_norm = dot3(&cross, &cross).sqrt();
                let norm = cross_norm.max(1e-6);
                let a_cross = cross3(&a, &cross);
                let b_cross = cross3(&b, &cross);
                for axis in 0..3 {
                    gradient[axis] = -a_cross[axis] / (norm_a * norm);
                    gradient[6 + axis] = b_cross[axis] / (norm_b * norm);
                    gradient[3 + axis] = -gradient[axis] - gradient[6 + axis];
                }
                if norm_a == 0.0 || norm_b == 0.0 {
                    f64::NAN
                } else {
                    cross_norm.atan2(dot3(&a, &b))
                }
            }
            Self::Dihedral => {
                let a = delta(1, 0);
                let b = delta(2, 1);
                let c = delta(3, 2);
                let normal1 = cross3(&a, &b);
                let normal2 = cross3(&b, &c);
                let norm1 = dot3(&normal1, &normal1);
                let norm2 = dot3(&normal2, &normal2);
                let length_squared = dot3(&b, &b);
                let length = length_squared.sqrt();
                let ab = dot3(&a, &b) / length_squared;
                let cb = dot3(&c, &b) / length_squared;
                for axis in 0..3 {
                    gradient[axis] = -normal1[axis] * length / norm1;
                    gradient[9 + axis] = normal2[axis] * length / norm2;
                    gradient[3 + axis] = (-1.0 - ab) * gradient[axis] + cb * gradient[9 + axis];
                    gradient[6 + axis] = -gradient[axis] - gradient[3 + axis] - gradient[9 + axis];
                }
                if norm1 == 0.0 || norm2 == 0.0 || length == 0.0 {
                    f64::NAN
                } else {
                    (dot3(&cross3(&normal1, &normal2), &b) / length).atan2(dot3(&normal1, &normal2))
                }
            }
        };
        (value, gradient)
    }
}
