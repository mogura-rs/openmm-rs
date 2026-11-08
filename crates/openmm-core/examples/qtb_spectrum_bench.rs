use openmm_core::integrators::qtb::quantum_thermal_bath_spectrum;
use std::time::Instant;

fn main() {
    for n in [2, 3, 5, 6, 16, 64, 128, 512, 1500] {
        for temperature in [0.0, 70.0, 300.0, 600.0] {
            for friction in [1.0, 10.0] {
                if n > 128 && (temperature != 300.0 || friction != 10.0) {
                    continue;
                }
                let started = Instant::now();
                let (theta, deconvolved) =
                    quantum_thermal_bath_spectrum(temperature, friction, 0.001, n).unwrap();
                println!("{{\"n\":{n},\"temperature\":{temperature},\"friction\":{friction},\"seconds\":{},\"theta\":{theta:?},\"deconvolved\":{deconvolved:?}}}", started.elapsed().as_secs_f64());
            }
        }
    }
}
