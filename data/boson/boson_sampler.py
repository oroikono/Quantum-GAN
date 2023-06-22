import strawberryfields as sf
from strawberryfields.ops import Sgate, BSgate, MeasureFock
import numpy as np
import torch
import os

def generate_boson_data(n_samples, n_photons, n_modes):
    eng = sf.Engine("fock", backend_options={"cutoff_dim": n_photons + 1})
    prog = sf.Program(n_modes)
    results = []

    with prog.context as q:
        for mode in q:
            Sgate(np.sqrt(n_photons)) | mode
        BSgate(np.pi/4, np.pi) | (q[0], q[1])
        MeasureFock() | q

    for _ in range(n_samples):
        result = eng.run(prog)
        results.append(result.samples[0])

    # Save the boson data
    output_dir = 'outputs/samples'
    np.savetxt(os.path.join(output_dir, "boson_data.txt"), results)

    return torch.Tensor(results)