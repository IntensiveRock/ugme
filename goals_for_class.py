import numpy as np
from ugme import UGME

# pass UGME the tpms and test a few things
model = UGME(dt=1).fit_fetch("data.npy")
assert model.check_U_first_steps() # testing that U[0:2] = TPM[0:2]
assert np.allclose(model.tpm, model.reconstruct_tpm()) # test that U(t) reconstructs dynamics

# get tau_R that sits below user-defined threshold
tau_R = model.select_tau_R(threshold=0.05)

# predict dynamics with and without averaging
predicted_dynamics = model.predict(tau_R)
predicted_with_ave = model.predict_averaged(t_r=tau_R - 10, tau_R=tau_R)
