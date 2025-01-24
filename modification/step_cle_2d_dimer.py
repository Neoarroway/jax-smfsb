#!/usr/bin/env python3

import jsmfsb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jsmfsb.models
import time
import numpy as np

time_cost=np.zeros(20,)
for kk in range(20):
    start_time = time.time()
    M = 20
    N = 30 + 10 * kk
    T = 2
    x0 = jnp.zeros((2, M, N))
    dimer = jsmfsb.models.dimer()
    #x0 = x0.at[:, int(M / 2), int(N / 2)].set(lv.m)
    x0 = x0.at[0, 8:12, :].set(dimer.m[0])
    x0 = x0.at[1, 8:12, :].set(dimer.m[1])
    step_dimer_2d = dimer.step_cle_2d(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    x1 = step_dimer_2d(k0, x0, 0, T)

    for i in range(2):
        plt.subplots()
        v_max = [160, 50]
        im = plt.imshow(x1[i, :, :], cmap='viridis', vmin=0, vmax=v_max[i])
        plt.title('SCLE ' + dimer.n[i])
        plt.colorbar(im)
        plt.savefig(f"stepCLE2Ddimer{i}.pdf")

    end_time = time.time()
    time_1loop = end_time - start_time
    time_cost[kk] = time_1loop
    print(kk)
np.savetxt('scletime.txt', time_cost, fmt='%f')


# eof
