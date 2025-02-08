#!/usr/bin/env python3

import jsmfsb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jsmfsb.models
import time
import numpy as np

time_cost=np.zeros(1,)
for kk in range(1):
    start_time = time.time()
    M = 40 + 10 * kk
    N = 50
    T = 40
    x0 = jnp.zeros((2, M, N))
    dimer = jsmfsb.models.dimer2()
    x0 = x0.at[0, 10:30, 0:-20].set(8)
    x0 = x0.at[1, 10:30, 0:-20].set(dimer.m[1])
    step_dimer_2d = dimer.step_gillespie_2d_dhz(jnp.array([0.01, 0.01]))
    k0 = jax.random.key(42)
    x1 = step_dimer_2d(k0, x0, 0, T)

    for i in range(2):
        plt.subplots()
        v_max = [4, 4]
        im = plt.imshow(x1[i, :, :], cmap='viridis', vmin=0, vmax=v_max[i])
        plt.title('Gillespie ' + dimer.n[i])
        plt.colorbar(im)
        plt.savefig(f"stepGillespie2Ddimer{i}.pdf")
    end_time = time.time()
    time_1loop = end_time - start_time
    time_cost[kk] = time_1loop
    print(kk)
np.savetxt('gillespietime2.txt', time_cost, fmt='%f')
plt.show()
# eof
