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
    M = 20
    N = 30
    H = 25
    T = 2
    x0 = jnp.zeros((2, M, N, H))
    dimer = jsmfsb.models.dimer()
    #x0 = x0.at[:, int(M / 2), int(N / 2)].set(lv.m)
    x0 = x0.at[0, 8:12, :, 0:-10].set(dimer.m[0])
    x0 = x0.at[1, 8:12, :, 0:-10].set(dimer.m[1])
    step_dimer_3d = dimer.step_gillespie_3d_dhz(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    x1 = step_dimer_3d(k0, x0, 0, T)

    for i in range(2):
        plt.subplots()
        v_max = [160, 50]
        im = plt.imshow(x1[i, :, 14, :], cmap='viridis', vmin=0, vmax=v_max[i])
        plt.title('Gillespie ' + dimer.n[i])
        plt.colorbar(im)
        plt.savefig(f"stepGillespie3Ddimer_xoz{i}.pdf")
    end_time = time.time()
    time_1loop = end_time - start_time
    time_cost[kk] = time_1loop
    print(kk)
np.savetxt('gillespietime3d.txt', time_cost, fmt='%f')
plt.show()
# eof
