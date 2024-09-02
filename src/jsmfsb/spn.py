#!/usr/bin/env python3
# spn.py

import numpy as np
import jax
from jax import grad, jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.lax as jl

# Spn class definition, including methods

class Spn:
    """Class for stochastic Petri net models.
    """

    
    def __init__(self, n, t, pre, post, h, m):
        """Constructor method for Spn objects

        Create a Spn object for representing a stochastic Petri net model that
        can be simulated using a variety of algorithms.

        Parameters
        ----------
        n : list of strings
            Names of the species/tokens in the model
        t : list of strings
            Names of the reactions/transitions in the model
        pre : matrix
            Matrix representing the LHS stoichiometries
        post: matrix
            Matrix representing the RHS stoichiometries
        h: function
            A function to compute the rates of the reactions from the current state and time of
            the system. The function should return a numpy array of rates.
        m: list of integers
            The intial state/marking of the model/net

        Returns
        -------
        A object of class Spn.

        Examples
        --------
        >>> import jsmfsb
        >>> import jax
        >>> import jax.numpy as jnp
        >>> sir = jsmfsb.Spn(["S", "I", "R"], ["S->I", "I->R"],
              [[1,1,0],[0,1,0]], [[0,2,0],[0,0,1]],
	      lambda x, t: jnp.array([0.3*x[0]*x[1]/200, 0.1*x[1]]),
	      [197, 3, 0])
        >>> stepSir = sir.stepGillespie()
        >>> jsmfsb.simSample(jax.random.key(42), 10, sir.m, 0, 20, stepSir)
        """
        self.n = n # species names
        self.t = t # reaction names
        self.pre = jnp.array(pre).astype(jnp.float32)
        self.post = jnp.array(post).astype(jnp.float32)
        self.h = h # hazard function
        self.m = jnp.array(m).astype(jnp.float32) # initial marking


        
    def __str__(self):
        """A very simple string representation of the Spn object, mainly for debugging.
        """
        return "n: {}\n t: {}\npre: {}\npost: {}\nh: {}\nm: {}".format(str(self.n),
                str(self.t), str(self.pre), str(self.post), str(self.h), str(self.m))


    
    def stepGillespie(self, minHaz=1e-10, maxHaz=1e07):
        """Create a function for advancing the state of a SPN by using the
        Gillespie algorithm

        This method returns a function for advancing the state of an SPN
        model using the Gillespie algorithm. The resulting function
        (closure) can be used in conjunction with other functions (such as
        `simTs`) for simulating realisations of SPN models.

        Parameters
        ----------
        minHaz : float
          Minimum hazard to consider before assuming 0. Defaults to 1e-10.
        maxHaz : float
          Maximum hazard to consider before assuming an explosion and
          bailing out. Defaults to 1e07.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using the Gillespie algorithm. The function closure
        has interface `function(key, x0, t0, deltat)`, where `key` is an
        unused JAX random key, `x0` and `t0` represent the initial state 
        and time, and `deltat` represents the amount of time by which the 
        process should be advanced. The function closure returns a vector 
        representing the simulated state of the system at the new time.

        Examples
        --------
        >>> import jsmfsb.models
        >>> import jax
        >>> lv = jsmfsb.models.lv()
        >>> stepLv = lv.stepGillespie()
        >>> stepLv(jax.random.key(42), lv.m, 0, 1)
        """
        S = (self.post - self.pre).T
        u, v = S.shape
        @jit
        def advance(state):
            key, xo, x, t = state
            h = self.h(x, t)
            h0 = jnp.sum(h)
            key, k1, k2 = jax.random.split(key, 3)
            t = jnp.where(h0 > maxHaz, 1e90, t)
            t = jnp.where(h0 < minHaz, 1e90,
                          t + jax.random.exponential(k1)/h0)
            j = jax.random.choice(k2, v, p=h/h0)
            xn = jnp.add(x, S[:,j])
            return (key, x, xn, t)
        @jit
        def step(key, x0, t0, deltat):
            t = t0
            x = x0
            termt = t0 + deltat
            key, x, xn, t = jl.while_loop(lambda state: state[3] < termt,
                                      advance, (key, x, x, t))
            return x
        return step

 
    def stepPTS(self, dt = 0.01):
        """Create a function for advancing the state of an SPN by using a 
        simple approximate Poisson time stepping method

        This method returns a function for advancing the state of an SPN
        model using a simple approximate Poisson time stepping method. The
        resulting function (closure) can be used in conjunction with other
        functions (such as ‘simTs’) for simulating realisations of SPN
        models.

        Parameters
        ----------
        dt : float
            The time step for the time-stepping integration method. Defaults to 0.01.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using a Poisson time stepping method with step size
        ‘dt’. The function closure has interface
        ‘function(key, x0, t0, deltat)’, where ‘x0’ and ‘t0’ represent the
        initial state and time, and ‘deltat’ represents the amount of time
        by which the process should be advanced. The function closure
        returns a vector representing the simulated state of the system at
        the new time.

        Examples
        --------
        >>> import jsmfsb.models
        >>> import jax
        >>> lv = jsmfsb.models.lv()
        >>> stepLv = lv.stepPTS(0.001)
        >>> k = jax.random.key(42)
        >>> stepLv(k, lv.m, 0, 1)
        """
        S = (self.post - self.pre).T
        u, v = S.shape
        @jit
        def advance(state):
            key, x, t = state
            key, k1 = jax.random.split(key)
            h = self.h(x, t)
            r = jax.random.poisson(k1, h*dt)
            x = jnp.add(x, S.dot(r))
            # TODO: sort out negative values
            #x = jnp.where(x < 0, -x, x)
            t = t + dt
            return (key, x, t)
        @jit
        def step(key, x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            key, x, t = jl.while_loop(lambda state: state[2] < termt,
                                      advance, (key, x, t))
            return x
        return step

   
    def stepEuler(self, dt = 0.01):
        """Create a function for advancing the state of an SPN by using a simple
        continuous deterministic Euler integration method

        This method returns a function for advancing the state of an SPN
        model using a simple continuous deterministic Euler integration
        method. The resulting function (closure) can be used in
        conjunction with other functions (such as ‘simTs’) for simulating
        realisations of SPN models.

        Parameters
        ----------
        dt : float
            The time step for the time-stepping integration method. Defaults to 0.01.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using an Euler method with step size ‘dt’. The
        function closure has interface ‘function(x0, t0, deltat)’, where
        ‘x0’ and ‘t0’ represent the initial state and time, and ‘deltat’
        represents the amount of time by which the process should be
        advanced. The function closure returns a vector representing the
        simulated state of the system at the new time.

        Examples
        --------
        >>> import jsmfsb.models
        >>> import jax
        >>> lv = jsmfsb.models.lv()
        >>> stepLv = lv.stepEuler(0.001)
        >>> k = jax.random.key(42)
        >>> stepLv(k, lv.m, 0, 1)
        """
        S = (self.post - self.pre).T
        @jit
        def advance(state):
            key, x, t = state
            key, k1 = jax.random.split(key)
            h = self.h(x, t)
            x = jnp.add(x, S.dot(h*dt))
            x = jnp.where(x < 0, -x, x)
            t = t + dt
            return (key, x, t)
        @jit
        def step(key, x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            key, x, t = jl.while_loop(lambda state: state[2] < termt,
                                      advance, (key, x, t))
            return x
        return step

    
    def stepCLE(self, dt = 0.01):
        """Create a function for advancing the state of an SPN by using a simple
        Euler-Maruyama integration method for the associated CLE

        This method returns a function for advancing the state of an SPN
        model using a simple Euler-Maruyama integration method
        method for the chemical Langevin equation form of the model.The 
        resulting function (closure) can be used in
        conjunction with other functions (such as `simTs`) for simulating
        realisations of SPN models.

        Parameters
        ----------
        dt : float
            The time step for the time-stepping integration method. Defaults to 0.01.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using an Euler-Maruyama method with step size ‘dt’. The
        function closure has interface ‘function(key, x0, t0, deltat)’, where
        ‘x0’ and ‘t0’ represent the initial state and time, and ‘deltat’
        represents the amount of time by which the process should be
        advanced. The function closure returns a vector representing the
        simulated state of the system at the new time.

        Examples
        --------
        >>> import jsmfsb.models
        >>> import jax
        >>> lv = jsmfsb.models.lv()
        >>> stepLv = lv.stepCLE(0.001)
        >>> stepLv(jax.random.key(42), lv.m, 0, 1)
        """
        S = (self.post - self.pre).T
        v = S.shape[1]
        sdt = np.sqrt(dt)
        @jit
        def advance(state):
            key, x, t = state
            key, k1 = jax.random.split(key)
            h = self.h(x, t)
            dw = jax.random.normal(k1, [v])*sdt
            x = jnp.add(x, S.dot(h*dt + jnp.sqrt(h)*dw))
            x = jnp.where(x < 0, -x, x)
            t = t + dt
            return (key, x, t)
        @jit
        def step(key, x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            key, x, t = jl.while_loop(lambda state: state[2] < termt,
                                      advance, (key, x, t))
            return x
        return step


    # spatial simulation functions, from chapter 9

    def stepGillespie1D(self, d, minHaz=1e-10, maxHaz=1e07):
        """Create a function for advancing the state of an SPN by using the
        Gillespie algorithm on a 1D regular grid

        This method creates a function for advancing the state of an SPN
        model using the Gillespie algorithm. The resulting function
        (closure) can be used in conjunction with other functions (such as
        `simTs1D`) for simulating realisations of SPN models in space and
        time.

        Parameters
        ----------
        d : array
          A vector of diffusion coefficients - one coefficient for each
          reacting species, in order. The coefficient is the reaction
          rate for a reaction for a molecule moving into an adjacent
          compartment. The hazard for a given molecule leaving the
          compartment is therefore twice this value (as it can leave to
          the left or the right).
        minHaz : float
          Minimum hazard to consider before assuming 0. Defaults to 1e-10.
        maxHaz : float
          Maximum hazard to consider before assuming an explosion and
          bailing out. Defaults to 1e07.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using the Gillespie algorithm. The function closure
        has arguments `key`, `x0`, `t0`, `deltat`, where `key` is a JAX
        random key, `x0` is a matrix with rows corresponding to species
        and columns corresponding to voxels, representing the initial
        condition, `t0` represent the initial state and time, and
        `deltat` represents the amount of time by which the process should
        be advanced. The function closure returns a matrix representing
        the simulated state of the system at the new time.

        Examples
        --------
        >>> import jsmfsb.models
        >>> import jax.numpy as jnp
        >>> lv = jsmfsb.models.lv()
        >>> stepLv1d = lv.stepGillespie1D(np.array([0.6,0.6]))
        >>> N = 20
        >>> x0 = jnp.zeros((2,N))
        >>> x0 = x0.at[:,int(N/2)].set(lv.m)
        >>> k0 = jax.random.key(42)
        >>> stepLv1d(k0, x0, 0, 1)
        """
        S = (self.post - self.pre).T
        u, v = S.shape
        @jit
        def advance(state):
            key, xo, x, t = state
            key, k1, k2, k3 = jax.random.split(key, 4)
            hr = jnp.apply_along_axis(lambda xi: self.h(xi, t), 0, x)
            hrs = jnp.apply_along_axis(jnp.sum, 0, hr)
            hrss = hrs.sum()
            hd = jnp.apply_along_axis(lambda xi: xi*d*2, 0, x)
            hds = jnp.apply_along_axis(jnp.sum, 0, hd)
            hdss = hds.sum()
            h0 = hrss + hdss
            t = jnp.where(h0 > maxHaz, 1e90, t)
            t = jnp.where(h0 < minHaz, 1e90,
                          t + jax.random.exponential(k1)/h0)
            def diffuse(key, x):
                n = x.shape[1]
                k1, k2, k3 = jax.random.split(key, 3)
                j = jax.random.choice(k1, n, p=hds/hdss) # pick a box
                i = jax.random.choice(k2, u, p=hd[:,j]/hds[j]) # pick species
                x[i,j] = x[i,j]-1 # decrement chosen box
                if (jax.random.uniform(k3) < 0.5):
                    # left
                    if (j>0):
                        x[i,j-1] = x[i,j-1] + 1
                    else:
                        x[i,n-1] = x[i,n-1] + 1
                else:
                    # right
                    if (j<n-1):
                        x[i,j+1] = x[i,j+1] + 1
                    else:
                        x[i,0] = x[i,0] + 1
                return x
            def react(key, x):
                n = x.shape[1]
                k1, k2 = jax.random.split(key, 2)
                j = jax.random.choice(k1, n, p=hrs/hrss) # pick a box
                i = jax.random.choice(k2, v, p=hr[:,j]/hrs[j]) # pick a reaction
                x = x.at[:,j].set(jnp.add(x[:,j], S[:,i]))
                return x
            xn = jnp.where(jax.random.uniform(k2)*h0 < hdss,
                           diffuse(k3, x), react(k3, x))
            return (key, x, xn, t)
        @jit
        def step(key, x0, t0, deltat):
            t = t0
            x = x0
            termt = t0 + deltat
            key, x, xn, t = jl.while_loop(lambda state: state[3] < termt,
                                          advance, (key, x, x, t))
            return x
        return step


    


# eof

