
import dolfin

from dolfin import Function
from dolfin import Constant
from dolfin import assemble
from dolfin import dot
from dolfin import ds
from dolfin import dx
from dolfin import grad
from dolfin import inner


class WaveSolver:

    def __init__(self, u, u_ic, dudt_ic, bcs_u, bcs_dudt, bcs_d2udt2, set_bcs_time):

        if not isinstance(u, dolfin.Function):
            raise TypeError("Parameter `u` must be a `dolfin.Function`")

        if not isinstance(bcs_u, (list, tuple)):
            bcs_u = (bcs_u,)
        if not all(isinstance(a, dolfin.DirichletBC) for a in bcs_u):
            raise TypeError("Parameter `bcs_u` must be a `list` "
                            "of `dolfin.DirichletBC`\'s.")

        if not isinstance(bcs_dudt, (list, tuple)):
            bcs_dudt = (bcs_dudt,)
        if not all(isinstance(a, dolfin.DirichletBC) for a in bcs_dudt):
            raise TypeError("Parameter `bcs_dudt` must be a `list` "
                            "of `dolfin.DirichletBC`\'s.")

        if not isinstance(bcs_d2udt2, (list, tuple)):
            bcs_d2udt2 = (bcs_d2udt2,)
        if not all(isinstance(a, dolfin.DirichletBC) for a in bcs_d2udt2):
            raise TypeError("Parameter `bcs_d2udt2` must be a `list` "
                            "of `dolfin.DirichletBC`\'s.")

        try:
            set_bcs_time(0.0)
        except:
            raise TypeError("Parameter `set_bcs_time` must be "
                            "callabel with a `float` argument.")

        self._u = u
        self._V = V = u.function_space()

        self._dudt = dolfin.Function(V, name="dudt")
        self._d2udt2 = dolfin.Function(V, name="d2udt2")

        v0 = dolfin.TestFunction(V)
        v1 = dolfin.TrialFunction(V)

        self._form_M = dot(v0, v1) * dx
        self._form_K = inner(grad(v0), grad(v1)) * dx

        self.u_ic = u_ic
        self.dudt_ic = dudt_ic

        self.bcs_u = bcs_u
        self.bcs_dudt = bcs_dudt
        self.bcs_d2udt2 = bcs_d2udt2

        self.set_bcs_time = set_bcs_time

    def solve(self, dt, nk, c=1.0, beta=0.25, gamma=0.5, callback=None):
        '''
        Parameters
        ----------
        dt : float
            Type step.
        nk : int
            Number of time steps.
        c : float
            Wave propagation speed.
        callback : callable
            Handle of a function that will be called (without arguments) at
            at each iteration time.

        Returns
        -------
        None

        '''

        if not isinstance(dt, float) or dt <= 0.0:
            raise TypeError("Parameter `dt` must be a positive `float`")

        if not isinstance(nk, int) or nk <= 0:
            raise TypeError("Parameter `dt` must be a positive `int`")

        if not isinstance(c, (float, int)):
            raise TypeError("Parameter `c` must be a `float`")
        elif c <= 0:
            raise TypeError("Parameter `c` must be positive")

        if not isinstance(beta, float) or not (0.0 <= beta <= 0.5):
            raise TypeError("Parameter `beta` must be in range [0.0,0.5]")

        if not isinstance(gamma, float) or not (0.0 <= gamma <= 1.0):
            raise TypeError("Parameter `gamma` must be in range [0.0,1.0]")

        if callback is None:
            callback = lambda : None
        elif not callable(callback):
            raise TypeError("Parameter `callback` must be "
                            "callable without parameters.")

        A0 = assemble(self._form_M)
        if c != 1.0: A0 *= 1.0/c**2

        K = assemble(self._form_K)
        A = A0 + K*(beta*dt**2)

        for bc_i in self.bcs_d2udt2:
            bc_i.apply(A0)
            bc_i.apply(A)

        init_solver = dolfin.LUSolver(A0)
        loop_solver = dolfin.LUSolver(A, "mumps")

        u_vec = self._u.vector()
        v_vec = self._dudt.vector()
        a_vec = self._d2udt2.vector()
        a_vec_star = a_vec.copy()

        # Assign initial conditions
        u_vec[:] = dolfin.interpolate(self.u_ic, self._V).vector()
        v_vec[:] = dolfin.interpolate(self.dudt_ic, self._V).vector()

        try:

            k = 0
            t = 0.0

            self.set_bcs_time(t)

            for bc_i in self.bcs_u:
                bc_i.apply(u_vec)

            for bc_i in self.bcs_dudt:
                bc_i.apply(v_vec)

            rhs = -(K*u_vec)

            for bc_i in self.bcs_d2udt2:
                bc_i.apply(rhs)

            init_solver.solve(a_vec, rhs)

            while k != nk:

                rhs = -(K*(u_vec + v_vec*dt + a_vec*((0.5-beta)*dt**2)))

                a_vec_star[:] = a_vec

                for bc_i in self.bcs_d2udt2:
                    bc_i.apply(rhs)

                loop_solver.solve(a_vec, rhs)

                u_vec += v_vec*dt + (a_vec_star*(0.5-beta) + a_vec*beta)*(dt**2)
                v_vec += (a_vec_star*(1.0-gamma) + a_vec*gamma)*dt

                k += 1
                t += dt

                self.set_bcs_time(t)

                for bc_i in self.bcs_u:
                    bc_i.apply(u_vec)

                for bc_i in self.bcs_dudt:
                    bc_i.apply(v_vec)

                callback() # Usually writes `u`

                print("{0:4d}/{1:d}".format(k, nk))

        except KeyboardInterrupt:
            print("\nStopping because of a `KeyboardInterrupt`.\n")
