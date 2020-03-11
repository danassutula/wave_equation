'''
Solve the wave equation using finite elements for the spatial discretization
and the Newmark method for time integration.

'''

import dolfin
from dolfin import *
from dolfin import pi as PI

EPS = 1e-12

### Solution file

# class NemarkSolver:
#     def __init__(self, form_M, form_K, beta=0.25, gamma=0.5)
#     pass
#     def solve(self, x, rhs)

class PeriodicWriter:
    '''Writes ".pvd" files.'''

    def __init__(self, filename:str, func:dolfin.Function, start:int=0, step:int=1):

        if not isinstance(filename, str):
            raise TypeError('Parameter `filename` must be a `str`')

        if not isinstance(func, dolfin.Function):
            raise TypeError('Parameter `func` must be a `dolfin.Function`')

        if not filename.endswith(".pvd"):
            filename += ".pvd"

        self.file = dolfin.File(filename)
        self.func = func

        self.count = -1
        self.start = start
        self.step  = step

    def write(self, force=False):

        self.count += 1

        if self.count < self.start:
            return

        if (self.count-self.start) % self.step == 0 or force:
            self.file << self.func

    def reset(self):
        self.count = -1


class Solver:

    def __init__(self, u, u_bc, dudt_bc, d2udt2_bc, set_bcs_time,
                 dirichlet_boundary, homogeneous_boundary=None):

        self._u = u
        self._V = u.function_space()

        self._dudt = Function(V, name="dudt")
        self._d2udt2 = Function(V, name="d2udt2")

        ### Weak form

        v0 = TestFunction(V)
        v1 = TrialFunction(V)

        self._form_M = dot(v1, v0) * dx
        self._form_K = 1/c**2*inner(grad(v1), grad(v0)) * dx

        ### BC's

        self.bcs_u      = [DirichletBC(V, u_bc, dirichlet_boundary),]
        self.bcs_dudt   = [DirichletBC(V, dudt_bc, dirichlet_boundary),]
        self.bcs_d2udt2 = [DirichletBC(V, d2udt2_bc, dirichlet_boundary),]

        if homogeneous_boundary is not None:
            zeros = Constant((0.0, 0.0))
            self.bcs_u.append(DirichletBC(V, zeros, homogeneous_boundary))
            self.bcs_dudt.append(DirichletBC(V, zeros, homogeneous_boundary))
            self.bcs_d2udt2.append(DirichletBC(V, zeros, homogeneous_boundary))

        self.set_bcs_time = set_bcs_time

    def solve(self, u_ic, dudt_ic, dt, T, beta=0.25, gamma=0.5, solution_writer=None):
        '''
        Newmark parameters
        beta = 0.25 # 0..0.5
        gamma = 0.5 # 0..1.0
        '''

        # Linear systems
        M = assemble(self._form_M)
        K = assemble(self._form_K)

        # LHS for the Newmark method
        A = M + K*(beta*dt**2)

        for bc_i in self.bcs_d2udt2:
            bc_i.apply(M)
            bc_i.apply(A)

        init_solver = LUSolver(M)
        loop_solver = LUSolver(A, "mumps")

        u_vec = self._u.vector()
        v_vec = self._dudt.vector()
        a_vec = self._d2udt2.vector()
        a_vec_star = a_vec.copy()

        # Assign initial conditions
        u_vec[:] = interpolate(u_ic, self._V).vector()
        v_vec[:] = interpolate(dudt_ic, self._V).vector()

        if solution_writer is None:
            solution_writer = type("DummyClass", (),
                {"write": lambda : None})

        try:

            i = 0
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

            while i != nt:

                rhs = -(K*(u_vec + v_vec*dt + a_vec*((0.5-beta)*dt**2)))

                a_vec_star[:] = a_vec

                for bc_i in self.bcs_d2udt2:
                    bc_i.apply(rhs)

                loop_solver.solve(a_vec, rhs)

                u_vec += v_vec*dt + (a_vec_star*(0.5-beta) + a_vec*beta)*(dt**2)
                v_vec += (a_vec_star*(1.0-gamma) + a_vec*gamma)*dt

                i += 1
                t += dt

                self.set_bcs_time(t)

                for bc_i in self.bcs_u:
                    bc_i.apply(u_vec)

                for bc_i in self.bcs_dudt:
                    bc_i.apply(v_vec)

                solution_writer.write()

                print("{0:d}/{1:d}: Energy={2:.3e}"
                    .format(i, nt, 0.5*(M*u_vec).inner(u_vec)))

        except KeyboardInterrupt:
            print("\nCought a `KeyboardInterrupt`\n")


if __name__ == "__main__":

    maximum_writes = 200

    ### Physical parameters

    c = 1.0 # Speed of light 300e6 (m/s)
    T = 4.0

    ### Mesh

    nx = ny = 160

    mesh = UnitSquareMesh(nx, ny, diagonal="crossed")

    ### Solver parameters

    CFL = 1/3
    dt = CFL*mesh.hmin()/c

    nt = int(T*(1+EPS) / dt)

    solution_writing_period = (nt // maximum_writes)

    ### Initial conditions

    u_ic = Expression(('0.0','0.0'), degree=0)
    dudt_ic = Expression(('0.0','0.0'), degree=0)

    ### Boundary conditions

    # omega = PI
    # omega = 2*PI
    omega = 4*PI
    alpha = 0.1

    kernel_u = "exp(-(pow((xc-x[0])/a,2) + pow((yc-x[1])/a,2)))"

    u_bc = Expression((f"sin(w*t)*{kernel_u}",)*2, t=0.0, xc=0.0, yc=0.0, w=omega, a=alpha, degree=2)
    dudt_bc = Expression((f"w*cos(w*t)*{kernel_u}",)*2, t=0.0, xc=0.0, yc=0.0, w=omega, a=alpha, degree=2)
    d2udt2_bc = Expression((f"-w*w*sin(w*t)*{kernel_u}",)*2, t=0.0, xc=0.0, yc=0.0, w=omega, a=alpha, degree=2)

    def dirichlet_boundary(x, on_boundary):
        return ((x[0] < 1e-6) or (x[1] < 1e-6)) and on_boundary

    def homogeneous_boundary(x, on_boundary):
        return ((x[0] > 1-1e-6) or (x[1] > 1-1e-6)) and on_boundary

    # homogeneous_boundary = None

    ### Function space

    degree = 1

    V = VectorFunctionSpace(mesh, 'CG', degree)

    u = Function(V, name="u")

    ### Define solver

    def set_bcs_time(t):
        u_bc.t = t
        dudt_bc.t = t
        d2udt2_bc.t = t

    solution_writer = PeriodicWriter("./out/u.pvd",
        u, start=1, step=solution_writing_period)

    solver = Solver(u, u_bc, dudt_bc, d2udt2_bc, set_bcs_time,
                    dirichlet_boundary, homogeneous_boundary)

    solver.solve(u_ic, dudt_ic, dt, T, solution_writer=solution_writer)

    print("Done")
