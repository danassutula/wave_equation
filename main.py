'''
Solve the wave equation using finite elements for the spatial discretization
and the Newmark method for time integration.

The Newmark time integration method
-----------------------------------
u^{k+1} = u^{k} + du/dt^{k} dt + ((0.5-beta) d2u/dt2^{k} + beta d2u/dt2^{k+1}) dt^2
du/dt^{k+1} = du/dt^{k} + ((1-gamma) d2u/dt2^{k} + gamma d2u/dt2^{k+1}) dt

'''

import dolfin

class PeriodicWriter:
    '''Class for writing ".pvd" files every n'th time '''

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

    def write(self, force_write=False):

        self.count += 1

        if self.count < self.start:
            return

        if (self.count-self.start) % self.step == 0 or force_write:
            self.file << self.func

    def reset(self):
        self.count = -1


if __name__ == "__main__":

    from dolfin import *
    from matplotlib import pyplot as plt
    from solver import WaveSolver

    eps = 1e-12
    maximum_writes = 200

    ### Physical parameters

    c = 1.0 # Speed of light 300e6 (m/s)
    T = 8.0 # Time period

    ### Mesh

    nx = ny = 80
    mesh = UnitSquareMesh(nx, ny, diagonal="crossed")

    ### Function space

    degree = 2

    V = VectorFunctionSpace(mesh, 'CG', degree)
    u = Function(V, name="u")

    ### Time integration parameters

    CFL = 0.5 # < 0.5

    dt = CFL*mesh.hmin()/c
    nk = int(T*(1+eps) / dt)

    ### Initial conditions

    u_ic = Expression(('0.0','0.0'), degree=0)
    dudt_ic = Expression(('0.0','0.0'), degree=0)

    ### Boundary conditions

    # omega = pi
    omega = 2*pi
    # omega = 4*pi
    alpha = 0.1

    kernel_u = "1.0"
    # kernel_u = "exp(-(pow((xc-x[0])/a,2) + pow((yc-x[1])/a,2)))"

    u_bc = Expression((f"sin(w*t)*{kernel_u}", "0.0"), t=0.0, xc=0.0, yc=0.0, w=omega, a=alpha, degree=2)
    dudt_bc = Expression((f"w*cos(w*t)*{kernel_u}", "0.0"), t=0.0, xc=0.0, yc=0.0, w=omega, a=alpha, degree=2)
    d2udt2_bc = Expression((f"-w*w*sin(w*t)*{kernel_u}", "0.0"), t=0.0, xc=0.0, yc=0.0, w=omega, a=alpha, degree=2)

    def dirichlet_boundary(x, on_boundary):
        return (x[0] < 1e-6) and on_boundary
        # return ((x[0] < 1e-6) or (x[1] < 1e-6)) and on_boundary

    def homogeneous_boundary(x, on_boundary):
        return (x[0] > 1-1e-6) and on_boundary
        # return ((x[0] > 1-1e-6) or (x[1] > 1-1e-6)) and on_boundary

    # homogeneous_boundary = None

    bcs_u = [DirichletBC(V, u_bc, dirichlet_boundary),]
    bcs_dudt = [DirichletBC(V, dudt_bc, dirichlet_boundary),]
    bcs_d2udt2 = [DirichletBC(V, d2udt2_bc, dirichlet_boundary),]

    if homogeneous_boundary is not None:
        zeros = Constant((0.0, 0.0))
        bcs_u.append(DirichletBC(V, zeros, homogeneous_boundary))
        bcs_dudt.append(DirichletBC(V, zeros, homogeneous_boundary))
        bcs_d2udt2.append(DirichletBC(V, zeros, homogeneous_boundary))

    def set_bcs_time(t):
        u_bc.t = t
        dudt_bc.t = t
        d2udt2_bc.t = t

    ### Define solver

    solution_writer = PeriodicWriter("./out/u.pvd",
        u, start=1, step=int(nk/maximum_writes+0.5))

    # Time-step implicitness (NOTE: Require `gamma >= 0.5` for stability)
    # gamma = 1.0
    # gamma = 3/4
    gamma = 1/2

    solver = WaveSolver(u, u_ic, dudt_ic, bcs_u, bcs_dudt, bcs_d2udt2, set_bcs_time)
    solver.solve(dt, nk, c=1.0, beta=gamma/2, gamma=gamma, callback=solution_writer.write)

    plt.figure()
    dolfin.plot(u)
    plt.show()
