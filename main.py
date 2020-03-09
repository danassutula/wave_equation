
from dolfin import *

import numpy as np
import matplotlib.pyplot as plt

nx = ny = 60
diagonal = "crossed"
degree = 1
mesh = UnitSquareMesh(nx, ny, diagonal=diagonal)

outfile_u = File("./out/u.pvd")

### Parameters

c = 1.0
omega = 2*np.pi

nk = 2400
t_final = 8
dt = t_final / nk

theta = 1.0

c_ = Constant(c)
dt_ = Constant(dt)
theta_ = Constant(theta)



### Source term

# g1 = Expression(" exp(-50*(pow(x[0], 2) + pow(x[1], 2)))", degree=3)
# g2 = Expression(" exp(-50*(pow((x[0]-1.0), 2) + pow((x[1]-1.0), 2)))", degree=3)
# g3 = Expression(" exp(-50*(pow((x[0]-0.5), 2) + pow((x[1]-0.5), 2)))", degree=4)

# gn = Constant(0.0)
# N = FacetNormal(mesh)
# g = gn * N

### Boundary conditions

u_D = Expression((f"sin({omega}*t)", "0.0"), t=0.0, degree=3)
dudt_D = Expression((f"{omega}*cos({omega}*t)", "0.0"), t=0.0, degree=3)

### Function space

V = VectorFunctionSpace(mesh, 'CG', degree)
S = FunctionSpace(mesh, ('CG' if degree > 1 else 'DG'), degree-1)

bc_u    = [DirichletBC(V, u_D, lambda x, on_boundary: (x[0] < 1e-6) and on_boundary),]
bc_dudt = [DirichletBC(V, dudt_D, lambda x, on_boundary: (x[0] < 1e-6) and on_boundary),]

bc_dofs = list(bc_u[0].get_boundary_values().keys())

def compute_bc_values_u():
    return list(bc_u[0].get_boundary_values().values())

def compute_bc_values_dudt():
    return list(bc_dudt[0].get_boundary_values().values())

v = TestFunction(V)
u = TrialFunction(V)

M = dot(u, v) * dx
K = inner(grad(u), grad(v)) * dx

lhs = assemble((1/c_**2)*M + (0.5*theta_*dt_**2)*K)
rhs_1 = assemble((1/c_**2)*M - (0.5*(1.0-theta_)*dt_**2)*K)
rhs_2 = assemble(-dt_*K)

u = Function(V, name="u")
dudt = Function(V, name="du/dt")
dudt_k = Function(V, name="du/dt (prev.)")

t = 0

# This only zeros-out the rows and puts ones on the diagonal (DO ONCE)
for bc_i in bc_dudt:
    bc_i.apply(lhs)

try:

    i = 0
    while t < t_final:

        u_D.t = t
        dudt_D.t = t

        dudt_k.vector()[:] = dudt.vector()

        u.vector()[bc_dofs] = compute_bc_values_u()
        dudt_k.vector()[bc_dofs] = compute_bc_values_dudt()

        rhs = rhs_1*dudt_k.vector() + rhs_2*u.vector()

        t += dt

        u_D.t = t
        dudt_D.t = t

        for bc_i in bc_dudt:
            bc_i.apply(rhs)

        print("Solve for {0:.3f}".format(t))
        solve(lhs, dudt.vector(), rhs)

        u.vector()[:] += (dudt_k.vector()*((1.0-theta)*dt) + dudt.vector()*(theta*dt))

        if i % 20 == 0:
            outfile_u << u

        i += 1

except KeyboardInterrupt:
    print("\nCought a `KeyboardInterrupt`\n")
    pass


# div_E = project(div(u), S)
# div_E.rename("div_E", "")
#
# curl_E = project(curl(u), S)
# curl_E.rename("curl_E", "")
#
# fh = plt.figure(1)
# fh.clear()
# plot(u)
# plt.show()
#
# File("electric_field.pvd") << u
#
# File("error_div.pvd") << div_E
# File("error_curl.pvd") << curl_E
#
# print("norm(u):\n", "{0:.6f}".format(norm(u)))
# print("norm(div(u)-g)\n", "{0:.6f}".format(assemble((div_E-g)**2*dx)))
# print("norm(curl(u):\n", "{0:.6f}".format(assemble(curl_E**2*dx)))
