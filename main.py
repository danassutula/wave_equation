
from dolfin import *

import numpy as np
import matplotlib.pyplot as plt

nx = ny = 240
diagonal = "crossed"
degree = 2
mesh = UnitSquareMesh(nx, ny, diagonal=diagonal)

### Source term

g1 = Expression(" exp(-50*(pow(x[0], 2) + pow(x[1], 2)))", degree=3)
g2 = Expression(" exp(-50*(pow((x[0]-1.0), 2) + pow((x[1]-1.0), 2)))", degree=3)
g3 = Expression(" exp(-50*(pow((x[0]-0.5), 2) + pow((x[1]-0.5), 2)))", degree=4)

g = g3

### Function space

V = VectorFunctionSpace(mesh, 'CG', degree)
S = FunctionSpace(mesh, ('CG' if degree > 1 else 'DG'), degree-1)

v = TestFunction(V)
u = TrialFunction(V)

F = ((div(u)-g)*div(v) + curl(u)*curl(v)) * dx
a, L = lhs(F), rhs(F)

bc = [# DirichletBC(V, Constant(( 0.0,-1.0)), lambda x, on_boundary: (x[0] < 1e-6) and on_boundary),
      # DirichletBC(V, Constant(( 1.0, 0.0)), lambda x, on_boundary: (x[1] < 1e-6) and on_boundary),
      DirichletBC(V, Constant(( 0.0, 0.0)), lambda x, on_boundary: (x[0] > 1.0-1e-6) and on_boundary)]

u = Function(V, name="electric_field")

A = assemble(a)
b = assemble(L)

for bc_i in bc:
    bc_i.apply(A, b)

solve(A, u.vector(), b)

div_E = project(div(u), S)
div_E.rename("div_E", "")

curl_E = project(curl(u), S)
curl_E.rename("curl_E", "")

fh = plt.figure(1)
fh.clear()
plot(u)
plt.show()

File("electric_field.pvd") << u

File("error_div.pvd") << div_E
File("error_curl.pvd") << curl_E

print("norm(u):\n", "{0:.6f}".format(norm(u)))
print("norm(div(u)-g)\n", "{0:.6f}".format(assemble((div_E-g)**2*dx)))
print("norm(curl(u):\n", "{0:.6f}".format(assemble(curl_E**2*dx)))
