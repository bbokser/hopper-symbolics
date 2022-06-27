import os
import symforce
symforce.set_backend("sympy")
symforce.set_log_level("warning")

from symforce import codegen
from symforce import geo
from symforce import sympy as sp
from symforce.values import Values


# Define symbols
L = geo.V6.symbolic("L").T  # Link Lengths
m = geo.V4.symbolic("m").T  # Link Masses
l_c0 = geo.V3.symbolic("l_c0").T  # center of mass locations
l_c1 = geo.V3.symbolic("l_c1").T
l_c2 = geo.V3.symbolic("l_c2").T
l_c3 = geo.V3.symbolic("l_c3").T
I = geo.V4.symbolic("I")
t = sp.Symbol('t')
q0 = sp.Function('q0')(t)
q1 = sp.Function('q1')(t)
q2 = sp.Function('q2')(t)
q3 = sp.Function('q3')(t)
q0d = sp.Function('q0d')(t)
q1d = sp.Function('q1d')(t)
q2d = sp.Function('q2d')(t)
q3d = sp.Function('q3d')(t)
q0dd = sp.Symbol('q0dd')
q1dd = sp.Symbol('q1dd')
q2dd = sp.Symbol('q2dd')
q3dd = sp.Symbol('q3dd')
g = geo.V3.symbolic("g")  # Gravity

q = geo.V4.symbolic("q")
qd = geo.V4.symbolic("qd")
inputs = Values()
inputs["q"] = q
inputs["qd"] = qd
inputs["g"] = g

# with inputs.scope("constants"):  # TODO: Figure out how tf this works
inputs["m"] = m
inputs["L"] = L
inputs["l_c0"] = l_c0
inputs["l_c1"] = l_c1
inputs["l_c2"] = l_c2
inputs["l_c3"] = l_c3
inputs["I"] = I

# with inputs.scope("params"):
#     inputs["g"] = g

# --- Forward Kinematics --- #
l0 = L[0]
l1 = L[1]
l2 = L[2]
l3 = L[3]
l4 = L[4]
l5 = L[5]

m0 = m[0]
m1 = m[1]
m2 = m[2]
m3 = m[3]

I0 = I[0]
I1 = I[1]
I2 = I[2]
I3 = I[3]

x0 = l_c0[0] * sp.cos(q0)
y0 = l_c0[1]
z0 = l_c0[2] * sp.sin(q0)

x1 = l0 * sp.cos(q0) + l_c1[0] * sp.cos(q0 + q1)
y1 = l_c1[1]
z1 = l0 * sp.sin(q0) + l_c1[2] * sp.sin(q0 + q1)

x2 = l_c2[0] * sp.cos(q2)
y2 = l_c2[1]
z2 = l_c2[2] * sp.sin(q2)

x3 = l2 * sp.cos(q2) + l_c3[0] * sp.cos(q2 + q3)
y3 = l_c3[1]
z3 = l2 * sp.sin(q2) + l_c3[2] * sp.sin(q2 + q3)

# Potential energy
r0 = sp.Matrix([x0, y0, z0])
r1 = sp.Matrix([x1, y1, z1])
r2 = sp.Matrix([x2, y2, z2])
r3 = sp.Matrix([x3, y3, z3])

U0 = m0 * (g.T * r0)
U1 = m1 * (g.T * r1)
U2 = m2 * (g.T * r2)
U3 = m3 * (g.T * r3)

U = (U0 + U1 + U2 + U3)[0, 0]  # + Uk

# Kinetic energy
x0d = sp.diff(x0, t)
z0d = sp.diff(z0, t)
x1d = sp.diff(x1, t)
z1d = sp.diff(z1, t)
x2d = sp.diff(x2, t)
z2d = sp.diff(z2, t)
x3d = sp.diff(x3, t)
z3d = sp.diff(z3, t)

v0_sq = x0d ** 2 + z0d ** 2
v1_sq = x1d ** 2 + z1d ** 2
v2_sq = x2d ** 2 + z2d ** 2
v3_sq = x3d ** 2 + z3d ** 2
T0 = 0.5 * m0 * v0_sq + 0.5 * I0 * q0d ** 2
T1 = 0.5 * m1 * v1_sq + 0.5 * I1 * q1d ** 2
T2 = 0.5 * m2 * v2_sq + 0.5 * I2 * q2d ** 2
T3 = 0.5 * m3 * v3_sq + 0.5 * I3 * q3d ** 2
T = T0 + T1 + T2 + T3

# Le Lagrangian
L = sp.trigsimp(T - U)
L = L.subs(sp.Derivative(q0, t), q0d)  # substitute d/dt q2 with q2d
L = L.subs(sp.Derivative(q1, t), q1d)  # substitute d/dt q1 with q1d
L = L.subs(sp.Derivative(q2, t), q2d)  # substitute d/dt q2 with q2d
L = L.subs(sp.Derivative(q3, t), q3d)  # substitute d/dt q2 with q2d

# Euler-Lagrange Equation
LE0 = sp.diff(sp.diff(L, q0d), t) - sp.diff(L, q0)
LE1 = sp.diff(sp.diff(L, q1d), t) - sp.diff(L, q1)
LE2 = sp.diff(sp.diff(L, q2d), t) - sp.diff(L, q2)
LE3 = sp.diff(sp.diff(L, q3d), t) - sp.diff(L, q3)
LE = sp.Matrix([LE0, LE1, LE2, LE3])

# subs first derivative
LE = LE.subs(sp.Derivative(q0, t), q0d)  # substitute d/dt q1 with q1d
LE = LE.subs(sp.Derivative(q1, t), q1d)  # substitute d/dt q1 with q1d
LE = LE.subs(sp.Derivative(q2, t), q2d)  # substitute d/dt q2 with q2d
LE = LE.subs(sp.Derivative(q3, t), q3d)  # substitute d/dt q1 with q1d

# subs second derivative
LE = LE.subs(sp.Derivative(q0d, t), q0dd)  # substitute d/dt q1d with q1dd
LE = LE.subs(sp.Derivative(q1d, t), q1dd)  # substitute d/dt q1d with q1dd
LE = LE.subs(sp.Derivative(q2d, t), q2dd)  # substitute d/dt q2d with q2dd
LE = LE.subs(sp.Derivative(q3d, t), q3dd)  # substitute d/dt q1d with q1dd
LE = sp.expand(sp.simplify(LE))

# Generalized mass matrix
M = sp.zeros(4, 4)
M[0, 0] = sp.collect(LE[0], q0dd).coeff(q0dd)
M[0, 1] = sp.collect(LE[0], q1dd).coeff(q1dd)
M[0, 2] = sp.collect(LE[0], q2dd).coeff(q2dd)
M[0, 3] = sp.collect(LE[0], q3dd).coeff(q3dd)
M[1, 0] = sp.collect(LE[1], q0dd).coeff(q0dd)
M[1, 1] = sp.collect(LE[1], q1dd).coeff(q1dd)
M[1, 2] = sp.collect(LE[1], q2dd).coeff(q2dd)
M[1, 3] = sp.collect(LE[1], q3dd).coeff(q3dd)
M[2, 0] = sp.collect(LE[2], q0dd).coeff(q0dd)
M[2, 1] = sp.collect(LE[2], q1dd).coeff(q1dd)
M[2, 2] = sp.collect(LE[2], q2dd).coeff(q2dd)
M[2, 3] = sp.collect(LE[2], q3dd).coeff(q3dd)
M[3, 0] = sp.collect(LE[3], q0dd).coeff(q0dd)
M[3, 1] = sp.collect(LE[3], q1dd).coeff(q1dd)
M[3, 2] = sp.collect(LE[3], q2dd).coeff(q2dd)
M[3, 3] = sp.collect(LE[3], q3dd).coeff(q3dd)

# Gravity Matrix
G = LE
G = G.subs(q0d, 0)
G = G.subs(q1d, 0)  # must remove q derivative terms manually
G = G.subs(q2d, 0)
G = G.subs(q3d, 0)
G = G.subs(q0dd, 0)
G = G.subs(q1dd, 0)
G = G.subs(q2dd, 0)
G = G.subs(q3dd, 0)

# Coriolis Matrix
# assume anything without qdd minus G is C
C = LE
C = C.subs(q0dd, 0)
C = C.subs(q1dd, 0)
C = C.subs(q2dd, 0)
C = C.subs(q3dd, 0)
C = C - G

# ------------- #
M = M.subs(q0, q[0])
M = M.subs(q1, q[1])
M = M.subs(q2, q[2])
M = M.subs(q3, q[3])
M = M.subs(q0d, qd[0])
M = M.subs(q1d, qd[1])
M = M.subs(q2d, qd[2])
M = M.subs(q3d, qd[3])

G = G.subs(q0, q[0])
G = G.subs(q1, q[1])
G = G.subs(q2, q[2])
G = G.subs(q3, q[3])

C = C.subs(q0, q[0])
C = C.subs(q1, q[1])
C = C.subs(q2, q[2])
C = C.subs(q3, q[3])
C = C.subs(q0d, qd[0])
C = C.subs(q1d, qd[1])
C = C.subs(q2d, qd[2])
C = C.subs(q3d, qd[3])

outputs_DEL = Values(M=geo.M(M), C=geo.M(C), G=geo.M(G))

gen_DEL = codegen.Codegen(
    inputs=inputs,
    outputs=outputs_DEL,
    config=codegen.CppConfig(),
    name="DEL",
    # return_key="DEL",
)
lagrange_data = gen_DEL.generate_function()

# Print what we generated
print("Files generated in {}:\n".format(lagrange_data.output_dir))
for f in lagrange_data.generated_files:
    print("  |- {}".format(os.path.relpath(f, lagrange_data.output_dir)))


