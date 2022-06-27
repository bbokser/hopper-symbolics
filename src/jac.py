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
t = sp.Symbol('t')
q0 = sp.Function('q0')(t)
q1 = sp.Function('q1')(t)
q2 = sp.Function('q2')(t)
q3 = sp.Function('q3')(t)
q0d = sp.Function('q0d')(t)
q1d = sp.Function('q1d')(t)
q2d = sp.Function('q2d')(t)
q3d = sp.Function('q3d')(t)
g = geo.V3.symbolic("g")  # Gravity

q = geo.V4.symbolic("q")
qd = geo.V4.symbolic("qd")
inputs = Values()
inputs["q"] = q
inputs["qd"] = qd
inputs["g"] = g
inputs["L"] = L

# --- actuator forward kinematics --- #
l0 = L[0]
l1 = L[1]
l2 = L[2]
l3 = L[3]
l4 = L[4]
l5 = L[5]

d_0 = 0
x0a = l0 * sp.cos(q0)
z0a = l0 * sp.sin(q0)
rho = sp.sqrt((x0a + d_0) ** 2 + z0a ** 2)
x1a = l2 * sp.cos(q2)
z1a = l2 * sp.sin(q2)
h = sp.sqrt((x0a - x1a) ** 2 + (z0a - z1a) ** 2)
mu = sp.acos((l3 ** 2 + h ** 2 - l1 ** 2) / (2 * l3 * h))
eta = sp.acos((h ** 2 + l2 ** 2 - rho ** 2) / (2 * h * l2))
alpha = sp.pi - (eta + mu) + q2
xa = l2 * sp.cos(q2) + (l3 + l4) * sp.cos(alpha) - d_0 + l5 * sp.cos(alpha - sp.pi / 2)
ya = 0
za = l2 * sp.sin(q2) + (l3 + l4) * sp.sin(alpha) + l5 * sp.cos(alpha - sp.pi / 2)
fwd_kin = sp.Matrix([xa, ya, za])

# compute end effector actuator jacobian
Ja = fwd_kin.jacobian([q0, q2])

# compute del/delq(Ja(q)q_dot)q_dot of ee actuator jacobian
qa_dot = sp.Matrix([q0d, q2d])
Ja_dqdot = Ja.multiply(qa_dot)
da = Ja_dqdot.jacobian([q0, q2]) * qa_dot

Ja_0 = Ja.subs(q0, q[0])
Ja_0 = Ja_0.subs(q2, q[2])
Ja_0 = Ja_0.subs(q0d, qd[0])
Ja_0 = Ja_0.subs(q2d, qd[2])

outputs_Jac = Values(Ja=geo.M(Ja_0))
gen_Jac = codegen.Codegen(
    inputs=inputs,
    outputs=outputs_Jac,
    config=codegen.CppConfig(),
    name="Jac",
    # return_key="Jac",
)
Jac_data = gen_Jac.generate_function()

# Print what we generated
print("Files generated in {}:\n".format(Jac_data.output_dir))
for f in Jac_data.generated_files:
    print("  |- {}".format(os.path.relpath(f, Jac_data.output_dir)))