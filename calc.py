from sympy import symbols, exp, diff
from sympy import Symbol, Matrix

# 定义变量
x, y = symbols('x y')

# 定义向量函数
f1 = exp((x + 3 * y) / 4)
f2 = exp((3 * x + y) / 4)

# 计算偏导数
J = [[diff(f1, x), diff(f1, y)],
     [diff(f2, x), diff(f2, y)]]
# J = np.linalg.inv(J)
invJ = J**-1
print("invJ = ", invJ)

# 计算雅可比行列式
jac_det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
print("jac = ", jac_det)

print(jac_det.simplify())  # 打印简化后的雅可比行列式
