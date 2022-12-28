from sympy import *
import numpy as np
import scipy as sp
import math
import time
from sympy.plotting import plot3d

x = Symbol('x')
y = Symbol('y')


def uniVariateSearch(iterations, obj_fun, initial):
    start = time.time()
    e = 0.01
    x_point = np.copy(initial)
    s = np.array([[1], [0]])
    a = np.zeros(shape=(20, 2))

    f_sub = obj_fun.subs([(x, x_point[0][0]), (y, x_point[1][0])])
    f_pos_pt = x_point + e*s
    f_obj_pos = obj_fun.subs([(x, f_pos_pt[0][0]), (y, f_pos_pt[1][0])])
    f_neg_pt = x_point - e*s
    f_obj_neg = obj_fun.subs([(x, f_neg_pt[0][0]), (y, f_neg_pt[1][0])])

    if f_obj_neg < f_sub:
        s_temp = -s
    else:
        s_temp = s

    i = 0
    while i < iterations:
        l = Symbol('l') # lambda
        x_point_temp = x_point + l*s_temp
        # optimize step length
        lf = obj_fun.subs([(x, x_point_temp[0][0]), (y, x_point_temp[1][0])])
        res_lf = lambdify(l, lf)
        l = sp.optimize.minimize(res_lf, 0).x
        x_point = x_point + l*s_temp
        a[i] = [x_point[0][0], x_point[1][0]]

        # flip x and y in direction
        if s[0][0] == 1 and s[1][0] == 0:
            s = np.array([[0], [1]])
        elif s[0, 0] == 0 and s[1, 0] == 1:
            s = np.array([[1], [0]])

        f_sub = obj_fun.subs([(x, x_point[0][0]), (y, x_point[1][0])])
        f_pos_pt = x_point + e * s
        f_obj_pos = obj_fun.subs([(x, f_pos_pt[0][0]), (y, f_pos_pt[1][0])])
        f_neg_pt = x_point - e * s
        f_obj_neg = obj_fun.subs([(x, f_neg_pt[0][0]), (y, f_neg_pt[1][0])])

        if f_obj_neg < f_sub:
            s_temp = -s
        else:
            s_temp = s

        i = i + 1

    end = time.time()

    print("plot: ")
    plot3d(obj_fun, markers=[{'args': [[a[19][0]], [a[19][1]], [obj_fun.subs([(x, a[19][0]), (y, a[19][1])])], "bo"]}])
    print("Initial guess: \n", initial)
    print("Initial objective function value: \n", obj_fun.subs([(x, initial[0][0]), (y, initial[1][0])]))
    print("Point of minima: \n", x_point)
    print("Objective function minimum value after optimization: \n", obj_fun.subs([(x, x_point[0][0]), (y, x_point[1][0])]))
    print("Time taken: \n", end - start, "seconds")


iterationNum = 20
# initial_one = np.array([[0], [0]])

initial_one = np.array([[-0.54545108], [1.50490216]])
initial_two = np.array([[0.43843696], [-1.06639331]])
initial_three = np.array([[-9.4534024], [-2.30101525]])
initial_four = np.array([[-2.06663461], [-6.84905814]])
initial_five = np.array([[2.37736264], [-7.37994077]])


dejong_fun = x**2 + y**2
rosenbrock_fun = 100*(y - x**2)**2 + (1 - x)**2
rastrigin_fun = 20 + (x**2 - 10*cos(2*math.pi*x)) + (y**2 - 10*cos(2*math.pi*y))
easom_fun = -cos(x)*cos(y) * exp(-(x-math.pi)**2-(y-math.pi)**2)
branin_fun = (y-(5.1/(4*math.pi**2))*x**2+(5/math.pi)*x-6)**2+10*(1-(1/(8*math.pi)))*cos(x)+10

print("dejong stuff")
uniVariateSearch(iterationNum, dejong_fun, initial_one) # solution: [0, 0]
uniVariateSearch(iterationNum, dejong_fun, initial_two)
uniVariateSearch(iterationNum, dejong_fun, initial_three)
uniVariateSearch(iterationNum, dejong_fun, initial_four)
uniVariateSearch(iterationNum, dejong_fun, initial_five)

print("rosenbrocks")
uniVariateSearch(iterationNum, rosenbrock_fun, initial_one) # solution: [1, 1]
uniVariateSearch(iterationNum, rosenbrock_fun, initial_two)
uniVariateSearch(iterationNum, rosenbrock_fun, initial_three)
uniVariateSearch(iterationNum, rosenbrock_fun, initial_four)
uniVariateSearch(iterationNum, rosenbrock_fun, initial_five)


print("rastrigin")
uniVariateSearch(iterationNum, rastrigin_fun, initial_one) # solution: [0, 0]
uniVariateSearch(iterationNum, rastrigin_fun, initial_two)
uniVariateSearch(iterationNum, rastrigin_fun, initial_three)
uniVariateSearch(iterationNum, rastrigin_fun, initial_four)
uniVariateSearch(iterationNum, rastrigin_fun, initial_five)

print("easom")
uniVariateSearch(iterationNum, easom_fun, initial_one) # solution: [pi,pi]
uniVariateSearch(iterationNum, easom_fun, initial_two)
uniVariateSearch(iterationNum, easom_fun, initial_three)
uniVariateSearch(iterationNum, easom_fun, initial_four)
uniVariateSearch(iterationNum, easom_fun, initial_five)

print("branin")
uniVariateSearch(iterationNum, branin_fun, initial_one) # solution: [-pi,12.275], [pi,2.275], [9.42478,2.475]
uniVariateSearch(iterationNum, branin_fun, initial_two)
uniVariateSearch(iterationNum, branin_fun, initial_three)
uniVariateSearch(iterationNum, branin_fun, initial_four)
uniVariateSearch(iterationNum, branin_fun, initial_five)



