def lambda_a(p_a, p):
    global lambda_max, n
    return lambda_max * (1 - (p_a / p) ** ((n - 1) / n)) ** 0.5


def dzeta(lam):
    global z, k
    return (1 - z) + z * (2 * k) / (k + 1) * (lam) ** 2 / (lam ** 2 + 1)


def eps(lam):
    global k
    return (1 - (k - 1) / (k + 1) * lam ** 2) ** (1 / (k - 1))


def pi(lam):
    global k
    return (1 - (k - 1) / (k + 1) * lam ** 2) ** (k / (k - 1))


def q(lam):
    global k
    return lam * eps(lam) / ((2 / (k + 1)) ** (1 / (k - 1)))


def tau(lam):
    global k
    return 1 - ((k - 1) / (k + 1)) * lam ** 2


def I_ud(p, lam):
    global betta, p_h
    return betta * ((lam + lam ** -1) * eps(1) * dzeta(lam) - p_h / p * 1 / q(lam))


def f_a_bound1(p, lam):
    global etta_f, p_h
    return etta_f / (p / p_h * (lam ** 2 + 1) * eps(lam) * dzeta(lam) - 1)


def f_a_bound2(p, lam):
    global etta_f, p_h
    return etta_f / (p / p_h * (lam ** 2 + 1) / tau(lam) * dzeta(lam) - 1)


def f_a_bound3(I_ud3, lam):
    global etta_f, betta
    return etta_f * (betta / I_ud3 * (lam + lam ** -1) * eps(1) * dzeta(lam) - 1)


def v_a(lam):
    return 1 / q(lam)


def p_a3(p, lam):
    return p * pi(lam)

def solver(a, b, c):
    # находим дискриминант
    D = b*b - 4*a*c
    if D >= 0:
        x1 = (-b + np.sqrt(D)) / (2*a)
        x2 = (-b - np.sqrt(D)) / (2*a)
        res = [x1, x2]
    else:
        res = "The discriminant is: %s \n This equation has no solutions" % D
    return res

def u(p):
    return u_1 * (p) ** mu