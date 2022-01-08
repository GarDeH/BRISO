import numpy as np
import pandas as pd




class get_ed_avg():
    """Класс get_ed_avg используется для определения типа формы заряда.
     В данной работе рассмотрены два типа формы заряда: щелевой и звездообразный.
    Methods
    -------
    lambda_a(self, p):
        При заданном давлении (р = pном) приведённая скорость потока в выходном
        сечении сопла для заданных точек находится из газодинамической (ГД) функции.

    dzeta(self, lam):
        Конденсированная фаза в продуктах сгорания.

    eps(self, lam):
        ГД функция.

    pi(self, lam):
        ГД функция.

    q(self, lam):
        ГД функция.

    tau(self, lam):
        ГД функция.

    I_ud(self, p, lam):
        Удельный импульс двигателя, реализуемый в точках 1 и 2.

    f_a_bound1(self, p, lam):
        Относительная площадь выходного сечения сопла на линии 1-2.

    f_a_bound2(self, p, lam):
        Относительная площадь выходного сечения сопла на линии 3-2.

    f_a_bound3(self, I_ud3, lam):
        Относительная площадь выходного сечения сопла на линии 3-1.

    v_a(self, lam):
        Степень расширения сопла.

    p_a3(self, p, lam):
        ГД функция.

    u(self, p):
        Закон горения.

    solver(self, a, b, c):
        Функция для определения корней квадратного уравнения. Необходима для
        определния минимального значения lambda_a диапазона.

    get_lambda_a_max(self, lambda_a):
        Функция для определния максимального значения lambda_a диапазона.

    get_result(self, s_df=0):
        Функция, в которой представлено решение задачи
        (s_df=0 - не сохранять dataFrames, s_df=1 - сохранять dataFrames).

        ----------------------------------------------------------
    Класс Volume используется для определения объёмов частей ДУ (используется в классе get_powder).
    Methods
    -------
    cil_ks(self):
        Функция для определения объёма цилиндричекой оболочки обечайки и защитно-крепящего слоя.

    def ell_pd(self):
        Функция для определения объёма эллиптического переднего днища обечайки и теплозащитного попрытия.

    def ell_sd(self):
        Функция для определения объёма эллиптического соплового днища обечайки и теплозащитного попрытия.

    def cil_st(self):
        Функция для определения объёма цилиндричекой оболочки соплового стакана и
        вкладыша критического сечения.

    def kon_1(self):
        Функция для определения объёма конической дозвуковой части сопла и теплозащитного попрытия.

    def kon_2(self):
        Функция для определения объёма конической сверхзвуковой части сопла и теплозащитного попрытия."""

    def __init__(self, INIT):
        """# Пример начальных данных класса get_ed_avg:
        INIT_1 = {'p_min': 4 * 1e6, 'p_max': 20 * 1e6, 'p_h': 0.1 * 1e6, 'p_a': 0.1 * 1e6,
        'f_a_max': 0.9, 'D_n': 610 * 1e-3, 'hi': 0.98, 'R_g': 550.84, 'n_kr': 1.1604, 'k': 1.1755,
        'z': 0.3239, 'T_r': 2999.5, 'I_P': 2250 * 1e3, 't_nom': 20, 'nu': 0.29, 'u_1': 4.38}

        res_1 = get_ed_avg(INIT_1).get_result(s_df=0)

        # Пример начальных данных класса Volume:
        INIT_V = [self.D_n, l_ob_arr, delta_ob, self.delta_zks, b_1, b_2, delta_dn1, delta_dn2, self.delta_p1,
        self.delta_p2, d, D_vh,
        D_vih, D_kr, self.tetta_1, self.tetta_2, l_kr, delta_s1, delta_s2, delta_st, self.delta_ps1,
        self.delta_ps2, self.delta_vks]"""

        self.p_min = INIT['p_min'] * 1e6
        self.p_max = INIT['p_max'] * 1e6
        self.p_h = INIT['p_h'] * 1e6
        self.p_a = INIT['p_a'] * 1e6
        self.f_a_max = INIT['f_a_max']
        self.D_n = INIT['D_n'] * 1e-3
        self.hi = INIT['hi']
        self.R_g = INIT['R_g']
        self.n_kr = INIT['n_kr']
        self.k = INIT['k']
        self.z = INIT['z']
        self.T_r = INIT['T_r']
        self.I_P = INIT['I_P'] * 1e3
        self.t_nom = INIT['t_nom']
        self.nu = INIT['nu']
        self.u_1 = INIT['u_1'] * 1e-3
        self.n_t = self.k
        self.F_m = np.pi * self.D_n ** 2 / 4
        self.P = self.I_P / self.t_nom
        self.lambda_max = ((self.n_t + 1) / (self.n_t - 1)) ** 0.5
        self.R_sm = self.R_g * (1 - self.z)
        self.A_n = (self.n_kr * (2 / (self.n_kr + 1)) **
                    ((self.n_kr + 1) / (self.n_kr - 1))) ** 0.5
        self.betta = (self.R_sm * self.hi * self.T_r) ** 0.5 / self.A_n
        self.etta_f = self.P / (self.p_h * self.F_m)
        self.D_ks = 0.96 * self.D_n  # m

    def lambda_a(self, p):
        return self.lambda_max * (1 - (self.p_a / p) ** ((self.n_t - 1) / self.n_t)) ** 0.5

    def dzeta(self, lam):
        return (1 - self.z) + self.z * (2 * self.k) / (self.k + 1) * (lam) ** 2 / (lam ** 2 + 1)

    def eps(self, lam):
        return (1 - (self.k - 1) / (self.k + 1) * lam ** 2) ** (1 / (self.k - 1))

    def pi(self, lam):
        return (1 - (self.k - 1) / (self.k + 1) * lam ** 2) ** (self.k / (self.k - 1))

    def q(self, lam):
        return lam * get_ed_avg.eps(self, lam) / ((2 / (self.k + 1)) ** (1 / (self.k - 1)))

    def tau(self, lam):
        return 1 - ((self.k - 1) / (self.k + 1)) * lam ** 2

    def I_ud(self, p, lam):
        return self.betta * ((lam + lam ** -1) * get_ed_avg.eps(self, 1) * get_ed_avg.dzeta(self,
                                                                                            lam) - self.p_h / p * 1 / get_ed_avg.q(
            self, lam))

    def f_a_bound1(self, p, lam):
        return self.etta_f / (
                    p / self.p_h * (lam ** 2 + 1) * get_ed_avg.eps(self, lam) * get_ed_avg.dzeta(self, lam) - 1)

    def f_a_bound2(self, p, lam):
        return self.etta_f / (
                    p / self.p_h * (lam ** 2 + 1) / get_ed_avg.tau(self, lam) * get_ed_avg.dzeta(self, lam) - 1)

    def f_a_bound3(self, I_ud3, lam):
        return self.etta_f * (
                    self.betta / I_ud3 * (lam + lam ** -1) * get_ed_avg.eps(self, 1) * get_ed_avg.dzeta(self, lam) - 1)

    def v_a(self, lam):
        return 1 / get_ed_avg.q(self, lam)

    def p_a3(self, p, lam):
        return p * get_ed_avg.pi(self, lam)

    def u(self, p):
        return self.u_1 * (p) ** self.nu

    def solver(self, a, b, c):
        # находим дискриминант
        D = b * b - 4 * a * c
        if D >= 0:
            x1 = (-b + np.sqrt(D)) / (2 * a)
            x2 = (-b - np.sqrt(D)) / (2 * a)
            res = [x1, x2]
        else:
            res = "The discriminant is: %s \n This equation has no solutions" % D
        return res

    def get_lambda_a_max(self, lambda_a):
        return self.etta_f / (
                    self.p_max / self.p_h * (lambda_a ** 2 + 1) * get_ed_avg.eps(self, lambda_a) * get_ed_avg.dzeta(
                self, lambda_a) - 1)

    def get_result(self, s_df=0):
        lambda_a1 = get_ed_avg.lambda_a(self, self.p_min)
        f_a1 = get_ed_avg.f_a_bound1(self, self.p_min, lambda_a1)
        v_a1 = get_ed_avg.v_a(self, lambda_a1)
        I_ud1 = get_ed_avg.I_ud(self, self.p_min, lambda_a1)
        print('Первая точка:')
        print('lambda_a1 =', lambda_a1)
        print('v_a1 =', v_a1)
        print('f_a1 =', f_a1)
        print('p_nom1 =', self.p_min)
        print('p_a1 / p_h1 =', get_ed_avg.p_a3(self,
                                               self.p_min, lambda_a1) / self.p_h)
        print('I_ud1 =', I_ud1)
        print('\n')

        print('Вторая точка:')
        lambda_a2 = get_ed_avg.lambda_a(self, self.p_max)
        f_a2 = round(get_ed_avg.f_a_bound1(self, self.p_max, lambda_a2), 2)
        v_a2 = get_ed_avg.v_a(self, lambda_a2)
        I_ud2 = get_ed_avg.I_ud(self, self.p_max, lambda_a2)
        print('lambda_a2 =', lambda_a2)
        print('v_a2 =', v_a2)
        print('f_a2 =', f_a2)
        print('p_nom2 =', self.p_max)
        print('p_a2 / p_h2 =', get_ed_avg.p_a3(self,
                                               self.p_max, lambda_a2) / self.p_h)
        print('I_ud2 =', I_ud2)
        print('\n')

        print('Третья точка:')
        N = 1000
        i = 1
        while i <= lambda_a1 - 0.1:
            if get_ed_avg.I_ud(self, self.p_max, i) < I_ud1:
                I_ud3 = get_ed_avg.I_ud(self, self.p_max, i)
                lambda_a3 = i
            else:
                break
            i += 0.001
        f_a3 = get_ed_avg.f_a_bound3(self, I_ud3, lambda_a3)
        v_a3 = get_ed_avg.v_a(self, lambda_a3)
        print('lambda_a3 =', lambda_a3)
        print('v_a3 =', v_a3)
        print('f_a3 =', f_a3)
        print('p_nom3 =', self.p_max)
        print('p_a3 / p_h3 =', get_ed_avg.p_a3(self,
                                               self.p_max, lambda_a3) / self.p_h)
        print('I_ud3 =', I_ud3)

        f_a_list_1 = [f_a1, f_a2, f_a3]  # для построения графиков
        v_a_list_1 = [v_a1, v_a2, v_a3]

        f_1_2, f_3_2, f_3_1, v_1_2, v_3_2, v_3_1 = [], [], [], [], [], []
        for i in range(int(lambda_a1 * N), int(lambda_a2 * N)):
            f_1_2.append(get_ed_avg.f_a_bound2(self, self.p_a, i / N))
            v_1_2.append(get_ed_avg.v_a(self, i / N))

        for i in range(int(lambda_a3 * N), int(lambda_a2 * N)):
            f_3_2.append(get_ed_avg.f_a_bound1(self, self.p_max, i / N))
            v_3_2.append(get_ed_avg.v_a(self, i / N))

        for i in range(int(lambda_a3 * N), int(lambda_a1 * N)):
            f_3_1.append(get_ed_avg.f_a_bound3(self, I_ud1, i / N))
            v_3_1.append(get_ed_avg.v_a(self, i / N))

        print('Определение lambda_a_min и lambda_a_max:')

        if f_a2 < self.f_a_max:
            f_a = f_a2
            a = 1 + self.z * (self.n_t - 1) / (self.n_t + 1)
            b = - I_ud1 / (self.betta * get_ed_avg.eps(self, 1)
                           ) * (1 + f_a / self.etta_f)
            c = 1 - self.z
            lambda_a_min = round(max(get_ed_avg.solver(self, a, b, c)), 2)
            lambda_a_max = round(lambda_a2, 2)
        else:
            f_a = self.f_a_max
            a = 1 + self.z * (self.n_t - 1) / (self.n_t + 1)
            b = - I_ud1 / (self.betta * get_ed_avg.eps(self, 1)
                           ) * (1 + f_a / self.etta_f)
            c = 1 - self.z
            lambda_a_min = round(max(get_ed_avg.solver(self, a, b, c)), 2)
            i = 0
            while i < 5:
                if 0.8999 <= get_ed_avg.get_lambda_a_max(self, i) <= 0.9001:
                    lambda_a_max = i
                    break
                i += 0.00001
        print('a =', a)
        print('b =', b)
        print('c =', c)
        print('lambda_a_min =', lambda_a_min)
        print('lambda_a_max =', lambda_a_max)

        print('Определение параметров на линии f_a:')
        lambda_range = np.linspace(lambda_a_min, lambda_a_max, 11)

        v_a_arr, p_nom_arr, p_div_p_a_arr, I_ud_arr, f_a_arr = np.array(
            []), np.array([]), np.array([]), np.array([]), np.array([])
        for l in lambda_range:
            I_udp = self.betta * \
                    (l + l ** -1) * get_ed_avg.eps(self, 1) * \
                    get_ed_avg.dzeta(self, l)
            I_ud = I_udp * self.etta_f / (self.etta_f + f_a)
            p_nom = self.p_h * self.betta * \
                    get_ed_avg.v_a(self, l) / (I_udp - I_ud)
            p_a = get_ed_avg.p_a3(self, p_nom, l)

            v_a_arr = np.append(v_a_arr, [get_ed_avg.v_a(self, l)])
            p_nom_arr = np.append(p_nom_arr, [p_nom / 1e6])
            p_div_p_a_arr = np.append(p_div_p_a_arr, [p_a / self.p_h])
            I_ud_arr = np.append(I_ud_arr, [I_ud])
            f_a_arr = np.append(f_a_arr, [f_a])

        data_1 = [np.round(lambda_range, 3), np.round(v_a_arr, 3), np.round(f_a_arr, 3),
                  np.round(p_nom_arr, 3), np.round(p_div_p_a_arr, 3), np.round(I_ud_arr, 3)]
        data_1_show = pd.DataFrame(data_1, index=["lambda_a", "v_a", "f_a_arr", "p_nom_arr, МПа",
                                                  "p/p_a_arr", "I_ud_arr, м/с"], columns=[i for i in range(1, 12)]).T
        pd.set_option('max_colwidth', 15)
        display(data_1_show)
        if s_df == 1:
            data_1_show.to_excel('data_1.xlsx')
        print('Определение среднего значения e_d:')
        e_d_arr, u_arr = np.array([]), np.array([])
        for i in p_nom_arr:
            e_d = 2 / self.D_ks * get_ed_avg.u(self, i) * self.t_nom
            e_d_arr = np.append(e_d_arr, e_d)
            u_arr = np.append(u_arr, get_ed_avg.u(self, i) * 1e3)

        avg = np.average(e_d_arr)
        print('e_d_list:', *e_d_arr, sep='\n')
        print('\naverage =', avg)
        if 0.25 > avg:
            print('\ne_d < 0.25 -> топливо не является пригодным')
        elif 0.75 < avg:
            print('\ne_d > 0.75 -> топливо не является пригодным')
        elif 0.5 <= avg <= 0.75:
            print('\n0.5 <= e_d <= 0.75 -> заряд щелевой формы')
        elif 0.25 <= avg < 0.5:
            print('\n0.25 <= e_d <= 0.5 -> заряд звездообразной формы')
            data_1 = [np.round(lambda_range, 3), np.round(v_a_arr, 3), np.round(f_a_arr, 3),
                      np.round(p_nom_arr, 3), np.round(p_div_p_a_arr, 3), np.round(I_ud_arr, 3)]
        return {'lambda_range': lambda_range, 'v_a_arr': v_a_arr, 'f_a_arr': f_a_arr,
                'e_d_arr': e_d_arr, 'p_nom_arr': p_nom_arr, 'p_div_p_a_arr': p_div_p_a_arr,
                'I_ud_arr': I_ud_arr, 'u_arr': u_arr, 'f_a_list_1': f_a_list_1, 'v_a_list_1': v_a_list_1,
                'f_1_2': f_1_2, 'f_3_2': f_3_2, 'f_3_1': f_3_1, 'v_1_2': v_1_2, 'v_3_2': v_3_2, 'v_3_1': v_3_1}


class Volume():
    def __init__(self, INIT):
        self.D_n = INIT[0]
        self.l_ob = INIT[1]
        self.delta_ob = INIT[2]
        self.delta_zks = INIT[3]
        self.b_1 = INIT[4]
        self.b_2 = INIT[5]
        self.delta_dn1 = INIT[6]
        self.delta_dn2 = INIT[7]
        self.delta_p1 = INIT[8]
        self.delta_p2 = INIT[9]
        self.d = INIT[10]
        self.D_vh = INIT[11]
        self.D_vih = INIT[12]
        self.D_kr = INIT[13]
        self.tetta_1 = INIT[14]
        self.tetta_2 = INIT[15]
        self.l_kr = INIT[16]
        self.delta_s1 = INIT[17]
        self.delta_s2 = INIT[18]
        self.delta_st = INIT[19]
        self.delta_ps1 = INIT[20]
        self.delta_ps2 = INIT[21]
        self.delta_vks = INIT[22]

    def cil_ks(self):
        d_ob = self.D_n - 2 * self.delta_ob
        S_ob = np.pi / 4 * (self.D_n ** 2 - d_ob ** 2)
        V_ob = S_ob * self.l_ob
        d_zks = d_ob - 2 * self.delta_zks
        S_zks = np.pi / 4 * (d_ob ** 2 - d_zks ** 2)
        V_zks = S_zks * self.l_ob
        return np.array([V_ob, V_zks])

    def ell_pd(self):
        V_dn1 = 2 / 3 * np.pi * self.delta_dn1 * \
                (self.D_n ** 2 / 4 + self.D_n * self.b_1 - (self.D_n + self.b_1)
                 * self.delta_dn1 + self.delta_dn1 ** 2)
        V_p1 = 2 / 3 * np.pi * self.delta_p1 * ((self.D_n - 2 * self.delta_dn1) ** 2 / 4 + (
                self.D_n - 2 * self.delta_dn1) * (self.b_1 - self.delta_dn1) - (
                                                            self.D_n - 2 * self.delta_dn1 + self.b_1 - self.delta_dn1) * self.delta_p1 + self.delta_p1 ** 2)
        return np.array([V_dn1, V_p1])

    def ell_sd(self):
        V_dn2 = np.pi / 6 * self.D_n ** 2 * self.b_1 * (1 - (self.d / self.D_n) ** 2) ** (3 / 2) - np.pi / 6 * (
                self.D_n - 2 * self.delta_dn1) ** 2 * (self.b_1 - self.delta_dn1) * (
                            1 - self.d ** 2 / (self.D_n - 2 * self.delta_dn1) ** 2) ** (3 / 2)
        V_p2 = np.pi / 6 * (self.D_n - 2 * self.delta_dn1) ** 2 * (self.b_1 - self.delta_dn1) * (
                    1 - (self.d / (self.D_n - 2 * self.delta_dn1)) ** 2) ** (3 / 2) - np.pi / 6 * (
                       (self.D_n - 2 * self.delta_dn1) - 2 * self.delta_p1) ** 2 * (
                           self.b_1 - self.delta_dn1 - self.delta_p1) * (
                           1 - self.d ** 2 / (self.D_n - 2 * self.delta_dn1 - 2 * self.delta_p1) ** 2) ** (3 / 2)
        return np.array([V_dn2, V_p2])

    def cil_st(self):
        d_st = self.D_kr + 2 * self.delta_vks
        S_st = np.pi / 4 * ((self.D_kr + 2 * self.delta_st +
                             2 * self.delta_vks) ** 2 - d_st ** 2)
        V_st = S_st * self.l_kr
        D_vks = self.D_kr + 2 * self.delta_vks
        S_vks = np.pi / 4 * (D_vks ** 2 - self.D_kr ** 2)
        V_vks = S_vks * self.l_kr
        return np.array([V_st, V_vks])

    def kon_1(self):
        l_kon_1 = (self.D_vh - self.D_kr) / 2 / np.tan(self.tetta_1)
        V_s1 = np.pi * l_kon_1 * self.delta_s1 / np.cos(self.tetta_1) * (
                (self.D_vh + 2 * self.delta_ps1 / np.cos(self.tetta_1) + self.D_kr + 2 * self.delta_ps1 / np.cos(
                    self.tetta_1)) / 2 + self.delta_s1 / np.cos(self.tetta_1))
        V_ps1 = np.pi * l_kon_1 * self.delta_ps1 / \
                np.cos(self.tetta_1) * ((self.D_vh + self.D_kr) /
                                        2 + self.delta_ps1 / np.cos(self.tetta_1))
        return np.array([V_s1, V_ps1])

    def kon_2(self):
        l_kon_2 = (self.D_vih - self.D_kr) / 2 / np.tan(self.tetta_2)
        V_s2 = np.pi * l_kon_2 * self.delta_s2 / np.cos(self.tetta_2) * (
                (self.D_vih + 2 * self.delta_ps2 / np.cos(self.tetta_2) + self.D_kr + 2 * self.delta_ps2 / np.cos(
                    self.tetta_2)) / 2 + self.delta_s2 / np.cos(self.tetta_2))
        V_ps2 = np.pi * l_kon_2 * self.delta_ps2 / np.cos(self.tetta_2) * (
                (self.D_vih + self.D_kr) / 2 + self.delta_ps2 / np.cos(self.tetta_2))
        return np.array([V_s2, V_ps2])


class get_powder():
    """Класс get_powder используется для определения параметров ДУ для заряда звездообразного или щелевого типа.
    Methods
    -------
    fi_kappa(kappa):
        Функция для определения значения поправки, учитывающей параметр Победоносцева.

    get_result(self, s_df=0):
        Функция, в которой представлено решение задачи."""

    def __init__(self, INIT):
        """# Пример начальных данных класса get_powder:
        INIT_1 = {'p_min': 4 * 1e6, 'p_max': 20 * 1e6, 'p_h': 0.1 * 1e6, 'p_a': 0.1 * 1e6,
        'f_a_max': 0.9, 'D_n': 610 * 1e-3, 'hi': 0.98, 'R_g': 550.84, 'n_kr': 1.1604, 'k': 1.1755,
        'z': 0.3239, 'T_r': 2999.5, 'I_P': 2250 * 1e3, 't_nom': 20, 'nu': 0.29, 'u_1': 4.38}

        res_1 = get_ed_avg(INIT_1).get_result(s_df=0)

        INIT_2_1 = {**INIT_1, **res_1}
        INIT_2_2 = {'delta_p1':6, 'delta_p2':6, 'delta_zks':1, 'delta_vks':15,
        'delta_ps1':6, 'delta_ps2':3, 'ro_p':1500, 'ro_zks':920,
        'ro_ps':1750, 'ro_vks':2200, 'sigma_vr':1830, 'sigma_02':1830,
        'D_t':0.002, 'tetta_1':30,'tetta_2':15, 'ro_k': 7800}

        INIT_2 = {**INIT_2_1,**INIT_2_2}
        INIT_2['count'] = 4
        INIT_2['a_'] = 0.3
        INIT_2['c_'] = 0.05
        INIT_2['r_'] = 0.1
        INIT_2['q_arr'] = [0.14, 0.66, 0.2]
        INIT_2['ro_arr'] = [920, 1950, 1500]
        INIT_2['shape'] = 'slot'

        res_2 = get_powder(INIT_2).get_result(s_df=1)"""

        self.D_n = INIT['D_n'] * 1e-3
        self.F_m = np.pi * self.D_n ** 2 / 4
        self.D_ks = 0.96 * self.D_n  # m
        self.I_P = INIT['I_P'] * 1e3
        self.nu = INIT['nu']
        if INIT['shape'] == 'slot':
            self.flag = INIT['shape']
            self.a_ = INIT['a_']
            self.c_ = INIT['c_']
        elif INIT['shape'] == 'star':
            self.flag = INIT['shape']
            self.r_ = INIT['r_']
        self.count = INIT['count']
        self.lambda_range = INIT['lambda_range']
        self.v_a_arr = INIT['v_a_arr']
        self.f_a_arr = INIT['f_a_arr']
        self.p_nom_arr = INIT['p_nom_arr']
        self.p_div_p_a_arr = INIT['p_div_p_a_arr']
        self.I_ud_arr = INIT['I_ud_arr']
        self.e_d_arr = INIT['e_d_arr']
        self.u_arr = INIT['u_arr']
        self.q_arr = INIT['q_arr']
        self.ro_arr = INIT['ro_arr']
        self.delta_p1 = INIT['delta_p1'] * 1e-3
        self.delta_p2 = INIT['delta_p2'] * 1e-3
        self.delta_zks = INIT['delta_zks'] * 1e-3
        self.delta_vks = INIT['delta_vks'] * 1e-3
        self.delta_ps1 = INIT['delta_ps1'] * 1e-3
        self.delta_ps2 = INIT['delta_ps2'] * 1e-3
        self.ro_p = INIT['ro_p']
        self.ro_zks = INIT['ro_zks']
        self.ro_ps = INIT['ro_ps']
        self.ro_vks = INIT['ro_vks']
        self.sigma_vr = INIT['sigma_vr']
        self.sigma_02 = INIT['sigma_02']
        self.D_t = INIT['D_t']
        self.tetta_1 = np.radians(INIT['tetta_1'])
        self.tetta_2 = np.radians(INIT['tetta_2'])
        self.ro_k = INIT['ro_k']

    def fi_kappa(kappa):
        if kappa > kappa_por:
            return 1 + 0.003 * (kappa - kappa_por)
        else:
            return 1

    def get_result(self, s_df=0):
        if self.flag == 'slot':
            p_nom_fit = self.p_nom_arr[self.e_d_arr < 1 - self.c_]
            e_d_fit = self.e_d_arr[self.e_d_arr < 1 - self.c_]
            u_fit = self.u_arr[self.e_d_arr < 1 - self.c_]
            I_ud_fit = self.I_ud_arr[self.e_d_arr < 1 - self.c_]
            f_a_fit = self.f_a_arr[self.e_d_arr < 1 - self.c_]
            v_a_fit = self.v_a_arr[self.e_d_arr < 1 - self.c_]
            d__arr = 1 - e_d_fit
            eps_f_arr = 1 - d__arr ** 2
            f_sh_arr = self.count / np.pi * (self.c_ * (1 - self.c_ ** 2) ** 0.5 - self.c_ *
                                             (d__arr ** 2 - self.c_ ** 2) ** 0.5 + np.arcsin(
                        self.c_) - d__arr ** 2 * np.arcsin(self.c_ / d__arr))
            eps_omega_arr = eps_f_arr - self.a_ * f_sh_arr
        elif self.flag == 'star':
            i = 0.001
            beta = np.pi / self.count
            while i < 180:
                if np.pi / 2 + beta + 0.001 >= np.cos(np.radians(i)) / np.sin(np.radians(i)) + np.radians(
                        i) >= np.pi / 2 + beta - 0.001:
                    tetta = np.radians(i)
                    break
                i += 0.01
            print(tetta)
            p_nom_fit = self.p_nom_arr[self.e_d_arr >= (
                    np.sin(beta) - self.r_ * np.cos(tetta)) / (np.sin(beta) + np.cos(tetta))]
            e_d_fit = self.e_d_arr[self.e_d_arr >= (
                    np.sin(beta) - self.r_ * np.cos(tetta)) / (np.sin(beta) + np.cos(tetta))]
            u_fit = self.u_arr[self.e_d_arr >= (
                    np.sin(beta) - self.r_ * np.cos(tetta)) / (np.sin(beta) + np.cos(tetta))]
            I_ud_fit = self.I_ud_arr[self.e_d_arr >= (
                    np.sin(beta) - self.r_ * np.cos(tetta)) / (np.sin(beta) + np.cos(tetta))]
            f_a_fit = self.f_a_arr[self.e_d_arr >= (
                    np.sin(beta) - self.r_ * np.cos(tetta)) / (np.sin(beta) + np.cos(tetta))]
            v_a_fit = self.v_a_arr[self.e_d_arr >= (
                    np.sin(beta) - self.r_ * np.cos(tetta)) / (np.sin(beta) + np.cos(tetta))]
            alpha_1 = np.pi / 2 + beta - tetta
            eps_f_arr = 1 - 1 / beta * ((1 - e_d_fit) / (1 + self.r_)) ** 2 * (
                        alpha_1 * self.r_ ** 2 + (np.sin(beta) / np.sin(
                    tetta) - self.r_ * np.cos(tetta) / np.sin(tetta)) * (self.r_ + np.cos(alpha_1)) + self.r_ * np.sin(
                    alpha_1))
            eps_omega_arr = eps_f_arr
        omega_arr = self.I_P / I_ud_fit
        q_i = self.q_arr
        ro_i = self.ro_arr
        total = 0

        for i in range(len(ro_i)):
            total += q_i[i] / ro_i[i]

        ro_t = round(1 / total)
        F_ks = np.pi * (self.D_ks) ** 2 / 4  # m**2

        l_zar_arr = omega_arr / (ro_t * eps_omega_arr * F_ks)
        if self.flag == 'slot':
            kappa_arr = 4 * l_zar_arr / (self.D_ks * (1 - e_d_fit))
        elif self.flag == 'star':
            kappa_arr = 4 * l_zar_arr / (self.D_ks * (1 - eps_f_arr)) * 1 / beta * (1 - e_d_fit) / (
                    1 + self.r_) * (np.sin(beta) / np.sin(tetta) - self.r_ * np.cos(tetta) / np.sin(
                tetta) + self.r_ * alpha_1)
        data_2 = [np.round(u_fit, 3), np.round(e_d_fit, 3), np.round(eps_omega_arr, 3), np.round(
            omega_arr, 3), np.round(l_zar_arr, 3), np.round(kappa_arr, 3)]

        data_2_show = pd.DataFrame(data_2, index=["u_fit, мм/с", "e_d_fit", "eps_omega_arr", "omega_arr, кг",
                                                  "l_zar_arr, м", "kappa_arr"],
                                   columns=[i for i in range(1, len(e_d_fit) + 1)]).T
        # pd.set_option('display.float_format', '{:.5}'.format)
        display(data_2_show)
        if s_df == 1:
            data_2_show.to_excel('data_2.xlsx')

        k_1 = 1.1
        k_2 = 1.2
        etta = 1.25

        T_0 = 273.15 + 50
        T_ref = 293.15
        kappa_por = 100

        p_50 = p_nom_fit * \
               np.exp((self.D_t * (T_0 - T_ref)) / (1 - self.nu)) * \
               1 ** (1 / (1 - self.nu))

        p_p_arr = p_50 * self.sigma_vr / self.sigma_02 * k_1 * k_2 * etta

        b = self.D_n / 4

        delta_ob_arr = self.D_n / 2 * p_p_arr / self.sigma_vr

        delta_dn_arr = self.D_n / 2 * p_p_arr / \
                       self.sigma_vr * ((self.D_n / b) ** 2 / 24 + 1 / 3)

        D_a_arr = np.sqrt(4 / np.pi * f_a_fit * self.F_m)

        D_kr_arr = D_a_arr / np.sqrt(v_a_fit)
        l_ob_arr = l_zar_arr

        delta_ob = delta_ob_arr
        b_1 = b
        b_2 = b
        delta_dn1 = delta_dn_arr
        delta_dn2 = delta_dn_arr
        d = self.D_n / 2

        D_vh = self.D_n / 2
        D_vih = D_a_arr
        D_kr = D_a_arr / np.sqrt(v_a_fit)
        l_kr = D_kr / 2
        delta_s1 = 2 * delta_ob
        delta_s2 = delta_ob
        delta_st = 3 * delta_ob
        INIT_V = [self.D_n, l_ob_arr, delta_ob, self.delta_zks, b_1, b_2, delta_dn1, delta_dn2, self.delta_p1,
                  self.delta_p2, d, D_vh,
                  D_vih, D_kr, self.tetta_1, self.tetta_2, l_kr, delta_s1, delta_s2, delta_st, self.delta_ps1,
                  self.delta_ps2, self.delta_vks]
        V_cil = Volume(INIT_V).cil_ks()
        print('V_об =', V_cil[0])
        print('V_зкс =', V_cil[1])
        V_dn_1 = Volume(INIT_V).ell_pd()
        print('V_дн1 =', V_dn_1[0])
        print('V_п1 =', V_dn_1[1])
        V_dn_2 = Volume(INIT_V).ell_sd()
        print('V_дн2 =', V_dn_2[0])
        print('V_п2 =', V_dn_2[1])
        V_st = Volume(INIT_V).cil_st()
        print('V_ст =', V_st[0])
        print('V_вкс =', V_st[1])
        V_kon_1 = Volume(INIT_V).kon_1()
        print('V_с1 =', V_kon_1[0])
        print('V_пс1 =', V_kon_1[1])
        V_kon_2 = Volume(INIT_V).kon_2()
        print('V_с2 =', V_kon_2[0])
        print('V_пс2 =', V_kon_2[1])

        ro_kor_arr = np.array([self.ro_k] * len(e_d_fit))
        ro_zks_arr = np.array([self.ro_zks] * len(e_d_fit))
        ro_p_arr = np.array([self.ro_p] * len(e_d_fit))
        ro_vks_arr = np.array([self.ro_vks] * len(e_d_fit))
        ro_ps_arr = np.array([self.ro_ps] * len(e_d_fit))
        ro_cil = np.array([ro_kor_arr, ro_zks_arr])
        ro_dn_1 = np.array([ro_kor_arr, ro_p_arr])
        ro_dn_2 = np.array([ro_kor_arr, ro_p_arr])
        ro_st = np.array([ro_kor_arr, ro_vks_arr])
        ro_kon_1 = np.array([ro_kor_arr, ro_ps_arr])
        ro_kon_2 = np.array([ro_kor_arr, ro_ps_arr])
        ro = np.array([ro_cil, ro_dn_1, ro_dn_2, ro_st, ro_kon_1, ro_kon_2])
        V = np.array([V_cil, V_dn_1, V_dn_2, V_st, V_kon_1, V_kon_2])
        m_dv_i = ro * V  # true
        m_dv_zer = [[0] for i in range(len(V))]

        for i in range(len(V)):
            m_dv_zer[i] = m_dv_i[i][0] + m_dv_i[i][1]
        m_dv_0 = sum(m_dv_zer)

        m_dv_arr = m_dv_0 + omega_arr
        alpha_dv_arr = m_dv_0 / omega_arr
        l_dv_arr = b + l_ob_arr + b * np.sqrt(1 - (D_vh / self.D_n) ** 2) + (D_vh - D_kr) / \
                   (2 * np.tan(self.tetta_1)) + D_kr / 2 + \
                   (D_a_arr - D_kr) / (2 * np.tan(self.tetta_2))

        # определение критерия
        c_dv_arr = np.sqrt((m_dv_arr * l_dv_arr) /
                           (min(m_dv_arr) * min(l_dv_arr)))
        c_dv_opt = min(c_dv_arr)
        m_dv_opt = m_dv_arr[c_dv_opt == c_dv_arr]
        l_dv_opt = l_dv_arr[c_dv_opt == c_dv_arr]
        p_nom_opt = p_nom_fit[c_dv_opt == c_dv_arr]
        omega_opt = omega_arr[c_dv_opt == c_dv_arr]
        e_d_opt = e_d_fit[c_dv_opt == c_dv_arr]
        delta_ob_opt = delta_ob[c_dv_opt == c_dv_arr]
        D_kr_opt = D_kr[c_dv_opt == c_dv_arr]
        data_3 = [np.round(delta_ob_arr * 1e3, 3), np.round(D_kr * 1e3, 3), np.round(m_dv_0, 3), np.round(
            m_dv_arr, 3), np.round(alpha_dv_arr, 3), np.round(l_dv_arr * 1e3, 3), np.round(c_dv_arr, 3)]
        data_3_show = pd.DataFrame(data_3, index=["delta_ob_arr, мм", "D_kr, мм", "m_dv_0, кг", "m_dv_arr, кг",
                                                  "alpha_dv_arr", "l_dv_arr, мм", "c_dv_arr"],
                                   columns=[i for i in range(1, len(e_d_fit) + 1)]).T
        # pd.set_option('display.float_format', '{:.5}'.format)
        display(data_3_show)
        if s_df == 1:
            data_3_show.to_excel('data_3.xlsx')
        if self.flag == 'slot':
            return {'p_nom_fit': p_nom_fit, 'e_d_fit': e_d_fit, 'eps_omega_arr': eps_omega_arr,
                    'kappa_arr': kappa_arr, 'u_fit': u_fit, 'omega_arr': omega_arr,
                    'l_zar_arr': l_zar_arr * 1e3, 'delta_ob_arr': delta_ob_arr * 1e3, 'D_kr_opt': D_kr_opt,
                    'm_dv_0': m_dv_0, 'm_dv_arr': m_dv_arr, 'alpha_dv_arr': alpha_dv_arr,
                    'l_dv_arr': l_dv_arr * 1e3, 'c_dv_arr': c_dv_arr, 'm_dv_opt': m_dv_opt,
                    'l_dv_opt': l_dv_opt * 1e3, 'p_nom_opt': p_nom_opt, 'omega_opt': omega_opt,
                    'e_d_opt': e_d_opt, 'ro_t': ro_t, 'delta_ob_opt': delta_ob_opt}
        elif self.flag == 'star':
            return {'p_nom_fit': p_nom_fit, 'e_d_fit': e_d_fit, 'eps_omega_arr': eps_omega_arr,
                    'kappa_arr': kappa_arr, 'u_fit': u_fit, 'omega_arr': omega_arr,
                    'l_zar_arr': l_zar_arr * 1e3, 'delta_ob_arr': delta_ob_arr * 1e3, 'D_kr_opt': D_kr_opt,
                    'm_dv_0': m_dv_0, 'm_dv_arr': m_dv_arr, 'alpha_dv_arr': alpha_dv_arr,
                    'l_dv_arr': l_dv_arr * 1e3, 'c_dv_arr': c_dv_arr, 'm_dv_opt': m_dv_opt,
                    'l_dv_opt': l_dv_opt * 1e3, 'p_nom_opt': p_nom_opt, 'omega_opt': omega_opt,
                    'e_d_opt': e_d_opt, 'ro_t': ro_t, 'delta_ob_opt': delta_ob_opt, 'tetta': tetta,
                    'alpha_1': alpha_1}


class get_flame():
    """Класс get_flame используется для определения изменения площади горения звездообразного заряда
    или щелевого типа.
    Methods
    -------
    Заряд щелевого типа:
    S_a(self, z):
        Горение по основной части поверхности канала.
    S_b(self, z):
        Площадь горения по поверхности канала в области щелей .

    fi_1(self, z):
        Условие выгорания поверхности В.

    S_c(self, z):
        Площадь горения по боковой поверхности щели.

    b1(self, z):
        Изменение высоты щели до выгорания поверхности В.

    b2(self, z):
        Изменение высоты щели после выгорания поверхности В.

    S_d(self, z):
        Суммарная площадь торца заряда со стороны щелей и торцевой поверхности щелей.

    S(self, z):
        Суммарная площадь поверхности горения.

    Заряд звездообразного типа:
    S(self, z):
        Площадь поверхности горения.

    Per(self, z):
        Периметр профиля поперечного сечения канала.

    alfa_1(self, z):
        Параметр, используемый для рассчётов.

    b(self, z):
        Параметр, используемый для рассчётов.

    alfa_2(self, z):
        Параметр, используемый для рассчётов.

    alfa_3(self, z):
        Параметр, используемый для рассчётов.

    dt_alfa_3(self, z):
        Параметр, используемый для рассчётов.

    def F_sv(self, z):
        Площадь свободного прохода канала заряда.

    F_sv_1(self, z):
        Площадь свободного прохода канала заряда для первой стадии.

    F_sv_2(self, z):
        Площадь свободного прохода канала заряда для второй стадии.

    F_sv_3(self, z):
        Площадь свободного прохода канала заряда для третьей стадии.

    dt_beta_3(self, z):
        Параметр, используемый для рассчётов.

    get_result(self, s_df=0):
        Функция, в которой представлено решение задачи."""

    def __init__(self, INIT):
        """# Пример начальных данных класса get_flame:
        INIT_1 = {'p_min': 4 * 1e6, 'p_max': 20 * 1e6, 'p_h': 0.1 * 1e6, 'p_a': 0.1 * 1e6,
        'f_a_max': 0.9, 'D_n': 610 * 1e-3, 'hi': 0.98, 'R_g': 550.84, 'n_kr': 1.1604, 'k': 1.1755,
        'z': 0.3239, 'T_r': 2999.5, 'I_P': 2250 * 1e3, 't_nom': 20, 'nu': 0.29, 'u_1': 4.38}

        res_1 = get_ed_avg(INIT_1).get_result(s_df=0)

        INIT_2_1 = {**INIT_1, **res_1}
        INIT_2_2 = {'delta_p1':6, 'delta_p2':6, 'delta_zks':1, 'delta_vks':15,
        'delta_ps1':6, 'delta_ps2':3, 'ro_p':1500, 'ro_zks':920,
        'ro_ps':1750, 'ro_vks':2200, 'sigma_vr':1830, 'sigma_02':1830,
        'D_t':0.002, 'tetta_1':30,'tetta_2':15, 'ro_k': 7800}

        INIT_2 = {**INIT_2_1,**INIT_2_2}
        INIT_2['count'] = 4
        INIT_2['a_'] = 0.3 # для щелевого заряда.
        INIT_2['c_'] = 0.05 # для щелевого заряда.
        INIT_2['r_'] = 0.1 # для звездообразного заряда.
        INIT_2['q_arr'] = [0.14, 0.66, 0.2]
        INIT_2['ro_arr'] = [920, 1950, 1500]
        INIT_2['shape'] = 'slot' # 'slot' - для щелевого заряда, 'star' - для звездообразного заряда.

        res_2 = get_powder(INIT_2).get_result(s_df=1)

        INIT_3 = {**INIT_2, **res_2}

        res_3 = get_flame(INIT_3).get_result()"""

        self.D_n = INIT['D_n'] * 1e-3
        if INIT['shape'] == 'slot':
            self.flag = INIT['shape']
            self.a_ = INIT['a_']
            self.c_ = INIT['c_']
        elif INIT['shape'] == 'star':
            self.flag = INIT['shape']
            self.r_ = INIT['r_']
        self.count = INIT['count']
        self.n = self.count
        self.delta_zks = INIT['delta_zks'] * 1e-3
        self.delta_ob = INIT['delta_ob_opt'][0]
        self.e_d = INIT['e_d_opt'][0]
        print(self.e_d)
        self.ro_t = INIT['ro_t']
        self.omega = INIT['omega_opt'][0]
        self.D_ks = self.D_n - 2 * self.delta_ob - 2 * self.delta_zks
        self.z_0 = self.e_d * self.D_ks / 2
        self.d = self.D_ks - 2 * self.z_0
        self.beta = np.pi / self.count
        self.F_ks = np.pi * (self.D_ks ** 2) / 4
        if self.flag == 'slot':
            self.d_ = self.d / self.D_ks
            self.epsilon_f = 1 - self.d_ ** 2
            self.f_cut = (self.n / np.pi) * (self.c_ * np.sqrt(1 - self.c_ ** 2) - self.c_ * np.sqrt(self.d_ **
                                                                                                     2 - self.c_ ** 2) + np.arcsin(
                self.c_) - self.d_ ** 2 * np.arcsin(self.c_ / self.d_))
            self.epsilon_w = self.epsilon_f - self.a_ * self.f_cut
            self.F_cut = self.F_ks * self.f_cut
        elif self.flag == 'star':
            self.alpha_1 = INIT['alpha_1']
            self.tetta = INIT['tetta']
            self.R = (self.D_ks / 2 - self.z_0) / (1 + self.r_)
            self.r = self.D_ks / 2 - self.z_0 - self.R
            self.epsilon_f = 1 - 1 / self.beta * ((1 - self.e_d) / (1 + self.r_)) ** 2 * (
                        self.alpha_1 * self.r_ ** 2 + (np.sin(self.beta) / np.sin(
                    self.tetta) - self.r_ * np.cos(self.tetta) / np.sin(self.tetta)) * (
                                    self.r_ + np.cos(self.alpha_1)) + self.r_ * np.sin(self.alpha_1))
            self.epsilon_w = self.epsilon_f
            self.z_1 = self.R * np.sin(self.beta) / np.cos(self.tetta) - self.r
            self.z_2 = self.z_0
            self.z_3 = np.sqrt(self.R ** 2 + (self.D_ks / 2) **
                               2 - self.R * self.D_ks * np.cos(self.beta)) - self.r
        self.l_gr = self.omega / (self.ro_t * self.epsilon_w * self.F_ks)
        if self.flag == 'slot':
            self.a = self.a_ * self.l_gr
            self.c = self.c_ * self.D_ks
            self.z_1 = 0.5 * (self.d * np.sin(self.beta) - self.c) / \
                       (1 - np.sin(self.beta))
            self.z_2 = (self.D_ks * np.sin(self.beta) - self.c) / 2

    def S_g_t(self, z):
        # Расчёт щелевого заряда
        def S_slot(self, z):
            def S_a(self, z):
                return np.pi * (self.d + 2 * z) * (self.l_gr - self.a - z)

            def S_b(self, z):
                if z > self.z_1:
                    return 0
                elif z <= self.z_1:
                    return self.n * (self.beta - fi_1(self, z)) * (self.d + 2 * z) * self.a

            def fi_1(self, z):
                return np.arcsin((self.c + 2 * z) / (self.d + 2 * z))

            def S_c(self, z):
                if 0 <= z <= self.z_1:
                    return 2 * self.n * self.a * b1(self, z)
                elif self.z_1 < z <= self.z_2:
                    return 2 * self.n * self.a * b2(self, z)
                elif z > self.z_2:
                    return 0

            def b1(self, z):
                return 0.5 * (np.sqrt(self.D_ks ** 2 - (self.c + 2 * z) ** 2) - np.sqrt(
                    (self.d + 2 * z) ** 2 - (self.c + 2 * z) ** 2))

            def b2(self, z):
                return 0.5 * (
                            np.sqrt(self.D_ks ** 2 - (self.c + z * 2) ** 2) - (self.c + 2 * z) * np.tan(self.beta) ** (
                        -1))

            def S_d(self, z):
                return np.pi / 4 * (self.D_ks ** 2 - (self.d + 2 * z) ** 2)

            def get_kappa(self, z):
                # return S_c(self, z) / (self.F_ks * self.f_cut)
                #
                return (S_a(self, z) + S_b(self, z)) / (self.F_ks * (1 - self.epsilon_f))

            def S(self, z):
                return S_a(self, z) + S_b(self, z) + S_c(self, z) + S_d(self, z)

            return {'S_g_t': S(self, z), 'kappa': get_kappa(self, z), 'S_a': S_a(self, z),
                    'S_b': S_b(self, z), 'S_c': S_c(self, z), 'S_d': S_d(self, z)}

        # функции для звезды
        def S_star(self, z):
            def Per(self, z):
                if z <= self.z_1:
                    return 2 * self.n * (alfa_1(self, z) * (self.r + z) + b(self, z))
                elif z <= self.z_2:
                    return 2 * self.n * alfa_2(self, z) * (self.r + z)
                elif z <= self.z_3:
                    return 2 * self.n * alfa_3(self, z) * (self.r + z)

            def alfa_1(self, z):

                return np.pi / 2 + self.beta - self.tetta

            def b(self, z):
                return self.R * np.sin(self.beta) / np.sin(self.tetta) - (self.r + z) * np.tan(self.tetta) ** (-1)

            def alfa_2(self, z):
                return self.beta + np.arcsin(self.R / (self.r + z) * np.sin(self.beta))

            def alfa_3(self, z):
                return alfa_2(self, z) - dt_alfa_3(self, z)

            def dt_alfa_3(self, z):
                return np.pi - np.arccos(
                    (self.R ** 2 + (self.r + z) ** 2 - (self.D_ks / 2) ** 2) / (2 * self.R * (self.r + z)))

            def get_F_sv(self, z):
                if z <= self.z_1:
                    return F_sv_1(self, z)
                elif z <= self.z_2:
                    return F_sv_2(self, z)
                elif z <= self.z_3:
                    return F_sv_3(self, z)

            def F_sv_1(self, z):
                return self.n * (
                            alfa_1(self, z) * (self.r + z) ** 2 + (self.R * np.sin(self.beta) / np.sin(self.tetta) -
                                                                   (self.r + z) * np.tan(self.tetta) ** (-1)) * (
                                        (self.r + z) + self.R * np.cos(alfa_1(self, z))) +
                            self.R * (self.r + z) * np.sin(alfa_1(self, z)))

            def F_sv_2(self, z):
                return self.n * (alfa_2(self, z) * (self.r + z) ** 2 + self.R * (self.r + z) * np.sin(alfa_2(self, z)))

            def F_sv_3(self, z):
                return self.n * (alfa_3(self, z) * (self.r + z) ** 2 + self.R * (self.r + z) * np.sin(alfa_2(self, z)) +
                                 dt_beta_3(self, z) * self.D_ks ** 2 / 4 - self.R * self.D_ks / 2 * np.sin(
                            dt_beta_3(self, z)))

            def dt_beta_3(self, z):
                return np.arccos((self.R ** 2 + (self.D_ks / 2) ** 2 - (self.r + z) ** 2) / (self.R * self.D_ks))

            def get_kappa(self, z):
                return Per(self, z) * self.l_gr / get_F_sv(self, z)

            return {'S_g_t': Per(self, z) * self.l_gr, 'kappa': get_kappa(self, z), 'F_sv': get_F_sv(self, z)}

        if self.flag == 'star':
            return S_star(self, z)
        elif self.flag == 'slot':
            return S_slot(self, z)

    def get_result(self, s_df=0):
        if self.flag == 'slot':
            z = np.linspace(0, self.z_0, 100)
            sa = np.array([get_flame.S_g_t(self, zx)['S_a'] for zx in z])
            sb = np.array([get_flame.S_g_t(self, zx)['S_b'] for zx in z])
            sc = np.array([get_flame.S_g_t(self, zx)['S_c'] for zx in z])
            sd = np.array([get_flame.S_g_t(self, zx)['S_d'] for zx in z])
            s = np.array([get_flame.S_g_t(self, zx)['S_g_t'] for zx in z])
            S_g_sr = self.omega / self.ro_t / self.z_0
            kappa_kan = (sa + sb) / (self.F_ks * (1 - self.epsilon_f))
            kappa_cut = sc / (self.F_ks * self.f_cut)
            print("f_cut =", self.f_cut)
            print("epsilon_f =", self.epsilon_f)
            print("epsilon_w =", self.epsilon_w)
            print("kappa_kan_0 =", kappa_kan[0])
            print("kappa_cut_0 =", kappa_cut[0])
            print("l_gr =", self.l_gr)
            print("S_sr =", s.mean())
            # print(S_g_sr)
            return {'a': self.a, 'c': self.c, 'D_ks': self.D_ks, 'epsilon_w': self.epsilon_w,
                    'epsilon_f': self.epsilon_f, 'kappa_0': kappa_kan[0], 'l_gr': self.l_gr,
                    'z_arr': z, 'sa': sa, 'sb': sb, 'sc': sc, 'sd': sd, 's': s, 'z_1': self.z_1,
                    'z_2': self.z_2, 'z_0': self.z_0, 'f_cut': self.f_cut, 'F_cut': self.F_cut}
        elif self.flag == 'star':
            z = np.linspace(0, self.z_3, 100)
            s = np.array([get_flame.S_g_t(self, zx)['S_g_t'] for zx in z])
            z1 = np.linspace(0, self.z_2, 100)
            s1 = np.array([get_flame.S_g_t(self, zx)['S_g_t'] for zx in z1])
            Fsv = np.array([get_flame.S_g_t(self, zx)['F_sv'] for zx in z])
            kappa = s / Fsv
            Fsv2 = np.array([get_flame.S_g_t(self, zx)['F_sv'] for zx in z1])
            kappa2 = s1 / Fsv2
            S_g_sr = self.omega / self.ro_t / self.z_0
            print("epsilon_f =", self.epsilon_f)
            print("epsilon_w =", self.epsilon_w)
            print("kappa_kan_0 =", kappa[0])
            print("l_gr =", self.l_gr)
            print("S_sr =", s.mean())
            return {'D_ks': self.D_ks, 'epsilon_f': self.epsilon_f, 'epsilon_w': self.epsilon_w, 'kappa_0': kappa[0],
                    'z_arr': z,
                    's': s, 'z_1': self.z_1, 'z_2': self.z_2, 'z_3': self.z_3, 'kappa': kappa,
                    'l_gr': self.l_gr, 's1': s1, 'kappa2': kappa2, 'z_0': self.z_0, 'R': self.R,
                    'r': self.r}


class get_p_graph():
    """Класс get_p_graph используется для определения параметров в процессе горения воспламенителя и ТРТ.
    Methods
    -------
    rk4(self, init, sys, t0, dt, n, stop):
        Численный метод получения решения дифференциальных уравнений (ДУ).
    System(self, t, y):
        Функция, содержащая систему ДУ и параметры, необходимые для её решения.

    S_g_v(self, e):
        Функция для определения площади горения воспламенителя в форме таблетки.

    get_X(self, G_v, T, G_t, dQ_t_dt, F_q_1):
        Комплекс X.

    get_Y(self, G_v, T, G_t, p, ro, G):
        Комплекс Y.

    get_u_t(self, p):
        Изменение высоты щели после выгорания поверхности В.

    get_u_v(self, p):
        Суммарная площадь торца заряда со стороны щелей и торцевой поверхности щелей.

    FI_kappa(self, kappa):
        Поправочная функция по параметру Победоносцева.

    F_q_1(self, etta_T):
        Функция Хевисайда по разнице температур.

    F_0(self, e):
        Функция Хевисайда по толщине воспламенителя.

    F_1(self, etta_T, e):
        Функция Хевисайда по толщине заряда ТРТ.

    get_q_zar(self, etta_T, G_v, T, F_sv_0):
        Плотность теплового потока.

    Pi(self, lamda):
        ГД функция.

    get_G(self, p, R, T):
        Секундный массовый расход продуктов сгорания через сопло.

    get_G_v(self, S, u, F_0):
        Секундный массоприход при сгорании воспламеительного состава.

    get_G_t(self, S, u, fi_kappa, f_1):
        Секундный массоприход при сгорании основного заряда.

    solve_rk4(self, T_0):
        Функция для решения системы ДУ с использованием метода Рунге-Кутты 4 порядка.

    get_result(self, s_df=0):
        Функция, в которой представлено решение задачи."""

    def __init__(self, INIT):
        """# Пример начальных данных класса get_p_graph:
        INIT_1 = {'p_min': 4 * 1e6, 'p_max': 20 * 1e6, 'p_h': 0.1 * 1e6, 'p_a': 0.1 * 1e6,
        'f_a_max': 0.9, 'D_n': 610 * 1e-3, 'hi': 0.98, 'R_g': 550.84, 'n_kr': 1.1604, 'k': 1.1755,
        'z': 0.3239, 'T_r': 2999.5, 'I_P': 2250 * 1e3, 't_nom': 20, 'nu': 0.29, 'u_1': 4.38}

        res_1 = get_ed_avg(INIT_1).get_result(s_df=0)

        INIT_2_1 = {**INIT_1, **res_1}
        INIT_2_2 = {'delta_p1':6, 'delta_p2':6, 'delta_zks':1, 'delta_vks':15,
        'delta_ps1':6, 'delta_ps2':3, 'ro_p':1500, 'ro_zks':920,
        'ro_ps':1750, 'ro_vks':2200, 'sigma_vr':1830, 'sigma_02':1830,
        'D_t':0.002, 'tetta_1':30,'tetta_2':15, 'ro_k': 7800}

        INIT_2 = {**INIT_2_1,**INIT_2_2}
        INIT_2['count'] = 4
        INIT_2['a_'] = 0.3 # для щелевого заряда.
        INIT_2['c_'] = 0.05 # для щелевого заряда.
        INIT_2['r_'] = 0.1 # для звездообразного заряда.
        INIT_2['q_arr'] = [0.14, 0.66, 0.2]
        INIT_2['ro_arr'] = [920, 1950, 1500]
        INIT_2['shape'] = 'slot' # 'slot' - для щелевого заряда, 'star' - для звездообразного заряда.

        res_2 = get_powder(INIT_2).get_result(s_df=1)

        INIT_3 = {**INIT_2, **res_2}

        res_3 = get_flame(INIT_3).get_result()

        INIT_4_1 = {**INIT_3, **res_3}
        INIT_4_2 = {'p_ref_t': 1e6, 'T_deg': [273.15-50, 273.15+20, 273.15+50], 'T_s': 750, 'T_ref': 273.15+20,
                    'c_p_nach': 1004.5, 'R_nach': 287, 'p_ref_v': 98066.5, 'ro_v': 1750, 'u_1_v': 11.7, 'nu_v': 0.226,
                    'D_t_v': 0.001, 'T_p_v': 1984.1, 'mu_g_v': 0.6130*1e-4, 'lamda_g_v': 0.11938, 'c_p_v': 1224.6,
                    'c_p_g_v': 1356.2, 'z_0_v': 0.4119, 'R_g_v': 228.08, 'T_p_1': 2999.5, 'n_t': 1.1755, 'n_kr': 1.1604,
                    'u_1_t': 4.38, 'nu_t': 0.29, 'D_t_t': 0.002, 'c_p_1': 2236.7, 'c_t': 1250, 'lamda_t': 0.3,
                    'kappa_ref': 100, 'z_0_t': 0.3239, 'R_t': 550.84, 'e_v_0': 2, 'm_v': 3}

        INIT_4 = {**INIT_4_1, **INIT_4_2}
        res_4 = get_p_graph(INIT_4).get_result(s_df=1)"""

        self.kappa_0 = INIT['kappa_0']
        self.l_gr = INIT['l_gr']
        self.D_ks = INIT['D_ks']
        self.z_0 = INIT['z_0']
        if INIT['shape'] == 'slot':
            self.flag = INIT['shape']
            self.z_1 = INIT['z_1']
            self.z_2 = INIT['z_2']
            self.z_3 = INIT['z_0']
            self.a_ = INIT['a_']
            self.a = INIT['a']
            self.c = INIT['c']
            self.c_ = self.c / self.D_ks
            self.d = self.D_ks - 2 * self.z_0
            self.d_ = self.d / self.D_ks
            self.f_cut = INIT['f_cut']
            self.F_cut = INIT['F_cut']
        if INIT['shape'] == 'star':
            self.flag = INIT['shape']
            self.r_ = INIT['r_']
            self.alpha_1 = INIT['alpha_1']
            self.tetta = INIT['tetta']
            self.R = INIT['R']
            self.r = INIT['r']
            self.z_1 = INIT['z_1']
            self.z_2 = INIT['z_2']
            self.z_3 = INIT['z_3']
            self.d = 4 * self.l_gr / self.kappa_0
        self.ro_t = INIT['ro_t']
        self.omega = INIT['omega_opt'][0]
        self.count = INIT['count']
        self.n = self.count
        self.F_ks = np.pi * (self.D_ks ** 2) / 4
        self.D_kr = INIT['D_kr_opt'][0]
        self.F_kr = np.pi * (self.D_kr ** 2) / 4
        self.e_d = INIT['e_d_opt'][0]
        self.beta = np.pi / self.count
        self.epsilon_f = INIT['epsilon_f']
        self.epsilon_w = INIT['epsilon_w']
        self.p_ref_t = INIT['p_ref_t']
        self.p_h = INIT['p_h'] * 1e6
        self.T_deg = INIT['T_deg']
        self.T_s = INIT['T_s']
        self.T_ref = INIT['T_ref']
        self.c_p_nach = INIT['c_p_nach']
        self.R_nach = INIT['R_nach']
        self.p_ref_v = INIT['p_ref_v']
        self.ro_v = INIT['ro_v']
        self.u_1_v = INIT['u_1_v'] * 1e-3
        self.nu_v = INIT['nu_v']
        self.D_t_v = INIT['D_t_v']
        self.T_p_v = INIT['T_p_v']
        self.mu_g = INIT['mu_g_v']
        self.lamda_g = INIT['lamda_g_v']
        self.c_p_v = INIT['c_p_v']
        self.c_p_g_v = INIT['c_p_g_v']
        self.z_0_v = INIT['z_0_v']
        self.R_g_v = INIT['R_g_v']
        self.R_v = self.R_g_v * (1 - self.z_0_v)
        self.T_p_1 = INIT['T_r']
        self.n_t = INIT['k']
        self.n_kr = INIT['n_kr']
        self.u_1_t = INIT['u_1'] * 1e-3
        self.nu_t = INIT['nu']
        self.D_t_t = INIT['D_t']
        self.c_p_1 = INIT['c_p_1']
        self.c_t = INIT['c_t']
        self.lamda_t = INIT['lamda_t']
        self.kappa_ref = INIT['kappa_ref']
        self.z_0_t = INIT['z']
        self.R_t = INIT['R_g']
        # print(self.T_p_1, self.n_t, self.n_kr, self.u_1_t, self.nu_t, self.D_t_t, self.c_p_1, self.z_0_t, self.R_t)
        self.R_1 = self.R_t * (1 - self.z_0_t)
        self.A_n = (self.n_kr * (2 / (self.n_kr + 1)) **
                    ((self.n_kr + 1) / (self.n_kr - 1))) ** 0.5
        self.W_0 = np.pi * self.D_ks ** 3 / 12 + np.pi * \
                   self.D_ks ** 2 / 4 * self.l_gr - self.omega / self.ro_t
        self.e_v_0 = INIT['e_v_0'] * 1e-3
        self.m_v = INIT['m_v']
        self.c_tabl = self.e_v_0 * 0.75
        self.h_tabl = self.e_v_0 - self.c_tabl / 2
        self.d_tabl = self.e_v_0 * 5
        self.r_tabl = self.d_tabl / 2
        self.R_tabl = (self.r_tabl ** 2 + self.h_tabl ** 2) / 2 / self.h_tabl
        self.V_tabl = np.pi * self.r_tabl ** 2 * self.c_tabl + 2 * \
                      np.pi * self.h_tabl ** 2 * (self.R_tabl - self.h_tabl / 3)
        self.V_v = self.m_v / self.ro_v
        self.N = np.round(self.V_v / self.V_tabl)
        print('\nКоличество зёрен:', self.N,'\n')

    def rk4(self, init, sys, t0, dt, n, stop):
        # declaration
        t = np.zeros(n)
        m = len(init) + 1
        res = np.zeros((n, m))
        # initialization
        i = 0
        X = np.array([i for i in init])
        t[i] = t0
        res[i, 0] = t[i]
        res[i, 1:m] = X
        # main loop
        # while stop(t[i], X) > 0 and i < n - 1:
        while stop(X[0]) and i < n - 1:
            k1 = sys(self, t[i], X)
            k2 = sys(self, t[i] + 0.5 * dt, X + k1 * 0.5 * dt)
            k3 = sys(self, t[i] + 0.5 * dt, X + k2 * 0.5 * dt)
            k4 = sys(self, t[i] + dt, X + k3 * dt)
            i += 1
            X += (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
            t[i] = t[i - 1] + dt
            res[i, 0] = t[i]
            res[i, 1:m] = X
        return res[0:i + 1, :]

    def System(self, t, y):
        p, T, W, c_p, R, e_v, e, etta_T = y
        k = c_p / (c_p - R)
        ro = p / (R * T)
        f_q_1 = get_p_graph.F_q_1(self, etta_T)
        f_0 = get_p_graph.F_0(self, e_v)
        f_1 = get_p_graph.F_1(self, etta_T, e)
        S_0 = get_flame.S_g_t(self, 0)['S_g_t']
        S_t = get_flame.S_g_t(self, e)['S_g_t']
        kappa = get_flame.S_g_t(self, e)['kappa']
        #         print(S_t, kappa)
        F_sv_0 = S_0 / self.kappa_0
        u_v = get_p_graph.get_u_v(self, p)
        u_t = get_p_graph.get_u_t(self, p)
        S_v = self.N * get_p_graph.S_g_v(self, e_v)
        fi_kappa = get_p_graph.FI_kappa(self, kappa)
        G_v = get_p_graph.get_G_v(self, S_v, u_v, f_0)
        #         print(S_t, G_v, S_v, u_v, f_0)
        G_t = get_p_graph.get_G_t(self, S_t, u_t, fi_kappa, f_1)
        #         print(S_t,G_t)
        G = get_p_graph.get_G(self, p, R, T)
        q_zar = get_p_graph.get_q_zar(self, etta_T, G_v, T, F_sv_0)
        dQ_t_dt = q_zar * S_0
        X = get_p_graph.get_X(self, G_v, T, G_t, dQ_t_dt, f_q_1)
        Y = get_p_graph.get_Y(self, G_v, T, G_t, p, ro, G)
        #         print(S_t, kappa, fi_kappa, G_v, G_t, G, q_zar, X, Y)
        dp_dt = (k - 1) / W * (X + Y * k / (k - 1))
        dT_dt = (k - 1) / (ro * W * R) * (X + Y)  # 1/
        dW_dt = G_v / self.ro_v + G_t / self.ro_t
        dc_p_dt = 1 / (ro * W) * (G_v * (self.c_p_v - c_p) + G_t * (self.c_p_1 - c_p))
        dR_dt = 1 / (ro * W) * (G_v * (self.R_v - R) + G_t * (self.R_1 - R))
        de_b_dt = u_v * f_0
        de_dt = u_t * fi_kappa * f_1
        detta_T_dt = (2 * q_zar ** 2) / (self.c_t * self.lamda_t * self.ro_t) * f_q_1
        return np.array([dp_dt, dT_dt, dW_dt, dc_p_dt, dR_dt, de_b_dt, de_dt, detta_T_dt])

    def S_g_v(self, e):
        def S1(self, e):
            return 4 * np.pi * (self.R_tabl - e) * h1(self, e) + 2 * np.pi * (self.r_tabl - e) * c1(self, e)

        def h1(self, e):
            return (self.R_tabl - e) - np.sqrt((self.R_tabl - e) ** 2 - (self.r_tabl - e) ** 2)

        def c1(self, e):
            return 2 * (self.e_v_0 - e - h1(self, e))

        def S2(self, e):
            return 4 * np.pi * (self.R_tabl - e) * (self.e_v_0 - e)

        def e1(self, e):
            return ((self.R_tabl ** 2 - self.r_tabl ** 2) - (self.R_tabl - self.e_v_0) ** 2) / (
                        2 * (self.R_tabl - self.r_tabl))

        e_cil = e1(self, e)
        if e <= e_cil:
            return S1(self, e)
        elif e <= self.e_v_0:
            return S2(self, e)
        else:
            return 0

    def get_X(self, G_v, T, G_t, dQ_t_dt, F_q_1):
        x = G_v * self.c_p_v * (self.T_p_v - T) + G_t * self.c_p_1 * \
            (self.T_p_1 - T) - dQ_t_dt * F_q_1  # c_p_1
        return x

    def get_Y(self, G_v, T, G_t, p, ro, G):
        y = G_v * self.R_v * T + G_t * self.R_1 * T - p / ro * \
            G - p * (G_v / self.ro_v + G_t / self.ro_t)
        return y

    # закон горения
    def get_u_t(self, p):
        return self.u_1_t * (p / self.p_ref_t) ** self.nu_t * np.exp(self.D_t_t * (self.T_0 - self.T_ref))

    def get_u_v(self, p):
        return self.u_1_v * (p / self.p_ref_v) ** self.nu_v * np.exp(self.D_t_v * (self.T_0 - self.T_ref))

    def FI_kappa(self, kappa):  # check
        if kappa < self.kappa_ref:
            return 1
        else:
            return 1 + 0.003 * (kappa - self.kappa_ref)

    def F_q_1(self, etta_T):
        if 0 < self.etta_T_s - etta_T:
            return 1
        else:
            return 0

    def F_0(self, e):
        if 0 <= self.e_v_0 - e:
            return 1
        else:
            return 0

    def F_1(self, etta_T, e):
        if self.z_0 < self.z_1 and self.z_0 < self.z_2:
            if 0 <= self.z_0 - e:
                return 1 - get_p_graph.F_q_1(self, etta_T)
            else:
                return 0
        else:
            if 0 <= max(self.z_1, self.z_2, self.z_3, self.z_0) - e:
                return 1 - get_p_graph.F_q_1(self, etta_T)
            else:
                return 0

    def get_q_zar(self, etta_T, G_v, T, F_sv_0):
        Pr = (self.c_p_g_v * self.mu_g) / self.lamda_g
        Re = G_v / F_sv_0 * self.d / self.mu_g
        Nu = 0.023 * Re ** 0.8 * Pr ** 0.4
        alpha = Nu * self.lamda_g / self.d
        return alpha * (T - (self.T_0 + etta_T ** 0.5))

    def Pi(self, lamda):
        return (1 - (self.n_t - 1) / (self.n_t + 1) * lamda ** 2) ** (self.n_t / (self.n_t - 1))

    def get_G(self, p, R, T):  # секундный массовый расход ПС через сопло
        if p * get_p_graph.Pi(self, 1) < self.p_h:
            return p * self.F_kr / np.sqrt(R * T) * (self.p_h / p) ** (1 / self.n_kr) * np.sqrt(
                2 * self.n_kr / (self.n_kr - 1) * np.abs(1 - (self.p_h / p) ** ((self.n_kr - 1) / self.n_kr)))
        else:
            return (self.A_n * self.F_kr * p) / ((R * T) ** 0.5)

    def get_G_v(self, S, u, F_0):  # расход от воспламенителя
        return S * u * self.ro_v * F_0

    def get_G_t(self, S, u, fi_kappa, f_1):  # расход от основного заряда
        return S * u * self.ro_t * f_1 * fi_kappa

    def solve_rk4(self, T_0):
        self.etta_T_s = (self.T_s - self.T_0) ** 2
        t_k_1 = 0.25
        d_t_1 = 5 * 1e-5
        step_1 = round(t_k_1 / d_t_1)
        t_k_2 = 1000
        d_t_2 = 5 * 1e-3
        step_2 = round(t_k_2 / d_t_2)
        INIT_1 = np.array(
            [self.p_h, self.T_0, self.W_0, self.c_p_nach, self.R_nach, 0, 0, 0])
        print(f'Начальные условия для первого этапа (T_0 = {round(self.T_0, 2)} K):', INIT_1, sep='\n')
        res_1 = get_p_graph.rk4(
            self, INIT_1, get_p_graph.System, t0=0, dt=d_t_1, n=step_1, stop=lambda x: True)
        INIT_2 = res_1[-1][1:]
        print(f'Начальные условия для второго этапа (T_0 = {round(self.T_0, 2)} K):', INIT_2, sep='\n')
        res_2 = get_p_graph.rk4(self, INIT_2, get_p_graph.System,
                                t0=t_k_1, dt=d_t_2, n=step_2, stop=lambda p: p * get_p_graph.Pi(self, 1) > self.p_h)
        return {'res_1': res_1, 'res_2': res_2}

    def get_result(self, s_df=0):
        result = []
        create_graph = []
        for i in range(len(self.T_deg)):
            self.T_0 = self.T_deg[i]
            result.append(get_p_graph.solve_rk4(self, self.T_0))
            e_topl_1, e_topl_2 = result[i]['res_1'][:, 7], result[i]['res_2'][:, 7]
            p_kam_1, p_kam_2 = result[i]['res_1'][:, 1], result[i]['res_2'][:, 1]
            e_vospl_1, e_vospl_2 = result[i]['res_1'][:, 6], result[i]['res_2'][:, 6]
            t_kam_1, t_kam_2 = result[i]['res_1'][:, 0], result[i]['res_2'][:, 0]
            T_kam_1, T_kam_2 = result[i]['res_1'][:, 2], result[i]['res_2'][:, 2]
            RESULT_1 = {'e_topl': e_topl_1, 'p_kam': p_kam_1, 'e_vospl': e_vospl_1,
                        't_kam': t_kam_1, 'T_kam': T_kam_1}
            RESULT_2 = {'e_topl': e_topl_2, 'p_kam': p_kam_2, 'e_vospl': e_vospl_2,
                        't_kam': t_kam_2, 'T_kam': T_kam_2}
            create_graph.append({'RESULT_1': RESULT_1, 'RESULT_2': RESULT_2})

        t, p, T = [], [], []
        for i in range(len(self.T_deg)):
            # Конец горения воспламенителя
            t_1 = create_graph[i]['RESULT_1']['t_kam'][create_graph[i]['RESULT_1']['e_topl'] == 0][-1]
            p_1 = create_graph[i]['RESULT_1']['p_kam'][create_graph[i]['RESULT_1']['e_topl'] == 0][-1]
            T_1 = create_graph[i]['RESULT_1']['T_kam'][create_graph[i]['RESULT_1']['e_topl'] == 0][-1]
            # Максимальное значение давления
            t_2 = create_graph[i]['RESULT_1']['t_kam'][
                create_graph[i]['RESULT_1']['p_kam'] == max(create_graph[i]['RESULT_1']['p_kam'])][-1]
            p_2 = create_graph[i]['RESULT_1']['p_kam'][
                create_graph[i]['RESULT_1']['p_kam'] == max(create_graph[i]['RESULT_1']['p_kam'])][-1]
            T_2 = create_graph[i]['RESULT_1']['T_kam'][
                create_graph[i]['RESULT_1']['p_kam'] == max(create_graph[i]['RESULT_1']['p_kam'])][-1]
            # Конец горения основного заряда
            t_3 = create_graph[i]['RESULT_2']['t_kam'][create_graph[i]['RESULT_2']['e_topl'] <= self.z_0][-1]
            p_3 = create_graph[i]['RESULT_2']['p_kam'][create_graph[i]['RESULT_2']['e_topl'] <= self.z_0][-1]
            T_3 = create_graph[i]['RESULT_2']['T_kam'][create_graph[i]['RESULT_2']['e_topl'] <= self.z_0][-1]
            # Конец интегрирования
            t_4 = create_graph[i]['RESULT_2']['t_kam'][create_graph[i]['RESULT_2']['e_topl'] >= self.z_0][-1]
            p_4 = create_graph[i]['RESULT_2']['p_kam'][create_graph[i]['RESULT_2']['e_topl'] >= self.z_0][-1]
            T_4 = create_graph[i]['RESULT_2']['T_kam'][create_graph[i]['RESULT_2']['e_topl'] >= self.z_0][-1]
            t.append({'t_1': t_1, 't_2': t_2, 't_3': t_4})
            p.append({'p_1': p_1, 'p_2': p_2, 'p_3': p_4})
            T.append({'T_1': T_1, 'T_2': T_2, 'T_3': T_4})

            data_4 = [np.array([np.round(t_1, 3), np.round(t_2, 3), np.round(t_3, 3), np.round(t_4, 3)]), np.array([
                np.round(p_1 / 1e6, 3), np.round(p_2 / 1e6, 3), np.round(p_3 / 1e6, 3), np.round(p_4 / 1e6, 3)]),
                      np.array([np.round(T_1, 3), np.round(T_2, 3), np.round(T_3, 3), np.round(T_4, 3)])]
            data_4_show = pd.DataFrame(data_4, index=["t, с", "p, МПа", "T, К"], columns=[
                i for i in range(1, 4 + 1)]).T
            # pd.set_option('display.float_format', '{:.5}'.format)
            display(data_4_show)
            if s_df == 1:
                data_4_show.to_excel(f'data_{i + 4}.xlsx')
        if self.flag == 'slot':
            print('D_ks:', self.D_ks*1e3)
            print('D_kr:', self.D_kr*1e3)
            print('m_v:', self.m_v)
            print('m_t:', self.omega)
            print('e_t_0:', self.z_0*1e3)
            print('n:', self.n)
            print('a:', self.a*1e3)
            print('c:', self.c*1e3)
            print('e_v_0:', self.e_v_0*1e3)
            print('d_1:', self.d_tabl*1e3)
            print('c_1:', self.c_tabl*1e3)
            print('T_s:', self.T_s)
            print('p_h:', self.p_h)
        elif self.flag == 'star':
            print('D_ks:', self.D_ks*1e3)
            print('D_kr:', self.D_kr*1e3)
            print('m_v:', self.m_v)
            print('m_t:', self.omega)
            print('e_t_0:', self.z_0*1e3)
            print('n:', self.n)
            print('tetta:', np.degrees(self.tetta))
            print('r:', self.r*1e3)
            print('e_v_0:', self.e_v_0*1e3)
            print('d_1:', self.d_tabl*1e3)
            print('c_1:', self.c_tabl*1e3)
            print('T_s:', self.T_s)
            print('p_h:', self.p_h)
        return result, create_graph, {'t': t, 'p': p, 'T': T}