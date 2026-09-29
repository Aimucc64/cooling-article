import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time

# Parameter
thetas = [30, 60, 90, 120, 150]
for n in thetas:
    theta1 = n  
    radii = [0.001, 0.005, 0.01, 0.015, 0.02]
    for m in radii:
        start_time = time.time()
        T_initial = 1043 
        T_end = 450  
        T_lunar = 300   
        T_dec = 833  
        radius = m  
        radius_mm = 1000 * radius
        k = 1.67    # W/(m.K) - Conductivity
        rho = 3151  # kg/m^3 - Density
        c = 1522.3  
        T_g = 900  
        Tc = 1043  
        epsilon1 = 0.85
        dt = 0.0002  
        dr = 0.0001  
        theta_h = theta1 * (math.pi/180)  
        omega1 = 2 * math.pi * (1 - math.cos(theta_h)) 
        omega2 = 2 * math.pi * (1 + math.cos(theta_h))  
        sigma = 5.67e-8  # W/m^2.K^4 - Stefan-Boltzmann constant
        # 层设置
        layer1 = round(radius/dr)  
        layer2 = round(5 * radius / dr)  
        r_down = np.arange(dr, 5 * radius + 0.9 * dr, dr)  
        r_down2 = r_down - dr  # 用作计算
        h_qq = r_down[layer1:] - radius * math.cos(theta_h)  
        h_qq2 = r_down2[layer1:] - radius * math.cos(theta_h)
        A_down = omega1 * r_down**2  
        A_down[layer1:] = 2 * math.pi * r_down[layer1:] * h_qq  
        V_down = 1/3 * omega1 * (r_down**3 - r_down2**3)  
        V_down[layer1:] = (math.pi * h_qq**2 * (r_down[layer1:] - h_qq/3) - math.pi * h_qq2**2 *
                           (r_down2[layer1:] - h_qq2/3))  
        T_down = np.full(layer2, float(T_lunar)) 
        T_down[0:layer1] = T_initial
        r_upper = np.arange(dr, radius + 0.9 * dr, dr)  
        r_upper2 = r_upper - dr
        A_upper = omega2 * r_upper**2  
        V_upper = 1/3 * omega2 * (r_upper**3 - r_upper2**3)  
        T_upper = np.full(layer1, float(T_initial))
        A_connect = 2 * math.pi * r_upper * dr  
        c_upper = np.full(layer1, float(c))
        rho_c_upper = rho * c_upper
        c_down = np.full(layer2, float(c))
        rho_c_down = rho * c_down
        # 作图温度
        temperatures1 = [T_upper[0]]  
        temperatures2 = [T_upper[-1]]
        #  temperatures3 = [T_down[0]]
        temperatures4 = [T_down[0]]
        # 时间迭代器
        time1 = 0  
        times = [time1]
        while T_upper[0] > T_end:
            # 热量变化
            if T_down[layer1-1] < T_g:
                for i in range(len(T_upper)):
                    if T_upper[i] < T_g:
                        rho_c_upper[i:] = rho * (957.6 + 0.2321 * T_upper[i:] - 23970000/T_upper[i:]**2)
                        break
                for i in range(len(T_upper)):
                    if T_down[i] < T_g:
                        rho_c_down[i:layer1] = rho * (957.6 + 0.2321 * T_down[i:layer1] - 23970000/T_down[i:layer1]**2)
                        break
            rho_c_down[layer1:] = rho * (957.6 + 0.2321 * T_down[layer1:] - 23970000/T_down[layer1:]**2)
            T_down2 = T_down[1:]  
            T_down2 = np.append(T_down2, [float(T_lunar)])  
            T_upper2 = T_upper[1:]  
            T_upper2 = np.append(T_upper2, [float(T_lunar)])
            q_down_cond = k * A_down * (T_down - T_down2) / dr  
            q_upper_cond = k * A_upper * (T_upper - T_upper2) / dr  
            q_upper_cond[-1] = float(0) 
            q_upper_down = k * A_connect * (T_upper - T_down[0:layer1]) / ((math.pi - theta_h) * r_upper/2) 
            q_upper_down[0] = q_upper_down[0] * ((math.pi - theta_h) * r_upper[0]/2) / (dr/math.sqrt(2))  
            q_upper_rad = epsilon1 * sigma * A_upper[-1] * (T_upper[-1] ** 4 - T_lunar ** 4)  
            q_down_rad = epsilon1 * sigma * A_down[layer1 - 1] * (T_down[layer1 - 1] ** 4 - T_down[layer1] ** 4)  
            # 温度变化
            # 向外层散热
            T_down[layer1-1] -= dt * q_down_rad / (rho_c_down[layer1-1] * V_down[layer1 - 1])  
            T_down -= dt * q_down_cond / (rho_c_down * V_down) 
            T_upper -= dt * q_upper_cond / (rho_c_upper * V_upper)  
            T_upper[-1] -= dt * q_upper_rad/(rho_c_upper[-1] * V_upper[-1])  
            # 从内层吸热
            q_down_cond2 = q_down_cond[0:-1]
            q_down_cond2 = np.insert(q_down_cond2, 0, 0)  
            T_down += dt * q_down_cond2 / (rho_c_down * V_down)
            q_upper_cond2 = q_upper_cond[0:-1]
            q_upper_cond2 = np.insert(q_upper_cond2, 0, 0)  
            T_upper += dt * q_upper_cond2 / (rho_c_upper * V_upper)
            # upper to lowwer
            T_upper -= dt * q_upper_down / (rho_c_upper * V_upper)
            T_down[0:layer1] += dt * q_upper_down / (rho_c_down[0:layer1] * V_down[0:layer1])

            # 迭代
            temperatures1.append(T_upper[0])
            temperatures2.append(T_upper[-1])
            #  temperatures3.append(T_down[0])
            temperatures4.append(T_down[0])
            time1 += dt
            times.append(time1)

        time_Tc = times[next(x[0] for x in enumerate(temperatures1) if x[1] <= Tc)]
        time_end = times[next(x[0] for x in enumerate(temperatures1) if x[1] <= T_end)]
        #time_dec = times[next(x[0] for x in enumerate(temperatures1) if x[1] <= T_dec)]
        # plot
        fig = plt.figure()
        ax1 = fig.subplots()
        ax1.plot(times, temperatures1, color='r', label="T of upper core")
        ax1.plot(times, temperatures2, color='blue', label="T of upper surface")
        ax1.plot(times, temperatures4, color='firebrick', label="T of lower core")
        #ax1.plot(times, temperatures4, color='c', label="T of lower surface")
        #ax1.axvline(x=time_Tc, color='orange', linestyle='--', label=f"Time of Tc: {time_Tc:.2f}s")
        #ax1.axvline(x=time_dec, color='dodgerblue', linestyle='--', label=f"Time of Tdec: {time_dec:.2f}s")
        ax1.axvline(x=time_end, color='forestgreen', linestyle='--', label=f"Time of Tend: {time_end:.2f}s")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Temperature (K)")
        ax1.set_title(f'basalt radius = {radius_mm} mm; θ = {theta1}')
        ax1.legend()

        # path
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        save_path1 = os.path.join(desktop_path, r'\')
        #save_path2 = os.path.join(desktop_path, r'\')
        # 存txt


        # save fig
        file_name = f'{theta1}-{radius_mm}-{T_initial}K-{T_end}K-{T_lunar}K.png'
        full_path = os.path.join(save_path1, file_name)
        plt.savefig(full_path, dpi=300)
        #file_name = f'{theta1}-{radius_mm}-{T_initial}-{T_end}K.svg'
        #full_path = os.path.join(save_path2, file_name)
        #plt.savefig(full_path, format='svg')
        end_time = time.time()
        execution_time = end_time - start_time
        print(radius, execution_time)
        # plt.show()

