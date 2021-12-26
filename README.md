# DZ_BRISO manual
Open the file 1-3 пункт (this file takes lace in the folder DZ) and input your initial data (  Определение диапазонов варьирования входных проектных параметров). 
Run the code. If in the part (Выбор формы топливного заряда и определение его геометрических характеристик) average more then 0,5 and less then 0,75 that you can continue use this code, 
else you have mistakes in the initial data or you have a star - shaped powder (this code is used for slot shaped powders).
In the picture with stars you can see the best conditions for your propulsion system (PS) (mass of PS and length of PS).
Next step. Open the file Параметры РДТТ. Bring there necessaraly parametrs from the file 1-3 пункт (Parametrs are pointed in 4 пункт).
There is you can determine the process of flame the powder and get the picture.
The end of home work. Open the file (Масса воспламенителя и индикаторная кривая) (this file takes lace in the folder DZ/5 пункт).
Bring in this file necessaraly parameters from other programms. 
The first step. Run the minimization and determine count of grains and the width of the one grain.
The second step. Run the function for determenation of the net and get the optimal parameters of count of grains and the width of the one grain.
The next step. Using the optimal parameters run the INIT1_1 and get indicator curve for -50 degrees then change the tempreture to +20 in init conditons (T_0 = T_20) and run the INIT1_2
then change the tempreture to +50 in init conditons (T_0 = T_50) and run the INIT1_3. Now you can get the picture with indicator curve for (-50, +20m, +50) for time = (0, 0.25) seconds.
The last step. Using the optimal parameters run the INIT2_1 and get indicator curve for -50 degrees then change the tempreture to +20 in init conditons (T_0 = T_20) and run the INIT2_2
then change the tempreture to +50 in init conditons (T_0 = T_50) and run the INIT2_3. Now you can get the picture with indicator curve for (-50, +20m, +50) 
for time = (0.25, while the flame of powder won't be over).
