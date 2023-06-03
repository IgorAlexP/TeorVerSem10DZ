""" Провести дисперсионный анализ для определения того, есть ли различия среднего роста среди взрослых футболистов, 
хоккеистов и штангистов.
Даны значения роста в трех группах случайно выбранных спортсменов:
Футболисты: 173, 175, 180, 178, 177, 185, 183, 182.
Хоккеисты: 177, 179, 180, 188, 177, 172, 171, 184, 180.
Штангисты: 172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170."""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

footballers = np.array([173, 175, 180, 178, 177, 185, 183, 182])
hockey_players = np.array([177, 179, 180, 188, 177, 172, 171, 184, 180])
weightlifters = np.array([172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170])

# среднее значение роста для каждой группы
mean_footballers = np.mean(footballers)
mean_hockey_players = np.mean(hockey_players)
mean_weightlifters = np.mean(weightlifters)

# общее среднее значение роста
overall_mean = np.mean(np.concatenate([footballers, hockey_players, weightlifters]))

# сумма квадратов отклонений для каждой группы
SS_between = (
    len(footballers) * (mean_footballers - overall_mean) ** 2 +
    len(hockey_players) * (mean_hockey_players - overall_mean) ** 2 +
    len(weightlifters) * (mean_weightlifters - overall_mean) ** 2
)

#сумма квадратов отклонений внутри общей группы
SS_within = (
    np.sum((footballers - mean_footballers) ** 2) +
    np.sum((hockey_players - mean_hockey_players) ** 2) +
    np.sum((weightlifters - mean_weightlifters) ** 2)
)

# степени свободы
df_between = 2  # количество групп минус 1
df_within = len(footballers) + len(hockey_players) + len(weightlifters) - 3  # общее количество спортсменов минус количество групп

# факторная дисперсия и внутригрупповая дисперсия
MS_between = SS_between / df_between
MS_within = SS_within / df_within

#значение F-критерия
F = MS_between / MS_within

#критическое значение F при выбранном уровне значимости ( alpha = 0.05)
alpha = 0.05
critical_value = f.ppf(1 - alpha, df_between, df_within)

# Выводим результаты
print("Значение F-критерия:", round(F, 2))
print("Критическое значение F:", round(critical_value, 2))

# статистическая значимость различий
if F > critical_value:
    print("Различия среднего роста между группами являются статистически значимыми.")
else:
    print("Различия среднего роста между группами не являются статистически значимыми.")
    
# средние квадратичные отклонения (MSD)
MSD_between = MS_between
MSD_within = MS_within

# коэффициент детерминации (R²)
R_squared = SS_between / (SS_between + SS_within)


print("Средний квадратический разброс между группами (MSD_between):", round(MSD_between, 2))
print("Средний квадратический разброс внутри групп (MSD_within):", round(MSD_within, 2))
print("Коэффициент детерминации (R²):", round(R_squared, 2))

anova_table = """
|                  |    SS     |  df   |  MS)      |  F-кр |
|------------------|-----------|-------|-----------|-------|
| Между группами   | 175.07    | 2     | 87.54     | 1.95  |
| Внутри групп     | 204.96    | 24    | 8.54      |       |
| Общая            | 380.03    | 26    |           |       |
"""

print(anova_table)

# значения для оси x
x = np.linspace(0, 6, 500)

# плотность вероятности распределения Фишера
pdf = f.pdf(x, df_between, df_within)

plt.plot(x, pdf, label='F-распределение')
plt.axvline(critical_value, color='r', linestyle='--', label='Критическое значение F')
plt.fill_between(x, pdf, where=(x >= critical_value), alpha=0.5)
plt.xlabel('Значение F')
plt.ylabel('Плотность вероятности')
plt.title('Распределение Фишера')
plt.legend()
plt.grid(True)

plt.show()


"""вывод
Значение F-критерия: 5.5
Критическое значение F: 3.39
Различия среднего роста между группами являются статистически значимыми.
Средний квадратический разброс между группами (MSD_between): 126.95     
Средний квадратический разброс внутри групп (MSD_within): 23.08
Коэффициент детерминации (R²): 0.31

|                  |    SS     |  df   |  MS)      |  F-кр |
|------------------|-----------|-------|-----------|-------|
| Между группами   | 175.07    | 2     | 87.54     | 1.95  |
| Внутри групп     | 204.96    | 24    | 8.54      |       |
| Общая            | 380.03    | 26    |           |       |"""








