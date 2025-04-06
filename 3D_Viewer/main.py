import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Параметры
N = 1000  # количество зондирующих импульсов
num_trials = 10000  # количество испытаний для метода Монте-Карло

# Функции для расчета вероятностей
def false_alarm_probability(threshold):
    return 1 - np.exp(-threshold)

def detection_probability(threshold, q):
    return 1 - np.exp(-q * threshold)

# Генерация порогов
thresholds = np.linspace(0, 10, 100)

# Расчет вероятностей ложного обнаружения
F_values = [false_alarm_probability(threshold) for threshold in thresholds]

# Построение графика вероятности ложного обнаружения
plt.figure(figsize=(10, 6))
plt.plot(thresholds, F_values)
plt.xscale('linear')
plt.yscale('log')
plt.title('Вероятность ложного обнаружения от порога принятия решения F(C)')
plt.xlabel('Порог C')
plt.ylabel('Вероятность ложного обнаружения F')
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(0.001, 1)
plt.show()

# Расчет вероятности правильного обнаружения для набора отношений сигнал/шум
q_values = np.concatenate([np.linspace(1, 3, 5), np.linspace(3.5, 10, 10)])  # Неравномерный шаг по q
D_values = {q: [detection_probability(threshold, q) for threshold in thresholds] for q in q_values}

# Построение графиков вероятности правильного обнаружения
plt.figure(figsize=(10, 6))
for q, D in D_values.items():
    plt.plot(thresholds, D, label=f'q={q:.2f}')
plt.xscale('linear')
plt.yscale('linear')
plt.title('Вероятность правильного обнаружения от порога принятия решения D(C,q)')
plt.xlabel('Порог C')
plt.ylabel('Вероятность правильного обнаружения D')
plt.legend()
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(0, 1)
plt.show()

# Метод Монте-Карло для оценки вероятностей
def monte_carlo_detection(threshold, q):
    signal = np.random.normal(loc=q, scale=1, size=num_trials)
    noise = np.random.normal(loc=0, scale=1, size=num_trials)
    detection = (signal + noise) > threshold
    return np.mean(detection)

# Расчет вероятностей правильного обнаружения методом Монте-Карло
D_mc_values = {q: [monte_carlo_detection(threshold, q) for threshold in thresholds] for q in q_values}

# Построение графиков вероятности правильного обнаружения (Метод Монте-Карло)
plt.figure(figsize=(10, 6))
for q, D_mc in D_mc_values.items():
    plt.plot(thresholds, D_mc, label=f'q={q:.2f} (MC)')
plt.xscale('linear')
plt.yscale('linear')
plt.title('Вероятность правильного обнаружения от порога принятия решения (Метод Монте-Карло)')
plt.xlabel('Порог C')
plt.ylabel('Вероятность правильного обнаружения D')
plt.legend()
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(0, 1)
plt.show()

# Пороговое отношение сигнал/шум для заданных вероятностей
desired_probabilities = [(0.1, 0.9), (0.01, 0.95), (0.001, 0.99)]
threshold_d_f_pairs = []

for F_target, D_target in desired_probabilities:
    C = -np.log(1 - F_target)
    q = -np.log(1 - D_target) / C
    threshold_d_f_pairs.append((F_target, D_target, C, q))

# Создание DataFrame для таблицы результатов