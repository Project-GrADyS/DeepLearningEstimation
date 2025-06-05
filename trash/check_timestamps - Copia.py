import numpy as np
import matplotlib.pyplot as plt

# Carregar o dataset
print("Carregando dataset.npz...")
data = np.load("dataset.npz")

# Obter os timestamps
timestamps = data['timestamps']

# Exibir informações sobre os timestamps
print(f"\nInformações sobre os timestamps:")
print(f"Shape: {timestamps.shape}")
print(f"Tipo de dados: {timestamps.dtype}")
print(f"Primeiros 10 timestamps: {timestamps[:10]}")
print(f"Últimos 10 timestamps: {timestamps[-10:]}")
print(f"Valor mínimo: {timestamps.min()}")
print(f"Valor máximo: {timestamps.max()}")

# Calcular o intervalo entre timestamps consecutivos
if len(timestamps) > 1:
    intervals = np.diff(timestamps)
    print(f"\nIntervalo médio entre timestamps: {np.mean(intervals):.6f}")
    print(f"Intervalo mínimo: {np.min(intervals):.6f}")
    print(f"Intervalo máximo: {np.max(intervals):.6f}")

# Plotar os timestamps
plt.figure(figsize=(12, 6))
plt.plot(timestamps, label='Timestamps')
plt.title('Sequência de Timestamps')
plt.xlabel('Índice')
plt.ylabel('Timestamp')
plt.grid(True)
plt.legend()
plt.savefig('timestamps_plot.png')
print("\nGráfico dos timestamps salvo como 'timestamps_plot.png'")

# Fechar o arquivo
data.close() 