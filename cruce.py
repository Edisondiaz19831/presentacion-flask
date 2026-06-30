import matplotlib.pyplot as plt

# Nota: (altura en semitonos desde C, tensión tonal hipotética)
notas = {
    "C": (0, 0),
    "D": (2, 4),
    "E": (4, 2),
    "F": (5, 6),
    "G": (7, 2),
    "A": (9, 4),
    "B": (11, 8)
}

melodia = ["E", "F","F","E"]

tiempo = list(range(len(melodia)))
altura = [notas[n][0] for n in melodia]
tension = [notas[n][1] for n in melodia]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Trayectoria de la melodía
ax.plot(tiempo, altura, tension, marker="o", linewidth=2)

# Etiquetas de cada punto
for i, nota in enumerate(melodia):
    ax.text(tiempo[i], altura[i], tension[i] + 0.3, nota)

ax.set_xlabel("Tiempo")
ax.set_ylabel("Altura (semitonos)")
ax.set_zlabel("Tensión tonal")

ax.set_title("Trayectoria melódica en 3D")

plt.show()