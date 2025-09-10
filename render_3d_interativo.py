#!/usr/bin/env python3
# render_3d_interativo.py
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (ativa proj. 3d)
from matplotlib.tri import Triangulation

def main(path="output.txt"):
    # Lê x, y, u
    try:
        data = np.loadtxt(path)
    except Exception as e:
        print(f"Erro ao ler {path}: {e}")
        sys.exit(1)

    if data.ndim != 2 or data.shape[1] < 3:
        print("Arquivo deve ter ao menos 3 colunas: x y u")
        sys.exit(1)

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x,y)")
    ax.set_title("Superfície 3D de u(x,y)")

    # Detecta se é grade regular
    ux = np.unique(x)
    uy = np.unique(y)
    is_grid = (ux.size * uy.size == x.size)

    if is_grid:
        # Preenche Z numa grade (robusto ao ordenamento do arquivo)
        ix = {val: i for i, val in enumerate(ux)}
        iy = {val: j for j, val in enumerate(uy)}
        Z = np.full((ux.size, uy.size), np.nan, dtype=float)
        for xi, yi, zi in zip(x, y, z):
            Z[ix[xi], iy[yi]] = zi
        X, Y = np.meshgrid(ux, uy, indexing="ij")

        # Plot como superfície (grade regular)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               linewidth=0.2, antialiased=True)
    else:
        # Plot com triangulação (pontos não estruturados)
        tri = Triangulation(x, y)
        surf = ax.plot_trisurf(tri, z, linewidth=0.2, antialiased=True)

    # Interação: arraste com o mouse para rotacionar, rodinha para zoom
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "output.txt"
    main(path)
