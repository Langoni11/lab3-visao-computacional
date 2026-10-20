
import sys

import cv2
import numpy as np

# ---------------- Parâmetros ajustáveis ----------------
AREA_MINIMA = 150            # abaixo disso é ruído, não objeto
LIMIAR_ASPECTO_PARAFUSO = 1.8   # razão de aspecto (lado maior / lado menor)
LIMIAR_COMPACIDADE_PORCA = 0.70  # compacidade mínima para ser "porca"
FATOR_DEFEITO = 0.60         # área < 60% da média da classe => DEFEITUOSO


def gerar_imagem_demo(caminho="demo_esteira.png"):
    """Cria uma imagem sintética: fundo branco, porcas, parafusos e 1 peça pequena."""
    img = np.full((800, 800, 3), 255, np.uint8)
    # Porcas (círculo com furo)
    for cx, cy, r in [(150, 150, 55), (400, 150, 55), (650, 150, 22)]:  # a última é defeituosa
        cv2.circle(img, (cx, cy), r, (30, 30, 30), -1)
        cv2.circle(img, (cx, cy), r // 2, (255, 255, 255), -1)
    # Parafusos (retângulo alongado + cabeça)
    for cx, cy in [(150, 450), (400, 450)]:
        cv2.rectangle(img, (cx - 90, cy - 12), (cx + 90, cy + 12), (30, 30, 30), -1)
        cv2.rectangle(img, (cx - 100, cy - 28), (cx - 80, cy + 28), (30, 30, 30), -1)
    # Parafuso defeituoso (curto)
    cv2.rectangle(img, (570, 444), (660, 456), (30, 30, 30), -1)
    cv2.rectangle(img, (562, 436), (574, 464), (30, 30, 30), -1)
    cv2.imwrite(caminho, img)
    return caminho


def segmentar(img):
    """Limiarização de Otsu invertida + morfologia."""
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(cinza, (5, 5), 0)
    # Objetos escuros sobre fundo claro -> INV deixa o objeto branco (255)
    _, binaria = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)   # remove ruído
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=2)  # fecha buracos
    return binaria


def extrair_atributos(c):
    area = cv2.contourArea(c)
    perimetro = cv2.arcLength(c, True)
    compacidade = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
    # Retângulo rotacionado: a razão de aspecto não depende da orientação da peça
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    lado_maior, lado_menor = max(w, h), min(w, h)
    aspecto = lado_maior / lado_menor if lado_menor > 0 else 0
    return area, compacidade, aspecto


def classificar(compacidade, aspecto):
    if aspecto >= LIMIAR_ASPECTO_PARAFUSO:
        return "PARAFUSO"
    if compacidade >= LIMIAR_COMPACIDADE_PORCA:
        return "PORCA"
    return "IRREGULAR"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    caminho = gerar_imagem_demo() if sys.argv[1] == "--demo" else sys.argv[1]

    img = cv2.imread(caminho)
    if img is None:
        print(f"Erro: imagem '{caminho}' não encontrada.")
        return

    # 1. Segmentação
    binaria = segmentar(img)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2. Atributos + classificação de cada objeto
    objetos = []
    for c in contornos:
        area, comp, asp = extrair_atributos(c)
        if area < AREA_MINIMA:
            continue
        objetos.append({"contorno": c, "area": area, "comp": comp, "asp": asp,
                        "classe": classificar(comp, asp)})

    # 3. Detecção de defeitos: compara a área com a média da própria classe
    medias = {}
    for classe in {o["classe"] for o in objetos}:
        areas = [o["area"] for o in objetos if o["classe"] == classe]
        medias[classe] = np.mean(areas)

    resultado = img.copy()
    contagem = {}
    n_defeitos = 0
    for i, o in enumerate(objetos, 1):
        defeituoso = len(
            [x for x in objetos if x["classe"] == o["classe"]]
        ) > 1 and o["area"] < FATOR_DEFEITO * medias[o["classe"]]
        o["defeito"] = defeituoso
        contagem[o["classe"]] = contagem.get(o["classe"], 0) + 1
        n_defeitos += defeituoso

        cor = (0, 0, 255) if defeituoso else (0, 180, 0)
        rotulo = f"{o['classe']}" + (" - DEFEITUOSO" if defeituoso else "")
        x, y, w, h = cv2.boundingRect(o["contorno"])
        cv2.drawContours(resultado, [o["contorno"]], -1, cor, 2)
        cv2.putText(resultado, rotulo, (x, max(y - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2)
        print(f"Objeto {i}: {rotulo:<22} área={o['area']:.0f}  "
              f"compacidade={o['comp']:.3f}  aspecto={o['asp']:.2f}")

    print("\n===== RELATÓRIO =====")
    print(f"Total de objetos: {len(objetos)}")
    for classe, n in contagem.items():
        print(f"  {classe}: {n}")
    print(f"Defeituosos: {n_defeitos}")

    cv2.imshow("Mascara Binaria", binaria)
    cv2.imshow("Controle de Qualidade", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
