import cv2
import numpy as np
from tkinter import Tk, filedialog

# Variabili globali
drawing = False
center = None
axes = (0, 0)
img = None
img_original = None
ellipses = []

# Parametri dell'ellisse attualmente in editing
angle = 0
start_angle = 0
end_angle = 360

# File picker
def seleziona_immagine():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Seleziona immagine",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

# Callback mouse
def draw_ellipse(event, x, y, flags, param):
    global drawing, center, axes, img, angle, start_angle, end_angle

    if event == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img = img_original.copy()
        for e in ellipses:
            cv2.ellipse(img, e["center"], e["axes"], e["angle"],
                        e["startAngle"], e["endAngle"], (0, 255, 0), 2)

        axes = (abs(x - center[0]), abs(y - center[1]))
        cv2.ellipse(img, center, axes, angle, start_angle, end_angle, (0, 0, 255), 2)
        cv2.imshow("Editor di ellissi", img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        axes = (abs(x - center[0]), abs(y - center[1]))
        # Rimane in preview fino a ENTER
        update_preview()

# Disegna ellissi incluse quella corrente
def update_preview():
    preview = img_original.copy()
    for e in ellipses:
        cv2.ellipse(preview, e["center"], e["axes"], e["angle"],
                    e["startAngle"], e["endAngle"], (0, 255, 0), 2)
    if center and axes != (0, 0):
        cv2.ellipse(preview, center, axes, angle, start_angle, end_angle, (0, 0, 255), 2)
    cv2.imshow("Editor di ellissi", preview)

# Main
file_path = seleziona_immagine()
if not file_path:
    print("Nessuna immagine selezionata.")
    exit()

img_original = cv2.imread(file_path)
if img_original is None:
    raise ValueError("Errore nel caricamento dell'immagine.")

img = img_original.copy()
cv2.namedWindow("Editor di ellissi")
cv2.setMouseCallback("Editor di ellissi", draw_ellipse)

print("""
Editor attivo.
 - Trascina con il mouse per definire un'ellisse
 - Modifica con:
    a/d → rotazione (angle)
    w/s → startAngle
    e/q → endAngle
 - ENTER per confermare ellisse
 - r per reset
 - s per salvare (ellissi.npy)
 - q per uscire
""")

while True:
    update_preview()
    key = cv2.waitKeyEx(0)

    if key == 27:  # ESC
        break

    elif key == ord('r'):
        ellipses.clear()
        center = None
        axes = (0, 0)
        print("Reset effettuato.")

    elif key == ord('s'):
        np.save("ellissi.npy", ellipses)
        print(f"Salvati {len(ellipses)} ellissi in 'ellissi.npy'.")

    elif key == 13:  # ENTER
        if center and axes != (0, 0):
            ellipses.append({
                "center": center,
                "axes": axes,
                "angle": angle,
                "startAngle": start_angle,
                "endAngle": end_angle
            })
            center = None
            axes = (0, 0)
            print(f"Ellisse aggiunta. Totale: {len(ellipses)}")

    elif key == ord('a'):
        angle -= 5
    elif key == ord('d'):
        angle += 5
    elif key == ord('w'):
        start_angle += 5
    elif key == ord('s'):
        start_angle -= 5
    elif key == ord('e'):
        end_angle += 5
    elif key == ord('q'):
        end_angle -= 5
    elif key == 2424832:  # ←
        print("key pressed")
        center = (center[0] - 5, center[1])
    elif key == 2555904:  # →
        center = (center[0] + 5, center[1])
    elif key == 2490368:  # ↑
        center = (center[0], center[1] - 5)
    elif key == 2621440:  # ↓
        center = (center[0], center[1] + 5)

cv2.destroyAllWindows()
