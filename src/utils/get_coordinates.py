import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Funzione per ridimensionare l'immagine
def resize_image(image, width=None, height=None):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is not None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        r = height / float(h)
        dim = (int(w * r), height)

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

# Funzione per aggiustare le coordinate ai valori originali
def adjust_coordinates(points, image, image_resized):
    adjusted_points = []
    for point in points:
        original_x = int(point[0] * (image.shape[1] / image_resized.shape[1]))
        original_y = int(point[1] * (image.shape[0] / image_resized.shape[0]))
        adjusted_points.append([original_x, original_y])
    return adjusted_points

# Funzione per selezionare l'immagine tramite interfaccia grafica
def select_image():
    root = tk.Tk()
    root.withdraw()  # Nasconde la finestra principale di tkinter

    file_path = filedialog.askopenfilename()
    return file_path

# Funzione principale per selezionare i punti
def select_points(image_path):
    # Lista per salvare i gruppi di punti selezionati
    all_points = []
    current_points = []

    zone_types = {0: 'no_passing_zone',
            1: 'road',
            2: 'pedestrian_crossing'
    }
    zones = []

    # Carica l'immagine
    image = cv2.imread(image_path)

    # Controlla se l'immagine è stata caricata correttamente
    if image is None:
        print("Errore: impossibile caricare l'immagine.")
        return

    # Ridimensiona l'immagine
    image_resized = resize_image(image, width=1200) 

    # Crea una finestra e mostra l'immagine
    cv2.namedWindow("Image")
    cv2.imshow("Image", image_resized)

    # Imposta la funzione di callback per il click del mouse
    def select_point(event, x, y, flags, param):
        nonlocal current_points, image_resized
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
            # Disegna un piccolo cerchio sul punto selezionato
            cv2.circle(image_resized, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow("Image", image_resized)

    cv2.setMouseCallback("Image", select_point)

    print("Seleziona i punti nell'immagine. Premi Invio per terminare un gruppo di punti, 'r' per rimuovere l'ultimo punto, 'q' per terminare.")


    # Aspetta che l'utente interagisca con i tasti
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if current_points:
                current_points.pop()
                # Ricarica l'immagine originale ridimensionata e ridisegna i punti rimanenti
                image_resized = resize_image(image, width=1200)
                for point in current_points:
                    cv2.circle(image_resized, (point[0], point[1]), 3, (0, 255, 0), -1)
                cv2.imshow("Image", image_resized)
        elif key == 13:  # Tasto Invio
            if current_points:
                adjusted_points = adjust_coordinates(current_points, image, image_resized)
                all_points.append(adjusted_points)
                
                while True:
                    zone_input = int(input('indica la classe della zona segmentata:\n\t0: no_passing_zone\n\t1: road\n\t2: roundabout\n\t3: cycling_path\n\t4: sidewalk\n\t5: pedestrian_crossing\n'))
                    if zone_input in [0,1,2,3,4,5]:
                        break
                zones.append(zone_types.get(zone_input))

                current_points = []
                # Ricarica l'immagine originale ridimensionata per eliminare i punti precedenti
                image_resized = resize_image(image, width=1200)
                cv2.imshow("Image", image_resized)
                print(f"{zones[-1]}: np.array({adjusted_points})")
        elif key == ord('q'):
            break

    # Chiudi tutte le finestre
    cv2.destroyAllWindows()

    # Salva i gruppi di punti aggiustati in un file di testo
    output_path = image_path.split('/')[:-1]
    output_path.append('coordinates.npy')
    
    all_points.append(zones)
    all_points = np.asarray(all_points, dtype="object")

    # output_path = r'C:\Users\Mattia\Desktop\test\coordinates.npy'

    out = ''
    for i in output_path:
        out = out+i+'\\'
    out = out[:-1]
    print(out)
    with open(out, 'wb') as file:
        np.save(file, all_points)

    print(f"Coordinate salvate in {out}")


if __name__ == "__main__":
    # Interfaccia per selezionare l'immagine
    selected_image_path = select_image()

    if selected_image_path:
        # Avvia la selezione dei punti sull'immagine selezionata
        select_points(selected_image_path)
    else:
        print("Nessuna immagine selezionata.")
