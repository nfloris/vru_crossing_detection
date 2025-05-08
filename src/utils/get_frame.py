import cv2
import os
import tkinter as tk
from tkinter import filedialog

# Funzione per selezionare il video tramite interfaccia grafica
def select_video():
    root = tk.Tk()
    root.withdraw()  # Nasconde la finestra principale di tkinter

    file_path = filedialog.askopenfilename()
    return file_path

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

def main(video_path):
    # Apri il video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Errore nell'aprire il file video.")
        return

    print('comandi:\n',
          '\t n -> avanzare di 5 frame\n',
          '\t b -> retrocedere di 1 frame\n',
          '\t s -> salvare il frame\n',
          '\t q -> uscire\n')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        # Imposta il frame corrente
        print(f'sei al frame numero: {current_frame} di {frame_count}')
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        resized_frame = resize_image(frame, width=1200)
        
        if not ret:
            print("Errore nel leggere il frame.")
            break

        # Mostra il frame
        cv2.imshow('Frame', resized_frame)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('n'):  # Vai avanti di 5 frame
            current_frame = min(current_frame + 5, frame_count - 1)
        elif key == ord('b'):  # Vai al frame precedente
            current_frame = max(current_frame - 1, 0)
        elif key == ord('s'):  # salva il frame
            output_path = video_path.split('/')[:-1]
            output_path.append(f'frame {current_frame}.jpg')
            output_path = os.path.join(*output_path)
            cv2.imwrite(output_path, frame)
            print('frame salvato in:', output_path)
        elif key == ord('q'):  # Esci
            break

    # Rilascia la cattura e chiudi tutte le finestre
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = select_video()
    main(video_path)
