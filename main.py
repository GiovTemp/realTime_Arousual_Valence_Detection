import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import threading
import cv2
from tensorflow.keras.models import load_model
import dlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns

def detect_landmarks(image_path):
    # Load the pre-trained face detector model
    detector_face = dlib.get_frontal_face_detector()

    # Load an image
    image = cv2.imread(image_path)

    # Convert the image to grayscale (required by Dlib)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector_face(gray)

    # Loop through each face and detect landmarks
    landmarks = predictor(gray, faces[0])
    landmarks_coordinates = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

    return image, landmarks_coordinates


def landmarks_combination_df(landmarks):
    # Inizializzazione di liste per le combinazioni lineari di coordinate x e y
    vec_comb_lin_x = []
    vec_comb_lin_y = []

    # Dividi l'array in coordinate x e y
    land_x = landmarks[:, 0]  # Estrae tutte le colonne 0 (coordinate x)
    land_y = landmarks[:, 1]  # Estrae tutte le colonne 1 (coordinate y)

    # Iterazione su ogni punto di landmark (68 in totale)
    for i in range(68):
        sum_x = 0
        sum_y = 0

        # Calcolo delle combinazioni lineari rispetto alle altre coordinate x e y
        for j in range(68):
            sum_x = sum_x + land_x[i] * (1 / land_x[j])
            sum_y = sum_y + land_y[i] * (1 / land_y[j])

        # Aggiunta delle combinazioni lineari alle liste
        vec_comb_lin_x.append(sum_x)
        vec_comb_lin_y.append(sum_y)

    # Concatenazione delle liste di combinazioni lineari di x e y
    all_landmark = vec_comb_lin_x + vec_comb_lin_y

    # Creazione di un DataFrame contenente tutte le combinazioni lineari
    return pd.DataFrame([all_landmark])

def analyze_frame(aus_sequence, model):
    # Utilizza il modello per fare previsioni sul frame
    predictions = model.predict(aus_sequence)

    # Estrazione arousal e valence dalle previsioni
    arousal, valence = predictions[0][0], predictions[0][1]

    return arousal, valence

def save_frame(frame, frame_index, folder="temp_frames"):
    # Verifica se la cartella specificata esiste, altrimenti creala
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Genera il percorso del frame utilizzando l'indice e il formato 'frame_0.jpg', 'frame_1.jpg', ..., 'frame_9.jpg'
    frame_path = os.path.join(folder, f"frame_{frame_index % 10}.jpg")

    # Salva il frame come un'immagine JPG nel percorso specificato
    cv2.imwrite(frame_path, frame)

    # Restituisce il percorso del frame appena salvato
    return frame_path


def analyze_frames_async(aus_list, emotion, model):
    # Creazione di un DataFrame con le Action Units
    test_df = pd.DataFrame([aus_list])

    # Aggiunta di colonne per le emozioni
    emotions_columns = ['happiness', 'sadness', 'surprise', 'fear', 'anger', 'neutral']
    test_df[emotions_columns] = 0
    test_df[emotion] = 1

    # Analisi del frame utilizzando il modello
    arousal, valence = analyze_frame(test_df, model)

    # Aggiornamento delle variabili globali di arousal e valence
    global global_arousal_valence
    with lock:
        global_arousal_valence = (arousal, valence)

def predict_emotion(prediction):
    # Definizione di intervalli di indici associati a diverse emozioni
    emotions = {'happiness': [4, 9], 'sadness': [0, 2, 11], 'surprise': [0, 1, 3, 17], 'fear': [0, 1, 2, 3, 5, 13, 17], 'anger': [2, 3, 5, 14]}

    # Inizializzazione dell'emozione predetta come 'neutral'
    emotion = 'neutral'

    # Inizializzazione della differenza temporanea come 0
    diff_temp = 0

    # Iterazione su ciascuna emozione e valutazione della predizione
    for key, value in emotions.items():
        sum = 0
        v_len = len(value) / 2

        # Calcolo del conteggio degli indici di predizione che superano una soglia
        for v in value:
            if prediction[v] >= 0.48:
                sum += 1

        # Calcolo della differenza rispetto alla metà del numero totale di indici
        diff = sum - v_len

        # Aggiornamento dell'emozione prevista se la differenza è maggiore o uguale
        if diff >= diff_temp:
            emotion = key
            diff_temp = diff

    # Restituzione dell'emozione prevista
    return emotion

def process_frame(frame, frame_index, model, scaler):
    # Salva il frame
    frame_path = save_frame(frame, frame_index)
    # Calcola i valori AU

    # Stampa il percorso del frame salvato
    print(f'Frame path: {frame_path}')

    # Rileva i landmark nell'immagine
    image, landmarks = detect_landmarks(frame_path)

    # Verifica se sono stati rilevati landmark
    if len(landmarks) > 0:
        # Calcola le combinazioni lineari dei landmark e crea un DataFrame
        landmarks_df = landmarks_combination_df(landmarks)

        # Effettua la previsione delle Action Units utilizzando il modello
        prediction = au_pred_model.predict(landmarks_df)

        # Determina l'emozione basata sulla previsione delle Action Units
        emotion = predict_emotion(prediction[0])

        # Stampa l'emozione predetta
        print(emotion)

        # Analizza in modo asincrono i frame e aggiorna le variabili globali di arousal e valence
        analyze_frames_async(prediction[0], emotion, model)


sns.set()

# Variabili globali per memorizzare arousal e valence
global_arousal_valence = (0.0, 0.0)
lock = threading.Lock()

au_pred_model = load_model('data/au_pred_model.h5')
arousal_valence_pred_model = load_model('data/arousal_valence_pred_model.h5')

# Load the pre-trained face landmark predictor model
predictor_path = "data/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Inizializzazione di Matplotlib
fig, ax = plt.subplots(figsize=(6, 6))
canvas = FigureCanvas(fig)

# Inizializza la cattura video
cap = cv2.VideoCapture(0)

frame_counter = 0
capture_interval = 20  # Analizza un frame ogni 20 frames
frame_index = 0
aus_array = []

arousal, valence = 0.0, 0.0  # Inizializza i valori fuori dal loop
points_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    if frame_counter == capture_interval:
        frame_counter = 0
        # Crea un thread per processare il frame
        threading.Thread(target=process_frame, args=(frame, frame_index, arousal_valence_pred_model, MinMaxScaler())).start()
        frame_index += 1

    with lock:
        arousal, valence = global_arousal_valence
        points_history.append((valence, arousal))
    # Pulisci il grafico precedente
    ax.clear()

    # Definisci i limiti del grafico

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # Disegna gli assi cartesiani
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)


    # Aggiungi la storia dei punti come mappa di calore
    if points_history:
        x, y = zip(*points_history)
        ax.scatter(x, y, c=range(len(x)), cmap='viridis', alpha=0.6)
    # Aggiungi il punto al grafico
    ax.scatter(valence, arousal, color='blue')
    ax.text(0.5, 0.9, f"Arousal: {arousal:.2f}\nValence: {valence:.2f}", fontsize=12, transform=ax.transAxes)

    # Visualizza la griglia
    ax.grid(True)


    # Aggiorna il canvas di Matplotlib
    canvas.draw()

    # Converti il canvas in un'immagine OpenCV
    buf = canvas.buffer_rgba()
    graph_image = np.asarray(buf)
    graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2RGB)

    # Ridimensiona l'immagine del grafico e del frame della webcam per adattarli affiancati
    graph_image = cv2.resize(graph_image, (frame.shape[1], frame.shape[0]))
    combined_image = np.hstack((frame, graph_image))

    # Mostra l'immagine combinata
    cv2.imshow('Frame + Graph', combined_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
