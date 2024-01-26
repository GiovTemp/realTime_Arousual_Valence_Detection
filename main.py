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
    landmarks_list = []
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_coordinates = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
        landmarks_list.append(landmarks_coordinates)

        # Draw landmarks on the image
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Draw a green circle for each landmark
            cv2.putText(image, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        middle_point = np.mean(landmarks_coordinates, axis=0).astype(int)

        # Draw a circle at the middle point on the image
        cv2.circle(image, tuple(middle_point), 4, (0, 0, 255), -1)

    return image, landmarks_list


def landmarks_combination_df(landmarks):
    vec_comb_lin_x = []
    vec_comb_lin_y = []

    # Dividi l'array in coordinate x e y
    land_x = landmarks[:, 0]  # Estrae tutte le colonne 0 (coordinate x)
    land_y = landmarks[:, 1]  # Estrae tutte le colonne 1 (coordinate y)

    for i in range(68):
        sum_x = 0
        sum_y = 0

        for j in range(68):
            sum_x = sum_x + land_x[i] * (1 / land_x[j])
            sum_y = sum_y + land_y[i] * (1 / land_y[j])

        vec_comb_lin_x.append(sum_x)
        vec_comb_lin_y.append(sum_y)

    all_landmark = vec_comb_lin_x + vec_comb_lin_y

    return pd.DataFrame([all_landmark], columns=all_columns_names)


def analyze_frame(aus_sequence, model):
    # Utilizza il modello per fare previsioni sulla sequenza
    predictions = model.predict(aus_sequence)

    # Estrai arousal e valence dalle previsioni
    arousal, valence = predictions[0][0], predictions[0][1]

    return arousal, valence


def save_frame(frame, frame_index, folder="temp_frames"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    frame_path = os.path.join(folder, f"frame_{frame_index % 10}.jpg")
    cv2.imwrite(frame_path, frame)
    return frame_path


def analyze_frames_async(aus_array, model):
    # Crea una sequenza ripetendo il frame
    aus_np = np.array(aus_array)
    #au_sequence = aus_np.reshape(1, 10, -1)
    df = pd.DataFrame(aus_np)

    # X = aus_np[:, :-2]  # tutte le colonne tranne le ultime due
    # y = aus_np[:, -2:]  # solo le ultime due colonne (arousal e valence)
    #
    num_samples = len(df) // 10
    #
    X_seq = np.array([df[i * 10:(i + 1) * 10] for i in range(num_samples)])

    arousal, valence = analyze_frame(X_seq, model)

    global global_arousal_valence
    with lock:
        global_arousal_valence = (arousal, valence)

def predict_emotion(prediction):
    emotions = {'happiness': [4, 9], 'sadness': [0, 2, 11], 'surprise': [0, 1, 3, 17], 'fear': [0, 1, 2, 3, 5, 13, 17], 'anger': [2, 3, 5, 14]}

    emotion = 'neutral'
    diff_temp = 0

    for key, value in emotions.items():
      sum = 0
      v_len = len(value)/2

      for v in value:
        if prediction[v] >= 0.48:
          sum = sum + 1

      diff = sum - v_len

      if diff >= diff_temp:
        emotion = key
        diff_temp = sum - v_len

    return emotion

def process_frame(frame, frame_index, model, scaler):
    # Salva il frame
    frame_path = save_frame(frame, frame_index)
    # Calcola i valori AU

    print(f'Frame path: {frame_path}')

    image, landmarks = detect_landmarks(frame_path)

    if len(landmarks) > 0:
        landmarks_df = landmarks_combination_df(landmarks[0])
        prediction = au_pred_model.predict(landmarks_df)

        print(predict_emotion(prediction[0]))

        au_values = np.array(list(prediction[0]))
        #au_values_scaled = scaler.fit_transform(au_values.reshape(1, -1))

        # Aggiungi i valori AU all'array e applica FIFO
        aus_array.append(au_values)

        if len(aus_array) > 10:
            aus_array.pop(0)
        # Se abbiamo raccolto 10 frame, analizzali
        if len(aus_array) == 10:
            analyze_frames_async(aus_array.copy(), model)


# Variabili globali per memorizzare arousal e valence
global_arousal_valence = (0.0, 0.0)
lock = threading.Lock()

columns_names = [f'Landmark_x_{i}' for i in range(68)]
columns_names1 = [f'Landmark_y_{i}' for i in range(68)]

all_columns_names = columns_names + columns_names1

au_pred_model = load_model('data/au_pred_model.h5')
arousal_valence_pred_model = load_model('data/arousal_valence_pred_model.h5')

# Load the pre-trained face landmark predictor model
predictor_path = "data/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Percorso del file CSV
file_path = 'data/new_aus_df.csv'

# Carica il DataFrame dal file CSV
new_aus_df = pd.read_csv(file_path)

new_aus_df = new_aus_df.dropna()

print(new_aus_df.head())

# Inizializza la grafica di Matplotlib
fig, ax = plt.subplots()
canvas = FigureCanvas(fig)

# Inizializza la cattura video
cap = cv2.VideoCapture(0)

frame_counter = 0
capture_interval = 20  # Analizza un frame ogni 20 frames
frame_index = 0
aus_array = []

arousal, valence = 0.0, 0.0  # Inizializza i valori fuori dal loop

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
    # Pulisci il grafico precedente
    ax.clear()

    # Definisci i limiti del grafico
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

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

