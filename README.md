🩺 AI Medical Diagnosis Assistant
An AI-powered web application that predicts diseases based on natural language symptom descriptions using a deep learning model (LSTM) built with TensorFlow/Keras and deployed via Streamlit.

🚀 Demo
Enter your symptoms in plain English (e.g., "I have a fever, sore throat, and body aches") and the model will predict the most likely disease along with a confidence score.

🧠 How It Works

User inputs symptoms as free-form text in the Streamlit UI.
The text is tokenized and padded to a fixed length of 50 tokens.
A trained LSTM deep learning model processes the sequence.
The model outputs a predicted disease name and confidence score.


🗂️ Project Structure
AI-Medical-Diagnosis-Assistant/
│
├── app.py                  # Streamlit web application (main entry point)
├── train_model.py          # Model training script
├── preprocess.py           # Data preprocessing utilities
│
├── disease_dl_model.h5     # Trained Keras LSTM model
├── model.pkl               # Serialized scikit-learn model
├── tokenizer.pkl           # Fitted Keras tokenizer
├── label_encoder.pkl       # Fitted label encoder
│
└── requirements.txt        # Python dependencies

🏗️ Model Architecture
The deep learning model is built with Keras and consists of:
LayerDetailsEmbeddinginput_dim=5000, output_dim=128, input_length=50LSTM128 units, return_sequences=TrueLSTM64 unitsDense128 units, ReLU activationDense (output)num_classes units, Softmax activation

Loss: Sparse Categorical Crossentropy
Optimizer: Adam
Early Stopping: Monitors val_loss with patience of 2


📦 Installation
1. Clone the repository
bashgit clone https://github.com/Loguraja33/AI-Medical-Diagnosis-Assistant.git
cd AI-Medical-Diagnosis-Assistant
2. Install dependencies
bashpip install -r requirements.txt
3. Run the app
bashstreamlit run app.py

🔁 Retraining the Model
If you want to retrain with your own dataset:

Prepare a CSV file with two columns: text (symptom descriptions) and label (disease names).
Update the file path in train_model.py:

python   df = pd.read_csv("your_dataset.csv")

Run the training script:

bash   python train_model.py
This will generate fresh disease_dl_model.h5, tokenizer.pkl, and label_encoder.pkl files.

📊 Dataset
This project was trained on the Symptom2Disease dataset, which contains labeled symptom descriptions mapped to disease names.

🛠️ Tech Stack
TechnologyPurposePythonCore languageTensorFlow / KerasDeep learning model (LSTM)StreamlitWeb UIscikit-learnLabel encoding, train/test splitNumPy / PandasData processingPickleModel serialization
