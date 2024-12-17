import pandas as pd
import openai
import uuid
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import speech_recognition as sr
import io

# Set your OpenAI API key
openai.api_key = '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'


class AudioProcessor(AudioProcessorBase):
    """Custom audio processor using speech recognition."""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.last_audio_transcription = ""

    def recv(self, frame):
        """Process incoming audio frames and transcribe speech to text."""
        audio_data = frame.to_ndarray().tobytes()
        sample_rate = frame.sample_rate
        audio_clip = sr.AudioData(audio_data, sample_rate, 2)  # 2: 16-bit PCM

        try:
            text = self.recognizer.recognize_google(audio_clip)
            self.last_audio_transcription = text
            st.session_state['last_audio_transcription'] = text
        except sr.UnknownValueError:
            self.last_audio_transcription = "Audio not clear, please try again."
        except sr.RequestError:
            self.last_audio_transcription = "Speech recognition service unavailable."
        return frame


class Chatbot:
    def __init__(self):
        """Initialize the chatbot, setting up session state and chat history."""
        self.initialize_session()

    def initialize_session(self):
        """Initialize session state variables."""
        if 'session_id' not in st.session_state:
            st.session_state['session_id'] = str(uuid.uuid4())

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        if 'last_audio_transcription' not in st.session_state:
            st.session_state['last_audio_transcription'] = ""

    def load_dataset(self):
        """Load the dataset from session state, return the DataFrame and its name."""
        dataset_data = st.session_state.get('df_to_chat', None)
        dataset_name = st.session_state.get('dataset_name_to_chat', None)

        if dataset_data is None:
            st.error("No dataset found in session state.")
            return None, None

        df = self.convert_dataset_to_dataframe(dataset_data)
        return df, dataset_name

    def convert_dataset_to_dataframe(self, dataset_data):
        """Convert the dataset bytes into a pandas DataFrame."""
        try:
            return pd.read_csv(io.BytesIO(dataset_data), encoding='utf-8', on_bad_lines='skip')
        except Exception as e:
            st.error(f"Failed to load dataset: {str(e)}")
            return None

    def call_openai(self, prompt, df):
        """Generate responses using OpenAI API with dataset context."""
        full_prompt = self.build_openai_prompt(prompt, df)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI that helps answer questions about datasets."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=200,
                temperature=0.6
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"

    def build_openai_prompt(self, user_prompt, df):
        """Build the prompt for OpenAI by including dataset context."""
        dataset_summary = self.generate_dataset_summary(df)
        full_prompt = f"{user_prompt}\n\nDataset Context:\n{dataset_summary}"
        return full_prompt

    def generate_dataset_summary(self, df):
        """Generate a summary of the dataset."""
        summary = f"Columns: {', '.join(df.columns)}\n"
        summary += f"Missing Values: {df.isnull().sum().sum()} total\n"
        summary += f"Summary Statistics:\n{df.describe().to_string()}\n"
        return summary

    def run(self):
        """Run the chatbot, handling user input and generating responses."""
        df, dataset_name = self.load_dataset()

        if df is not None and dataset_name is not None:
            st.header(f"Chat with your Dataset: {dataset_name} 🧠")

            user_input = st.chat_input(f"Ask a question about {dataset_name}!")
            if user_input:
                self.append_chat_history('user', user_input)
                response = self.call_openai(user_input, df)
                self.append_chat_history('assistant', response)

            self.setup_voice_input()

            for chat in st.session_state['chat_history']:
                st.chat_message(chat['role']).write(chat['message'])

            self.export_chat_history()
        else:
            st.error("No dataset selected. Please upload a dataset.")

    def setup_voice_input(self):
        """Set up Streamlit WebRTC for voice input."""
        st.write("🎙️ Use your voice to interact with the chatbot:")
        webrtc_streamer(
            key="voice_input",
            mode=WebRtcMode.SENDRECV,
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={
                "audio": {"deviceId": "default"},  # Use system default microphone
                "video": False
            },
        )

        if st.session_state['last_audio_transcription']:
            st.write(f"Transcribed Audio: {st.session_state['last_audio_transcription']}")
            self.append_chat_history('user', st.session_state['last_audio_transcription'])
            response = self.call_openai(st.session_state['last_audio_transcription'], self.load_dataset()[0])
            self.append_chat_history('assistant', response)
            st.session_state['last_audio_transcription'] = ""

    def append_chat_history(self, role, message):
        """Append a chat message to the history."""
        st.session_state['chat_history'].append({'role': role, 'message': message})
        st.chat_message(role).write(message)

    def export_chat_history(self):
        """Allow the user to export chat history."""
        if st.session_state.get('chat_history'):
            chat_history_text = "\n".join(
                [f"{chat['role'].capitalize()}: {chat['message']}" for chat in st.session_state['chat_history']]
            )
            st.download_button(
                label="Download Chat History",
                data=chat_history_text,
                file_name="chat_history.txt",
                mime="text/plain"
            )


# Run the chatbot
chatbot = Chatbot()
chatbot.run()
