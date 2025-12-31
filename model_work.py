import os
from io import BytesIO
from typing import Literal

import numpy as np
import soundfile
import torch
import torchaudio
from openai import OpenAI
from transformers import AutoModel, AutoProcessor, Pipeline, pipeline


class TextGenerationModel:

    def __init__(self):

        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def generate_text(self, prompt):

        completion = self.client.chat.completions.create(
            model="google/gemma-3-1b",
            messages=[
                {"role": "system", "content": "Always answer in rhymes."},
                {"role": "user", "content": "Introduce yourself."},
            ],
            temperature=0.7,
            max_tokens=-1,
            stream=False,
        )

        if completion and completion.choices and len(completion.choices) > 0:

            return completion.choices[0].message
        else:
            return "Error: No response generated from the model."


class PipelineTextGenModel:

    def __init__(self):
        self.pipline = None
        self.device_name = None

        if torch.cuda.is_available():
            self.device_name = "cuda"

        else:
            self.device_name = "cpu"

    def load_pipeline(self):

        print("Loading Pipeline_text_gen_Model...")

        self.pipe = pipeline(
            "text-generation",
            model="gpt2",
            torch_dtype=torch.bfloat16,
            device=self.device_name,
        )

        print("Pipeline_text_gen_Model is loaded bro")

    def predict(self, user_message):

        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate",
            },
            {"role": "user", "content": user_message},
        ]
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )

        outputs_data = outputs[0]["generated_text"].split("<|assistant|>")
        if len(outputs_data) != 2:

            return "unexpected output len"

        result = outputs_data[1].strip()
        return result

    def non_temp_predict(self, user_message, max_new_tokens=100, temperature=0.7):
        """Generate text for chat models that support templates"""
        if self.pipe is None:
            return "Error: Model not loaded."

        try:
            # Create messages in chat format
            messages = [
                {"role": "system", "content": "You are a friendly chatbot"},
                {"role": "user", "content": user_message},
            ]

            # Check if tokenizer has chat template
            if (
                hasattr(self.pipe.tokenizer, "chat_template")
                and self.pipe.tokenizer.chat_template
            ):

                prompt = self.pipe.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:

                prompt = f"System: {messages[0]['content']}\nUser: {user_message}\nAssistant:"

            outputs = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )

            print(outputs)

            return outputs[0]["generated_text"]

        except Exception as e:
            return f"Error: {str(e)}"


class AudioGenModel:

    def __init__(self):

        self.speaker = "[spkr_63]"

        self.pipe = None
        self.device_name = None

        if torch.cuda.is_available():
            self.device_name = "cuda"

        else:

            self.device_name = "cpu"

    def load_audio_pipeline(self):

        print("Loading audio model bro...")

        self.pipe = pipeline(
            task="indri-tts",
            model="11mlabs/indri-0.1-124m-tts",
            device=self.device_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        print("Audio model is loaded bro..........")

        return self

    def generate_audio(self, prompt: str, chunk_size: int = 1024):

        output = self.pipe([prompt], speaker=self.speaker)

        audio_array = output[0]["audio"][0].numpy().squeeze()

        sample_rate = 24000

        buffer = BytesIO()

        soundfile.write(buffer, audio_array, sample_rate, format="WAV")

        buffer.seek(0)

        while True:
            chunk = buffer.read(chunk_size)
            if not chunk:
                break
            yield chunk
