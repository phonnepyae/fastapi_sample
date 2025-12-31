import asyncio
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Body, FastAPI, Query, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from model_work import AudioGenModel, PipelineTextGenModel, TextGenerationModel
from schemas import (StudentRequestModel, StudentResponseModel,
                     textRequestModel, textResponseModel)

# This tells httpx/openai to ignore system proxies
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)


ml_models = {}


@asynccontextmanager
async def startup_lifespan(app: FastAPI):

    text_gen_model = TextGenerationModel()

    ml_models["TextGenModel"] = text_gen_model

    pipeline_text_model = PipelineTextGenModel()

    pipeline_text_model.load_pipeline()

    ml_models["PipeLineTextModel"] = pipeline_text_model

    audio_gen_model = AudioGenModel()

    ml_models["PipeLineAudioModel"] = audio_gen_model.load_audio_pipeline()

    yield

    ml_models.clear()


app = FastAPI(lifespan=startup_lifespan)


@app.get("/")
def home():
    return "Hello World"



@app.post("/text_gen")
async def text_gen(
    request: Request, body: textRequestModel = Body(...)
) -> textResponseModel:

    start_time = time.time()

    generated_text = ml_models["TextGenModel"].generate_text(body.prompt)

    execution_time = time.time() - start_time

    print(f"Response: {generated_text}")
    print(f"Execution time: {execution_time:.2f} seconds")

    response = textResponseModel(
        execution_time=float(execution_time), result=generated_text
    )

    return response


@app.post("/pipe_text")
async def pipe_text_prediction(
    request: Request, body: textRequestModel = Body(...)
) -> textResponseModel:

    start_time = time.time()

    if ml_models.get("PipeLineTextModel") is None:

        print(ml_models)

        print("Model is not loaded")

        return textResponseModel(
            execution_time=int(time.time() - start_time), result="Model is not loaded"
        )

    generated_text = ml_models["PipeLineTextModel"].non_temp_predict(
        user_message=body.prompt
    )

    print(ml_models)

    return textResponseModel(
        execution_time=int(time.time() - start_time), result=generated_text
    )


@app.get(
    "/text_to_speech",
    responses={status.HTTP_200_OK: {"content": {"audio/wav": {}}}},
    response_class=StreamingResponse,
)
async def audio_generation(prompt=Query(...)) -> StreamingResponse:

    if ml_models.get("PipeLineAudioModel") is None:

        print("Audio Model is not loaded")

        return StreamingResponse(
            Response(content="audio model not loaded", media_type="text/plain")
        )

    output_audio = ml_models["PipeLineAudioModel"].generate_audio(prompt)

    if output_audio is None:

        print("Audio generation failed")

        return StreamingResponse(
            Response(content="audio model not loaded", media_type="text/plain")
        )

    else:

        print(output_audio)

    return StreamingResponse(
        output_audio,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=generated_audio.wav"},
    )


if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
