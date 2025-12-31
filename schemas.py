from typing import Literal

from pydantic import BaseModel, Field


class StudentRequestModel(BaseModel):
    
    """To add Student info"""

    class_name: Literal["class1", "class2", "class3"]

    stud_name: str = "Lin Lin"
    stud_id: int = Field(
        ..., gt=0, lt=101, description="Student ID must be between 1 and 100"
    )
    stud_age: int = Field(
        ..., gt=16, lt=30, description="Student age must be between 17 and 29"
    )


class StudentResponseModel(BaseModel):
    
    """To get and show student info"""

    execution_time: float = 0.0
    message: str = ""


class textRequestModel(BaseModel):
    
    """to get prompt for text generation"""
    
    prompt: str = "What is deep learning?"


class textResponseModel(BaseModel):
    
    """to return generated text"""

    response: str = ""
