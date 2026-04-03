import dspy 
from pydantic import BaseModel
import base64

 
class ObjectDetectionModel(BaseModel):
    ymin: int 
    xmin: int 
    ymax: int 
    xmax: int 
    label: str
    description: str
    estimated_width_cm: float 
    estimated_height_cm: float




class ObjectDetectionSinature(dspy.Signature):
    '''
    1. Identify every object relevant to the task instruction in the attached overhead camera image. 
    2. Return their bounding boxes as a JSON array with labels. Never return masks or code fencing.
    If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
    
    For each object, provide:
    1. **box_2d**: Bounding box as `[ymin, xmin, ymax, xmax]` normalized to 0–1000 (integers only).
    2. **label**: A short, unique label (e.g., "red cup", "blue plate").
    3. **description**: Brief visual description (color, shape, approximate size).
    4. **estimated_size_cm**: Rough width × height in centimeters.
    '''
    image: dspy.Image = dspy.Input(description="설명") 
    instruction: str = dspy.Input(description="설명")
    detections: list[ObjectDetectionModel] = dspy.Output(description="설명")


class ObjectDetectionModule(dspy.Module):
   def __init__(self):
       super().__init__() 
       self.model = dspy.Predict(ObjectDetectionSinature) 

    def forward(self, instruction, image_file) -> list[ObjectDetectionModel]:
        with open("image.png", "rb") as image_file:
            base64_data = base64.b64encode(
                image_file.read()
            ).decode("utf-8")
        image_data_uri = f"data:image/png;base64,{base64_data}"
        return self.model(instruction=instruction, image=image_data_uri).detections
   
 