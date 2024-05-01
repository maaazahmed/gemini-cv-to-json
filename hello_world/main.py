from fastapi import FastAPI, UploadFile, File
import vertexai
from vertexai.generative_models import GenerativeModel
import os
print(os.getcwd())
from io import BytesIO
import pdfminer.high_level
import pdfminer.layout
import json
from fastapi.staticfiles import StaticFiles
from vertexai.preview.language_models import TextGenerationModel
from starlette.responses import FileResponse 

from hello_world.prompt import JSON_DATA

app:FastAPI = FastAPI()


# Initialize Vertex AI
project_id = "langchain-420611"
vertexai.init(project=project_id, location="us-central1")
model = GenerativeModel("gemini-1.0-pro-001")



# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('./index.html')



def text_to_JSON(extracted_text:str):
    # """Ideation example with a Large Language Model"""

    # TODO developer - override these parameters as needed:
    parameters = {
        "temperature": .0,
        "max_output_tokens": 1024,
        "top_p": .8,
        "top_k": 40,

    }


    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
            f"""
                promt Format this resume in JSON format using the JSON Resume schema (htt ps://raw .githubusercontent.com/infineco/resume-s chema/master/schema.json). Fill in the "environment" section of each experiment with the relevant technologies (languages, frameworks, tools or methodologies), in the form of a single-level list. If an experiment end date is not available, remove the "endDate" field. if a date does not contain the day, put the first day of the month, if a date does not contain the month, put the first day of the year, must create basic propery object, do not write word "json```" at start of the json object
                
                DATA:
                {extracted_text}
                """,
        
        **parameters,
    )
    return response.text



def remove_json_markers(text):
  """
  This function removes the leading and trailing ```json characters from a string.

  Args:
      text: The string to be processed.

  Returns:
      The string without the ```json markers.
  """
  if text.startswith("```json") and text.endswith("```"):
    return text[7:-3]  # Slice the string to remove the first 6 and last 3 characters
  else:
    return text

# Example usage



@app.post("/convert-pdf-to-json")
async def convert_pdf_to_text(pdf_file: UploadFile = File(...)):
    """
    Converts uploaded PDF file to text and returns the extracted text as a string.
    Args:
        pdf_file (UploadFile): The uploaded PDF file.

    Returns:
        str: The extracted text from the PDF file.
    """
    
    try:
        contents = await pdf_file.read()  # Read the uploaded file content

        # Use pdfminer.high_level for efficient text extraction
        extracted_text = pdfminer.high_level.extract_text(BytesIO(contents))
        cv_json = text_to_JSON(extracted_text)
        cleaned_data = remove_json_markers(cv_json)
        return json.loads(cleaned_data)

    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        return {"error": "Failed to convert PDF to text. Please check the file format or try again later."}







