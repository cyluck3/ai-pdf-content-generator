import os
import asyncio
import json

# Mis librerías
from utils.agent import flowtask

# Módulos externos
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# PARA GENERAR PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY  # Import TA_JUSTIFY for text justification

# --- Configuration ---c
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

MODEL_NAME_MAIN = "gemini-2.0-flash"
MODEL_NAME_AGENT2 = "gemma-3-27b-it" # Las API'S de OpenRouter tienen límite diario, si hay errores por ello, probar con alguna otra diferente de OpenRouter, como gemma-3-27b-it o algun otro modelo. (Aunque deepseek-r1 de open router es de lo mejor)
# CUANDO SE RECIBEN RESULTADOS CON ERRORES, ES POSIBLE QUE LA IA NO ESTË RESPONDIENDO COMO SE DEBE O QUE NO ESTÉ RESPONDIENDO POR LÍMITE DE USO. SE RECOMIENDA REVISAR PARA DESCARTAR.

# --- Model Instantiation ---
llm = ChatGoogleGenerativeAI(model=MODEL_NAME_MAIN, google_api_key=GOOGLE_API_KEY)

# --- Prompt ---
SYSTEM_PROMPT = """
You are a master prompt engineer. Your task is to analyze user requests and create a well-structured JSON dictionary of prompts for a series of writing tasks. 
The output should be a JSON dictionary with keys ("title for the prompt 1", "title for the prompt 2", "title for the prompt 3", etc.) and corresponding prompts as values. The last key-value will be a title for the subject covered for all the prompts.
The prompts should be ordered sequentially. Do not include any markdown or additional text outside of the JSON dictionary, you will ONLY response with the JSON dictionary, in Spanish.
If the user's request is not specific, ask them what they want.

Example of the JSON dictionary:
{{
    "subtitle 1": "Write a detailed introduction about...",
    "subtitle 2": "Expand on the main points...",
    "subttile 3": "...",
    ...
    "subtitle n": "Conclude the text by",
    "title": "general title for the subject"
}}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | llm

# --- Función para generar PDF ---
def create_pdf(filename, title, content_data):
    """Creates a PDF document with the given title and content."""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()

    # Definir estilos personalizado
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.black,
        spaceAfter=0.2 * inch,
        alignment=1,  # Center
    )

    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.black,
        spaceAfter=0.1 * inch,
    )

    normal_style = styles['Normal']
    normal_style.fontSize = 12
    normal_style.spaceAfter = 0.1 * inch
    normal_style.alignment = TA_JUSTIFY  # Justificar texto

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.5 * inch))

    for subtitle, text in content_data.items():
        story.append(Paragraph(subtitle, subtitle_style))
        story.append(Paragraph(text, normal_style))
        story.append(Spacer(1, 0.2 * inch))

    doc.build(story)


# --- Función principal ---
async def runai():
    print("Welcome to the Google AI Prompt Generator!")

    agent2 = flowtask("prompt-processor", MODEL_NAME_AGENT2)

    chat_history = []
    interaction_count = 0
    max_interactions = 25

    while True:
        user_input = input("You: ")

        if user_input.lower() == "salir":
            print("Goodbye!")
            break

        if user_input.lower() == "resetear":
            chat_history = []
            interaction_count = 0
            print("Memory reset.")
            continue

        if interaction_count >= max_interactions:
            chat_history = []
            interaction_count = 0
            print("Memory reset due to maximum interactions reached.")

        human_message = HumanMessage(content=user_input)
        chat_history.append(human_message)

        try:
            response = chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response.content))
            interaction_count += 1

            try:
                fresponse = response.content.replace("`", "").replace("json", "").replace("```", "")
                response_json = json.loads(fresponse)
                print("Plan designed for you:", fresponse, "\n\nExecuting the plan...")

                previous_prompt = ""
                title = list(response_json.items())[-1]
                title_value = title[1]
                title_value = title_value.replace(":", ",").replace("*", "").replace('"', "") #eliminar posibles caracteres especiales prohibidos para nombres de archivo que impidan un guardado exitoso

                last_key = list(response_json.keys())[-1]
                del response_json[last_key]
                print(response_json)

                print(title_value, "TITULO")

                # --- Preparar contenido para el PDF ---
                pdf_content_data = {}

                for i, element in enumerate(response_json.values()):
                    print(f"Procesando Prompt {i+1}: {element}...")
                    agent2_instruction = (
                        f"{element}\n"
                        f"You are a professional writer. Write high-quality paragraphs, avoid lists, and do not use markdown. "
                        f"Consider the previous prompt that was already covered, don't repeat things covered in this prompt: '{previous_prompt}', this is only given to maintain context."
                    )

                    agent2_response = await agent2.add_instruction(agent2_instruction + "\nEscribe en español.")

                    if agent2_response is not None:
                        subtitle = list(response_json.keys())[i]
                        pdf_content_data[subtitle] = agent2_response

                        print(agent2_response + "\n\n")
                        previous_prompt = element

                    else:
                        print(f"Error: agent2 did not return a response for prompt {i+1}. Skipping this prompt.")

                # Guardar en Documentos del sistema operativo
                # Obtener la ruta de la carpeta "Documentos"
                documentos_path = os.path.join(os.path.expanduser("~"), "Documents\\Docsaicg\\pdf")

                # Asegurarse de que la carpeta existe (por si acaso)
                if not os.path.exists(documentos_path):
                    os.makedirs(documentos_path)  # Crear la carpeta si no existe

                print(documentos_path, "DOCUMENTOS PATH")

                # Se crean las carpetas docs/pdf si no existen, directorios personalizados en el mismo proyecto
                # os.makedirs("./docs/pdf/", exist_ok=True) 

                # --- Generar archivo PDF ---
                pdf_filename = documentos_path + "\\" + title_value + ".pdf"
                print("Generating PDF file: ", pdf_filename, "...")
                create_pdf(pdf_filename, title_value, pdf_content_data)

                # Renombrar el archivo para evitar que se muestre como "(anonymous)" en el header tab del navegador al abrirlo
                # Construir la nueva ruta del archivo
                new_pdf_filename = os.path.join(documentos_path, title_value + ".pdf")
                os.rename(pdf_filename, new_pdf_filename) #rename the file once it has been saved to ensure the file has the right name
               
                print(f"Tu documento sobre {title_value}, ha sido creado y guardado exitosamente.")

            except json.JSONDecodeError as e:
                print(response.content)

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")

async def runaiload():
    directorio = './inputs/'
    archivos = os.listdir(directorio)

    archivosleidos = [] # Esto almacena el contenido de cada archivo leído, necesito por tanto que este modo sea capaz de generar un pdf para cada archivo de texto

    for archivo in archivos:
        if archivo.endswith('.txt'):
            ruta_archivo = os.path.join(directorio, archivo)
            with open(ruta_archivo, 'r') as file:
                contenido = file.read()
                # print(f'Contenido del archivo {archivo}:')
                archivosleidos.append(contenido)
                # print(contenido)
                # print('------------------------')


    for texto in archivosleidos:
            
        SYSTEM_PROMPTCU = """
    You are a master prompt engineer. Your task is to analyze a given text and create a well-structured JSON dictionary of 10 prompts for a series of writing tasks, as a researcher and writer. 
    The output should be a JSON dictionary with keys ("title for the prompt 1", "title for the prompt 2", "title for the prompt 3", etc.) and corresponding prompts as values. The last key-value will be a title for the subject covered for all the prompts.
    The prompts should be ordered sequentially. Do not include any markdown or additional text outside of the JSON dictionary, you will ONLY response with the JSON dictionary, in Spanish.

    Example of the JSON dictionary:
    {{
        "subtitle 1": "Write a detailed introduction about...",
        "subtitle 2": "Expand on the main points...",
        "subttile 3": "...",
        ...
        "subtitle n": "Conclude the text by",
        "title": "general title for the subject"
    }}

        The text to process its the next one: 
    """
        try:
            agente = flowtask("procesador-carga-texto", MODEL_NAME_MAIN)  
            agente2 = flowtask("prompt-processor", MODEL_NAME_AGENT2)
            

            # --- Registrar el penúltimo prompt procesado para mantener contexto de lo que ya se explicó antes
            previous_prompt = ""
            
            # --- Preparar contenido para el PDF ---
            pdf_content_data = {}

            resa = await agente.add_instruction(SYSTEM_PROMPTCU + texto) # prompt for customizing info process mode and generating list of prompts

            fresponse = resa.replace("`", "").replace("json", "").replace("```", "")
            response_json = json.loads(fresponse)
            print("Plan designed for you:", fresponse, "\n\nExecuting the plan...")

            title = list(response_json.items())[-1]
            title_value = title[1]
            title_value = title_value.replace(":", ",").replace("*", "").replace('"', "") #eliminar posibles caracteres especiales prohibidos para nombres de archivo que impidan un guardado exitoso

            last_key = list(response_json.keys())[-1]
            del response_json[last_key]
            print(response_json)

            print(title_value, "TITULO")

            for i, element in enumerate(response_json.values()):
                print(f"Procesando Prompt {i+1}: {element}...")
                agent2_instruction = (
                    f"{element}\n"
                    f"You are a professional writer. Write high-quality paragraphs, avoid lists, and do not use markdown. "
                    f"Consider the previous prompt that was already covered, don't repeat things covered in this prompt: '{previous_prompt}', this is only given to maintain context."
                )

                agent2_response = await agente2.add_instruction(agent2_instruction + "\nEscribe en español.")

                if agent2_response is not None:
                    subtitle = list(response_json.keys())[i]
                    pdf_content_data[subtitle] = agent2_response

                    print(agent2_response + "\n\n")
                    previous_prompt = element

                else:
                    print(f"Error: agent2 did not return a response for prompt {i+1}. Skipping this prompt.")

            # Guardar en Documentos del sistema operativo
            # Obtener la ruta de la carpeta "Documentos"
            documentos_path = os.path.join(os.path.expanduser("~"), "Documents\\Docsaicg\\pdf")

            # Asegurarse de que la carpeta existe (por si acaso)
            if not os.path.exists(documentos_path):
                os.makedirs(documentos_path)  # Crear la carpeta si no existe

            print(documentos_path, "DOCUMENTOS PATH")

            # Se crean las carpetas docs/pdf si no existen, directorios personalizados en el mismo proyecto
            # os.makedirs("./docs/pdf/", exist_ok=True) 

            # --- Generar archivo PDF ---
            pdf_filename = documentos_path + "\\" + title_value + ".pdf"
            print("Generating PDF file: ", pdf_filename, "...")
            create_pdf(pdf_filename, title_value, pdf_content_data)

            # Renombrar el archivo para evitar que se muestre como "(anonymous)" en el header tab del navegador al abrirlo
            # Construir la nueva ruta del archivo
            new_pdf_filename = os.path.join(documentos_path, title_value + ".pdf")
            os.rename(pdf_filename, new_pdf_filename) #rename the file once it has been saved to ensure the file has the right name
            
            print(f"Tu documento sobre {title_value}, ha sido creado y guardado exitosamente.")
            
        except Exception as e:
            print("Ocurrió une error: ", e)



mode = False # True para el modo de asistente interactivo, False para el modo carga de archivo (tomará los archivos del directorio inputs)

if __name__ == "__main__":
    if (mode):
        asyncio.run(runai()) # Asistente de IA interactivo
    else:
        asyncio.run(runaiload()) # Procesamiento de archivos cargados