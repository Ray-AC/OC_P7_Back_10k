from imports import *

with open("./data/interpratibilite_globale.png", "rb") as file:
    image_content = file.read()

async def tab_interpratibilite_globale():
    image_base64 = base64.b64encode(image_content).decode()
    return image_base64
