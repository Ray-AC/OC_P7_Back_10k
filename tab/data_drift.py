from imports import *

with open("./data/data_drift.png", "rb") as file:
    image_content = file.read()
    
async def tab_data_drift():
    image_base64 = base64.b64encode(image_content).decode()
    return image_base64