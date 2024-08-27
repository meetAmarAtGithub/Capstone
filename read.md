1. Install requirement files - pip install -r requirements.txt

2. check installed framework versions - 
i) python -m pip show ultralytics
ii) python -m pip show fastapi
iii) python -m pip show uvicorn
iv) python -m pip show fastapi-cors
v) pip list | grep -E 'ultralytics|fastapi|uvicorn'

3. Run the Flask application using the following command: python app.py

4. You can test the API using a tool like Postman or curl. Here is an example using curl:
curl -X POST "http://0.0.0.0:8000/predict/" -F "file=@/home/amar/Desktop/Reva/CarDamageDetection/0200.JPEG"

5.Use ngrok for HTTPS Tunneling
Ngrok can expose your local server to the internet via a secure https tunnel.
Download and install ngrok from ngrok.com.
Run ngrok http 8000 to start a tunnel to your local server.
Ngrok will provide you with a public https URL that tunnels to your local server.
i) install ngrok - amar@amar-Inspiron-N4050:~/Desktop$ sudo snap install ngrok
ii) sign up at ngrok and get authtoken.
iii) Authenticate your ngrok agent. You only have to do this once. The Authtoken is saved in the default configuration file. - 
amar@amar-Inspiron-N4050:~/Desktop$ ngrok config add-authtoken 2gsNsYJlDHxDEUPbSB23DwjGr5T_4wTJFZraE6fYT9tpyag8Z
Authtoken saved to configuration file: /home/amar/snap/ngrok/148/.config/ngrok/ngrok.yml



Note - 
1. Name: ultralytics
Version: 8.2.17
Summary: Ultralytics YOLOv8 for SOTA object detection, multi-object tracking, instance segmentation, pose estimation and image classification.
Home-page: 
Author: Glenn Jocher, Ayush Chaurasia, Jing Qiu
Author-email: 
License: AGPL-3.0
Location: /home/amar/.miniconda3/lib/python3.12/site-packages
Requires: matplotlib, opencv-python, pandas, pillow, psutil, py-cpuinfo, pyyaml, requests, scipy, seaborn, thop, torch, torchvision, tqdm
Required-by: 

2. Name: fastapi
Version: 0.111.0
Summary: FastAPI framework, high performance, easy to learn, fast to code, ready for production
Home-page: 
Author: 
Author-email: =?utf-8?q?Sebasti=C3=A1n_Ram=C3=ADrez?= <tiangolo@gmail.com>
License: 
Location: /home/amar/.miniconda3/lib/python3.12/site-packages
Requires: email_validator, fastapi-cli, httpx, jinja2, orjson, pydantic, python-multipart, starlette, typing-extensions, ujson, uvicorn
Required-by: fastapi-cli

3. Name: uvicorn
Version: 0.29.0
Summary: The lightning-fast ASGI server.
Home-page: 
Author: 
Author-email: Tom Christie <tom@tomchristie.com>
License: 
Location: /home/amar/.miniconda3/lib/python3.12/site-packages
Requires: click, h11
Required-by: fastapi, fastapi-cli

4. Name: fastapi_cors
Version: 0.0.6
Summary: Simple env support of CORS settings for Fastapi applications
Home-page: 
Author: 
Author-email: Ian Cleary <github@iancleary.me>
License: MIT
Location: /home/amar/.miniconda3/lib/python3.12/site-packages
Requires: environs, fastapi
Required-by: 