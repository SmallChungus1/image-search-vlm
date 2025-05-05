A webapp for searching images using text prompts made using FAISS and CLIP's image and text encoders to perform image search with text prompts<br><br>
To use the webapp:<br>
-Copy your CLIP model folder from huggingface under 'models' folder: https://huggingface.co/docs/transformers/model_doc/clip<br><br>
-Create an image using <br>
`docker build -t {image name}:{tag name} .`<br><br>
-Mount the image folder you want to search with at container initalization using <br>
`docker run --rm -p 8000:8000 -v {path to your image folder}:/app/static/images {image name}:{tag name}`

