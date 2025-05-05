A webapp for searching images using text prompts made using FAISS and CLIP's image and text encoders to perform image search with text prompts

To use the webapp:
-Copy your CLIP model folder from huggingface under 'models' folder

-Create a container using 
`docker build -t {image name}:{tag name} .`

-Mount the image folder you want to search in during container initalization using 
`docker run --rm -p8000:8000 -v {path to your image folder}:/app/static/images {image name}:{tag name}`

