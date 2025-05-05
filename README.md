A webapp for searching images using text prompts made using FAISS and CLIP's image and text encoders to perform image search with text prompts<br><br>
You can use the FinetuneClip.ipynb notebook to finetune your own CLIP model from hugginface. You'll need image-caption paris where captions are stored as individual txt files per image. It'll generate a CLIP model folder for use with the webapp.<br><br>
To use the containerized webapp:<br><br>
-`git clone https://github.com/SmallChungus1/image-search-vlm.git` into specified folder<br><br>
-Copy your CLIP model folder from huggingface under 'models' folder: https://huggingface.co/docs/transformers/model_doc/clip<br><br>
-Create an image using <br>
`docker build -t {image name}:{tag name} .`<br><br>
-Mount the image folder you want to search with at container initalization using <br>
`docker run --rm -p 8000:8000 -v {path to your image folder}:/app/static/images {image name}:{tag name}`

