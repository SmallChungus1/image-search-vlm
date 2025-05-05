import torch
from transformers import pipeline, CLIPProcessor, CLIPModel
import faiss
from PIL import Image
import os
from flask import Flask, render_template, request, send_file, url_for, send_from_directory

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #need this to get past omp error #15

#class for handling entire image search process
class clip_image_search():
  def __init__(self):
    #clip model init, specify model from huggingface or change path to your own models
    self.clip_model = CLIPModel.from_pretrained("models/clip-finetune-results").to(device) #using my finetuned weight on car make model data
    self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    self.clip_model.eval()

    #faiss for indexing img embeddings
    self.faiss_index = None

  def init_and_index_faiss(self, embedding_dimension, embeddings, img_paths, search_metric='l2'):
    #L2 or cosine similarity search metric
    if search_metric == 'l2':
      self.faiss_index = faiss.IndexFlatL2(embedding_dimension)
    elif search_metric == 'ip':
      self.faiss_index = faiss.IndexFlatIP(embedding_dimension)

    #attach ids to faiss index
    img_ids = [i for i in range(len(img_paths))]
    #wrap with IndexIDMap to associate vectors with ids:
    #https://medium.com/data-science/building-an-image-similarity-search-engine-with-faiss-and-clip-2211126d08fa
    self.faiss_index_wIds = faiss.IndexIDMap(self.faiss_index)
    #add vectors to index
    self.faiss_index_wIds.add_with_ids(embeddings, img_ids)

    return self.faiss_index_wIds

  def encode_imgs(self, img_folder):
    VALID_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}

    embedded_imgs_list = []
    img_paths = [os.path.join(img_folder, fname) for fname in os.listdir(img_folder) if os.path.splitext(fname)[1].lower() in VALID_EXTS]
    imgs = [Image.open(img_path).convert('RGB') for img_path in img_paths]

    with torch.no_grad():
      processed_imgs = self.clip_processor(text=None, images=imgs, return_tensors="pt", padding=True, use_fast=True)
      img_embeddings = self.clip_model.get_image_features(**processed_imgs.to(device))

      #faiss needs embeedings in f32
      img_embeddings = img_embeddings.cpu().numpy().astype('float32')
      embedding_dimenstion = img_embeddings.shape[1]

      #init faiss index
      self.init_and_index_faiss(embedding_dimenstion, img_embeddings, img_paths, search_metric="l2")

      for an_img_path, an_img_embedding in zip(img_paths, img_embeddings):
        embedded_imgs_list.append([an_img_path, an_img_embedding])
    return embedded_imgs_list

  def encode_text_prompt(self, text):
    with torch.no_grad():
      text_inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
      text_embeddings = self.clip_model.get_text_features(**text_inputs.to(device))

    return text_embeddings.cpu().numpy().astype('float32')

  def search_imgs(self, text_query, result_limit=5):
    text_query_embedding = self.encode_text_prompt(text_query)
    distance, indices = self.faiss_index_wIds.search(text_query_embedding, result_limit)
    return distance, indices

  def retrieve_images_as_pil(self, img_folder, indices):
    img_pil_objs = []
    img_paths = [os.path.join(img_folder, img_path) for img_path in os.listdir(img_folder)]
    for index in indices[0]:
      img = Image.open(img_paths[index])
      img_pil_objs.append(img)
    return img_pil_objs
  
  def retrieve_images_as_uris(self, img_folder, indices):
    img_paths = [os.path.join(img_folder, img_path) for img_path in os.listdir(img_folder)]
    selected_img_paths = [img_paths[i] for i in indices[0]]
    return selected_img_paths

###Flask section###
app = Flask(__name__)

#init image serach obj
clip_img_search_obj = clip_image_search()

#Specify image folder for running image search on (would be specified by user before app start up)
# IMAGE_FOLDER = os.environ.get(
#     'IMAGE_FOLDER',
#     os.path.join(app.static_folder, 'images')
# )

IMAGE_FOLDER="testImages"

#don't need to do anything until user posts a search prompt
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

#main routing for image search pipeline
@app.route('/searchImage', methods=['POST'])
def search_images():
    search_prompt = request.form['searchPromptInput']
    #result_limit = request.form['resultLimitInput']
    global IMAGE_FOLDER

    #encode imgs first
    clip_img_search_obj.encode_imgs(IMAGE_FOLDER)

    #then encode prompt and search imgs
    _, queried_indicies = clip_img_search_obj.search_imgs(search_prompt, result_limit=3)

    matched_img_paths = clip_img_search_obj.retrieve_images_as_uris(IMAGE_FOLDER, queried_indicies)
    matched_img_paths_basenames = [os.path.basename(img_path) for img_path in matched_img_paths]
    print(matched_img_paths_basenames)

    return render_template('index.html', images=matched_img_paths_basenames)

@app.route('/preview_img_folder', methods=['GET'])
def get_all_img_paths():
    VALID_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
    img_paths = [os.path.join(IMAGE_FOLDER, fname) for fname in os.listdir(IMAGE_FOLDER) if os.path.splitext(fname)[1].lower() in VALID_EXTS]
    
    return render_template('previewFolder.html', images=[os.path.basename(img_path) for img_path in img_paths])

#need this for serving images to frontend, if not using static folder for storing images
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)