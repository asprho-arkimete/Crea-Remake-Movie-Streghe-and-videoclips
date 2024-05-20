import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
from PIL import Image, ImageTk
from tkinter import PhotoImage, ttk
from tkinter import messagebox
from PIL import Image, ImageTk,ImageSequence
from PIL import Image, ImageDraw, ImageTk
import cv2
from tkinter import filedialog
from googletrans import Translator
from diffusers import StableDiffusionPipeline
from compel import Compel
import torch
import os
import time
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import shutil
from tqdm import tqdm
import subprocess
import random as ra
from moviepy.editor import VideoFileClip, concatenate_videoclips
from elevenlabs import Voice, voices, VoiceSettings, generate, play, set_api_key, save
import json
import pygame
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageTk
from pydub import AudioSegment
from pytube import YouTube
from youtube_search import YoutubeSearch
import random
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageTk
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import concatenate_audioclips
from moviepy.editor import CompositeAudioClip
from moviepy.audio.fx.all import audio_fadeout
import zipfile

 

Loading=False

# Inizializza il modulo mixer
pygame.mixer.init()

with open("ele.json") as f:
    secrets = json.load(f)

api_key_ele = secrets["api_key_eleventlabs"]
set_api_key(api_key_ele)
voiceelevellabs = [v.name for v in voices()]


 

# Ottieni la directory corrente
d = os.getcwd()

# Crea il percorso per la nuova directory "sounds"
d = os.path.join(d, "sounds")

# Crea il percorso per la sottodirectory "private"
private = os.path.join(d, "private")

# Crea la directory "sounds"
os.makedirs(d, exist_ok=True)  # Imposta exist_ok=True

# Crea la sottodirectory "private"
os.makedirs(private, exist_ok=True)  # Imposta exist_ok=Tru

     













w= 512
h=720
translator = Translator()

photosclips= []
listvary= []
listlong=[]
# Crea delle liste vuote per memorizzare i riferimenti agli oggetti
list_grafici = []
list_combomodelaudios = []
list_combovoices = []
list_dir=[]
list_sound=[]
list_seed= []
# Crea delle liste vuote per memorizzare i riferimenti agli oggetti
list_positives = []
list_negatives = []
 



def crea_elementi(root, frame, canvas, scrollbar, scrollable_frame, photos, text):
    
    global check_var,checkboximagevideo,photosclips,listvary,listlong,longclip,steps,cfg,list_seed,button,voiceelevellabs,list_dir,list_sound,list_combovoices,list_positives,list_negatives,list_combomodelaudios,Loading,pathfacebox,combostreghe,comboswap,swapface_var
    
    for widget in scrollable_frame.winfo_children():
        widget.destroy()
    
   

   
    def generaimmagine(desc_englist,k,negativo,c,s):
        global w,h,combobox,comboboxresolution,list_seed,check_seed,checkboxseed,pathfacebox,combostreghe,swapface_var
        if check_seed.get()==False:
            # Genera un seed casuale
            seed = random.randint(0, 2**32 - 1)
            # Se list_seed[k] esiste già, sostituiscilo con il nuovo seed
            if k < len(list_seed):
                list_seed[k] = seed
            # Altrimenti, aggiungi il nuovo seed alla lista
            else:
                list_seed.append(seed)
        else:
            # Utilizza l'ultimo seed generato
            seed = list_seed[k]
        print(f"seed image: {k} seed: {seed}")
        
        #'Realist V6','Rocco_1','Rocco_2'
        model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE" 
        P_id= "stablediffusionapi/uber-realistic-merge"
        P_id2="ductridev/uber-realistic-porn-merge-urpm"
        device = "cuda"
        token = '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'
        if combobox.get()== 'Realist V6':
                model_id= "SG161222/Realistic_Vision_V6.0_B1_noVAE" 
        elif combobox.get()=='Rocco_1':
            model_id= P_id
        elif combobox.get()=='Rocco_2':
            model_id= P_id2 
        else:
            model_id= model_id
            
        try:
            if comboboxresolution.get() is None:
                w= 512
                h=720
            else:
                w,h = map(int, comboboxresolution.get().split(','))
        except ValueError as error:
            w= 512
            h=720
        finally:
            pass
        
        try:
            cfg = c.get()
        except (ValueError, AttributeError):
            print("errore")
        finally:
            print("OK cfg")
             

        try:
            steps = s.get()
        except (ValueError, AttributeError):
            print ("errore steps")
        finally:
            print ("ok steps")

        #['Phoebe','Piper','Prue','Pagie','Billie']
            
        pipe= StableDiffusionPipeline.from_pretrained(model_id,use_auth_token=token,torch_dtype=torch.float16).to(device)
       
        pipe.enable_xformers_memory_efficient_attention()  
        pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
        
        pathface="nessunviso"
        prompt = desc_englist  
        negative= negativo 
        
        if combostreghe.get()== 'Phoebe':
            print("Phoebe")
            pipe.load_lora_weights(".\\Lora\\4lyss4m.safetensors", weight_name="4lyss4m.safetensors", adapter_name="4lyss4m")
            prompt= prompt+"<lora:4lyss4m:1.0>,"
            pathface= ".\\faces_streghe\\phoebe"
        elif combostreghe.get()=='Piper':
            print("Piper")
            pipe.load_lora_weights(".\\Lora\\Holly_Marie_Combs_PMv1_Lora.safetensors", weight_name="Holly_Marie_Combs_PMv1_Lora.safetensors", adapter_name="Holly_Marie_Combs_PMv1_Lora")
            prompt=prompt+"<lora:Holly_Marie_Combs_PMv1_Lora:1.3>,"
            pathface= ".\\faces_streghe\\piper"
        elif combostreghe.get()== 'Prue':
            print("Prue")
            pipe.load_lora_weights(".\\Lora\\PrueHalliwell.safetensors", weight_name="PrueHalliwell.safetensors", adapter_name="PrueHalliwell")
            prompt= prompt+"<lora:PrueHalliwell:1.0>,"
            pathface= ".\\faces_streghe\\prue"
        elif combostreghe.get()== 'Pagie':
            print("Pagie")
            pipe.load_lora_weights(".\\Lora\\PaigeMatthews.safetensors", weight_name="PaigeMatthews.safetensors", adapter_name="PaigeMatthews")
            prompt=prompt+ "<lora:PaigeMatthews:1.0>,"
            pathface= ".\\faces_streghe\\paige"
        elif combostreghe.get()== 'Billie':
            print("Billie")
            pipe.load_lora_weights(".\\Lora\\k4l3yc.safetensors", weight_name="k4l3yc.safetensors", adapter_name="k4l3yc")
            prompt=prompt+"<lora:k4l3yc:1>,"
            pathface= ".\\faces_streghe\\billie"
        else:
           print("No PERSONAGGIO")
           
       
        #espandi token
        compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        conditioning = compel(prompt)
        # or: conditioning = compel.build_conditioning_tensor(prompt)
        negative_conditioning = compel(negative)
        [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
        print(f"CFG: {cfg}")    
        print(f"Steps: {steps}")
        # Crea un generatore con il seed
        generator = torch.Generator(device="cuda").manual_seed(seed)
        if isinstance(pipe, StableDiffusionPipeline):
            with torch.cuda.amp.autocast():
                image = pipe(prompt_embeds=conditioning,  generator=generator,width=w,height=h, guidance_scale=float(cfg),num_inference_steps=int(steps),negative_propmt_embeds=negative_conditioning).images[0]
                pathimage= f".\PIA\example\keyframe_{k}.png"
                if os.path.exists(pathimage):
                    os.remove(pathimage)
                image.save(pathimage)
                selezionavolto="nessuna foto"
                print(f"pathface: {pathfacebox}")
                if swapface_var.get() and pathfacebox != "nessuna foto":
                    selezionavolto = os.path.join(pathfacebox)
                    print(f" volto UPLOdato: {selezionavolto}")
                     # cambia viso con foto
                    generato = os.path.join(pathimage)
                    
                    if pathfacebox!= "nessuna foto": 
                        shutil.copyfile(selezionavolto,".\\swapseed\\volto.png")
                    shutil.copyfile(generato,".\\swapseed\\generato.png")
                    
                    os.chdir(".\\swapseed")
                    os.system("python main.py")
                    time.sleep(1)
                    os.chdir("..")
                    if os.path.exists(pathimage): 
                        os.remove(pathimage)
                    if os.path.exists(".\\swapseed\\generatedimagewithface.png"):
                        shutil.move(".\\swapseed\\generatedimagewithface.png", pathimage)
                elif combostreghe.get()!= "" and combostreghe.get()!= "Nessun Personaggio":
                    # seleziona volto
                    print(f"path face: {pathface}")
                    volti = [volto for volto in os.listdir(pathface) if volto.endswith(("jpg","png"))]    
                    selezionavolto= random.choice(volti)
                    # Get the full path of the selected face and the generated image
                    selezionavolto = os.path.join(pathface, selezionavolto)
                    print(f"foto selezionata: {selezionavolto}")
                     # cambia viso con foto
                    generato = os.path.join(pathimage)
                    
                    if pathfacebox!= "nessuna foto": 
                        shutil.copyfile(selezionavolto,".\\swapseed\\volto.png")
                    shutil.copyfile(generato,".\\swapseed\\generato.png")
                    
                    os.chdir(".\\swapseed")
                    os.system("python main.py")
                    time.sleep(1)
                    os.chdir("..")
                    if os.path.exists(pathimage): 
                        os.remove(pathimage)
                    if os.path.exists(".\\swapseed\\generatedimagewithface.png"):
                        shutil.move(".\\swapseed\\generatedimagewithface.png", pathimage)

               
            
    def generavideoclip(pathimage, k, promptpositivo, promptnegativo):
        global listvary, num_prompts
        os.chdir('Pia')
        nomeimage = os.path.basename(pathimage)
        nomeimage = os.path.splitext(nomeimage)[0]  # Remove the file extension

        # Check if the file exists and, if so, remove it
        if os.path.exists(".\example\config\hard_animation2.yaml"):
            os.remove(".\example\config\hard_animation2.yaml")

        # Modify prompts based on the value of listvary[k].get()
        
        if k < len(listvary):
            variazioni= listvary[k].get()
            if variazioni== "":
                variazioni=1
            num_prompts = int(variazioni)
        else:
            print("Index out of range")
            num_prompts=1
        print(f"numero elementi lista variazioni: {len(listvary)}")
        print(f"numero variazioni: {num_prompts }")
       
            
           

        if num_prompts== 1:
            # Create the file and write the code in it
            with open(".\example\config\hard_animation2.yaml", 'w') as file:
                # Base configuration
                file.write(f"""base: 'example/config/base.yaml'
prompts:
- - {promptpositivo}
n_prompt:
  - '{promptnegativo}'
validation_data:
  input_name: '{nomeimage}'
  validation_input_path: '.\example'
  save_path: 'example/result'
  mask_sim_range: [0]
generate:
  use_lora: false
  use_db: true
  global_seed: 5658137986800322011
  lora_path: ""
  db_path: "models/DreamBooth_LoRA/realisticVisionV51_v51VAE.safetensors"
  lora_alpha: 0.8
""")
        elif num_prompts== 2:
            # Create the file and write the code in it
            with open(".\example\config\hard_animation2.yaml", 'w') as file:
                # Base configuration
                file.write(f"""base: 'example/config/base.yaml'
prompts:
- - {promptpositivo}
  - {promptpositivo}
n_prompt:
  - '{promptnegativo}'
validation_data:
  input_name: '{nomeimage}'
  validation_input_path: '.\example'
  save_path: 'example/result'
  mask_sim_range: [0]
generate:
  use_lora: false
  use_db: true
  global_seed: 5658137986800322011
  lora_path: ""
  db_path: "models/DreamBooth_LoRA/realisticVisionV51_v51VAE.safetensors"
  lora_alpha: 0.8
""")
                
        elif num_prompts== 3:
                # Create the file and write the code in it
            with open(".\example\config\hard_animation2.yaml", 'w') as file:
                # Base configuration
                file.write(f"""base: 'example/config/base.yaml'
prompts:
- - {promptpositivo}
  - {promptpositivo}
  - {promptpositivo}
n_prompt:
  - '{promptnegativo}'
validation_data:
  input_name: '{nomeimage}'
  validation_input_path: '.\example'
  save_path: 'example/result'
  mask_sim_range: [0]
generate:
  use_lora: false
  use_db: true
  global_seed: 5658137986800322011
  lora_path: ""
  db_path: "models/DreamBooth_LoRA/realisticVisionV51_v51VAE.safetensors"
  lora_alpha: 0.8
""")
                
        os.system("python inference.py --config=example/config/hard_animation2.yaml --loop")
        os.chdir('..')
    def make_regenerate_function_audio(i, t, g, v):
            global waves
            waves= []
            def playsoundaudio():
                os.makedirs(".\\elevenlabs",exist_ok=True)
                if v.get()!= "nessuna voce":
                    v_id= [voce.voice_id for voce in voices() if voce.name== v.get()][0] 
                    print(f"PLAY AUDIO {i}, testo: {t.get('1.0','end')}, combovoice: {v.get()}, voce_id: {v_id}")
                    if not t.get('1.0','end')== "":
                        audio = generate(model="eleven_multilingual_v2",text=t.get('1.0','end'),voice=Voice(voice_id=v_id,settings=VoiceSettings(stability=0.50, similarity_boost=1.0, style=0.10, use_speaker_boost=True)))
                        if os.path.exists(f".\\elevenlabs\\voice_{i}.mp3"):
                            os.remove(f".\\elevenlabs\\voice_{i}.mp3")
                            time.sleep(1)
                        save(audio,filename= f".\\elevenlabs\\voice_{i}.mp3")
                        time.sleep(1)
                        

                    def waveForm(p):
                        print("create wave form")
                        audio = AudioSegment.from_mp3(p)
                        audio.export(p.replace('mp3','wav'), format='wav')
                        # Leggi il file audio
                        samplerate, data = read(p.replace('mp3','wav'))

                        # Crea la figura e l'asse
                        fig, ax = plt.subplots()

                        # Crea la forma d'onda
                        ax.plot(data)
                        ax.axis('off')

                        # Crea un canvas da figura
                        canvas = FigureCanvas(fig)
                        canvas.draw()

                      
                        # Converti il canvas in un'immagine PIL e poi in un'immagine Tkinter
                        pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
                        pil_image.save(f".\\wave\\w_{i}.png")  # Salva l'immagine PIL qui
                        tk_image = ImageTk.PhotoImage(pil_image)
                        
                        if os.path.exists(p.replace('mp3','wav')):
                            os.remove(p.replace('mp3','wav'))

                        return tk_image
                            
                    waveForm(f".\\elevenlabs\\voice_{i}.mp3")
                    # Carica l'immagine con PIL
                    waveform_image = Image.open(f".\\wave\\w_{i}.png")
                    
                   # Ridimensiona l'immagine
                    waveform_image.thumbnail((g[i].winfo_width(),g[i].winfo_height()), Image.LANCZOS)
                                        
                    # Converte l'immagine PIL in un'immagine Tkinter
                    waveform_image_tk = ImageTk.PhotoImage(waveform_image)
                    
                    # Aggiungi l'immagine alla lista
                    waves.append(waveform_image_tk)
                    
                    # Crea l'immagine sul canvas
                    g[i].create_image(0, 0, image=waves[-1], anchor='nw')
            
                    
                    if os.path.exists(f".\\elevenlabs\\voice_{i}.mp3"):
                        # Carica il file audio
                        pygame.mixer.Sound(f".\\elevenlabs\\voice_{i}.mp3").play()
            return playsoundaudio
        
            
    def make_regenerate_function(i, entry, entryneg,cfg,steps):
        global photosclips,listvary, list_positives, list_negatives
        def regenerate_image():
            en= entry.get('1.0', 'end').strip()
            if en != "" and len(en) > 2:      
                try:
                    english = translator.translate(en, src='it', dest='en').text
                except Exception as e:
                    print(f"Errore durante la traduzione dell'elemento {i}: {e}")
                finally:
                    pass
            
            englishneg = translator.translate(entryneg.get('1.0', 'end'), src='it', dest='en').text

            generaimmagine(english, i, englishneg,cfg,steps)  # Generate the image
            img = Image.open(f".\PIA\example\keyframe_{i}.png")  # Open the new image
            img.thumbnail((300, 300))  # Resize it
            photo = ImageTk.PhotoImage(img)  # Convert it to PhotoImage
            photos[i] = photo  # Update the corresponding photo in the list
            labels[i].config(image=photo)  # Update the corresponding label's image
            # Aggiorna i dati dell'entry nella lista
            list_positives[i] = entry 
            list_negatives[i] = entryneg
        

        def regenerate_video():
            global photosclips,listvary
            english = translator.translate(entry.get('1.0', 'end'), src='it', dest='en').text
            englishneg = translator.translate(entryneg.get('1.0', 'end'), src='it', dest='en').text

            generavideoclip(f".\PIA\example\keyframe_{i}.png", i, english, englishneg)
            
            #indice bottone cliccatp
            print(f"INDICE BOTTONE PREMUTO: {i}")
            pathvideo = ".\\PIA\\example\\result"
            dirs = [int(d) for d in os.listdir(pathvideo) if d.isdigit()]
            numeromaggiore = max(dirs)
            if numeromaggiore!= i:
                if os.path.exists(os.path.join(pathvideo, str(i))):
                    for f in os.listdir(os.path.join(pathvideo, str(i))):
                        if os.path.exists(os.path.join(pathvideo, str(i), f)):
                            os.remove(os.path.join(pathvideo, str(i), f)) 
                    os.rmdir(os.path.join(pathvideo, str(i)))
                os.rename(os.path.join(pathvideo, str(numeromaggiore)), os.path.join(pathvideo, str(i)))
            v=listvary[i].get()
            if v== "":
               v=1
            if os.path.exists(f'.\\PIA\\example\\result\\{i}\\{v}_sim_3.gif'):
                # mostra primo frame della gif nel riquadro  player_frame
                # Apri l'immagine con PIL e ridimensionala
                img = Image.open(f'.\\PIA\\example\\result\\{i}\\{v}_sim_3.gif')
                # Ottieni le dimensioni originali dell'immagine
                original_width, original_height = img.size

                # Calcola il rapporto tra le dimensioni originali
                ratio = original_width / original_height

                # Ottieni le dimensioni del canvas
                canvas_width = canvas_frames[i].winfo_width()
                canvas_height = canvas_frames[i].winfo_height()

                # Calcola le nuove dimensioni mantenendo lo stesso rapporto
                if canvas_width / canvas_height > ratio:
                    # Se il canvas è più largo rispetto all'immagine, adatta l'altezza dell'immagine al canvas
                    new_height = canvas_height
                    new_width = int(new_height * ratio)
                else:
                    # Altrimenti, adatta la larghezza dell'immagine al canvas
                    new_width = canvas_width
                    new_height = int(new_width / ratio)

                # Ridimensiona l'immagine
                new_img = img.resize((new_width, new_height), Image.LANCZOS)

                # Converti l'immagine ridimensionata in PhotoImage
                photo = ImageTk.PhotoImage(new_img)
                photosclips.append(photo)  # Memorizza l'immagine nella lista

             
                # Cambia lo sfondo del canvas
                canvas_frames[i].config(bg='pink')

                # Cancella tutto dal canvas
                canvas_frames[i].delete('all')

                # Aggiorna l'immagine sul canvas
                canvas_frames[i].create_image(0, 0, image=photosclips[-1], anchor='nw')
                # Aggiorna i dati dell'entry nella lista
                if list_positives[i].get('1.0', 'end').strip() != entry.get('1.0', 'end').strip():
                    list_positives[i] = entry
                if list_negatives[i].get('1.0', 'end').strip() != entryneg.get('1.0', 'end').strip():
                    list_negatives[i] = entryneg
                
       
            
        return regenerate_image, regenerate_video
    
    def make_video_functions(i):
        global photosclips,listvary
        def play():
            global listvary
            v=listvary[i].get()
            if v=='':
               v= int(1)
            if os.path.exists(f'.\\PIA\\example\\result\\{i}\\{v}_sim_3.gif'):   
                file_path = os.path.join(os.getcwd(),'PIA','example','result',str(i),f'{v}_sim_3.gif')
                # Open the file with its default application
                os.startfile(file_path)
        
        return play
    
   
        
    t=1
    # Add a trailing '|' if not present
    if text[-1] != '|':
        text = text + '|'
    # Create a list to hold the label objects
    labels = []
    # Create a list to hold the Canvas objects
    canvas_frames = []
    #elimina tutti i file e cartelle vecchie nella cartelle : \PIA\example\result
    # Percorso della cartella
    
    
    folder_path = os.path.join(os.getcwd(),'PIA','example','result')
    # Verifica se la cartella esiste
    if os.path.exists(folder_path):
        # Elenca tutti i file e le cartelle nel percorso
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Se è un file, rimuovilo
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            
            # Se è una cartella, rimuovi la cartella e tutto il suo contenuto
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    
    def make_function_combo(i):
        def update_combosound(event):
            # Ottieni il valore corrente di combodir
            current_dir = list_dir[i].get()
            # Ottieni la lista dei file nella directory corrente
            files = os.listdir(os.path.join(d, current_dir))
            # Aggiorna i valori di combosound con i file
            list_sound[i]['values'] = ["nessun suono"] + files

        def playsoundcombo(event):
            # Ottieni il percorso corrente
            d = os.getcwd()
            # Se il mixer sta riproducendo un suono, lo ferma
            if pygame.mixer.get_busy():
                try:
                    pygame.mixer.stop()
                except Exception  as errorstop:
                    print(f"errore stopping audio mixer: {errorstop}")
                finally:
                    pass
            # Riproduci il suono selezionato
            try:
                if list_sound[i]!= "nessun suono":
                    pygame.mixer.Sound(os.path.join(d,'sounds', list_dir[i].get(), list_sound[i].get())).play()
                else:
                    # Se il mixer sta riproducendo un suono, lo ferma
                    if pygame.mixer.get_busy():
                        try:
                            pygame.mixer.stop()
                        except Exception  as errorstop:
                            print(f"errore stopping audio mixer: {errorstop}")
                        finally:
                            pass
                    
            except Exception as errorplay:
                print(f"errore playing audio: {errorplay}")
            finally:
                pass
            
                
            

        return update_combosound, playsoundcombo

  
    
   
    # Define combodir before the for loop
    for i, elemento,in enumerate(text.split('|')):
        button.config(text="New Genera elementi")
        
        print(f"elemento: {elemento}")
        elemento= elemento.strip()
        if elemento != "" and len(elemento) > 2:      
            try:
                english = translator.translate(elemento, src='it', dest='en').text
            except Exception as e:
                print(f"Errore durante la traduzione dell'elemento {i}: {e}")
            finally:
               pass
                
            negative = "(blurry),( duplicate),(deformed),(octane render, render, drawing, anime, bad photo, bad photography:1.3), (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured"
            if Loading==False:
                generaimmagine(english,i,negative,cfg,steps)
            time.sleep(2)
            img= Image.open(f".\PIA\example\keyframe_{i}.png")
            # Provide the target size. The aspect ratio will be preserved.
            img.thumbnail((300, 300))
            # Convert the image object to a PhotoImage object
            photo = ImageTk.PhotoImage(img)
            photos.append(photo)  # Save a reference to the PhotoImage object
            ttk.Label(scrollable_frame, image=photo).grid(row=0, column=i*t)  # PictureBox
            ttk.Label(scrollable_frame, text="Prompt positivo").grid(row=1, column=i*t)
            entry_positivo = tk.Text(scrollable_frame, height=3)
            entry_positivo.grid(row=2, column=i*t)
            entry_positivo.insert('1.0', english)  # Insert the text into the Text widge
            # Aggiungi i riferimenti alle entry nelle liste
            list_positives.append(entry_positivo)
            
            
            ttk.Label(scrollable_frame, text="Prompt negativo").grid(row=3, column=i*t)
            
            entry_negativo=  tk.Text(scrollable_frame, height=3)
            entry_negativo.grid(row=4, column=i*t)
            entry_negativo.insert('1.0',negative)
            list_negatives.append(entry_negativo)
           
            label = ttk.Label(scrollable_frame)
            label.grid(row=0, column=i*t)  # PictureBox
            
            labels.append(label)  # Save a reference to the Label object
            regenerate_image, regenerate_video= make_regenerate_function(i, entry_positivo,entry_negativo,cfg,steps)
           
            ttk.Button(scrollable_frame, text="rigenera imagine", command=regenerate_image).grid(row=5, column=i*t)
            ttk.Button(scrollable_frame, text="rigenera video", command=regenerate_video).grid(row=6, column=i*t)
    
            
            player_frame = tk.Frame(scrollable_frame)
            player_frame.grid(row=7, column=i*t)  # Posiziona il frame del player nel layout

            canvas_frame = tk.Canvas(player_frame,bg='pink')
            canvas_frame.grid(row=0, column=0)  # Posiziona il frame del canvas nel layout del player_frame
            canvas_frames.append(canvas_frame)  # Save a reference to the Canvas object
            button_frame = tk.Frame(player_frame)
            button_frame.grid(row=1, column=0)  # Position the button frame in the layout
            if check_var.get()== 1 and Loading== False:
                generavideoclip(f".\PIA\example\keyframe_{i}.png",i,english,negative)
                if os.path.exists(f'.\\PIA\\example\\result\\{i}\\1_sim_3.gif'):
                    # Apri l'immagine con PIL e ridimensionala
                    with Image.open(f'.\\PIA\\example\\result\\{i}\\1_sim_3.gif') as img:
                        # Converti l'immagine ridimensionata in PhotoImage
                        photo = ImageTk.PhotoImage(img)
                        photosclips.append(photo)  # Memorizza l'immagine nella lista

                    # Aggiorna l'immagine sul canvas
                    canvas_frames[i].create_image(0, 0, image=photosclips[-1], anchor='nw')
            else:
                os.makedirs(os.path.join(f'.\\PIA\\example\\result\\', str(i)), exist_ok=True)
                
                            
                
            
            # Play button
            play = make_video_functions(i)
            tk.Button(button_frame, text="play", command=play).grid(row=0, column=0)

            # Variants
            labelvarianti = tk.Label(button_frame, text="Varianti clip")
            labelvarianti.grid(row=0, column=1)
            varianti = ttk.Combobox(button_frame, values=['1','2','3'])
            varianti.grid(row=1, column=1)
            listvary.append(varianti)

            # Duration
            labeldurata = tk.Label(button_frame, text="Durata clip")
            labeldurata.grid(row=0, column=2)
            longclip = ttk.Combobox(button_frame, values=[f'{4*i}s' for i in range(1, 31)])  # values from '4s' to '120s'
            longclip.grid(row=1, column=2)
            listlong.append(longclip)
          
            #AUDIOBOX
            style = ttk.Style()
            style.configure("BG.TLabel", background="#005fcc")
            #AUDIOBOX
            frameETaudiobox = tk.Frame(scrollable_frame, bg="#005fcc")
            frameETaudiobox.grid(column=i,row=11)  # Posizionato nella colonna i

            ETaudio = tk.Label(frameETaudiobox, text="AUDIO_BOX", font="Algerian",fg="#5fa4fe",bg="#005fcc")
            ETaudio.grid(column=0, row=0, columnspan=1)  # Centered in the frame

            # Crea gli oggetti e li aggiunge alle rispettive liste
            testo = tk.Text(frameETaudiobox,width= 20,height=3)
            testo.grid(column=0, row=1, sticky='w', pady=(0, 0))  # Positioned to the left, reduced vertical space

            grafico= tk.Canvas(frameETaudiobox,width= 100,height=100,bg="white")
            grafico.grid(column=0,row=2, pady=(0, 0))  # Reduced vertical space
            list_grafici.append(grafico)

            
            #comboboxaudio
            framecombo= tk.Frame(frameETaudiobox, bg="#005fcc")
            framecombo.grid(column=1, row=0, rowspan=4)  # Positioned to the right of the text box, rowspan increased to 4

            labecombomodelaudio= ttk.Label(framecombo,text="select model",style="BG.TLabel")
            labecombomodelaudio.grid(column=0 ,row= 0, pady=(0,0))

            combomodelaudio = ttk.Combobox(framecombo)
            combomodelaudio.grid(column=0, row=1, pady=(0, 0))  # Positioned at the top of the frame
            combomodelaudio.config(values=["ElevenLabs", "Coqui"])
            list_combomodelaudios.append(combomodelaudio)
            
            labecombovoice=ttk.Label(framecombo,text="select voice",style="BG.TLabel")
            labecombovoice.grid(column=0,row=2,pady=(0,0))
            


            combovoice= ttk.Combobox(framecombo)
            combovoice.grid(column=0, row=3, pady=(0, 0))  # Positioned below the first combobox
            if "nessuna voce" not in voiceelevellabs:
                voiceelevellabs = ["nessuna voce"] + voiceelevellabs
            combovoice.config(values=voiceelevellabs)
            list_combovoices.append(combovoice)
           
            playsoundaudio= make_regenerate_function_audio(i, testo, list_grafici, combovoice)
            tk.Button(framecombo,text="Play Audio",command= playsoundaudio).grid(column=0,row=4, pady=(0, 0))  # Add a new Button to playaudios
            
             
                
            
                
            update_combosound, playsoundcombo = make_function_combo(i)
            labecombodir=ttk.Label(framecombo,text="select dir suoni",style="BG.TLabel")
            labecombodir.grid(column=0,row=5,pady=(0,0))  # Corrected variable name

            combodir=ttk.Combobox(framecombo)  # Add a new Combobox to combodir
            combodir.grid(column=0, row=6, pady=(0, 0))  # Positioned below the first combobox
            combodir.config(values=[dir for dir in os.listdir(d)])
            combodir.bind('<<ComboboxSelected>>', update_combosound)  # Aggiungi l'evento di binding
            list_dir.append(combodir)
            labecombosound=ttk.Label(framecombo,text="select suoni dir",style="BG.TLabel")
            labecombosound.grid(column=0,row=7,pady=(0,0))
            
            combosound=ttk.Combobox(framecombo)  # Add a new Combobox to combosound
            combosound.grid(column=0, row=8, pady=(0, 0))  # Positioned below the first combobox
          # combosound.config(values=[audio for audio in os.listdir(os.path.join(d,combodir.get()))])
            combosound.bind('<<ComboboxSelected>>',playsoundcombo)
            list_sound.append(combosound)
         
        
        
                
            
           

        
             

       
        
        

    canvas.pack(side="left", fill="both", expand=True)

    scrollbar_x = tk.Scrollbar(root, orient="horizontal")
    scrollbar_x.pack(side="bottom", fill="x")

    scrollbar_y = tk.Scrollbar(root, orient="vertical")
    scrollbar_y.pack(side="right", fill="y")

    # Configura il canvas per usare le scrollbar
    canvas.config(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)
    scrollbar_x.config(command=canvas.xview)
    scrollbar_y.config(command=canvas.yview)
def main():
    global combobox,comboboxresolution,steps,cfg,check_var,checkboximagevideo,listlong,longclip,AUDIO_DOWNLOAD_DIR,check_seed,checkboxseed,button,nomesaving,text_box,list_positives,list_negatives,listvary,listlong,list_combomodelaudios,list_combovoices,list_dir,list_sound,root, frame2, canvas, scrollbar, scrollable_frame, photos, text_box,Loading,picturebox,pathfacebox,foto_tk,combostreghe,swapface_var,comboswap
    # Imposta il percorso predefinito del file
    nomesaving = os.path.join(os.path.dirname(os.path.realpath(__file__)), "salvataggioprogetto.zip")
    root = tk.Tk()
    frame = tk.Frame(root)
    frame.pack()
    
    framesave=  tk.Frame(root)
    framesave.place(x=0, y=0)
    
    def saving_as():
        global nomesaving, text_box, combobox, steps, cfg,list_positives,list_negatives,listvary,listlong,list_combomodelaudios,list_combovoices,list_dir,list_sound
        nomesaving = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("Zip files", "*.zip")])
        if nomesaving:
            saving()

    def saving():
        global nomesaving, text_box, combobox, steps, cfg,list_positives,list_negatives,listvary,listlong,list_combomodelaudios,list_combovoices,list_dir,list_sound
        # Crea una nuova finestra
        savingform = tk.Toplevel()
        savingform.title("Saving progetto")

        # Crea una progress bar
        progresssaving = ttk.Progressbar(savingform, length=200)
        progresssaving.pack()

        # Crea una label
        labelsaving = tk.Label(savingform)
        labelsaving.pack()
        temp_filename = "temp.txt"
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        if os.path.exists(nomesaving):
            os.remove(nomesaving)
         # Apri il file in scrittura
        with open(temp_filename, 'w') as fsave:
            # Scrivi ogni linea nel file
            for i, line in enumerate([
                f"MODEL:{combobox.get()}",
                f"RISOLUZIONE:{comboboxresolution.get()}",
                f"STEPS:{steps.get()}",
                f"CFG:{cfg.get()}",
                f"PROMPTS_POSITIVE: {[entry.get('1.0', 'end-1c') for entry in list_positives]}",
                f"PROMPTS_NEGATIVE: {[entry.get('1.0', 'end-1c') for entry in list_negatives]}",
                f"VARIANTI_CLIPS: {[listvary[i].get() for i in range(len(listvary))]}",
                f"DURATA_CLIPS: {[listlong[i].get() for i in range(len(listlong))]}",
                f"COMBOSELECT: {[list_combomodelaudios[i].get() for i in range(len(list_combomodelaudios))]}",
                f"VOCESELEZIONATA: {[list_combovoices[i].get() for i in range(len(list_combovoices))]}",
                f"CARTELLA_SUONI: {[list_dir[i].get() for i in range(len(list_dir))]}",
                f"SUONI: {[list_sound[i].get() for i in range(len(list_sound))]}"
            ]):
                fsave.write(line + "\n")
                # Aggiorna la progress bar e la label
                progresssaving['value'] = (i + 1) / 12 * 100  # Aggiorna il valore della progress bar (da 0 a 100)
                labelsaving['text'] = "Saving file: " + line  # parametro che vine scritto nel  temp.txt
                savingform.update_idletasks()  # Aggiorna la finestra
        time.sleep(2)
        # Resetta la progress bar
        progresssaving['value'] = 0
        savingform.update_idletasks()  # Aggiorna la finestra

        # Crea un nuovo file ZIP
        with zipfile.ZipFile(nomesaving, 'w') as mio_zip:
            mio_zip.write(temp_filename)
            # Aggiorna la progress bar e la label
            progresssaving['value'] = 25  # Aggiorna il valore della progress bar (da 0 a 100)
            labelsaving['text'] = "Saving file: " + temp_filename  # parametro che vine scritto nel  temp.txt
            savingform.update_idletasks()  # Aggiorna la finestra

            # Aggiungi tutti i file dalla cartella '.\elevenlabs' al file ZIP
            for folderName, subfolders, filenames in os.walk('.\\elevenlabs'):
                for filename in filenames:
                    filePath = os.path.join(folderName, filename)
                    mio_zip.write(filePath)
            # Aggiorna la progress bar e la label
            progresssaving['value'] = 50  # Aggiorna il valore della progress bar (da 0 a 100)
            labelsaving['text'] = "Saving file: " + filePath  # parametro che vine scritto nel  temp.txt
            savingform.update_idletasks()  # Aggiorna la finestra

            # Aggiungi tutti i file .png che contengono la parola 'keyframe_' dalla cartella '\example' al file ZIP
            for filename in os.listdir('.\\PIA\\example'):
                if 'keyframe_' in filename and filename.endswith('.png'):
                    filePath = os.path.join('.\\PIA\\example', filename)
                    mio_zip.write(filePath)
            # Aggiorna la progress bar e la label
            progresssaving['value'] = 75  # Aggiorna il valore della progress bar (da 0 a 100)
            labelsaving['text'] = "Saving file: " + filePath  # parametro che vine scritto nel  temp.txt
            savingform.update_idletasks()  # Aggiorna la finestra

            # Aggiungi tutte le cartelle e i file dalla cartella '\example\result' al file ZIP
            for folderName, subfolders, filenames in os.walk('.\\PIA\\example\\result'):
                for filename in filenames:
                    filePath = os.path.join(folderName, filename)
                    mio_zip.write(filePath)
        # Aggiorna la progress bar e la label
        progresssaving['value'] = 100  # Aggiorna il valore della progress bar (da 0 a 100)
        labelsaving['text'] = "Salvataggio completato"  # parametro che vine scritto nel  temp.txt
        savingform.update_idletasks()  # Aggiorna la finestra

        # Resetta la progress bar e la label per il prossimo salvataggio
        progresssaving['value'] = 0
        labelsaving['text'] = ""
        
        # Chiudi la finestra
        savingform.destroy()
    
    def caricaproggetto():
        global combobox, comboboxresolution,steps,cfg,root, frame2, canvas, scrollbar, scrollable_frame, photos, text_box,Loading,listvary,listlong,list_combomodelaudios,list_combovoices,list_dir,list_sound
        print("CARICA PROGETTO")
        apriprogetto= filedialog.askopenfilename(defaultextension='.zip')
        #elimina old files e directorys
        pathelevenlabs= ".\\elevenlabs"
        pathkeyframe= ".\\PIA\\example"
        pathclips= ".\\PIA\\example\\result"
        
        # numero di files contenuti nella directory pathelevenlabs
        nfilesele= len([f for f in os.listdir(pathelevenlabs) if os.path.isfile(os.path.join(pathelevenlabs, f))])
        #numero dei files nella cartella pathkeyframe che finiscono con .png e contengono la parola keyframe
        nfileskey= len([f for f in os.listdir(pathkeyframe) if f.endswith(".png") and "keyframe" in f])
        #numero di files contenuti in tutte le sotto directorys nella directory pathclips 
        nfilesclips = sum([len(files) for r, d, files in os.walk(pathclips)])

        # una di queste cartelle contiene files 
        if nfilesclips > 0 or nfilesele > 0 or nfileskey > 0:
            if not messagebox.askyesno("Attenzione", "Ci sono ancora dei files nelle cartelle. Vuoi cancellarli?"):
                return  # Se l'utente sceglie "No", esce dalla funzione

        # elimina tutti file e cartelle nei paths: 
        for folder in [pathelevenlabs, pathclips]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        # elimina i file png nella cartella examples di tipo png e che contengono la parola keyframe_
        for filename in os.listdir(pathkeyframe):
            if filename.endswith(".png") and "keyframe_" in filename:
                file_path = os.path.join(pathkeyframe, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
                    
        print(f"percorso file:{apriprogetto}")
        #estrai file zip
                   #sorgente    cartella dello script ; copia files e directory 
        dirscripts= os.getcwd()
        with zipfile.ZipFile(apriprogetto, 'r') as file_zip:
            file_zip.extractall(dirscripts)
        Loading= True    
        # Crea una nuova finestra
        progress_window = tk.Toplevel(root)
        progress_window.title("Caricamento in corso...")
        # Crea la barra di progresso
        progress_bar = ttk.Progressbar(progress_window, length=500, mode='determinate')
        progress_bar.pack(pady=10)
        # Ottieni il numero totale di linee nel file
        num_lines = sum(1 for line in open(".\\temp.txt"))
        for i, line in enumerate(open(".\\temp.txt").readlines()):
            if "MODEL:" in line:
                combobox.set(line.replace("MODEL:", "").strip())
            elif "RISOLUZIONE:" in line:
                comboboxresolution.set(line.replace("RISOLUZIONE:", "").strip())
            elif "STEPS:" in line:
                steps.delete(0, 'end')
                steps.insert(0, line.replace("STEPS:", "").strip())
            elif "CFG:" in line:
                cfg.delete(0, 'end')
                cfg.insert(0, line.replace("CFG:", "").strip())
            elif "PROMPTS_POSITIVE:" in line:
                line= line.replace("PROMPTS_POSITIVE:","")
                testo = line.replace("['", "").replace("']", "").replace("', '", "|").strip() + "|"
                text_box.delete('1.0', 'end')  # Clear the existing text
                text_box.insert('1.0', testo)  # Insert the new text
                
                crea_elementi(root, frame2, canvas, scrollbar, scrollable_frame, photos, text_box.get("1.0", "end-1c"))
                time.sleep(0.100)
            elif "VARIANTI_CLIPS:" in line:
                line = line.replace("VARIANTI_CLIPS:", "")
                valori = line.replace("[", "").replace("]", "").replace("'", "").split(", ")
                for i, valore in enumerate(valori):
                    listvary[i].set(valore)
            elif "DURATA_CLIPS:" in line:
                    line = line.replace("DURATA_CLIPS:", "")
                    valori2 = line.replace("[", "").replace("]", "").replace("'", "").split(", ")
                    for i, valore in enumerate(valori2):
                        listlong[i].set(valore)
            elif "COMBOSELECT:" in line:
                    line = line.replace("COMBOSELECT:", "")
                    valori3 = line.replace("[", "").replace("]", "").replace("'", "").split(", ")
                    for i, valore in enumerate(valori3):
                        list_combomodelaudios[i].set(valore)
            elif "VOCESELEZIONATA:" in line:
                    line = line.replace("VOCESELEZIONATA:", "")
                    valori4 = line.replace("[", "").replace("]", "").replace("'", "").split(", ")
                    for i, valore in enumerate(valori4):
                        list_combovoices[i].set(valore)
            elif "CARTELLA_SUONI:" in line:
                    line = line.replace("CARTELLA_SUONI:", "")
                    valori5 = line.replace("[", "").replace("]", "").replace("'", "").split(", ")
                    for i, valore in enumerate(valori5):
                        list_dir[i].set(valore)
            elif "SUONI:" in line:
                    line = line.replace("SUONI:", "")
                    valori6 = line.replace("[", "").replace("]", "").replace("'", "").split(", ")
                    for i, valore in enumerate(valori6):
                        list_sound[i].set(valore)
            # Calcola il progresso come una percentuale del numero totale di linee
            progress = (i + 1) / num_lines * 100
            progress_bar['value'] = progress
            progress_window.update()
            time.sleep(0.01)
       
        Loading= False
        # Distrugge la finestra alla fine del caricamento
        progress_window.destroy()
                
            
                
                
            
               
        
                    
        
         
            
            
            
            
            
    
    save= ttk.Button(framesave,text="Save")
    save.pack(side="top")
    save.config(command=saving) # Pass 'nomesaving' to the 'saving' function
    save_as= ttk.Button(framesave,text="Save As")
    save_as.pack(side="top")
    save_as.config(command=saving_as)
    caricaprogetto= ttk.Button(framesave,text="Carica Progetto")
    caricaprogetto.pack(side="top")
    caricaprogetto.config(command=caricaproggetto)
    

   
           
        

    model_frame = tk.Frame(frame)
    model_frame.pack(side="left", anchor="n")
    labemodel= ttk.Label(model_frame,text= 'SELEZIONA MODELLO')
    labemodel.pack(side="top")
    combobox = ttk.Combobox(model_frame, values=['Realist V6','Rocco_1','Rocco_2'])
    combobox.pack(side="top")
    labelresolu= ttk.Label(model_frame,text="RISOLUZIONE FRAMES")
    labelresolu.pack(side="top")
    comboboxresolution = ttk.Combobox(model_frame, values=['256,256', '3640,360', '256,360', '360,256', '512,512', '512,720', '720,512', '720,720', '960,544', '1920,1080', '2560,1440', '3840,2160', '7680,4320'])
    comboboxresolution.pack(side="top")
    
    labelsteps = ttk.Label(model_frame, text="STEPS")
    labelsteps.pack(side="top")
    steps = tk.Entry(model_frame)
    steps.pack(side="top")
    steps.insert(0, "25")  # Inserting text at the beginning

    labelcfg = ttk.Label(model_frame, text="CONFIG")
    labelcfg.pack(side="top")
    cfg = tk.Entry(model_frame)
    cfg.pack(side="top")
    cfg.insert(0, "7")  # Inserting text at the beginning
    
    from tkinter import font as tkFont
    from tkinter import Canvas
    # Crea un nuovo font
    customFont = tkFont.Font(family="A Charming Font", size=20,weight="bold")
    # Applica il font al tuo widget
    labelstreghe = ttk.Label(model_frame, text="STREGHE", font=customFont)
    labelstreghe.pack(side="top")
    
    def selezionapersonaggio(event=None):  # Aggiungi event=None per gestire l'evento
        global combostreghe, text_box
        if combostreghe.get()== "Nessun Personaggio":
            print("nessun personaggio")
        elif combostreghe.get() == "Phoebe":
            text_box.insert('end', "una ragazza di 20 anni (((4lyss4m)))  'inserisci qui la descrizione  '|")
        elif combostreghe.get() == "Piper":
            text_box.insert('end', "una ragazza di 23 anni (((Holly_Marie_Combs_PMv1_Lora)))  'inserisci qui la descrizione  '|")
        elif combostreghe.get() == "Prue":
            text_box.insert('end', "una ragazza di 27 anni (((PrueHalliwell)))  'inserisci qui la descrizione  '|")
        elif combostreghe.get() == "Pagie":
            text_box.insert('end', "una ragazza di 25 anni (((PaigeMatthews)))  'inserisci qui la descrizione  '|")
        elif combostreghe.get() == "Billie":
            text_box.insert('end', "una ragazza di 18 anni (((k4l3yc)))  'inserisci qui la descrizione  '|")

    combostreghe = ttk.Combobox(model_frame, values=['Nessun Personaggio','Phoebe', 'Piper', 'Prue', 'Pagie', 'Billie'])
    combostreghe.pack(side="top")
    combostreghe.bind('<<ComboboxSelected>>', selezionapersonaggio)  # Cambia l'evento e passa la funzione
    
    pathfacebox = "nessuna foto"
    
   #frame foto
    framefoto= ttk.Frame(model_frame)
    framefoto.pack(side="top")

    # Crea picturebox
    picturebox = Canvas(framefoto, width=128, height=128, background="pink")
    picturebox.pack(side="top")

    foto_tk=None

    def Uploadfoto():
        global pathfacebox, picturebox,foto_tk
        pathfacebox = filedialog.askopenfilename()
        if pathfacebox != "nessuna foto" and pathfacebox is not None and pathfacebox != "":
            print(f"path face: {pathfacebox}")
            foto = Image.open(pathfacebox)
            foto.thumbnail((128, 128))
            foto_tk = ImageTk.PhotoImage(foto)
            picturebox.create_image(10,10, image=foto_tk, anchor='nw')
            picturebox.update()
        else:
            print(f"path face: {pathfacebox} : nessuna foto")

    # Create a frame for the buttons
    button_frame = ttk.Frame(model_frame)
    button_frame.pack(side="top", anchor="e")

    uploadfoto = ttk.Button(button_frame, text="Upload Faccia")
    uploadfoto.pack(side="left")
    uploadfoto.config(command=Uploadfoto)
    
    def cancfoto():
        global pathfacebox, picturebox,swapface_var
        pathfacebox = "nessuna foto"
        picturebox.delete("all")
        picturebox.config(background="pink")
        picturebox.update()
        

    cancellafoto= ttk.Button(button_frame,text="cancella foto")
    cancellafoto.pack(side="left")
    cancellafoto.config(command=cancfoto)
    
     # Define swapface_var before using it
    swapface_var = tk.BooleanVar()
    swapface_var.set(False)  # The checkbox will be unchecked by default

    comboswap= ttk.Checkbutton(button_frame, text="swapface", variable=swapface_var)
    comboswap.pack(side="left")
    

    framecheckbox= tk.Frame(frame)
    framecheckbox.pack(side='right',anchor='n')
    # Creare una variabile tkinter IntVar
    check_var = tk.IntVar()
    # Impostare la variabile a 0 (false)
    check_var.set(0)
    checkboximagevideo= ttk.Checkbutton(framecheckbox,text="Genera img_keys e Clips", variable=check_var)
    checkboximagevideo.pack(side='top')
    
    check_seed= tk.IntVar()
    check_seed.set(0)
    checkboxseed = ttk.Checkbutton(framecheckbox, text="Genera stesse immagini", variable=check_seed)
    checkboxseed.pack(side="top")
    
    # Inizializza le variabili globali
    prev_progress = 0
    bar = None
    AUDIO_DOWNLOAD_DIR = ".//sounds//youtube"

    def progress_function(stream, chunk, bytes_remaining):
        global prev_progress
        current_progress = (stream.filesize - bytes_remaining)
        progress_increment = current_progress - prev_progress
        bar.update(progress_increment)
        prev_progress = current_progress

    def search_canzoneedownload(song_title):
        global bar,AUDIO_DOWNLOAD_DIR
        results = YoutubeSearch(song_title, max_results=1).to_dict()
        print(f"risultati: {results}")
        if results:
            video_url = 'https://www.youtube.com' + results[0]['url_suffix']
            video = YouTube(video_url, on_progress_callback=progress_function)
            audio = video.streams.filter(only_audio = True).first()
            bar = tqdm(total=audio.filesize, ncols=100, unit='B', unit_scale=True)
            try:
                # Definisci il nome del file
                file_name = f"{song_title.strip()}.mp3"
                # Scarica l'audio e salvalo con il nome del file definito
                try:
                    audio.download(output_path=AUDIO_DOWNLOAD_DIR, filename=file_name)
                except Exception as ex:
                    print(f"errore download: {ex}")
                finally:
                    pass 
                print("\naudio was downloaded successfully")
                prev_progress=100

                # Converti il file mp3 in wav
                audio = AudioSegment.from_mp3(f"{AUDIO_DOWNLOAD_DIR}/{file_name}")
                audio.export(f"{AUDIO_DOWNLOAD_DIR}/{song_title}.wav", format="wav")
                print(f"audio was converted to wav and saved as {song_title}.wav")
            except:
                print("Failed to download audio")
            finally:
                bar.close()
        else:
            print( "Nessun risultato trovato")


    
    def newpath():
        if not text_box.get('1.0','end')=="" and "youtube:" in text_box.get('1.0','end').lower():
            y,suono= text_box.get('1.0','end').split(":")
            # scarica video da youtube
            print(f"nome suono: {suono}")
            search_canzoneedownload(suono)
        elif not text_box.get('1.0','end')=="" and "https://www.youtube.com/watch?v=" in text_box.get('1.0','end').lower():
            link = text_box.get('1.0','end').strip()
            youtubeObject = YouTube(link)
            youtubeStream = youtubeObject.streams.get_highest_resolution()
            try:
                # Scarica il video
                file_path = youtubeStream.download(output_path=AUDIO_DOWNLOAD_DIR)
                print("Video download is completed successfully")

                # Estrai l'audio dal video
                video = AudioSegment.from_file(file_path)
                audio = video.set_channels(1)  # Converte l'audio in mono
                # Rimuovi o sostituisci i caratteri speciali nel titolo del video
                safe_title = ''.join(c if c.isalnum() else "_" for c in youtubeObject.title)
                # Usa il titolo sicuro per creare il nome del file
                file_name = safe_title + ".wav"
                # Esporta l'audio
                audio.export(os.path.join(AUDIO_DOWNLOAD_DIR, file_name), format="wav")
                print("Audio extraction is completed successfully")

                # Rimuovi il file video originale
                os.remove(file_path)
            except Exception as ex:
                print(f"An error has occurred: {ex}")
        else:
            os.makedirs(".\\sounds\\extra",exist_ok=True)
            #copia file da disco
            diagfile = filedialog.askopenfilename()  # Mostra la finestra di dialogo del file
            # Copia il file
            destination = os.path.join(".\\sounds\\extra", os.path.basename(diagfile))
            shutil.copy(diagfile, destination)
            


         

    
            

    pathaudio = ttk.Button(framecheckbox, text="path new Audio", command=newpath)
    pathaudio.pack(side="right", anchor='n')
    
    text_frame = tk.Frame(frame)
    text_frame.pack(side="left", fill="x")

    text_box = tk.Text(text_frame)
    text_box.pack(side="top", fill="x")
    button = ttk.Button(text_frame, text="Genera elementi")
    button.pack(side="top")

    
    
    def monta(comboboxup):
       
        global listlong,longclip,list_dir,list_sound,list_combovoices,list_combovoices,fileaudio
        secondisound=0
        secondivoice=0
        secondicombo=0
        percorso_clips = ".\\PIA\\example\\result"
        # pulizia
        for p in os.listdir(percorso_clips):
            if p.endswith(".mp4"):
                os.remove(os.path.join(percorso_clips,p))
        if os.path.exists(".\\CodeFormer\\finalvideo.mp4"):
           os.remove(".\\CodeFormer\\finalvideo.mp4")
        
       
        clips = os.listdir(percorso_clips)
        n_clips = len(clips)
        print(f"Montaggio di {n_clips} clips")
        tutti_nome_clips = []
        clips_Part = [[] for _ in range(n_clips)]
        fileaudio=[]
        
        
        tutti_nome_clips.append("[")

        for i, clip in enumerate(clips):
            soundpath = os.path.join(os.getcwd(), 'sounds', list_dir[i].get(), list_sound[i].get())
            voicepath = os.path.join(os.getcwd(), 'elevenlabs', f"voice_{i}.mp3")
            if not os.path.exists(soundpath) or list_sound[i]== "nessun suono" or list_sound[i]== '':
                soundpath=  ".\sounds\extra\silenzio.wav"
            if not os.path.exists(voicepath) or list_combovoices[i]== "nessuna voce" or list_combovoices[i]=='':
                voicepath= ".\sounds\extra\silenzio.wav"
                
            # Calcola la durata del file audio
            audio = AudioSegment.from_file(soundpath)
            secondisound = audio.duration_seconds
            print(f"secondi audio: {list_sound[i].get()}, {secondisound}")
            # Calcola la durata del file voce
            voice = AudioSegment.from_file(voicepath)
            secondivoice = voice.duration_seconds
            print(f"secondi voce: {list_combovoices[i].get()}, {secondivoice}")
            
            # Ottieni la durata specificata
            if listlong[i].get()=='':
                secondicombo='10s'
            else:
                secondicombo = listlong[i].get()
            secondicombo = int(secondicombo.replace('s', ''))
            print(f"secondi combo: {listlong[i].get()}")
            
            
            # Crea un dizionario con le durate
            durate = {"audio": secondisound, "voce": secondivoice, "combo": secondicombo}
            # Trova il massimo tra le tre durate
            audio_max = max(durate, key=durate.get)
            # Trova la chiave con il valore massimo
            secondi = max(secondisound, secondivoice, secondicombo)
            
            #applica audio alla clip
            if audio_max == "audio":
                print(f"AudiuoL: {soundpath}")
                if os.path.exists(soundpath):
                    fileaudio.append(soundpath)
            elif audio_max == "voce":
                print(f"VoceL: {voicepath}")
                if os.path.exists(voicepath):
                    fileaudio.append(voicepath)
            # se i secondi della combobox sono maggiori della durata del audio suono e audio voce
            elif audio_max=="combo":
                print(f"voce combo: {list_combovoices[i].get()}")
                print(f"sondcombo:{list_sound[i].get()}")
                
                #se la durata del audio sond è maggiore di quella della voce
                if not list_sound[i].get()== "nessun suono" and list_combovoices[i].get()== "nessuna voce":
                    print(f"AudiuoLC: {soundpath}")
                    soundpaths = []
                    # se il nome del file contiene Fd_io o Fd_i;o Fd_o ,rimuovilo
                    nomefile = list_sound[i].get().replace('Fd_io.mp3','').replace('Fd_i.mp3','').replace('Fd_o.mp3','')
                    #se nome file finisce con _p1 allora trova altri segmenti del altro suono e uniscili fino ad arrivare alla massima dirata della combolong
                    if nomefile.endswith('_p1'):
                        print("nome suono contiene P1")
                        soundpaths.append(os.path.join(os.getcwd(), 'sounds', list_dir[i].get(), list_sound[i].get()))
                        sommaaudio = (len(AudioSegment.from_file(soundpaths[-1]))/1000)
                        print(f"somma prima: {sommaaudio}")
                        for y in range(2,100):
                                # verifica se il file nomeframento_py esiste
                            if os.path.exists(os.path.join(os.getcwd(), 'sounds', list_dir[i].get(), nomefile.replace('_p1',f'_p{y}.mp3'))):
                                # calcola la durata del prossimo segmento
                                next_segment_duration = round(len(AudioSegment.from_file(os.path.join(os.getcwd(), 'sounds', list_dir[i].get(), nomefile.replace('_p1',f'_p{y}.mp3'))))/1000)
                                print(f"next segmento: {next_segment_duration}")
                                # verifica se l'aggiunta del prossimo segmento farebbe superare la somma totale il limite
                                print(f"somma : {int(sommaaudio + next_segment_duration)}, long: {int(listlong[i].get().replace('s', ''))}")
                                if int(sommaaudio + next_segment_duration) <=  int(listlong[i].get().replace('s', '')):
                                    # se non supera il limite, aggiungi la durata alla somma totale e aggiungi il segmento a soundpaths
                                    sommaaudio += next_segment_duration
                                    soundpaths.append(os.path.join(os.getcwd(), 'sounds', list_dir[i].get(), nomefile.replace('_p1',f'_p{y}.mp3')))
                            else:
                                print(f"FILE NON ESISTE: {os.path.join(os.getcwd(), 'sounds', list_dir[i].get(), nomefile.replace('_p1',f'_p{y}.mp3'))}")
                                break
                                

                        print(f"segmenti concatenati: {[os.path.basename(sound) for sound in soundpaths]}")
                        conc = sum([AudioSegment.from_file(sound) for sound in soundpaths])
                        conc.export(".\\conc.mp3", format='mp3')
                        fileaudio.append(".\\conc.mp3")
                    else:
                        fileaudio.append(soundpath)
                elif list_sound[i].get()== "nessun suono" and list_combovoices[i].get()== "nessuna voce":
                    print("video Muto")
                    #nessun audio
                    # Aggiungi un file audio silenzioso a fileaudio
                    fileaudio.append(".\\sounds\\extra\\silenzio.wav")
                elif list_sound[i].get()== "nessun suono" and not list_combovoices[i].get()== "nessuna voce":
                   print(f"VoceLC: {voicepath}")  
                   fileaudio.append(voicepath)
        
            clips_dir = [clipd for clipd in os.listdir(os.path.join(percorso_clips, clip)) if clipd.endswith(".gif")]
            print(f"{len(clips_dir)} clips nella cartella {i}")
            if len(clips_dir) == 0:
                tutti_nome_clips.append(f"{i}_[")
                for _ in range(int((secondi // 4) // len(clips_dir))):
                    clips_casuali = ra.sample(clips_dir, len(clips_dir))
                    tutti_nome_clips.extend(clips_casuali)
                    clips_Part[i].extend(clips_casuali)
                tutti_nome_clips.append("]")
            else:
                tutti_nome_clips.append(f"{i}_[")
                for _ in range(int(secondi // 4) // len(clips_dir)):
                    clips_casuali = ra.sample(clips_dir, len(clips_dir))
                    tutti_nome_clips.extend(clips_casuali)
                    clips_Part[i].extend(clips_casuali)
                tutti_nome_clips.append("]")
        print(([str(clip) for clip in tutti_nome_clips]))
        for i, _ in enumerate(clips):
                print(f"part clip: {i}")
                print(f"\n {clips_Part[i]}")
                
        
        all_temp_clips = []
        for i in tqdm(range(n_clips), desc="totale Rendering clips"):
            tempclip = []
            for cliptemp in tqdm(clips_Part[i], desc=f"rendering temp_{i}"):
                cliptemp = VideoFileClip(os.path.join(percorso_clips, str(i), cliptemp))
                tempclip.append(cliptemp)
            clip = concatenate_videoclips(tempclip)
            all_temp_clips.append(clip)
        
            # applica audio alla clip se esiste
            audio = AudioFileClip(fileaudio[i])
            print(f"\naudio clip: {i}, {fileaudio[i]}")
            clip = clip.set_audio(audio)
        
        
            clip.write_videofile(os.path.join(percorso_clips, f"temp_{i}.mp4"), codec='mpeg4', audio_codec='aac')
            clip.close()
            
        final_clips = []
        for i in range(n_clips):
            # Leggi ogni clip video (con audio) una per una
            clip = VideoFileClip(os.path.join(percorso_clips, f"temp_{i}.mp4"))
            # Aggiungi la clip alla lista delle clip finali
            final_clips.append(clip)

        # Concatena tutte le clip nella clip finale
        final_video = concatenate_videoclips(final_clips)   
        # Scrivi la clip finale su disco
        final_video.write_videofile(os.path.join(percorso_clips, "finalvideo.mp4"), codec='mpeg4', audio_codec='aac')
            
        # upscale
        print(f"upscalvalu {comboboxup.get()}")
        
            
        if comboboxup.get() != "noUpscale" and comboboxup.get() != '':
                print(f"Upscale...{comboboxup.get()}")
                if int(comboboxup.get().replace("X",''))== 2:
                    print("Upscale 2")
                    valore= int(comboboxup.get().replace("X",''))
                if int(comboboxup.get().replace("X",''))== 4:
                    print("Upscale 4")
                    valore= int(comboboxup.get().replace("X",''))
                if int(comboboxup.get().replace("X",''))== 8:
                    print("Upscale 8")
                    valore= int(comboboxup.get().replace("X",''))
        else:
            print(f"Low Resolution...{comboboxup.get()}")
            valore= 0
       # aggiungi musica
        if not combomusica.get().strip() == "nessuna musica":
            video = VideoFileClip(os.path.join(percorso_clips,"finalvideo.mp4"))
            
            # Crea un AudioFileClip per la musica di background
            musica = AudioFileClip(f".\\musica\\{combomusica.get()}")
            
            # Tronca la musica alla durata del video
            musica = musica.subclip(0, video.duration)
            
            # Imposta il volume della musica al 40%
            musica = musica.volumex(0.4)
            
            # Aggiungi una sfumatura di fade out di 2 secondi alla fine della musica
            musica = musica.fx(audio_fadeout, 2)
            
            # Crea un CompositeAudioClip con l'audio del video e la musica di background
            nuovo_audio = CompositeAudioClip([video.audio, musica])
            
            # Imposta il nuovo audio come l'audio del video
            video = video.set_audio(nuovo_audio)
            
            video.write_videofile(os.path.join(percorso_clips, "finalvideo_Musica.mp4"), codec='mpeg4', audio_codec='aac')
           
            
           
        
        
        #UPSCALE
        if valore==0:
           if os.path.exists(os.path.join(percorso_clips,"finalvideo_Musica.mp4")):
                  os.startfile(os.path.join(percorso_clips,"finalvideo_Musica.mp4"))  
           elif os.path.exists(os.path.join(percorso_clips,"finalvideo.mp4")):
              os.startfile(os.path.join(percorso_clips,"finalvideo.mp4"))        
        else:
            if os.path.exists(os.path.join(percorso_clips,"finalvideo_Musica.mp4")):
                shutil.move(os.path.join(percorso_clips,"finalvideo_Musica.mp4"),".\\CodeFormer\\finalvideo.mp4")
                os.chdir("CodeFormer")
            else:
                shutil.move(os.path.join(percorso_clips,"finalvideo.mp4"),".\\CodeFormer\\finalvideo.mp4")
                os.chdir("CodeFormer")
            
            # pulizia cartelle precedenti
            for p in os.listdir(".\\results\\finalvideo_1.0"):
                path = os.path.join(".\\results\\finalvideo_1.0", p)
                if os.path.isdir(path):
                        shutil.rmtree(path)
            
                
        if valore== 2:
            #upscaleX2
            print("total Upscale X2")
            os.system("python inference_codeformer.py --bg_upsampler realesrgan --face_upsample -w 1.0 --input_path finalvideo.mp4")
            time.sleep(2)
            os.chdir("..")
            
            if os.path.exists(".\\CodeFormer\\results\\finalvideo_1.0\\finalvideo.mp4"):
               shutil.move(".\\CodeFormer\\results\\finalvideo_1.0\\finalvideo.mp4",".\\PIA\\example\\result\\finalvideoUP_X2.mp4") 
            else:
                print("file non trovato")
            if os.path.exists(".\\PIA\\example\\result\\finalvideoUP_X2.mp4"):
               os.startfile(".\\PIA\\example\\result\\finalvideoUP_X2.mp4") 
               
        if valore== 4:
           #upscaleX2
           print("total Upscale X4: fist Upscale X2")
           os.system("python inference_codeformer.py --bg_upsampler realesrgan --face_upsample -w 1.0 --input_path finalvideo.mp4")
           time.sleep(2)
           os.chdir("..")
           #upscale X2 
           if os.path.exists(".\\CodeFormer\\results\\finalvideo_1.0\\finalvideo.mp4"):
              shutil.move(".\\CodeFormer\\results\\finalvideo_1.0\\finalvideo.mp4",".\\CodeFormer\\finalvideo.mp4") 
           else:
               print("file non trovato")
            # pulizia cartelle precedenti
           for p in os.listdir(".\\CodeFormer\\results\\finalvideo_1.0"):
               path = os.path.join(".\\CodeFormer\\results\\finalvideo_1.0", p)
               if os.path.isdir(path):
                    shutil.rmtree(path)
           #UPscale X4 
           os.chdir("CodeFormer")
           print("total Upscale X4: Second Upscale X2")
           os.system("python inference_codeformer.py --bg_upsampler realesrgan --face_upsample -w 1.0 --input_path finalvideo.mp4")
           time.sleep(2)
           os.chdir("..")
           if os.path.exists(".\\CodeFormer\\results\\finalvideo_1.0\\finalvideo.mp4"):
              shutil.move(".\\CodeFormer\\results\\finalvideo_1.0\\finalvideo.mp4",".\\PIA\\example\\result\\finalvideoUP_X4.mp4") 
           else:
                print("file non trovato")
           if os.path.exists(".\\PIA\\example\\result\\finalvideoUP_X4.mp4"):
              os.startfile(".\\PIA\\example\\result\\finalvideoUP_X4.mp4") 
              
        if valore== 8:
            #upscaleX2
            print("total Upscale X8: fist Upscale X2")
            os.system("python inference_codeformer.py --bg_upsampler realesrgan --face_upsample -w 1.0 --input_path finalvideo.mp4")
            time.sleep(2)
            os.chdir("..")
            #upscale X2 
            if os.path.exists(".\\CodeFormer\\results\\finalvideo_1.0\\finalvideo.mp4"):
                shutil.move(".\\CodeFormer\\results\\finalvideo_1.0\\finalvideo.mp4",".\\CodeFormer\\finalvideo.mp4") 
            else:
                print("file non trovato")
                # pulizia cartelle precedenti
            for p in os.listdir(".\\CodeFormer\\results\\finalvideo_1.0"):
                path = os.path.join(".\\CodeFormer\\results\\finalvideo_1.0", p)
                if os.path.isdir(path):
                        shutil.rmtree(path)
            #UPscale X4 
            os.chdir("CodeFormer")
            print("total Upscale X8: Second Upscale X2")
            os.system("python inference_codeformer.py --bg_upsampler realesrgan --face_upsample -w 1.0 --input_path finalvideo.mp4")
            time.sleep(2)
            os.chdir("..")
            if os.path.exists(".\\CodeFormer\\results\\finalvideo_1.0\\finalvideo.mp4"):
                shutil.move(".\\CodeFormer\\results\\finalvideo_1.0\\finalvideo.mp4",".\\CodeFormer\\finalvideo.mp4") 
            else:
                    print("file non trovato")
            #UPscale X8 
            os.chdir("CodeFormer")
            print("total Upscale X8: terzo Upscale X2")
            os.system("python inference_codeformer.py --bg_upsampler realesrgan --face_upsample -w 1.0 --input_path finalvideo.mp4")
            time.sleep(2)
            os.chdir("..")
            if os.path.exists(".\\CodeFormer\\results\\finalvideo_1.0\\finalvideo.mp4"):
                shutil.move(".\\CodeFormer\\results\\finalvideo_1.0\\finalvideo.mp4",".\\PIA\\example\\result\\finalvideoUP_X8.mp4") 
            else:
                    print("file non trovato")
            if os.path.exists(".\\PIA\\example\\result\\finalvideoUP_X8.mp4"):
                os.startfile(".\\PIA\\example\\result\\finalvideoUP_X8.mp4")     
                
            
           
            
            
    

    
        
        
    comboboxup = ttk.Combobox(root)
    comboboxup.pack(side="bottom", anchor="e")
    comboboxup.config(values=["noUpscale","X2","X4","X8"])

    buttonmount = ttk.Button(root, text="Mount")
    buttonmount.pack(side="bottom", anchor="e")
    buttonmount.config(command=lambda: monta(comboboxup))
    
    # Inizializza le variabili globali
    prev_progress = 0
    bar = None
    Musica_DOWNLOAD_DIR = ".//musica"
    
 

    
    
    def progress_function(stream, chunk, bytes_remaining):
        global prev_progress
        current_progress = (stream.filesize - bytes_remaining)
        progress_increment = current_progress - prev_progress
        bar.update(progress_increment)
        prev_progress = current_progress

    def search_canzoneedownload(song_title):
        global bar,AUDIO_DOWNLOAD_DIR,prev_progress
        prev_progress=0
        results = YoutubeSearch(song_title, max_results=1).to_dict()
        print(f"risultati: {results}")
       
        if results:
            video_url = 'https://www.youtube.com' + results[0]['url_suffix']
            video = YouTube(video_url, on_progress_callback=progress_function)
            audio = video.streams.filter(only_audio = True).first()
            bar = tqdm(total=audio.filesize, ncols=100, unit='B', unit_scale=True)
            try:
                # Definisci il nome del file
                file_name = f"{song_title.strip()}.mp3"
                # Scarica l'audio e salvalo con il nome del file definito
                try:
                    audio.download(output_path=AUDIO_DOWNLOAD_DIR, filename=file_name)
                except Exception as ex:
                    print(f"errore download: {ex}")
                finally:
                    pass 
                print("\naudio was downloaded successfully")
                prev_progress=100

                # Converti il file mp3 in wav
                audio = AudioSegment.from_mp3(f"{AUDIO_DOWNLOAD_DIR}/{file_name}")
                audio.export(f"{AUDIO_DOWNLOAD_DIR}/{song_title}.wav", format="wav")
                print(f"audio was converted to wav and saved as {song_title}.wav")
            except:
                print("Failed to download audio")
            finally:
                bar.close()
        else:
            print( "Nessun risultato trovato")
    
    def musicabackground():
        if not text_box.get('1.0','end')=="" and "youtube:" in text_box.get('1.0','end').lower():
            y,suono= text_box.get('1.0','end').split(":")
            # scarica video da youtube
            print(f"nome suono: {suono}")
            search_canzoneedownload(suono)
        elif not text_box.get('1.0','end')=="" and "https://www.youtube.com/watch?v=" in text_box.get('1.0','end').lower():
            link = text_box.get('1.0','end').strip()
            youtubeObject = YouTube(link)
            youtubeStream = youtubeObject.streams.get_highest_resolution()
            try:
                # Scarica il video
                file_path = youtubeStream.download(output_path=Musica_DOWNLOAD_DIR)
                print("Video download is completed successfully")

                # Estrai l'audio dal video
                video = AudioSegment.from_file(file_path)
                audio = video.set_channels(1)  # Converte l'audio in mono
                # Rimuovi o sostituisci i caratteri speciali nel titolo del video
                safe_title = ''.join(c if c.isalnum() else "_" for c in youtubeObject.title)
                # Usa il titolo sicuro per creare il nome del file
                file_name = safe_title + ".wav"
                # Esporta l'audio
                audio.export(os.path.join(Musica_DOWNLOAD_DIR, file_name), format="wav")
                print("Audio extraction is completed successfully")

                # Rimuovi il file video originale
                os.remove(file_path)
            except Exception as ex:
                print(f"An error has occurred: {ex}")
        else:
            os.makedirs(".\\sounds\\extra",exist_ok=True)
            #copia file da disco
            diagfile = filedialog.askopenfilename()  # Mostra la finestra di dialogo del file
            # Copia il file
            destination = os.path.join(".\\musica", os.path.basename(diagfile))
            shutil.copy(diagfile, destination)
  
    def caricamusica():
            os.makedirs(".\\musica", exist_ok=True)
            musicacollection = ["nessuna musica"]
            musicacollection += [musica for musica in os.listdir(".\musica") if musica.endswith(('.mp3', '.wav'))]
            combomusica.config(values=musicacollection)
                        
            if combomusica.get().strip() == "nessuna musica":
                print ("stop musica")
                try:
                    print("Musica in stopping")
                    pygame.mixer.music.stop()
                except Exception as errorstop:
                        print(f"errore stopping audio mixer: {errorstop}")
            else:
                # Riproduci il suono selezionato
                try:
                    if combomusica.get() != "nessuna musica":
                        print(os.path.abspath(os.path.join('.\musica', combomusica.get())))
                        pygame.mixer.music.load(os.path.join('.\musica', combomusica.get()))
                        pygame.mixer.music.play()
                except Exception as errorplay:
                    print(f"errore playing audio: {errorplay}")
                    
    frame_Musica= ttk.Frame(root)
    frame_Musica.pack(side="bottom", anchor="w")

    combomusica = ttk.Combobox(frame_Musica)
    combomusica.pack(side="top", anchor="w")
    combomusica.bind('<<ComboboxSelected>>', lambda event: caricamusica())

    frameMuscica2= ttk.Frame(frame_Musica)
    frameMuscica2.pack(side="top", anchor="w")

    musica = ttk.Button(frameMuscica2, text="Musica Background")
    musica.pack(side="left", anchor="w")
    musica.config(command=musicabackground)

    def stop():
        pygame.mixer.music.stop()

    musicstop= ttk.Button(frameMuscica2,text= "Stop")
    musicstop.pack(side="left",anchor="w")
    musicstop.config(command=stop)

    

    # Popola la combobox quando il programma viene eseguito
    caricamusica()
    
    frame2 = tk.Frame(root,bg='blue')  # Create a new frame for the canvas and scrollbar
    frame2.pack()
    
    

    canvas = tk.Canvas(frame2, width=500, height=500)
    scrollbar = ttk.Scrollbar(frame2, orient="horizontal", command=canvas.xview)
    scrollbar.pack(side="bottom", fill="x")  # Aggiungi questa linea
    # Crea una barra di scorrimento verticale
    v_scrollbar = ttk.Scrollbar(frame2, orient="vertical", command=canvas.yview)
    v_scrollbar.pack(side="right", fill="y")  # Aggiungi questa linea

    # Configura il tuo canvas per utilizzare la nuova barra di scorrimento verticale
    canvas.configure(yscrollcommand=v_scrollbar.set)
    scrollable_frame = ttk.Frame(canvas)
    scrollable_frame.place(relwidth=1, relheight=1)  # Imposta le dimensioni relative al canvas
    
    

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(xscrollcommand=scrollbar.set)

    photos = []  # List to hold references to PhotoImage objects

    button.config(command=lambda: crea_elementi(root, frame2, canvas, scrollbar, scrollable_frame, photos, text_box.get("1.0", "end-1c")))
    



    root.mainloop()

main()