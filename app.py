import gradio as gr
import os
from pathlib import Path
import argparse
import shutil
from train_dreambooth import run_training
from convertosd import convert
from PIL import Image
from slugify import slugify
import requests
import torch
import zipfile
import urllib.parse
import gc
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download

css = '''
    .instruction{position: absolute; top: 0;right: 0;margin-top: 0px !important}
    .arrow{position: absolute;top: 0;right: -110px;margin-top: -8px !important}
    #component-4, #component-3, #component-10{min-height: 0}
    .duplicate-button img{margin: 0}
'''
maximum_concepts = 3

#Pre download the files even if we don't use it here
model_to_load = snapshot_download(repo_id="multimodalart/sd-fine-tunable")
safety_checker = snapshot_download(repo_id="multimodalart/sd-sc")

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))

def swap_text(option):
    mandatory_liability = "You must have the right to do so and you are liable for the images you use, example:"
    if(option == "object"):
        instance_prompt_example = "cttoy"
        freeze_for = 50
        return [f"You are going to train `object`(s), upload 5-10 images of each object you are planning on training on from different angles/perspectives. {mandatory_liability}:", '''<img src="file/cat-toy.png" />''', f"You should name your concept with a unique made up word that has low chance of the model already knowing it (e.g.: `{instance_prompt_example}` here). Images will be automatically cropped to 512x512.", freeze_for]
    elif(option == "person"):
       instance_prompt_example = "julcto"
       freeze_for = 100
       return [f"You are going to train a `person`(s), upload 10-20 images of each person you are planning on training on from different angles/perspectives. {mandatory_liability}:", '''<img src="file/person.png" />''', f"You should name the files with a unique word that represent your concept (e.g.: `{instance_prompt_example}` here). Images will be automatically cropped to 512x512.", freeze_for]
    elif(option == "style"):
        instance_prompt_example = "trsldamrl"
        freeze_for = 10
        return [f"You are going to train a `style`, upload 10-20 images of the style you are planning on training on. Name the files with the words you would like  {mandatory_liability}:", '''<img src="file/trsl_style.png" />''', f"You should name your files with a unique word that represent your concept (e.g.: `{instance_prompt_example}` here). Images will be automatically cropped to 512x512.", freeze_for]

def count_files(*inputs):
    file_counter = 0
    concept_counter = 0
    for i, input in enumerate(inputs):
        if(i < maximum_concepts-1):
            files = inputs[i]
            if(files):
                concept_counter+=1
                file_counter+=len(files)
    uses_custom = inputs[-1] 
    type_of_thing = inputs[-4]
    if(uses_custom):
        Training_Steps = int(inputs[-3])
    else:
        if(type_of_thing == "person"):
            Training_Steps = file_counter*200*2
        else:
            Training_Steps = file_counter*200
    return([gr.update(visible=True), gr.update(visible=True, value=f"You are going to train {concept_counter} {type_of_thing}(s), with {file_counter} images for {Training_Steps} steps. This should take around {round(Training_Steps/1.5, 2)} seconds, or {round((Training_Steps/1.1)/3600, 2)} hours. As a reminder, the T4 GPU costs US$0.60 for 1h. Once training is over, don't forget to swap the hardware back to CPU.")])

def train(*inputs):
    torch.cuda.empty_cache()
    if 'pipe' in globals():
        del pipe
        gc.collect()

    if "IS_SHARED_UI" in os.environ:
        raise gr.Error("This Space only works in duplicated instances")
    
    if os.path.exists("output_model"): shutil.rmtree('output_model')
    if os.path.exists("instance_images"): shutil.rmtree('instance_images')
    if os.path.exists("diffusers_model.zip"): os.remove("diffusers_model.zip")
    if os.path.exists("model.ckpt"): os.remove("model.ckpt")
    if os.path.exists("hastrained.success"): os.remove("hastrained.success")
    file_counter = 0
    for i, input in enumerate(inputs):
        if(i < maximum_concepts-1):
            if(input):
                os.makedirs('instance_images',exist_ok=True)
                files = inputs[i+(maximum_concepts*2)]
                prompt = inputs[i+maximum_concepts]
                if(prompt == "" or prompt == None):
                    raise gr.Error("You forgot to define your concept prompt")
                for j, file_temp in enumerate(files):
                    file = Image.open(file_temp.name)
                    width, height = file.size
                    side_length = min(width, height)
                    left = (width - side_length)/2
                    top = (height - side_length)/2
                    right = (width + side_length)/2
                    bottom = (height + side_length)/2
                    image = file.crop((left, top, right, bottom))
                    image = image.resize((512, 512))
                    extension = file_temp.name.split(".")[1]
                    image = image.convert('RGB')
                    image.save(f'instance_images/{prompt}_({j+1}).jpg', format="JPEG", quality = 100)
                    file_counter += 1
    
    os.makedirs('output_model',exist_ok=True)
    uses_custom = inputs[-1] 
    type_of_thing = inputs[-4]
    model_name = inputs[-7]
    remove_attribution_after = inputs[-6]
    hf_token = inputs[-5]
    if(uses_custom):
        Training_Steps = int(inputs[-3])
        Train_text_encoder_for = int(inputs[-2])
    else:
        Training_Steps = file_counter*200
        if(type_of_thing == "object"):
            Train_text_encoder_for=30
        elif(type_of_thing == "person"):
            Train_text_encoder_for=60
        elif(type_of_thing == "style"):
            Train_text_encoder_for=15
    
    class_data_dir = None
    stptxt = int((Training_Steps*Train_text_encoder_for)/100)
    args_general = argparse.Namespace(
                image_captions_filename = True,
                train_text_encoder = True,
                stop_text_encoder_training = stptxt,
                save_n_steps = 0,
                pretrained_model_name_or_path = model_to_load,
                instance_data_dir="instance_images",
                class_data_dir=class_data_dir,
                output_dir="output_model",
                instance_prompt="",
                seed=42,
                resolution=512,
                mixed_precision="fp16",
                train_batch_size=1,
                gradient_accumulation_steps=1,
                use_8bit_adam=True,
                learning_rate=2e-6,
                lr_scheduler="polynomial",
                lr_warmup_steps = 0,
                max_train_steps=Training_Steps,     
    )
    print("Starting training...")
    lock_file = open("intraining.lock", "w")
    lock_file.close()
    run_training(args_general)
    gc.collect()
    torch.cuda.empty_cache()
    print("Adding Safety Checker to the model...")
    shutil.copytree(f"{safety_checker}/feature_extractor", "output_model/feature_extractor")
    shutil.copytree(f"{safety_checker}/safety_checker", "output_model/safety_checker")
    shutil.copy(f"model_index.json", "output_model/model_index.json")
    print("Zipping model file...")
    with zipfile.ZipFile('diffusers_model.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir('output_model/', zipf)
    print("Training completed!")
    if os.path.exists("intraining.lock"): os.remove("intraining.lock")
    trained_file = open("hastrained.success", "w")
    trained_file.close()
    if(remove_attribution_after):
        push(model_name, "My personal profile", hf_token, True)
        hardware_url = f"https://huggingface.co/spaces/{os.environ['SPACE_ID']}/hardware"
        headers = { "authorization" : f"Bearer {hf_token}"}
        body = {'flavor': 'cpu-basic'}
        requests.post(hardware_url, json = body, headers=headers)
    return [
        gr.update(visible=True, value=["diffusers_model.zip"]), #result
        gr.update(visible=True), #try_your_model
        gr.update(visible=True), #push_to_hub
        gr.update(visible=True), #convert_button
        gr.update(visible=False), #training_ongoing
        gr.update(visible=True) #completed_training
    ]

def generate(prompt):
    torch.cuda.empty_cache()
    from diffusers import StableDiffusionPipeline
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained("./output_model", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(prompt).images[0]  
    return(image)
    
def push(model_name, where_to_upload, hf_token, comes_from_automated=False):
    if(not os.path.exists("model.ckpt")):
        convert("output_model", "model.ckpt")
    from huggingface_hub import HfApi, HfFolder, CommitOperationAdd
    from huggingface_hub import create_repo
    model_name_slug = slugify(model_name)
    api = HfApi()
    your_username = api.whoami(token=hf_token)["name"]
    if(where_to_upload == "My personal profile"):    
        model_id = f"{your_username}/{model_name_slug}"
    else:
        model_id = f"sd-dreambooth-library/{model_name_slug}"
        headers = {"Authorization" : f"Bearer: {hf_token}", "Content-Type": "application/json"}
        response = requests.post("https://huggingface.co/organizations/sd-dreambooth-library/share/SSeOwppVCscfTEzFGQaqpfcjukVeNrKNHX", headers=headers)
    
    images_upload = os.listdir("instance_images")
    image_string = ""
    instance_prompt_list = []
    previous_instance_prompt = ''
    for i, image in enumerate(images_upload):
        instance_prompt = image.split("_")[0]
        if(instance_prompt != previous_instance_prompt):
            title_instance_prompt_string = instance_prompt
            instance_prompt_list.append(instance_prompt)
        else:
            title_instance_prompt_string = ''
        previous_instance_prompt = instance_prompt
        image_string = f'''{title_instance_prompt_string} (use that on your prompt)
{image_string}![{instance_prompt} {i}](https://huggingface.co/{model_id}/resolve/main/concept_images/{urllib.parse.quote(image)})'''
    readme_text = f'''---
license: creativeml-openrail-m
tags:
- text-to-image
---
### {model_name} Dreambooth model trained by {api.whoami(token=hf_token)["name"]} with [Hugging Face Dreambooth Training Space](https://huggingface.co/spaces/multimodalart/dreambooth-training)

You run your new concept via `diffusers` [Colab Notebook for Inference](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_inference.ipynb). Don't forget to use the concept prompts! 

Sample pictures of:
{image_string}
'''
    #Save the readme to a file
    readme_file = open("model.README.md", "w")
    readme_file.write(readme_text)
    readme_file.close()
    #Save the token identifier to a file
    text_file = open("token_identifier.txt", "w")
    text_file.write(', '.join(instance_prompt_list))
    text_file.close()
    create_repo(model_id,private=True, token=hf_token)
    operations = [
        CommitOperationAdd(path_in_repo="token_identifier.txt", path_or_fileobj="token_identifier.txt"),
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj="model.README.md"),
        CommitOperationAdd(path_in_repo=f"model.ckpt",path_or_fileobj="model.ckpt")
    ]
    api.create_commit(
    repo_id=model_id,
    operations=operations,
    commit_message=f"Upload the model {model_name}",
    token=hf_token
    )
    api.upload_folder(
    folder_path="output_model",
    repo_id=model_id,
    token=hf_token
    )
    api.upload_folder(
    folder_path="instance_images",
    path_in_repo="concept_images",
    repo_id=model_id,
    token=hf_token
    )
    if(not comes_from_automated):
        extra_message = "Don't forget to remove the GPU attribution after you play with it."
    else:
        extra_message = "The GPU has been removed automatically as requested, and you can try the model via the model page"
    api.create_discussion(repo_id=os.environ['SPACE_ID'], title=f"Your model {model_name} has finished trained from the Dreambooth Train Spaces!", description=f"Your model has been successfully uploaded to: https://huggingface.co/{model_id}. {extra_message}",token=hf_token)

    return [gr.update(visible=True, value=f"Successfully uploaded your model. Access it [here](https://huggingface.co/{model_id})"), gr.update(visible=True, value=["diffusers_model.zip", "model.ckpt"])]

def convert_to_ckpt():
    convert("output_model", "model.ckpt")
    return gr.update(visible=True, value=["diffusers_model.zip", "model.ckpt"])

def check_status():
    if os.path.exists("hastrained.success"):
        update_top_tag = gr.Update(value=f'''
        <div class="gr-prose" style="max-width: 80%">
            <h2>Your model has finished training ✅</h2>
            <p>Yay, congratulations on training your model. Scroll down to play with with it, save it (either downloading it or on the Hugging Face Hub). Once you are done, your model is safe, and you don't want to train a new one, go to the <a href="https://huggingface.co/spaces/{os.environ['SPACE_ID']}">settings page</a> and downgrade your Space to a CPU Basic</p> 
        </div>
        ''')
        show_outputs = True
    elif os.path.exists("intraining.lock"):
        update_top_tag = gr.Update(value='''
        <div class="gr-prose" style="max-width: 80%">
            <h2>Don't worry, your model is still training! ⌛</h2>
            <p>You closed the tab while your model was training, but it's all good! It is still training right now. You can click the "Open logs" button above here to check the training status. Once training is done, reload this tab to interact with your model</p> 
        </div>
        ''')
        show_outputs = False
    return [
        update_top_tag, #top_description
        gr.update(visible=show_outputs), #try_your_model
        gr.update(visible=show_outputs), #push_to_hub
        gr.update(visible=show_outputs, value=["diffusers_model.zip"]), #result
        gr.update(visible=show_outputs), #convert_button
    ]

def checkbox_swap(checkbox):
    reverse_bool = not checkbox
    return [gr.update(visible=reverse_bool), gr.update(visible=reverse_bool), gr.update(visible=reverse_bool)]

with gr.Blocks(css=css) as demo:
    with gr.Box():
        if "IS_SHARED_UI" in os.environ:
            top_description = gr.HTML(f'''
                <div class="gr-prose" style="max-width: 80%">
                <h2>Attention - This Space doesn't work in this shared UI</h2>
                <p>For it to work, you have to duplicate the Space and run it on your own profile using a (paid) private T4 GPU for training. As each T4 costs US$0.60/h, it should cost < US$1 to train a model with less than 100 images using default settings!&nbsp;&nbsp;<a class="duplicate-button" style="display:inline-block" href="https://huggingface.co/spaces/{os.environ['SPACE_ID']}?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></p>
                <img class="instruction" src="file/duplicate.png"> 
                <img class="arrow" src="file/arrow.png" />
                </div>
            ''')
        else:
            top_description = gr.HTML(f'''
                <div class="gr-prose" style="max-width: 80%">
                <h2>You have successfully duplicated the Dreambooth Training Space 🎉</h2>
                <p>If you haven't already, <a href="https://huggingface.co/spaces/{os.environ['SPACE_ID']}/settings">attribute a T4 GPU to it (via the Settings tab)</a> and run the training below. You will be billed by the minute from when you activate the GPU until when you turn it off.</p> 
                </div>
            ''')    
    gr.Markdown("# Dreambooth training")
    gr.Markdown("Customize Stable Diffusion by giving it a few examples. You can train up to three concepts by providing examples for each. This Space is based on TheLastBen's [fast-DreamBooth Colab](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast-DreamBooth.ipynb) with [🧨 diffusers](https://github.com/huggingface/diffusers)")
    
    with gr.Row() as what_are_you_training:
        type_of_thing = gr.Dropdown(label="What would you like to train?", choices=["object", "person", "style"], value="object", interactive=True)

    #Very hacky approach to emulate dynamically created Gradio components   
    with gr.Row() as upload_your_concept:
        with gr.Column():
            thing_description = gr.Markdown("You are going to train an `object`, please upload 5-10 images of the object you are planning on training on from different angles/perspectives. You must have the right to do so and you are liable for the images you use, example:")
            thing_image_example = gr.HTML('''<img src="file/cat-toy.png" />''')
            things_naming = gr.Markdown("You should name your concept with a unique made up word that has low chance of the model already knowing it (e.g.: `cttoy` here). Images will be automatically cropped to 512x512.")
        with gr.Column():
            file_collection = []
            concept_collection = []
            buttons_collection = []
            delete_collection = []
            is_visible = []

            row = [None] * maximum_concepts
            for x in range(maximum_concepts):
                ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])
                if(x == 0):
                    visible = True
                    is_visible.append(gr.State(value=True))
                else:
                    visible = False
                    is_visible.append(gr.State(value=False))

                file_collection.append(gr.File(label=f'''Upload the images for your {ordinal(x+1) if (x>0) else ""} concept''', file_count="multiple", interactive=True, visible=visible))
                with gr.Column(visible=visible) as row[x]:
                    concept_collection.append(gr.Textbox(label=f'''{ordinal(x+1) if (x>0) else ""} concept prompt - use a unique, made up word to avoid collisions'''))  
                    with gr.Row():
                        if(x < maximum_concepts-1):
                            buttons_collection.append(gr.Button(value="Add +1 concept", visible=visible))
                        if(x > 0):
                            delete_collection.append(gr.Button(value=f"Delete {ordinal(x+1)} concept"))
            
            counter_add = 1
            for button in buttons_collection:
                if(counter_add < len(buttons_collection)):
                    button.click(lambda:
                    [gr.update(visible=True),gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), True, None],
                    None, 
                    [row[counter_add], file_collection[counter_add], buttons_collection[counter_add-1], buttons_collection[counter_add], is_visible[counter_add], file_collection[counter_add]], queue=False)
                else:
                    button.click(lambda:[gr.update(visible=True),gr.update(visible=True), gr.update(visible=False), True], None, [row[counter_add], file_collection[counter_add], buttons_collection[counter_add-1], is_visible[counter_add]], queue=False)
                counter_add += 1
            
            counter_delete = 1
            for delete_button in delete_collection:
                if(counter_delete < len(delete_collection)+1):
                    delete_button.click(lambda:[gr.update(visible=False),gr.update(visible=False), gr.update(visible=True), False], None, [file_collection[counter_delete], row[counter_delete], buttons_collection[counter_delete-1], is_visible[counter_delete]], queue=False)
                counter_delete += 1
                  
    with gr.Accordion("Custom Settings", open=False):
        swap_auto_calculated = gr.Checkbox(label="Use custom settings")
        gr.Markdown("If not checked, the number of steps and % of frozen encoder will be tuned automatically according to the amount of images you upload and whether you are training an `object`, `person` or `style` as follows: The number of steps is calculated by number of images uploaded multiplied by 20. The text-encoder is frozen after 10% of the steps for a style, 30% of the steps for an object and is fully trained for persons.")
        steps = gr.Number(label="How many steps", value=800)
        perc_txt_encoder = gr.Number(label="Percentage of the training steps the text-encoder should be trained as well", value=30)

    with gr.Box(visible=False) as training_summary:
        training_summary_text = gr.Textbox("", visible=False, label="Training Summary")
        training_summary_checkbox = gr.Checkbox("Remove GPU After - automatically remove paid GPU attribution and upload model to the Hugging Face Hub after training")
        training_summary_model_name = gr.Textbox(label="Name of your model", visible=False)
        training_summary_token_message = gr.Markdown("[A Hugging Face write access token](https://huggingface.co/settings/tokens), go to \"New token\" -> Role : Write. A regular read token won't work here.", visible=False)            
        training_summary_token = gr.Textbox(label="Hugging Face Write Token", type="password", visible=False)
    
    train_btn = gr.Button("Start Training")
    
    training_ongoing = gr.Markdown("## Training is ongoing ⌛... You can close this tab if you like or just wait. If you did not check `Remove GPU After`, you can come back here to try your model and upload it after training. Don't forget to remove the GPU attribution after you are done. ", visible=False)
    
    #Post-training UI
    completed_training = gr.Markdown('''# ✅ Training completed. 
    ### Don't forget to remove the GPU attribution after you are done trying and uploading your model''', visible=False)
    
    with gr.Row():
        with gr.Box(visible=False) as try_your_model:
            gr.Markdown("## Try your model")
            prompt = gr.Textbox(label="Type your prompt")
            result_image = gr.Image()
            generate_button = gr.Button("Generate Image")
        
        with gr.Box(visible=False) as push_to_hub:
            gr.Markdown("## Push to Hugging Face Hub")
            model_name = gr.Textbox(label="Name of your model", placeholder="Tarsila do Amaral Style")
            where_to_upload = gr.Dropdown(["My personal profile", "Public Library"], label="Upload to")
            gr.Markdown("[A Hugging Face write access token](https://huggingface.co/settings/tokens), go to \"New token\" -> Role : Write. A regular read token won't work here.")
            hf_token = gr.Textbox(label="Hugging Face Write Token", type="password")
            push_button = gr.Button("Push to the Hub")
    
    result = gr.File(label="Download the uploaded models in the diffusers format", visible=True)
    success_message_upload = gr.Markdown(visible=False)
    convert_button = gr.Button("Convert to CKPT", visible=False)
    
    #Swap the examples and the % of text encoder trained depending if it is an object, person or style
    type_of_thing.change(fn=swap_text, inputs=[type_of_thing], outputs=[thing_description, thing_image_example, things_naming, perc_txt_encoder], queue=False)
    
    #Update the summary box below the UI according to how many images are uploaded and whether users are using custom settings or not 
    for file in file_collection:
        file.change(fn=count_files, inputs=file_collection+[type_of_thing]+[steps]+[perc_txt_encoder]+[swap_auto_calculated], outputs=[training_summary, training_summary_text], queue=False)
    steps.change(fn=count_files, inputs=file_collection+[type_of_thing]+[steps]+[perc_txt_encoder]+[swap_auto_calculated], outputs=[training_summary, training_summary_text], queue=False)
    perc_txt_encoder.change(fn=count_files, inputs=file_collection+[type_of_thing]+[steps]+[perc_txt_encoder]+[swap_auto_calculated], outputs=[training_summary, training_summary_text], queue=False)
    
    #Give more options if the user wants to finish everything after training
    training_summary_checkbox.change(fn=checkbox_swap, inputs=training_summary_checkbox, outputs=[training_summary_token_message, training_summary_token, training_summary_model_name])
    #Add a message for while it is in training
    train_btn.click(lambda:gr.Update(visible=True), inputs=None, outputs=training_ongoing)
    
    #The main train function
    train_btn.click(fn=train, inputs=is_visible+concept_collection+file_collection+[training_summary_model_name]+[training_summary_checkbox]+[training_summary_token]+[type_of_thing]+[steps]+[perc_txt_encoder]+[swap_auto_calculated], outputs=[result, try_your_model, push_to_hub, convert_button, training_ongoing, completed_training])
    
    #Button to generate an image from your trained model after training
    generate_button.click(fn=generate, inputs=prompt, outputs=result_image)
    #Button to push the model to the Hugging Face Hub
    push_button.click(fn=push, inputs=[model_name, where_to_upload, hf_token], outputs=[success_message_upload, result])
    #Button to convert the model to ckpt format 
    convert_button.click(fn=convert_to_ckpt, inputs=[], outputs=result)

demo.load(fn=check_status,inputs=None, outputs=[top_description, try_your_model, push_to_hub, result, convert_button])
demo.launch(debug=True)