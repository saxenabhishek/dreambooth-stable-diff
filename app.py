import gradio as gr
import os
from pathlib import Path
import argparse
import shutil
from train_dreambooth import run_training
from convertosd import convert
from PIL import Image
import torch

css = '''
    .instruction{position: absolute; top: 0;right: 0;margin-top: 0px !important}
    .arrow{position: absolute;top: 0;right: -8px;margin-top: -8px !important}
    #component-4, #component-3, #component-10{min-height: 0}
'''
shutil.unpack_archive("mix.zip", "mix")
model_to_load = "multimodalart/sd-fine-tunable"
maximum_concepts = 3

def swap_text(option):
    mandatory_liability = "You must have the right to do so and you are liable for the images you use"
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
    for i, input in enumerate(inputs):
        if(i < maximum_concepts-1):
            if(input):
                files = inputs[i+(maximum_concepts*2)]
                for j, tile_temp in enumerate(files):
                    file_counter+= 1
    uses_custom = inputs[-1] 
    type_of_thing = inputs[-4]
    if(uses_custom):
        Training_Steps = int(inputs[-3])
    else:
        if(type_of_thing == "person"):
            Training_Steps = file_counter*200*2
        else:
            Training_Steps = file_counter*200
    return(gr.update(visible=True, value=f"You are going to train {file_counter} files for {Training_Steps} steps. This should take around {round(Training_Steps/1.5, 2)} seconds, or {round((Training_Steps/1.5)/3600, 2)}. The T4 GPU costs US$0.60 for 1h, so the estimated costs for this training run should be {round(((Training_Steps/1.5)/3600)*0.6, 2)}"))
def train(*inputs):

    if os.path.exists("diffusers_model.zip"): os.remove("diffusers_model.zip")
    if os.path.exists("model.ckpt"): os.remove("model.ckpt")
    file_counter = 0
    for i, input in enumerate(inputs):
        if(i < maximum_concepts-1):
            if(input):
                os.makedirs('instance_images',exist_ok=True)
                files = inputs[i+(maximum_concepts*2)]
                prompt = inputs[i+maximum_concepts]
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
    if(uses_custom):
        Training_Steps = int(inputs[-3])
        Train_text_encoder_for = int(inputs[-2])
    else:
        Training_Steps = file_counter*200
    if(type_of_thing == "person"):
            class_data_dir = "mix"
            Train_text_encoder_for=100
            args_txt_encoder = argparse.Namespace(
                image_captions_filename = True,
                train_text_encoder = True,
                pretrained_model_name_or_path=model_to_load,
                instance_data_dir="instance_images",
                class_data_dir=class_data_dir,
                output_dir="output_model",
                with_prior_preservation=True,
                prior_loss_weight=1.0,
                instance_prompt="",
                seed=42,
                resolution=512,
                mixed_precision="fp16",
                train_batch_size=1,
                gradient_accumulation_steps=1,
                gradient_checkpointing=True,
                use_8bit_adam=True,
                learning_rate=2e-6,
                lr_scheduler="polynomial",
                lr_warmup_steps=0,
                max_train_steps=Training_Steps,
                num_class_images=200
            )
            args_unet = argparse.Namespace(
                image_captions_filename = True,
                train_only_unet=True,
                Session_dir="output_model",
                save_starting_step=0,
                save_n_steps=0,
                pretrained_model_name_or_path=model_to_load,
                instance_data_dir="instance_images",
                output_dir="output_model",
                instance_prompt="",
                seed=42,
                resolution=512,
                mixed_precision="fp16",
                train_batch_size=1,
                gradient_accumulation_steps=1,
                gradient_checkpointing=False,
                use_8bit_adam=True,
                learning_rate=2e-6,
                lr_scheduler="polynomial",
                lr_warmup_steps=0,
                max_train_steps=Training_Steps
            )
            run_training(args_txt_encoder)
            run_training(args_unet)
    elif(type_of_thing == "object" or type_of_thing == "style"):
            if(type_of_thing == "object"):
                Train_text_encoder_for=30
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
            run_training(args_general)
    torch.cuda.empty_cache()
    #convert("output_model", "model.ckpt")
    shutil.rmtree('instance_images')
    shutil.make_archive("diffusers_model", 'zip', "output_model")
    torch.cuda.empty_cache()
    return [gr.update(visible=True, value=["diffusers_model.zip"]), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)]

def generate(prompt):
    from diffusers import StableDiffusionPipeline
    
    pipe = StableDiffusionPipeline.from_pretrained("./output_model", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(prompt).images[0]  
    return(image)
    
def push(path):
    pass

def convert_to_ckpt():
    convert("output_model", "model.ckpt")
    return gr.update(visible=True, value=["diffusers_model.zip", "model.ckpt"])
   
with gr.Blocks(css=css) as demo:
    with gr.Box():
        if "IS_SHARED_UI" in os.environ:
            gr.HTML('''
                <div class="gr-prose" style="max-width: 80%">
                <h2>Attention - This Space doesn't work in this shared UI</h2>
                <p>For it to work, you have to duplicate the Space and run it on your own profile where a (paid) private GPU will be attributed to it during runtime. It will cost you < US$1 to train a model on default settings! 🤑</p> 
                <img class="instruction" src="file/duplicate.png"> 
                <img class="arrow" src="file/arrow.png" />
                </div>
            ''')
        else:
            gr.HTML('''
                <div class="gr-prose" style="max-width: 80%">
                <h2>You have successfully cloned the Dreambooth Training Space</h2>
                <p><a href="#">Now you can attribute a T4 GPU to it</a> (by going to the Settings tab) and run the training below. The GPU will be automatically unassigned after training is over. So you will be billed by the minute between when you activate the GPU and when it finishes training.</p> 
                </div>
            ''')    
    gr.Markdown("# Dreambooth training")
    gr.Markdown("Customize Stable Diffusion by giving it with few-shot examples")
    with gr.Row():
        type_of_thing = gr.Dropdown(label="What would you like to train?", choices=["object", "person", "style"], value="object", interactive=True)
       
    with gr.Row():
        with gr.Column():
            thing_description = gr.Markdown("You are going to train an `object`, upload 5-10 images of the object you are planning on training on from different angles/perspectives. You must have the right to do so and you are liable for the images you use")
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

                file_collection.append(gr.File(label=f"Upload the images for your {ordinal(x+1)} concept", file_count="multiple", interactive=True, visible=visible))
                with gr.Column(visible=visible) as row[x]:
                    concept_collection.append(gr.Textbox(label=f"{ordinal(x+1)} concept prompt - use a unique, made up word to avoid collisions"))  
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
                    [row[counter_add], file_collection[counter_add], buttons_collection[counter_add-1], buttons_collection[counter_add], is_visible[counter_add], file_collection[counter_add]])
                else:
                    button.click(lambda:[gr.update(visible=True),gr.update(visible=True), gr.update(visible=False), True], None, [row[counter_add], file_collection[counter_add], buttons_collection[counter_add-1], is_visible[counter_add]])
                counter_add += 1
            
            counter_delete = 1
            for delete_button in delete_collection:
                if(counter_delete < len(delete_collection)+1):
                    delete_button.click(lambda:[gr.update(visible=False),gr.update(visible=False), gr.update(visible=True), False], None, [file_collection[counter_delete], row[counter_delete], buttons_collection[counter_delete-1], is_visible[counter_delete]])
                counter_delete += 1
            
            
            
    with gr.Accordion("Advanced Settings", open=False):
        swap_auto_calculated = gr.Checkbox(label="Use these advanced setting")
        gr.Markdown("If not checked, the number of steps and % of frozen encoder will be tuned automatically according to the amount of images you upload and whether you are training an `object`, `person` or `style`.")
        steps = gr.Number(label="How many steps", value=800)
        perc_txt_encoder = gr.Number(label="Percentage of the training steps the text-encoder should be trained as well", value=30)

    type_of_thing.change(fn=swap_text, inputs=[type_of_thing], outputs=[thing_description, thing_image_example, things_naming, perc_txt_encoder], queue=False)
    training_summary = gr.Textbox("", visible=False, label="Training Summary")
    for file in file_collection:
        file.change(fn=count_files, inputs=file_collection+[type_of_thing]+[steps]+[perc_txt_encoder]+[swap_auto_calculated], outputs=[training_summary, training_summary])
    train_btn = gr.Button("Start Training")
    with gr.Box(visible=False) as try_your_model:
        gr.Markdown("Try your model")
        with gr.Row():
            prompt = gr.Textbox(label="Type your prompt")
            result = gr.Image()
        generate_button = gr.Button("Generate Image")
    with gr.Box(visible=False) as push_to_hub:
        gr.Markdown("Push to Hugging Face Hub")
        model_repo_tag = gr.Textbox(label="Model name or URL", placeholder="username/model_name")
        push_button = gr.Button("Push to the Hub")
    result = gr.File(label="Download the uploaded models in the diffusers format", visible=True)
    convert_button = gr.Button("Convert to CKPT", visible=False)

    train_btn.click(fn=train, inputs=is_visible+concept_collection+file_collection+[type_of_thing]+[steps]+[perc_txt_encoder]+[swap_auto_calculated], outputs=[result, try_your_model, push_to_hub, convert_button])
    generate_button.click(fn=generate, inputs=prompt, outputs=result)
    push_button.click(fn=push, inputs=model_repo_tag, outputs=[])
    convert_button.click(fn=convert_to_ckpt, inputs=[], outputs=result)
demo.launch()