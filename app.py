import gradio as gr
import os
from pathlib import Path
import argparse
import shutil
from train_dreambooth import run_training

css = '''
    .instruction{position: absolute; top: 0;right: 0;margin-top: 0px !important}
    .arrow{position: absolute;top: 0;right: -8px;margin-top: -8px !important}
    #component-4, #component-3, #component-10{min-height: 0}
'''
shutil.unpack_archive("mix.zip", "mix")
maximum_concepts = 3
def swap_values_files(*total_files):
    file_counter = 0
    for files in total_files:
        if(files):
            for file in files:
                filename = Path(file.orig_name).stem
                pt=''.join([i for i in filename if not i.isdigit()])
                pt=pt.replace("_"," ")
                pt=pt.replace("(","")
                pt=pt.replace(")","")
                instance_prompt = pt
                print(instance_prompt)
                file_counter += 1
            training_steps = (file_counter*200)
    return training_steps

def swap_text(option):
    mandatory_liability = "You must have the right to do so and you are liable for the images you use"
    if(option == "object"):
        instance_prompt_example = "cttoy"
        freeze_for = 50
        return [f"You are going to train `object`(s), upload 5-10 images of each object you are planning on training on from different angles/perspectives. {mandatory_liability}:", '''<img src="file/cat-toy.png" />''', f"You should name your concept with a unique made up word that has low chance of the model already knowing it (e.g.: `{instance_prompt_example}` here)", freeze_for]
    elif(option == "person"):
       instance_prompt_example = "julcto"
       freeze_for = 100
       return [f"You are going to train a `person`(s), upload 10-20 images of each person you are planning on training on from different angles/perspectives. {mandatory_liability}:", '''<img src="file/cat-toy.png" />''', f"You should name the files with a unique word that represent your concept (like `{instance_prompt_example}` in this example). You can train multiple concepts as well.", freeze_for]
    elif(option == "style"):
        instance_prompt_example = "mspolstyll"
        freeze_for = 10
        return [f"You are going to train a `style`, upload 10-20 images of the style you are planning on training on. Name the files with the words you would like  {mandatory_liability}:", '''<img src="file/cat-toy.png" />''', f"You should name your files with a unique word that represent your concept (as `{instance_prompt_example}` for example). You can train multiple concepts as well.", freeze_for]

def train(*inputs):
    file_counter = 0
    for i, input in enumerate(inputs):
        if(i < maximum_concepts-1):
            if(input):
                os.makedirs('instance_images',exist_ok=True)
                files = inputs[i+(maximum_concepts*2)]
                prompt = inputs[i+maximum_concepts]
                for j, file in enumerate(files):
                    shutil.copy(file.name, f'instance_images/{prompt} ({j+1}).jpg')
                    file_counter += 1
    
    uses_custom = inputs[-1] 
    if(uses_custom):
        Training_Steps = int(inputs[-3])
        Train_text_encoder_for = int(inputs[-2])
        stptxt = int((Training_Steps*Train_text_encoder_for)/100)
    else:
        Training_Steps = file_counter*200
        if(inputs[-4] == "person"):
            class_data_dir = "mix"
            args_txt_encoder = argparse.Namespace(
                image_captions_filename = True,
                train_text_encoder = True,
                pretrained_model_name_or_path="./stable-diffusion-v1-5",
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
        elif(inputs[-4] == "object"):
            class_data_dir = None
        elif(inputs[-4] == "style"):
            class_data_dir = None

    args = argparse.Namespace(
        image_captions_filename = True,
        train_text_encoder = True,
        stop_text_encoder_training = stptxt,
        save_n_steps = 0
        dump_only_text_encoder = True,
        pretrained_model_name_or_path = "./stable-diffusion-v1-5",
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
    run_training(args)
    os.rmdir('instance_images')
with gr.Blocks(css=css) as demo:
    with gr.Box():
        # You can remove this part here for your local clone
        gr.HTML('''
            <div class="gr-prose" style="max-width: 80%">
            <h2>Attention - This Space doesn't work in this shared UI</h2>
            <p>For it to work, you have to duplicate the Space and run it on your own profile where a (paid) private GPU will be attributed to it during runtime. It will cost you < US$1 to train a model on default settings! 🤑</p> 
            <img class="instruction" src="file/duplicate.png"> 
            <img class="arrow" src="file/arrow.png" />
            </div>
        ''')
    gr.Markdown("# Dreambooth training")
    gr.Markdown("Customize Stable Diffusion by giving it with few-shot examples")
    with gr.Row():
        type_of_thing = gr.Dropdown(label="What would you like to train?", choices=["object", "person", "style"], value="object", interactive=True)
        #with gr.Column():
            #with gr.Box():
            #    gr.Textbox(label="What prompt you would like to train it on", value="The photo of a cttoy", interactive=True).style(container=False, item_container=False)
            #    gr.Markdown("You should try using words the model doesn't know. Don't use names or well known concepts.")
    with gr.Row():
        with gr.Column():
            thing_description = gr.Markdown("You are going to train an `object`, upload 5-10 images of the object you are planning on training on from different angles/perspectives. You must have the right to do so and you are liable for the images you use")
            thing_image_example = gr.HTML('''<img src="file/cat-toy.png" />''')
            things_naming = gr.Markdown("For training, you should name the files with a unique word that represent your concept (like `cctoy` in this example). You can train multiple concepts by naming multiple images at once. Images will be automatically cropped to 512x512.")
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
    
    #for file in file_collection:
    #    file.change(fn=swap_values_files, inputs=file_collection, outputs=[steps])

    type_of_thing.change(fn=swap_text, inputs=[type_of_thing], outputs=[thing_description, thing_image_example, things_naming, perc_txt_encoder])
    train_btn = gr.Button("Start Training")
    train_btn.click(fn=train, inputs=is_visible+concept_collection+file_collection+[type_of_thing]+[steps]+[perc_txt_encoder]+[swap_auto_calculated], outputs=[])
demo.launch()