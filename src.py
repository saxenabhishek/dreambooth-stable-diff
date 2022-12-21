from nbox import operator
import argparse
import os

# os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/Data/hf_home"

from train_dreambooth import run_training
import torch
from PIL import Image
from huggingface_hub import snapshot_download

from time import time
from nbox import RelicsNBX

maximum_concepts = 3

"""
v1-5
v2-1-768
v2-1-512
"""


def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        new_image.paste(image, (0, (w - h) // 2))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        new_image.paste(image, ((h - w) // 2, 0))
        return new_image


def get_files(dir):
    files = []
    for file in os.listdir(dir):
        files.append(os.path.join(dir, file))
    return files


@operator()
def main():
    torch.cuda.empty_cache()

    which_model = "v1-5"
    model_v1_5 = snapshot_download(
        repo_id="multimodalart/sd-fine-tunable",
    )
    model_to_load = model_v1_5
    resolution = 512  # if which_model != "v2-1-768" else 768

    os.makedirs("instance_images", exist_ok=True)
    file_counter = 0

    files = get_files("./data")

    prompt = "chainsawman"
    if prompt == "" or prompt is None:
        raise ValueError("You forgot to define your concept prompt")

    for j, file_temp in enumerate(files):
        file = Image.open(file_temp)
        image = pad_image(file)
        image = image.resize((resolution, resolution))
        # extension = file_temp.name.split(".")[1]
        image = image.convert("RGB")
        image.save(f"instance_images/{prompt}_({j+1}).jpg", format="JPEG", quality=100)
        file_counter += 1

    os.makedirs("output_model", exist_ok=True)

    Train_text_encoder_for = 15  # 30 for object, 70 for person, 15 for style

    Training_Steps = file_counter * 150

    # if type_of_thing == "person" and Training_Steps > 2600:
    #     Training_Steps = 2600  # Avoid overfitting on people's faces

    stptxt = int((Training_Steps * Train_text_encoder_for) / 100)

    gradient_checkpointing = True if (which_model != "v1-5") else False

    cache_latents = True if which_model != "v1-5" else False

    args_general = argparse.Namespace(
        image_captions_filename=True,
        train_text_encoder=True if stptxt > 0 else False,
        stop_text_encoder_training=stptxt,
        save_n_steps=0,
        pretrained_model_name_or_path=model_to_load,
        instance_data_dir="instance_images",
        class_data_dir=None,
        output_dir="output_model",
        instance_prompt="",
        seed=42,
        resolution=resolution,
        mixed_precision="fp16",
        train_batch_size=1,
        gradient_accumulation_steps=1,
        use_8bit_adam=True,
        learning_rate=2e-6,
        lr_scheduler="polynomial",
        lr_warmup_steps=0,
        max_train_steps=Training_Steps,
        gradient_checkpointing=gradient_checkpointing,
        cache_latents=cache_latents,
    )
    print("Starting single training...")

    run_training(args_general)
    print("DONE")

    print("uploading to relics")
    relic = RelicsNBX("dreambooth", create=True)

    relic.put_to("./output_model/model.ckpt", f"output/{time()}_{prompt}.ckpt")


if __name__ == "__main__":
    main()
