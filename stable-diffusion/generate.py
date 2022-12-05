import os

import torch
import fire
import time
from typing import Optional


class Timer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        print(self.name)
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        dt = self.end-self.start
        for unit in ["s", "ms", "us", "ns"]:
            if dt < 1:
                dt *= 1000
            else:
                break

        if unit == "s" and dt >= 60:
            dt /= 60
            unit = "mins"

        print(f"Took {dt:.03f} {unit}")


def get_new_filename(prefix: str, ext: str) -> str:
    for i in range(1000):
        filename = f"{prefix}_{i:03d}.{ext}"
        if not os.path.exists(filename):
            return filename
    
    raise ValueError("Overflow!")


def main(
    prompt: str="a photo of an astronaut riding a horse on mars",
    outdir: str="generated",
    outfile: Optional[str]=None,
    outext: str="png",
    **kwargs,
):
    """
    kwargs: extra args to pass to pipe(prompt, **kwargs). defaults:
        --guidance-scale=7.5 \
        --num-inference-steps=10 \
        --height=512 \
        --width=512
    """
    t0 = time.time()
    with Timer("Importing StableDiffusionPipeline..."):
        from diffusers import StableDiffusionPipeline

    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cpu"

    if outfile is None:
        outfile = prompt.replace(" ", "_")

    with Timer(f"Setting up pipe for {model_id}..."):
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe = pipe.to(device)
    
    with Timer(f"Generating image from prompt {prompt}"):
        print(f"passing extra model kwargs: {kwargs}")
        result = pipe(prompt, **kwargs)

    print(result)

    for image in result.images:
        filename = get_new_filename(
            prefix=f"{outdir}/{outfile}",
            ext=outext,
        )
        with Timer(f"Saving image to {filename}..."):
            image.save(filename)

    print(f"Entire script took {time.time()-t0} s")


if __name__ == '__main__':
    fire.Fire(main)

