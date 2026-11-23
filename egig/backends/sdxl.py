"""SDXL + IP-Adapter (appearance from reference photos) + ControlNet canny (structure).

Memory plan for a 16 GB T4 (fp16): UNet ~5.1 GB + ControlNet ~2.5 GB stay on the GPU while
denoising; text encoders, the IP-Adapter image encoder and the VAE are moved to the GPU only
when needed (`enable_model_cpu_offload`).
"""

from __future__ import annotations

from typing import Any, Optional

from PIL import Image

from ..config import Settings
from . import GenerationJob, estimate_tokens


class SDXLBackend:
    name = "sdxl"

    def __init__(self, settings: Settings, pipe: Any = None, device: Optional[str] = None):
        self.s = settings
        self.pipe = pipe  # tests inject a tiny pipeline here
        self.device = device
        self._tokenizer = None

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        if self.pipe is None:
            self.pipe = self._build_pipeline()

    def _build_pipeline(self) -> Any:
        import torch
        from diffusers import (AutoencoderKL, ControlNetModel, DPMSolverMultistepScheduler,
                               StableDiffusionXLControlNetPipeline)
        from transformers import CLIPVisionModelWithProjection

        cuda = torch.cuda.is_available()
        dtype = torch.float16 if cuda else torch.float32
        variant = "fp16" if cuda else None
        s = self.s
        print("[sdxl] loading models (first run downloads ~14 GB, later runs use the cache)...")

        try:
            controlnet = ControlNetModel.from_pretrained(s.controlnet_model, torch_dtype=dtype, variant=variant)
        except (OSError, ValueError):  # repo without an fp16 variant
            controlnet = ControlNetModel.from_pretrained(s.controlnet_model, torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained(s.vae_model, torch_dtype=dtype)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            s.ip_adapter_repo, subfolder=s.ip_image_encoder_subfolder, torch_dtype=dtype
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            s.sdxl_model, controlnet=controlnet, vae=vae, image_encoder=image_encoder,
            torch_dtype=dtype, variant=variant, use_safetensors=True,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        pipe.load_ip_adapter(s.ip_adapter_repo, subfolder=s.ip_adapter_subfolder, weight_name=s.ip_adapter_weight)

        if cuda and s.cpu_offload:
            pipe.enable_model_cpu_offload(gpu_id=_gpu_index(self.device))
        else:
            pipe.to(self.device or ("cuda:0" if cuda else "cpu"))
        print("[sdxl] ready")
        return pipe

    # ------------------------------------------------------------------- tokens
    def count_tokens(self, text: str) -> int:
        """Exact CLIP token count (SDXL reads at most 77 tokens incl. start/end)."""
        tok = self.pipe.tokenizer if self.pipe is not None else self._load_tokenizer()
        if tok is None:
            return estimate_tokens(text)
        return len(tok(text, add_special_tokens=False).input_ids)

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is None:
            try:
                from transformers import CLIPTokenizer

                self._tokenizer = CLIPTokenizer.from_pretrained(self.s.sdxl_model, subfolder="tokenizer")
            except Exception as err:  # offline etc. -> fall back to the estimate
                print(f"[sdxl] tokenizer unavailable ({err}); estimating token counts")
                self._tokenizer = False
        return self._tokenizer or None

    # --------------------------------------------------------------- generation
    def generate(self, job: GenerationJob) -> Image.Image:
        import torch
        from diffusers.image_processor import IPAdapterMaskProcessor

        self.load()
        pipe = self.pipe
        extra: dict = {}
        if job.ip_images:
            # One IP-Adapter, several reference images, each limited to its own mask region.
            pipe.set_ip_adapter_scale([list(job.ip_scales)])
            masks = IPAdapterMaskProcessor().preprocess(job.ip_masks, height=job.height, width=job.width)
            masks = masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])
            extra["ip_adapter_image"] = [list(job.ip_images)]
            extra["cross_attention_kwargs"] = {"ip_adapter_masks": [masks]}
        else:
            # IP-Adapter is loaded, so it needs an image; scale 0 switches it off.
            pipe.set_ip_adapter_scale(0.0)
            extra["ip_adapter_image"] = [Image.new("RGB", (224, 224), (128, 128, 128))]

        if job.control_image is not None:
            control, control_scale = job.control_image, job.control_scale
        else:  # nothing rigid to trace: black image + scale 0 = ControlNet off
            control, control_scale = Image.new("RGB", (job.width, job.height)), 0.0

        result = pipe(
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            image=control,
            controlnet_conditioning_scale=control_scale,
            control_guidance_end=job.control_end,
            num_inference_steps=job.steps,
            guidance_scale=job.guidance,
            height=job.height,
            width=job.width,
            generator=torch.Generator(device="cpu").manual_seed(job.seed),
            **extra,
        )
        return result.images[0]


def _gpu_index(device: Optional[str]) -> int:
    if device and device.startswith("cuda:"):
        return int(device.split(":")[1])
    return 0
